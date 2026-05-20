import os, torch, time
import pandas as pd
from datasets import load_dataset
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, LogitsProcessor, LogitsProcessorList
from tqdm import tqdm

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
NUM_TEST_SAMPLES = 3100
WARMUP_SAMPLES = 30
device = "cuda" if torch.cuda.is_available() else "cpu"

class CUDATimer:
    def __init__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
    def start(self): self.start_event.record()
    def stop(self): self.end_event.record()
    def get_time(self): return self.start_event.elapsed_time(self.end_event) / 1000.0  # elapsed_time auto-syncs

class TTFTLogitsProcessor(LogitsProcessor):
    def __init__(self):
        self.first_token_event = torch.cuda.Event(enable_timing=True)
        self.is_first = True
    def __call__(self, input_ids, scores):
        if self.is_first:
            self.first_token_event.record()
            self.is_first = False
        return scores

print("Loading Qwen E2E Model...")
model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map=device).eval()
processor = AutoProcessor.from_pretrained(MODEL_ID)
dataset = load_dataset("detection-datasets/coco", split=f"val[:{NUM_TEST_SAMPLES}]", trust_remote_code=True)

vision_encoder = getattr(model, 'visual', getattr(getattr(model, 'model', None), 'visual', None))

results = []
print("Starting Qwen E2E Inference...")
with torch.no_grad():
    for idx, data in enumerate(tqdm(dataset)):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        image = data['image'].convert("RGB") if data['image'].mode != "RGB" else data['image']

        # 1b. Image preprocessing: CPU(image_processor) + H2D → wall clock
        _t = time.perf_counter()
        img_in = processor.image_processor(images=image, return_tensors="pt").to(device)
        pixel_values = img_in.pixel_values.to(dtype=torch.bfloat16)
        grid_thw = img_in.image_grid_thw
        torch.cuda.synchronize()  # H2D 전송 완료 대기
        r_img = time.perf_counter() - _t

        # 2a. ViT: vision_encoder.forward() — 이 모델 버전에서 Merger를 포함하지 않음
        # 2b. Merger: 명시적으로 별도 호출하여 CUDA Event로 각각 측정
        ev_vit_start    = torch.cuda.Event(enable_timing=True)
        ev_merger_start = torch.cuda.Event(enable_timing=True)
        ev_enc_end      = torch.cuda.Event(enable_timing=True)

        merger  = getattr(vision_encoder, 'merger', None)
        lm_dim  = model.get_input_embeddings().weight.shape[1]

        ev_vit_start.record()
        v_out = vision_encoder(pixel_values, grid_thw=grid_thw)
        if hasattr(v_out, 'last_hidden_state'):
            v_out = v_out.last_hidden_state
        elif isinstance(v_out, tuple):
            v_out = v_out[0]
        ev_merger_start.record()

        # Merger가 있고 dim이 LLM과 다를 때만 명시적 호출 (이미 포함된 경우 skip)
        if merger is not None and v_out.shape[-1] != lm_dim:
            image_embeds = merger(v_out)
            merger_ran = True
        else:
            image_embeds = v_out
            merger_ran = False
        ev_enc_end.record()
        torch.cuda.synchronize()

        r_vit = ev_vit_start.elapsed_time(ev_merger_start) / 1000.0
        r_mlp = ev_merger_start.elapsed_time(ev_enc_end)   / 1000.0 if merger_ran else 0.0

        N_patches = image_embeds.shape[0]

        # 1a. Text tokenization: CPU(string 구성 + tokenize) → wall clock
        # N_patches 계산 후 프롬프트 구성 (Qwen2-VL 특성상 인코딩 후 텍스트 구성)
        _t = time.perf_counter()
        image_token_str = "<|vision_start|>" + ("<|image_pad|>" * N_patches) + "<|vision_end|>"
        text_prompt = (f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                       f"<|im_start|>user\n{image_token_str}Describe this image in detail.<|im_end|>\n"
                       f"<|im_start|>assistant\n")
        txt_in = processor.tokenizer(text_prompt, return_tensors="pt").to(device)
        torch.cuda.synchronize()
        r_text = time.perf_counter() - _t

        # 3. Fusion: GPU (embedding lookup + in-place image token 교체) → CUDA Event
        t_fus = CUDATimer(); t_fus.start()
        in_embs = model.get_input_embeddings()(txt_in.input_ids)
        image_mask = (txt_in.input_ids == processor.tokenizer.convert_tokens_to_ids("<|image_pad|>"))
        in_embs[image_mask] = image_embeds
        t_fus.stop()
        torch.cuda.synchronize()
        r_fus = t_fus.get_time()

        # 4. Generate: pure GPU → CUDA Event
        hnd = TTFTLogitsProcessor()
        t_gen = CUDATimer(); t_gen.start()
        outs = model.generate(
            inputs_embeds=in_embs, attention_mask=txt_in.attention_mask,
            image_grid_thw=grid_thw, max_new_tokens=64,
            logits_processor=LogitsProcessorList([hnd])
        )
        t_gen.stop()
        torch.cuda.synchronize()

        if idx >= WARMUP_SAMPLES:
            r4 = t_gen.start_event.elapsed_time(hnd.first_token_event) / 1000.0
            true_ttft = r_text + r_img + r_vit + r_mlp + r_fus + r4
            decode = (t_gen.get_time() - r4) if outs.shape[1] > 1 else 0.0
            results.append([r_text, r_img, r_vit, r_mlp, r_fus, r4, true_ttft, decode, true_ttft + decode,
                            torch.cuda.max_memory_allocated() / (1024**3), outs.shape[1]])

df = pd.DataFrame(results, columns=['1a_text','1b_img','2a_vit','2b_mlp','3_fusion','4_gen',
                                     'true_ttft','decode_time','total_latency','vram','tokens'])
df.to_csv("qwen_e2e.csv", index=False)
