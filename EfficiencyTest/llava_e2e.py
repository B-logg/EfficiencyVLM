import os
import torch
import pandas as pd
from datasets import load_dataset
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor, LogitsProcessor, LogitsProcessorList
from tqdm import tqdm

MODEL_ID = "llava-hf/llava-v1.6-vicuna-7b-hf"
NUM_TEST_SAMPLES = 210
WARMUP_SAMPLES = 10
device = "cuda" if torch.cuda.is_available() else "cpu"

class CUDATimer:
    def __init__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
    def start(self): self.start_event.record()
    def stop(self): self.end_event.record()
    def get_time(self): return self.start_event.elapsed_time(self.end_event) / 1000.0

class TTFTLogitsProcessor(LogitsProcessor):
    def __init__(self):
        self.first_token_event = torch.cuda.Event(enable_timing=True)
        self.is_first = True
    def __call__(self, input_ids, scores):
        if self.is_first:
            self.first_token_event.record()
            self.is_first = False
        return scores

print("Loading LLaVA E2E Model...")
model = LlavaNextForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map=device).eval()
processor = LlavaNextProcessor.from_pretrained(MODEL_ID)
# 경고 메시지를 없애기 위해 trust_remote_code는 제거했습니다.
dataset = load_dataset("detection-datasets/coco", split=f"val[:{NUM_TEST_SAMPLES}]")
image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")

results = []
print("Starting LLaVA E2E Inference...")
with torch.no_grad():
    for idx, data in enumerate(tqdm(dataset)):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        
        image = data['image'].convert("RGB") if data['image'].mode != "RGB" else data['image']
        timer_text = CUDATimer(); timer_img = CUDATimer(); timer_vit = CUDATimer(); timer_mlp = CUDATimer(); timer_fusion = CUDATimer(); timer_gen = CUDATimer()
        
        # 1a. Text
        timer_text.start()
        txt_in = processor.tokenizer("USER: <image>\nDescribe this image in detail.\nASSISTANT:", return_tensors="pt").to(device)
        timer_text.stop()
        
        # 1b. Image
        timer_img.start()
        # [에러 해결] text="<image>" 를 추가하여 Processor 내부 버그(NoneType 에러)를 방지합니다.
        img_in = processor(text="<image>", images=image, return_tensors="pt").to(device, torch.bfloat16)
        pixel_values = img_in.pixel_values
        if pixel_values.dim() == 5:
            b, num_patches, c, h, w = pixel_values.shape
            pixel_values = pixel_values.view(b * num_patches, c, h, w)
        timer_img.stop()
        
        # 2a. ViT
        timer_vit.start()
        vision_tower = getattr(model, 'vision_tower', getattr(getattr(model, 'model', None), 'vision_tower', None))
        v_out = vision_tower(pixel_values, output_hidden_states=True)
        timer_vit.stop()
        
        # 2b. Projector
        timer_mlp.start()
        multi_modal_projector = getattr(model, 'multi_modal_projector', getattr(getattr(model, 'model', None), 'multi_modal_projector', None))
        img_embs = multi_modal_projector(v_out.hidden_states[-2])
        if img_embs.dim() == 4: img_embs = img_embs.flatten(1, 2) 
        if img_embs.dim() == 3 and img_embs.shape[0] != 1: img_embs = img_embs.view(1, -1, img_embs.shape[-1])
        timer_mlp.stop()
        
        # 3. Fusion
        timer_fusion.start()
        in_embs = model.get_input_embeddings()(txt_in.input_ids)
        idx_img = torch.where(txt_in.input_ids == image_token_id)[1][0]
        f_embs = torch.cat([in_embs[:, :idx_img, :], img_embs, in_embs[:, idx_img+1:, :]], dim=1)
        m_img = torch.ones((1, img_embs.shape[1]), dtype=txt_in.attention_mask.dtype, device=device)
        f_mask = torch.cat([txt_in.attention_mask[:, :idx_img], m_img, txt_in.attention_mask[:, idx_img+1:]], dim=1)
        timer_fusion.stop()
        
        # 4. Gen
        torch.cuda.synchronize()
        timer_gen.start()
        hnd = TTFTLogitsProcessor()
        outs = model.generate(inputs_embeds=f_embs, attention_mask=f_mask, max_new_tokens=64, logits_processor=LogitsProcessorList([hnd]))
        timer_gen.stop()
        torch.cuda.synchronize()

        if idx >= WARMUP_SAMPLES:
            r1a, r1b, r2a, r2b, r3 = timer_text.get_time(), timer_img.get_time(), timer_vit.get_time(), timer_mlp.get_time(), timer_fusion.get_time()
            r4 = timer_gen.start_event.elapsed_time(hnd.first_token_event) / 1000.0
            true_ttft = r1a + r1b + r2a + r2b + r3 + r4
            decode = (timer_gen.get_time() - r4) if outs.shape[1] > 1 else 0.0
            results.append([r1a, r1b, r2a, r2b, r3, r4, true_ttft, decode, true_ttft+decode, torch.cuda.max_memory_allocated()/(1024**3), outs.shape[1]])

df = pd.DataFrame(results, columns=['1a_text','1b_img','2a_vit','2b_mlp','3_fusion','4_gen','true_ttft','decode_time','total_latency','vram','tokens'])
df.to_csv("llava_e2e.csv", index=False)