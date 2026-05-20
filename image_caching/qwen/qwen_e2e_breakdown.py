import os, torch, time
import numpy as np
import pandas as pd
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from tqdm import tqdm

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
device = "cuda"

RESOLUTIONS = {
    "448x448":   (448,  448),
    "896x896":   (896,  896),
    "1344x1344": (1344, 1344),
    "1792x1792": (1792, 1792),
    "2520x2520": (2520, 2520),
}
WARMUP_ITER = 30
NUM_ITER = WARMUP_ITER + 500

class CUDATimer:
    def __init__(self): self.s = torch.cuda.Event(enable_timing=True); self.e = torch.cuda.Event(enable_timing=True)
    def start(self): self.s.record()
    def stop(self): self.e.record()
    def get_time(self): return self.s.elapsed_time(self.e) / 1000.0  # elapsed_time auto-syncs

print("Loading Qwen E2E Model for Breakdown...")
model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map=device).eval()
processor = AutoProcessor.from_pretrained(MODEL_ID)
vision_encoder = getattr(model, 'visual', getattr(getattr(model, 'model', None), 'visual', None))

results = []
with torch.no_grad():
    for label, size in RESOLUTIONS.items():
        print(f"\nTesting Qwen E2E Breakdown - Resolution: {label}")

        # 더미 이미지: 랜덤 픽셀, 루프 밖에서 생성
        dummy_array = np.random.randint(0, 256, (size[1], size[0], 3), dtype=np.uint8)
        dummy_image = Image.fromarray(dummy_array)

        avg_prep, avg_vit, avg_merger, avg_fus, avg_pref, measure_count = 0.0, 0.0, 0.0, 0.0, 0.0, 0

        for i in tqdm(range(NUM_ITER)):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

            # 1. Image Preprocessing: CPU(image_processor) + H2D → wall clock
            _t = time.perf_counter()
            img_in = processor.image_processor(images=dummy_image, return_tensors="pt").to(device)
            p_val = img_in.pixel_values.to(torch.bfloat16)
            g_thw = img_in.image_grid_thw
            torch.cuda.synchronize()  # H2D 전송 완료 대기
            t_prep_ms = (time.perf_counter() - _t) * 1000

            # 2. ViT + Merger: vision_encoder는 ViT만 실행, Merger 명시적 별도 호출
            ev_vit_start    = torch.cuda.Event(enable_timing=True)
            ev_merger_start = torch.cuda.Event(enable_timing=True)
            ev_enc_end      = torch.cuda.Event(enable_timing=True)

            merger = getattr(vision_encoder, 'merger', None)
            lm_dim = model.get_input_embeddings().weight.shape[1]

            ev_vit_start.record()
            v_out = vision_encoder(p_val, grid_thw=g_thw)
            if hasattr(v_out, 'last_hidden_state'):
                v_out = v_out.last_hidden_state
            elif isinstance(v_out, tuple):
                v_out = v_out[0]
            ev_merger_start.record()

            if merger is not None and v_out.shape[-1] != lm_dim:
                img_embs = merger(v_out)
                merger_ran = True
            else:
                img_embs = v_out
                merger_ran = False
            ev_enc_end.record()
            torch.cuda.synchronize()

            t_vit_ms    = ev_vit_start.elapsed_time(ev_merger_start)
            t_merger_ms = ev_merger_start.elapsed_time(ev_enc_end) if merger_ran else 0.0

            N_patches = img_embs.shape[0]

            # 3. Text tokenization + Fusion: CPU(string 구성+tokenize) + GPU(embedding+교체) → wall clock
            # Qwen2-VL: N_patches 확정 후 프롬프트 구성 가능, CPU+GPU 혼합이라 wall clock 사용
            _t = time.perf_counter()
            prompt = (f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                      f"<|im_start|>user\n<|vision_start|>{'<|image_pad|>' * N_patches}<|vision_end|>\n"
                      f"Describe.<|im_end|>\n<|im_start|>assistant\n")
            txt_in = processor.tokenizer(prompt, return_tensors="pt").to(device)
            in_embs = model.get_input_embeddings()(txt_in.input_ids)
            in_embs[txt_in.input_ids == processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")] = img_embs
            torch.cuda.synchronize()
            t_fus_ms = (time.perf_counter() - _t) * 1000

            # 4. Prefill (TTFT): generate 시작부터 첫 토큰까지 → wall clock + max_new_tokens=1
            _t = time.perf_counter()
            model.generate(
                inputs_embeds=in_embs, attention_mask=txt_in.attention_mask,
                image_grid_thw=g_thw, max_new_tokens=1
            )
            torch.cuda.synchronize()
            t_pref_ms = (time.perf_counter() - _t) * 1000

            if i >= WARMUP_ITER:
                avg_prep   += t_prep_ms
                avg_vit    += t_vit_ms
                avg_merger += t_merger_ms
                avg_fus    += t_fus_ms
                avg_pref   += t_pref_ms
                measure_count += 1

        results.append({
            "Resolution": label,
            "Image Preprocessing": avg_prep   / measure_count,
            "ViT":                 avg_vit    / measure_count,
            "Merger":              avg_merger / measure_count,
            "Fusion (Text+Visual)": avg_fus   / measure_count,
            "LLM Prefill (TTFT)":  avg_pref   / measure_count,
        })

pd.DataFrame(results).to_csv("qwen_e2e_breakdown.csv", index=False)
print("Qwen E2E Breakdown done.")
