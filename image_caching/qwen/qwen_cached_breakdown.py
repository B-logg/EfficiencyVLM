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

print("Loading Qwen Cached Model for Breakdown...")
model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map=device).eval()
processor = AutoProcessor.from_pretrained(MODEL_ID)
vision_encoder = getattr(model, 'visual', getattr(getattr(model, 'model', None), 'visual', None))

results = []
with torch.no_grad():
    for label, size in RESOLUTIONS.items():
        print(f"\nTesting Qwen Cached Breakdown - Resolution: {label}")

        # 사전 작업: 랜덤 이미지로 visual encoder 실행 후 .pt 저장 (타이밍 루프 밖)
        dummy_array = np.random.randint(0, 256, (size[1], size[0], 3), dtype=np.uint8)
        dummy_image = Image.fromarray(dummy_array)
        img_in = processor.image_processor(images=dummy_image, return_tensors="pt").to(device)
        v_out = vision_encoder(img_in.pixel_values.to(torch.bfloat16), grid_thw=img_in.image_grid_thw)
        img_embs_pre = v_out.last_hidden_state if hasattr(v_out, 'last_hidden_state') else (v_out[0] if isinstance(v_out, tuple) else v_out)
        merger = getattr(vision_encoder, 'merger', None)
        if merger and img_embs_pre.shape[-1] != model.get_input_embeddings().weight.shape[1]:
            img_embs_pre = merger(img_embs_pre)
        torch.save({"embeds": img_embs_pre.cpu(), "grid_thw": img_in.image_grid_thw.cpu()}, f"temp_qwen_{label}.pt")
        torch.cuda.synchronize()

        avg_db, avg_fus, avg_pref, measure_count = 0.0, 0.0, 0.0, 0

        for i in tqdm(range(NUM_ITER)):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

            # 1. DB Load: CPU(disk I/O) + H2D → wall clock
            _t = time.perf_counter()
            saved = torch.load(f"temp_qwen_{label}.pt")
            embs = saved["embeds"].to(device, torch.bfloat16)
            g_thw = saved["grid_thw"].to(device)
            N_patches = embs.shape[0]
            torch.cuda.synchronize()  # H2D 전송 완료 대기
            t_db_ms = (time.perf_counter() - _t) * 1000

            # 2. Text tokenization + Fusion: CPU(string 구성+tokenize) + GPU(embedding+교체) → wall clock
            # Qwen2-VL: N_patches 기반 프롬프트 구성이 CPU, in-place 교체가 GPU
            _t = time.perf_counter()
            prompt = (f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                      f"<|im_start|>user\n<|vision_start|>{'<|image_pad|>' * N_patches}<|vision_end|>\n"
                      f"Describe.<|im_end|>\n<|im_start|>assistant\n")
            txt_in = processor.tokenizer(prompt, return_tensors="pt").to(device)
            in_embs = model.get_input_embeddings()(txt_in.input_ids)
            in_embs[txt_in.input_ids == processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")] = embs
            torch.cuda.synchronize()
            t_fus_ms = (time.perf_counter() - _t) * 1000

            # 3. Prefill (TTFT): generate 시작부터 첫 토큰까지 → wall clock + max_new_tokens=1
            _t = time.perf_counter()
            model.generate(
                inputs_embeds=in_embs, attention_mask=txt_in.attention_mask,
                image_grid_thw=g_thw, max_new_tokens=1
            )
            torch.cuda.synchronize()
            t_pref_ms = (time.perf_counter() - _t) * 1000

            if i >= WARMUP_ITER:
                avg_db   += t_db_ms
                avg_fus  += t_fus_ms
                avg_pref += t_pref_ms
                measure_count += 1

        results.append({
            "Resolution": label,
            "Image Preprocessing": 0.0,
            "DB Load": avg_db / measure_count,
            "Fusion (Text+Visual)": avg_fus / measure_count,
            "LLM Prefill (TTFT)": avg_pref / measure_count,
        })

pd.DataFrame(results).to_csv("qwen_cached_breakdown.csv", index=False)
print("Qwen Cached Breakdown done.")
