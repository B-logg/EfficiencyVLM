import os, torch, time
import numpy as np
import pandas as pd
from PIL import Image
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
from tqdm import tqdm

MODEL_ID = "llava-hf/llava-v1.6-vicuna-7b-hf"
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

print("Loading LLaVA E2E Model for Breakdown...")
model = LlavaNextForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map=device).eval()
processor = LlavaNextProcessor.from_pretrained(MODEL_ID)
processor.image_processor.image_grid_pinpoints = [
    [336, 336], [336, 672], [672, 336], [672, 672],
    [336, 1008], [1008, 336], [672, 1008], [1008, 672],
    [1008, 1008], [1344, 1344], [1680, 1680], [2016, 2016]
]
processor.image_processor.max_image_patches = 100
image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")

vision_tower = getattr(model, 'vision_tower', getattr(getattr(model, 'model', None), 'vision_tower', None))
projector = getattr(model, 'multi_modal_projector', getattr(getattr(model, 'model', None), 'multi_modal_projector', None))

results = []
with torch.no_grad():
    for label, size in RESOLUTIONS.items():
        print(f"\nTesting LLaVA E2E Breakdown - Resolution: {label}")

        # 더미 이미지: 랜덤 픽셀, 루프 밖에서 생성
        dummy_array = np.random.randint(0, 256, (size[1], size[0], 3), dtype=np.uint8)
        dummy_image = Image.fromarray(dummy_array)

        avg_prep, avg_enc, avg_fus, avg_pref, measure_count = 0.0, 0.0, 0.0, 0.0, 0
        crash_flag = False

        for i in tqdm(range(NUM_ITER)):
            if crash_flag: break
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

            # 1. Image Preprocessing: CPU(processor/tiling) + H2D → wall clock
            _t = time.perf_counter()
            img_in = processor(text="<image>", images=dummy_image, return_tensors="pt").to(device, torch.bfloat16)
            pixel_values = img_in.pixel_values
            if pixel_values.dim() == 5:
                b, num_p, c, h, w = pixel_values.shape
                pixel_values = pixel_values.view(b * num_p, c, h, w)
            torch.cuda.synchronize()  # H2D 전송 완료 대기
            t_prep_ms = (time.perf_counter() - _t) * 1000

            # 2. Image Encoding: ViT + Projector → pure GPU, CUDA Event
            t_enc = CUDATimer(); t_enc.start()
            v_out = vision_tower(pixel_values, output_hidden_states=True)
            img_embs = projector(v_out.hidden_states[-2])
            if img_embs.dim() == 4: img_embs = img_embs.flatten(1, 2)
            if img_embs.dim() == 3 and img_embs.shape[0] != 1:
                img_embs = img_embs.view(1, -1, img_embs.shape[-1])
            t_enc.stop()
            torch.cuda.synchronize()
            t_enc_ms = t_enc.get_time() * 1000

            # 3. Fusion + Prefill
            try:
                txt_in = processor.tokenizer(
                    "USER: <image>\nDescribe.\nASSISTANT:", return_tensors="pt"
                ).to(device)

                max_total_len = 4096
                text_len = txt_in.input_ids.shape[1] - 1
                allowed_img_len = max_total_len - text_len - 100
                if img_embs.shape[1] > allowed_img_len:
                    img_embs = img_embs[:, :allowed_img_len, :]

                # Fusion: pure GPU → CUDA Event
                t_fus = CUDATimer(); t_fus.start()
                in_embs = model.get_input_embeddings()(txt_in.input_ids)
                idx_img = torch.where(txt_in.input_ids == image_token_id)[1][0]
                f_embs = torch.cat([in_embs[:, :idx_img, :], img_embs, in_embs[:, idx_img+1:, :]], dim=1)
                m_img = torch.ones((1, img_embs.shape[1]), dtype=txt_in.attention_mask.dtype, device=device)
                f_mask = torch.cat([txt_in.attention_mask[:, :idx_img], m_img, txt_in.attention_mask[:, idx_img+1:]], dim=1)
                t_fus.stop()
                torch.cuda.synchronize()
                t_fus_ms = t_fus.get_time() * 1000

                # Prefill (TTFT): generate 시작부터 첫 토큰까지 → wall clock + max_new_tokens=1
                _t = time.perf_counter()
                model.generate(inputs_embeds=f_embs, attention_mask=f_mask, max_new_tokens=1)
                torch.cuda.synchronize()
                t_pref_ms = (time.perf_counter() - _t) * 1000

                if i >= WARMUP_ITER:
                    avg_prep += t_prep_ms
                    avg_enc  += t_enc_ms
                    avg_fus  += t_fus_ms
                    avg_pref += t_pref_ms
                    measure_count += 1
            except Exception as e:
                print(f"Crash Details: {e}")
                avg_prep, avg_enc, avg_fus, avg_pref, measure_count, crash_flag = 0, 0, 0, 0, 1, True; break

        results.append({
            "Resolution": label,
            "Image Preprocessing": avg_prep / measure_count,
            "Image Encoding": avg_enc / measure_count,
            "Fusion": avg_fus / measure_count,
            "LLM Prefill (TTFT)": avg_pref / measure_count,
        })

pd.DataFrame(results).to_csv("llava_e2e_breakdown.csv", index=False)
print("LLaVA E2E Breakdown done.")
