import os, torch, time
import numpy as np
import pandas as pd
from PIL import Image
from transformers import AutoModel, AutoTokenizer, PreTrainedModel
from tqdm import tqdm
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

MODEL_ID = "OpenGVLab/InternVL3_5-8B"
device = "cuda"

PreTrainedModel.all_tied_weights_keys = {}

RESOLUTIONS = {
    "256 (Tokens)": (1, 1),
    "1K (Tokens)": (2, 2),
    "2K (Tokens)": (2, 4),
    "4K (Tokens)": (4, 4),
    "8K (Tokens)": (4, 8)
}
WARMUP_ITER = 30
NUM_ITER = WARMUP_ITER + 500

class CUDATimer:
    def __init__(self): self.s = torch.cuda.Event(enable_timing=True); self.e = torch.cuda.Event(enable_timing=True)
    def start(self): self.s.record()
    def stop(self): self.e.record()
    def get_time(self): return self.s.elapsed_time(self.e) / 1000.0  # elapsed_time auto-syncs both events

print("Loading InternVL E2E Model for Breakdown...")
model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True).eval().to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

transform = T.Compose([
    T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

results = []

with torch.no_grad():
    for label, (h, w) in RESOLUTIONS.items():
        print(f"\nTesting InternVL E2E Breakdown - Resolution: {label}")
        num_tiles = h * w

        # 더미 이미지: 랜덤 픽셀 (흰색/검정 등 상수값 회피), 루프 밖에서 한 번 생성
        dummy_array = np.random.randint(0, 256, (448, 448, 3), dtype=np.uint8)
        dummy_image = Image.fromarray(dummy_array)

        avg_prep, avg_enc, avg_pref, measure_count = 0.0, 0.0, 0.0, 0
        crash_flag = False

        for i in tqdm(range(NUM_ITER)):
            if crash_flag: break
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

            # 1. Preprocess: CPU(transform) + H2D → CPU wall clock
            # CUDA Event으로 재면 transform() CPU 시간이 누락됨
            _t = time.perf_counter()
            pixel_values = torch.stack([transform(dummy_image) for _ in range(num_tiles)]).to(device, dtype=torch.bfloat16)
            torch.cuda.synchronize()  # H2D 전송 완료 대기
            t_prep_ms = (time.perf_counter() - _t) * 1000

            # 2. Encoding: ViT + Unshuffle + MLP → pure GPU, CUDA Event
            t_enc = CUDATimer(); t_enc.start()
            vit = model.vision_model(pixel_values).last_hidden_state[:, 1:, :]
            b, s, c = vit.shape
            vit = (vit
                .reshape(b, int(s**0.5), int(s**0.5), c)
                .unfold(1, 2, 2).unfold(2, 2, 2)
                .reshape(b, int(s**0.5)//2, int(s**0.5)//2, 4, c)
                .reshape(b, int(s**0.5)//2, int(s**0.5)//2, c*4)
                .reshape(b, -1, c*4))
            img_embs = model.mlp1(vit)
            img_embs = img_embs.reshape(-1, img_embs.shape[-1])
            t_enc.stop()
            torch.cuda.synchronize()  # prefill 전에 인코딩 완료 보장
            t_enc_ms = t_enc.get_time() * 1000

            # 3. Prefill: tokenizer(CPU) + embed + assemble + 첫 토큰까지 → wall clock
            # max_new_tokens=1로 정확히 TTFT만 측정, _t를 tokenizer 앞에 설정
            try:
                _t = time.perf_counter()  # tokenizer부터 측정 시작
                tok_1 = tokenizer("User: ", return_tensors="pt", add_special_tokens=True).input_ids.to(device)
                tok_2 = tokenizer("\nDescribe.\nAssistant:", return_tensors="pt", add_special_tokens=False).input_ids.to(device)
                emb_1 = model.language_model.get_input_embeddings()(tok_1)
                emb_2 = model.language_model.get_input_embeddings()(tok_2)

                allowed = 4096 - emb_1.shape[1] - emb_2.shape[1] - 50
                if img_embs.shape[0] > allowed:
                    img_embs = img_embs[:allowed, :]

                f_embs = torch.cat([emb_1, img_embs.unsqueeze(0), emb_2], dim=1)
                f_mask = torch.cat([
                    torch.ones_like(tok_1),
                    torch.ones((1, img_embs.shape[0]), dtype=tok_1.dtype, device=device),
                    torch.ones_like(tok_2)
                ], dim=1)

                # max_new_tokens=1: prefill pass + 첫 토큰 샘플링만 → 정확한 TTFT
                model.language_model.generate(
                    inputs_embeds=f_embs, attention_mask=f_mask,
                    max_new_tokens=1,
                )
                torch.cuda.synchronize()  # 첫 토큰 GPU 작업 완료 대기
                t_pref_ms = (time.perf_counter() - _t) * 1000

                if i >= WARMUP_ITER:
                    avg_prep += t_prep_ms
                    avg_enc += t_enc_ms
                    avg_pref += t_pref_ms
                    measure_count += 1
            except Exception as e:
                print(f"Crash Details: {e}")
                avg_prep, avg_enc, avg_pref, measure_count, crash_flag = 0, 0, 0, 1, True; break

        results.append({
            "Resolution": label,
            "Image Preprocessing": avg_prep / measure_count,
            "Image Encoding": avg_enc / measure_count,
            "LLM Prefill": avg_pref / measure_count,
        })

pd.DataFrame(results).to_csv("internvl_e2e_breakdown.csv", index=False)
print("E2E Breakdown done.")
