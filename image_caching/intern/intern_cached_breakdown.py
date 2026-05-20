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

print("Loading InternVL Cached Model for Breakdown...")
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
        print(f"\nTesting InternVL Cached Breakdown - Resolution: {label}")
        num_tiles = h * w

        # 사전 작업: 랜덤 이미지로 ViT + pixel shuffle 실행 후 .pt 저장 (타이밍 루프 밖)
        dummy_array = np.random.randint(0, 256, (448, 448, 3), dtype=np.uint8)
        dummy_image = Image.fromarray(dummy_array)
        pixel_values = torch.stack([transform(dummy_image) for _ in range(num_tiles)]).to(device, dtype=torch.bfloat16)
        vit = model.vision_model(pixel_values).last_hidden_state[:, 1:, :]
        b, s, c = vit.shape
        pixel_shuffled = (vit
            .reshape(b, int(s**0.5), int(s**0.5), c)
            .unfold(1, 2, 2).unfold(2, 2, 2)
            .reshape(b, int(s**0.5)//2, int(s**0.5)//2, 4, c)
            .reshape(b, int(s**0.5)//2, int(s**0.5)//2, c*4)
            .reshape(b, -1, c*4))
        torch.save({"pixel_shuffled": pixel_shuffled.cpu()}, f"temp_internvl_breakdown_{h*w}.pt")
        torch.cuda.synchronize()

        avg_enc, avg_pref, measure_count = 0.0, 0.0, 0
        crash_flag = False

        for i in tqdm(range(NUM_ITER)):
            if crash_flag: break
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

            # --- Encoding: DB Load(CPU wall clock) + MLP(CUDA Event) ---

            # DB Load: torch.load()는 CPU 작업 → CUDA Event 쓰면 누락됨, wall clock 사용
            _t = time.perf_counter()
            pixel_shuffled = torch.load(f"temp_internvl_breakdown_{h*w}.pt")["pixel_shuffled"].to(device, torch.bfloat16)
            torch.cuda.synchronize()  # H2D 전송 완료 대기
            t_load_ms = (time.perf_counter() - _t) * 1000

            # MLP: pure GPU → CUDA Event
            t_mlp_enc = CUDATimer(); t_mlp_enc.start()
            img_embs = model.mlp1(pixel_shuffled)
            img_embs = img_embs.reshape(-1, img_embs.shape[-1])
            t_mlp_enc.stop()
            torch.cuda.synchronize()  # prefill 전에 MLP 완료 보장
            t_enc_ms = t_load_ms + t_mlp_enc.get_time() * 1000

            # --- Prefill: tokenizer(CPU) + embed + assemble + 첫 토큰까지 → wall clock ---
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
                    avg_enc += t_enc_ms
                    avg_pref += t_pref_ms
                    measure_count += 1
            except Exception as e:
                print(f"Crash Details: {e}")
                avg_enc, avg_pref, measure_count, crash_flag = 0, 0, 1, True; break

        results.append({
            "Resolution": label,
            "Image Preprocessing": 0.0,
            "Image Encoding (DB Load)": avg_enc / measure_count,
            "LLM Prefill": avg_pref / measure_count,
        })

pd.DataFrame(results).to_csv("internvl_cached_breakdown.csv", index=False)
print("Cached Breakdown done.")
