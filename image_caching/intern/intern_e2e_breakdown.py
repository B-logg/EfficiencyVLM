import os, torch
import pandas as pd
from PIL import Image
from transformers import AutoModel, AutoTokenizer, LogitsProcessor, LogitsProcessorList, PreTrainedModel
from tqdm import tqdm
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

MODEL_ID = "OpenGVLab/InternVL3_5-8B"
device = "cuda"

# [가중치 패치] 메타 텐서 & Qwen3 구조 충돌 방지
PreTrainedModel.all_tied_weights_keys = {}

# InternVL: 1 Tile (448x448) = 256 tokens.
RESOLUTIONS = {
    "256 (Tokens)": (1, 1),
    "1K (Tokens)": (2, 2),
    "2K (Tokens)": (2, 4),
    "4K (Tokens)": (4, 4),
    "8K (Tokens)": (4, 8)
}
NUM_ITER = 50

class CUDATimer:
    def __init__(self): self.s = torch.cuda.Event(enable_timing=True); self.e = torch.cuda.Event(enable_timing=True)
    def start(self): self.s.record()
    def stop(self): self.e.record()
    def time(self): return self.s.elapsed_time(self.e) / 1000.0

class TTFTLogitsProcessor(LogitsProcessor):
    def __init__(self): self.evt = torch.cuda.Event(enable_timing=True); self.is_first = True
    def __call__(self, i, s):
        if self.is_first: self.evt.record(); self.is_first = False
        return s

print("Loading InternVL E2E Model for Breakdown...")
model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True).eval().to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# 실제 환경과 동일한 전처리 파이프라인
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
        # 꼼수 대신 실제 PIL 이미지 사용
        dummy_image = Image.new('RGB', (448, 448), color='white')
        
        avg_prep, avg_enc, avg_pref, measure_count = 0.0, 0.0, 0.0, 0
        crash_flag = False
        
        for i in tqdm(range(NUM_ITER)):
            if crash_flag: break
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
            t_prep=CUDATimer(); t_enc=CUDATimer(); t_pref=CUDATimer()
            
            # 1. Preproc (CPU 정규화 -> GPU 이동)
            t_prep.start()
            pixel_values = torch.stack([transform(dummy_image) for _ in range(num_tiles)]).to(device, dtype=torch.bfloat16)
            t_prep.stop()
            
            # 2. Encoding (ViT + Unshuffle + MLP)
            t_enc.start()
            vit = model.vision_model(pixel_values).last_hidden_state[:, 1:, :]
            b, s, c = vit.shape
            vit = vit.reshape(b, int(s**0.5), int(s**0.5), c).unfold(1, 2, 2).unfold(2, 2, 2).reshape(b, int(s**0.5)//2, int(s**0.5)//2, 4, c).reshape(b, int(s**0.5)//2, int(s**0.5)//2, c*4).reshape(b, -1, c*4)
            img_embs = model.mlp1(vit)
            img_embs = img_embs.reshape(-1, img_embs.shape[-1])
            t_enc.stop()

            # 3. Prefill (Text + Fusion + Gen -> 샌드위치 기법)
            t_pref.start()
            tok_1 = tokenizer("User: ", return_tensors="pt", add_special_tokens=True).input_ids.to(device)
            tok_2 = tokenizer("\nDescribe.\nAssistant:", return_tensors="pt", add_special_tokens=False).input_ids.to(device)
            
            emb_1 = model.language_model.get_input_embeddings()(tok_1)
            emb_2 = model.language_model.get_input_embeddings()(tok_2)
            
            # Context Limit 크래시 방지
            allowed = 4096 - emb_1.shape[1] - emb_2.shape[1] - 50
            if img_embs.shape[0] > allowed: 
                img_embs = img_embs[:allowed, :]

            f_embs = torch.cat([emb_1, img_embs.unsqueeze(0), emb_2], dim=1)
            f_mask = torch.cat([
                torch.ones_like(tok_1), 
                torch.ones((1, img_embs.shape[0]), dtype=tok_1.dtype, device=device), 
                torch.ones_like(tok_2)
            ], dim=1)
            
            try:
                torch.cuda.synchronize(); hnd = TTFTLogitsProcessor()
                model.language_model.generate(inputs_embeds=f_embs, attention_mask=f_mask, max_new_tokens=10, logits_processor=LogitsProcessorList([hnd])); t_pref.stop(); torch.cuda.synchronize()
                if i >= 10:
                    avg_prep += t_prep.time() * 1000; avg_enc += t_enc.time() * 1000; avg_pref += t_pref.s.elapsed_time(hnd.evt); measure_count += 1
            except Exception as e:
                print(f"Crash Details: {e}")
                avg_prep, avg_enc, avg_pref, measure_count, crash_flag = 0, 0, 0, 1, True; break
                
        results.append({"Resolution": label, "Image Preprocessing": avg_prep/measure_count, "Image Encoding": avg_enc/measure_count, "LLM Prefill": avg_pref/measure_count})

pd.DataFrame(results).to_csv("internvl_e2e_breakdown.csv", index=False)
print("✅ E2E Breakdown 완료 및 CSV 저장됨!")