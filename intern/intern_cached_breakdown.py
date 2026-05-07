import os, torch, time
import pandas as pd
from transformers import AutoModel, AutoTokenizer, LogitsProcessor, LogitsProcessorList
from tqdm import tqdm

MODEL_ID = "OpenGVLab/InternVL3_5-8B"
device = "cuda"

RESOLUTIONS = {"256 (Tokens)": (1, 1), "1K (Tokens)": (2, 2), "2K (Tokens)": (2, 4), "4K (Tokens)": (4, 4), "8K (Tokens)": (4, 8)}
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

model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, trust_remote_code=True, _fast_init=False).eval().to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
results = []

with torch.no_grad():
    for label, (h, w) in RESOLUTIONS.items():
        dummy_pixels = torch.randn((h*w, 3, 448, 448), dtype=torch.bfloat16, device=device)
        vit = model.vision_model(dummy_pixels).last_hidden_state[:, 1:, :]
        torch.save({"vit_embeds": vit.cpu()}, f"temp_internvl_{label}.pt")

        avg_enc, avg_pref, measure_count = 0.0, 0.0, 0
        crash_flag = False
        
        for i in tqdm(range(NUM_ITER)):
            if crash_flag: break
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
            t_enc=CUDATimer(); t_pref=CUDATimer()
            
            # 2. Encoding (DB Load + Unshuffle + MLP)
            t_enc.start()
            vit = torch.load(f"temp_internvl_{label}.pt")["vit_embeds"].to(device, torch.bfloat16)
            b, s, c = vit.shape
            vit = vit.reshape(b, int(s**0.5), int(s**0.5), c).unfold(1, 2, 2).unfold(2, 2, 2).reshape(b, int(s**0.5)//2, int(s**0.5)//2, 4, c).reshape(b, int(s**0.5)//2, int(s**0.5)//2, c*4).reshape(b, -1, c*4)
            img_embs = model.mlp1(vit).reshape(-1, model.config.hidden_size)
            t_enc.stop()

            # 3. Prefill
            t_pref.start()
            txt_in = tokenizer("User: <image>\nDescribe.\nAssistant:", return_tensors="pt").to(device)
            in_embs = model.language_model.get_input_embeddings()(txt_in.input_ids)
            idx_img = torch.where(txt_in.input_ids == tokenizer.convert_tokens_to_ids("<image>"))[1][0]
            
            allowed = 4096 - in_embs.shape[1] - 50
            if img_embs.shape[0] > allowed: img_embs = img_embs[:allowed, :]

            f_embs = torch.cat([in_embs[:, :idx_img, :], img_embs.unsqueeze(0), in_embs[:, idx_img+1:, :]], dim=1)
            f_mask = torch.cat([txt_in.attention_mask[:, :idx_img], torch.ones((1, img_embs.shape[0]), device=device), txt_in.attention_mask[:, idx_img+1:]], dim=1)
            try:
                torch.cuda.synchronize(); hnd = TTFTLogitsProcessor()
                model.language_model.generate(inputs_embeds=f_embs, attention_mask=f_mask, max_new_tokens=10, logits_processor=LogitsProcessorList([hnd])); t_pref.stop(); torch.cuda.synchronize()
                if i >= 10:
                    avg_enc += t_enc.time() * 1000; avg_pref += t_pref.s.elapsed_time(hnd.evt); measure_count += 1
            except Exception:
                avg_enc, avg_pref, measure_count, crash_flag = 0, 0, 1, True; break
                
        results.append({"Resolution": label, "Image Preprocessing": 0.0, "Image Encoding": avg_enc/measure_count, "LLM Prefill": avg_pref/measure_count})

pd.DataFrame(results).to_csv("internvl_cached_breakdown.csv", index=False)