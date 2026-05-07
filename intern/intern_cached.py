import os, torch, time
import pandas as pd
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer, LogitsProcessor, LogitsProcessorList
from tqdm import tqdm

MODEL_ID = "OpenGVLab/InternVL2_5-2B"
EMBED_DIR = "./internvl_vision_embeddings"
NUM_TEST_SAMPLES = 210
WARMUP_SAMPLES = 10
device = "cuda" if torch.cuda.is_available() else "cpu"

class CUDATimer:
    def __init__(self): self.s = torch.cuda.Event(enable_timing=True); self.e = torch.cuda.Event(enable_timing=True)
    def start(self): self.s.record()
    def stop(self): self.e.record()
    def get_time(self): return self.s.elapsed_time(self.e) / 1000.0

class TTFTLogitsProcessor(LogitsProcessor):
    def __init__(self): self.evt = torch.cuda.Event(enable_timing=True); self.is_first = True
    def __call__(self, input_ids, scores):
        if self.is_first: self.evt.record(); self.is_first = False
        return scores

model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True, device_map=device).eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
dataset = load_dataset("detection-datasets/coco", split=f"val[:{NUM_TEST_SAMPLES}]", trust_remote_code=True)
results = []

print("Starting InternVL Cached Inference...")
with torch.no_grad():
    for idx, data in enumerate(tqdm(dataset)):
        pt_path = os.path.join(EMBED_DIR, f"embed_{data['image_id']}.pt")
        if not os.path.exists(pt_path): continue
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        
        t_db=CUDATimer(); t_unsh=CUDATimer(); t_mlp=CUDATimer(); t_txt=CUDATimer(); t_fus=CUDATimer(); t_gen=CUDATimer()
        
        # 2. DB Load (ViT 대체)
        t_db.start(); vit_embeds = torch.load(pt_path)["vit_embeds"].to(device, dtype=torch.bfloat16); t_db.stop()

        # 3. Pixel Unshuffle (Cached 모드에서도 직접 연산)
        t_unsh.start()
        b, s, c = vit_embeds.shape
        vit_embeds = vit_embeds.reshape(b, int(s**0.5), int(s**0.5), c).unfold(1, 2, 2).unfold(2, 2, 2).reshape(b, int(s**0.5)//2, int(s**0.5)//2, 4, c).reshape(b, int(s**0.5)//2, int(s**0.5)//2, c*4).reshape(b, -1, c*4)
        t_unsh.stop()
        
        # 4. MLP Projector (Cached 모드에서도 직접 연산)
        t_mlp.start(); img_embs = model.mlp1(vit_embeds).squeeze(0); t_mlp.stop()

        # 5. Text Encoding
        t_txt.start()
        txt_in = tokenizer("User: <image>\nDescribe this image.\nAssistant:", return_tensors="pt").to(device)
        in_embs = model.language_model.get_input_embeddings()(txt_in.input_ids)
        t_txt.stop()
        
        # 6. Fusion
        t_fus.start()
        idx_img = torch.where(txt_in.input_ids == tokenizer.convert_tokens_to_ids("<image>"))[1][0]
        f_embs = torch.cat([in_embs[:, :idx_img, :], img_embs.unsqueeze(0), in_embs[:, idx_img+1:, :]], dim=1)
        m_img = torch.ones((1, img_embs.shape[0]), dtype=txt_in.attention_mask.dtype, device=device)
        f_mask = torch.cat([txt_in.attention_mask[:, :idx_img], m_img, txt_in.attention_mask[:, idx_img+1:]], dim=1)
        t_fus.stop()
        
        # 7. Gen
        torch.cuda.synchronize(); t_gen.start(); hnd = TTFTLogitsProcessor()
        outs = model.language_model.generate(inputs_embeds=f_embs, attention_mask=f_mask, max_new_tokens=64, logits_processor=LogitsProcessorList([hnd])); t_gen.stop(); torch.cuda.synchronize()

        if idx >= WARMUP_SAMPLES:
            r_db, r_unsh, r_mlp, r_txt, r_fus = t_db.get_time(), t_unsh.get_time(), t_mlp.get_time(), t_txt.get_time(), t_fus.get_time()
            r_ttft = t_gen.start_event.elapsed_time(hnd.evt) / 1000.0
            true_ttft = r_db + r_unsh + r_mlp + r_txt + r_fus + r_ttft
            decode = (t_gen.get_time() - r_ttft) if outs.shape[1] > 1 else 0.0
            results.append([0.0, r_db, r_unsh, r_mlp, r_txt, r_fus, r_ttft, true_ttft, decode, true_ttft+decode, torch.cuda.max_memory_allocated()/(1024**3), outs.shape[1]])

df = pd.DataFrame(results, columns=['1_img_preproc','2_vit_or_db','3_unshuffle','4_mlp','5_text','6_fusion','7_gen','true_ttft','decode_time','total_latency','vram','tokens'])
df.to_csv("internvl_cached.csv", index=False)