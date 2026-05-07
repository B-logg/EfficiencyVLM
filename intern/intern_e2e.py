import os, torch, time
import pandas as pd
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer, LogitsProcessor, LogitsProcessorList
from tqdm import tqdm
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

MODEL_ID = "OpenGVLab/InternVL3.5-8B"
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

print("Loading InternVL E2E Model...")
model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True, device_map=device).eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
dataset = load_dataset("detection-datasets/coco", split=f"val[:{NUM_TEST_SAMPLES}]", trust_remote_code=True)

transform = T.Compose([T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC), T.ToTensor(), T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
results = []

print("Starting InternVL E2E Inference...")
with torch.no_grad():
    for idx, data in enumerate(tqdm(dataset)):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        t_img=CUDATimer(); t_vit=CUDATimer(); t_unsh=CUDATimer(); t_mlp=CUDATimer(); t_txt=CUDATimer(); t_fus=CUDATimer(); t_gen=CUDATimer()
        
        # 1. Preprocessing
        t_img.start(); pixel_values = transform(data['image'].convert('RGB')).unsqueeze(0).to(device, dtype=torch.bfloat16); t_img.stop()
        
        # 2. InternViT
        t_vit.start(); vit_embeds = model.vision_model(pixel_values).last_hidden_state[:, 1:, :]; t_vit.stop()
        
        # 3. Pixel Unshuffle
        t_unsh.start()
        b, s, c = vit_embeds.shape
        vit_embeds = vit_embeds.reshape(b, int(s**0.5), int(s**0.5), c).unfold(1, 2, 2).unfold(2, 2, 2).reshape(b, int(s**0.5)//2, int(s**0.5)//2, 4, c).reshape(b, int(s**0.5)//2, int(s**0.5)//2, c*4).reshape(b, -1, c*4)
        t_unsh.stop()
        
        # 4. MLP Projector
        t_mlp.start(); img_embs = model.mlp1(vit_embeds).squeeze(0); t_mlp.stop()

        # 5. Text Encoding (텍스트 임베딩 연산)
        t_txt.start()
        txt_in = tokenizer("User: <image>\nDescribe this image.\nAssistant:", return_tensors="pt").to(device)
        in_embs = model.language_model.get_input_embeddings()(txt_in.input_ids)
        t_txt.stop()
        
        # 6. Fusion (Sequence Assembly)
        t_fus.start()
        idx_img = torch.where(txt_in.input_ids == tokenizer.convert_tokens_to_ids("<image>"))[1][0]
        f_embs = torch.cat([in_embs[:, :idx_img, :], img_embs.unsqueeze(0), in_embs[:, idx_img+1:, :]], dim=1)
        m_img = torch.ones((1, img_embs.shape[0]), dtype=txt_in.attention_mask.dtype, device=device)
        f_mask = torch.cat([txt_in.attention_mask[:, :idx_img], m_img, txt_in.attention_mask[:, idx_img+1:]], dim=1)
        t_fus.stop()
        
        # 7. Generation (TTFT)
        torch.cuda.synchronize(); t_gen.start(); hnd = TTFTLogitsProcessor()
        outs = model.language_model.generate(inputs_embeds=f_embs, attention_mask=f_mask, max_new_tokens=64, logits_processor=LogitsProcessorList([hnd])); t_gen.stop(); torch.cuda.synchronize()

        if idx >= WARMUP_SAMPLES:
            r_img, r_vit, r_unsh, r_mlp, r_txt, r_fus = t_img.get_time(), t_vit.get_time(), t_unsh.get_time(), t_mlp.get_time(), t_txt.get_time(), t_fus.get_time()
            r_ttft = t_gen.start_event.elapsed_time(hnd.evt) / 1000.0
            true_ttft = r_img + r_vit + r_unsh + r_mlp + r_txt + r_fus + r_ttft
            decode = (t_gen.get_time() - r_ttft) if outs.shape[1] > 1 else 0.0
            results.append([r_img, r_vit, r_unsh, r_mlp, r_txt, r_fus, r_ttft, true_ttft, decode, true_ttft+decode, torch.cuda.max_memory_allocated()/(1024**3), outs.shape[1]])

df = pd.DataFrame(results, columns=['1_img_preproc','2_vit','3_unshuffle','4_mlp','5_text','6_fusion','7_gen','true_ttft','decode_time','total_latency','vram','tokens'])
df.to_csv("internvl_e2e.csv", index=False)