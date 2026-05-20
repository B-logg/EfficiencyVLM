import os, torch, time
import pandas as pd
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer, LogitsProcessor, LogitsProcessorList
from tqdm import tqdm
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers.modeling_utils import PreTrainedModel # [추가된 부분]

MODEL_ID = "OpenGVLab/InternVL3_5-8B"
NUM_TEST_SAMPLES = 3100
WARMUP_SAMPLES = 30
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


PreTrainedModel.all_tied_weights_keys = {}

model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True).eval().to(device)


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

        # 5. Text Encoding (샌드위치 방식)
        t_txt.start()
        tok_1 = tokenizer("User: ", return_tensors="pt", add_special_tokens=True).input_ids.to(device)
        tok_2 = tokenizer("\nDescribe this image.\nAssistant:", return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        
        emb_1 = model.language_model.get_input_embeddings()(tok_1)
        emb_2 = model.language_model.get_input_embeddings()(tok_2)
        t_txt.stop()
        
        # 6. Fusion (Sequence Assembly)
        t_fus.start()
        # 앞 텍스트 + 이미지 + 뒤 텍스트 결합
        f_embs = torch.cat([emb_1, img_embs.unsqueeze(0), emb_2], dim=1)
        
        # Attention Mask도 길이에 맞게 결합
        f_mask = torch.cat([
            torch.ones_like(tok_1), 
            torch.ones((1, img_embs.shape[0]), dtype=tok_1.dtype, device=device), 
            torch.ones_like(tok_2)
        ], dim=1)
        t_fus.stop()
        
        # 7. Generation (TTFT)
        torch.cuda.synchronize(); t_gen.start(); hnd = TTFTLogitsProcessor()
        outs = model.language_model.generate(inputs_embeds=f_embs, attention_mask=f_mask, max_new_tokens=64, logits_processor=LogitsProcessorList([hnd])); t_gen.stop(); torch.cuda.synchronize()

        if idx >= WARMUP_SAMPLES:
            r_img, r_vit, r_unsh, r_mlp, r_txt, r_fus = t_img.get_time(), t_vit.get_time(), t_unsh.get_time(), t_mlp.get_time(), t_txt.get_time(), t_fus.get_time()
            r_ttft = t_gen.s.elapsed_time(hnd.evt) / 1000.0
            true_ttft = r_img + r_vit + r_unsh + r_mlp + r_txt + r_fus + r_ttft
            decode = (t_gen.get_time() - r_ttft) if outs.shape[1] > 1 else 0.0
            results.append([r_img, r_vit, r_unsh, r_mlp, r_txt, r_fus, r_ttft, true_ttft, decode, true_ttft+decode, torch.cuda.max_memory_allocated()/(1024**3), outs.shape[1]])

df = pd.DataFrame(results, columns=['1_img_preproc','2_vit','3_unshuffle','4_mlp','5_text','6_fusion','7_gen','true_ttft','decode_time','total_latency','vram','tokens'])
df.to_csv("internvl_e2e.csv", index=False)