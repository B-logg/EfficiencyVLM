import os, torch, time
import pandas as pd
from PIL import Image
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor, LogitsProcessor, LogitsProcessorList
from tqdm import tqdm

MODEL_ID = "llava-hf/llava-v1.6-vicuna-7b-hf"
device = "cuda"

RESOLUTIONS = {"448x448": (448, 448), "896x896": (896, 896), "1344x1344": (1344, 1344), "1792x1792": (1792, 1792), "2520x2520": (2520, 2520)}
NUM_ITER = 110

class CUDATimer:
    def __init__(self): self.s = torch.cuda.Event(enable_timing=True); self.e = torch.cuda.Event(enable_timing=True)
    def start(self): self.s.record()
    def stop(self): self.e.record()
    def time(self): return self.s.elapsed_time(self.e) / 1000.0

class TTFTLogitsProcessor(LogitsProcessor):
    def __init__(self): self.evt = torch.cuda.Event(enable_timing=True); self.is_first = True
    def __call__(self, input_ids, scores):
        if self.is_first: self.evt.record(); self.is_first = False
        return scores

print("Loading LLaVA Cached Model...")
model = LlavaNextForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map=device).eval()

# Cached도 동일하게 안전장치 해제
processor = LlavaNextProcessor.from_pretrained(MODEL_ID, max_image_patches=200)
image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")

results = []
with torch.no_grad():
    for label, size in RESOLUTIONS.items():
        print(f"Testing LLaVA Cached - Resolution: {label} | 100 Measure")
        
        img_in = processor(text="<image>", images=Image.new('RGB', size, color='white'), return_tensors="pt").to(device, torch.bfloat16)
        pixel_values = img_in.pixel_values
        if pixel_values.dim() == 5:
            b, num_p, c, h, w = pixel_values.shape
            pixel_values = pixel_values.view(b * num_p, c, h, w)
        v_out = getattr(model, 'vision_tower', getattr(model.model, 'vision_tower', None))(pixel_values, output_hidden_states=True)
        img_embs = getattr(model, 'multi_modal_projector', getattr(model.model, 'multi_modal_projector', None))(v_out.hidden_states[-2])
        torch.save(img_embs.cpu(), f"temp_llava_{label}.pt")
        
        avg_preproc = 0.0; avg_encode = 0.0; avg_prefill = 0.0; measure_count = 0
        crash_flag = False
        
        for i in tqdm(range(NUM_ITER)):
            if crash_flag: break
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
            t1a=CUDATimer(); t_db=CUDATimer(); t3=CUDATimer(); t4=CUDATimer()
            
            t1a.start(); txt_in = processor.tokenizer("USER: <image>\nDescribe.\nASSISTANT:", return_tensors="pt").to(device); t1a.stop()
            
            t_db.start(); img_embs = torch.load(f"temp_llava_{label}.pt").to(device, torch.bfloat16)
            if img_embs.dim() == 4: img_embs = img_embs.flatten(1, 2) 
            if img_embs.dim() == 3 and img_embs.shape[0] != 1: img_embs = img_embs.view(1, -1, img_embs.shape[-1])
            t_db.stop()
            
            t3.start(); in_embs = model.get_input_embeddings()(txt_in.input_ids); idx_img = torch.where(txt_in.input_ids == image_token_id)[1][0]
            f_embs = torch.cat([in_embs[:, :idx_img, :], img_embs, in_embs[:, idx_img+1:, :]], dim=1)
            m_img = torch.ones((1, img_embs.shape[1]), dtype=txt_in.attention_mask.dtype, device=device)
            f_mask = torch.cat([txt_in.attention_mask[:, :idx_img], m_img, txt_in.attention_mask[:, idx_img+1:]], dim=1)
            t3.stop()
            
            try:
                torch.cuda.synchronize(); t4.start(); hnd = TTFTLogitsProcessor()
                model.generate(inputs_embeds=f_embs, attention_mask=f_mask, max_new_tokens=10, logits_processor=LogitsProcessorList([hnd])); t4.stop(); torch.cuda.synchronize()

                if i >= 10:
                    avg_preproc += (t1a.time()) * 1000
                    avg_encode += (t_db.time() + t3.time()) * 1000
                    avg_prefill += (t4.s.elapsed_time(hnd.evt))
                    measure_count += 1
            except Exception as e:
                print(f"\n[💥 EXPECTED CRASH] LLaVA Context Limit Exceeded at {label}! Logging as 0.")
                avg_preproc, avg_encode, avg_prefill, measure_count = 0, 0, 0, 1
                crash_flag = True
                break
                
        results.append({"Resolution": label, "Image Preprocessing": avg_preproc / measure_count, "Image Encoding (DB Load)": avg_encode / measure_count, "LLM Prefill": avg_prefill / measure_count})

pd.DataFrame(results).to_csv("llava_cached.csv", index=False)
print("Saved llava_cached.csv")