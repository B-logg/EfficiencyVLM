import os, torch, time
import pandas as pd
from datasets import load_dataset
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor, LogitsProcessor, LogitsProcessorList
from tqdm import tqdm

MODEL_ID, NUM_TEST_SAMPLES, WARMUP_SAMPLES, device = "llava-hf/llava-v1.6-vicuna-7b-hf", 1010, 10, "cuda"

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

model = LlavaNextForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map=device).eval()
processor = LlavaNextProcessor.from_pretrained(MODEL_ID)
dataset = load_dataset("detection-datasets/coco", split=f"val[:{NUM_TEST_SAMPLES}]", trust_remote_code=True)
img_id = processor.tokenizer.convert_tokens_to_ids("<image>")

results = []
with torch.no_grad():
    for idx, data in enumerate(tqdm(dataset)):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        image = data['image'].convert("RGB") if data['image'].mode != "RGB" else data['image']
        
        t1a=CUDATimer(); t1b=CUDATimer(); t2a=CUDATimer(); t2b=CUDATimer(); t3=CUDATimer(); t4=CUDATimer()
        
        t1a.start(); txt_in = processor.tokenizer("USER: <image>\nDescribe this image in detail. ASSISTANT:", return_tensors="pt").to(device); t1a.stop()
        t1b.start(); img_in = processor(images=image, return_tensors="pt").to(device, torch.bfloat16); p_val=img_in.pixel_values; t1b.stop()
        
        t2a.start(); v_out = model.vision_tower(p_val, output_hidden_states=True); t2a.stop()
        t2b.start(); img_embs = model.multi_modal_projector(v_out.hidden_states[-2]); t2b.stop()
        
        t3.start()
        # AnyRes feature shape formatting
        if img_embs.dim() == 4: img_embs = img_embs.flatten(1, 2) 
        if img_embs.dim() == 3 and img_embs.shape[0] != 1: img_embs = img_embs.view(1, -1, img_embs.shape[-1])

        in_embs = model.get_input_embeddings()(txt_in.input_ids)
        idx_img = torch.where(txt_in.input_ids == img_id)[1][0]
        f_embs = torch.cat([in_embs[:,:idx_img,:], img_embs, in_embs[:,idx_img+1:,:]], dim=1)
        m_img = torch.ones((1, img_embs.shape[1]), dtype=txt_in.attention_mask.dtype, device=device)
        f_mask = torch.cat([txt_in.attention_mask[:,:idx_img], m_img, txt_in.attention_mask[:,idx_img+1:]], dim=1)
        t3.stop()
        
        torch.cuda.synchronize()
        t4.start(); hnd = TTFTLogitsProcessor()
        outs = model.generate(inputs_embeds=f_embs, attention_mask=f_mask, max_new_tokens=64, logits_processor=LogitsProcessorList([hnd])); t4.stop()
        torch.cuda.synchronize()

        if idx >= WARMUP_SAMPLES:
            r1a, r1b, r2a, r2b, r3 = t1a.time(), t1b.time(), t2a.time(), t2b.time(), t3.time()
            r4 = t4.s.elapsed_time(hnd.evt) / 1000.0
            true_ttft = r1a + r1b + r2a + r2b + r3 + r4
            decode = outs.shape[1] > 1 and (t4.time() - r4) or 0.0
            results.append([r1a, r1b, r2a, r2b, r3, r4, true_ttft, decode, true_ttft+decode, torch.cuda.max_memory_allocated()/(1024**3), outs.shape[1]])

pd.DataFrame(results, columns=['1a_text','1b_img','2a_vit','2b_mlp','3_fusion','4_gen','true_ttft','decode_time','total_latency','vram','tokens']).to_csv("llava_e2e.csv", index=False)