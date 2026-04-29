import os, torch, time
import pandas as pd
from PIL import Image
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor, LogitsProcessor, LogitsProcessorList
from tqdm import tqdm

MODEL_ID = "llava-hf/llava-v1.6-vicuna-7b-hf"
device = "cuda"

SEQ_SIZES = {"256": (448, 448), "1k": (896, 896), "2k": (1260, 1260), "4k": (1792, 1792), "8k": (2520, 2520)}
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

print("Loading LLaVA E2E Model...")
model = LlavaNextForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map=device).eval()
processor = LlavaNextProcessor.from_pretrained(MODEL_ID)
image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")

results = []
with torch.no_grad():
    for label, size in SEQ_SIZES.items():
        print(f"Testing LLaVA E2E - Sequence Length: {label}")
        dummy_image = Image.new('RGB', size, color='white')
        avg_preproc = 0.0; avg_encode = 0.0; avg_prefill = 0.0
        measure_count = 0
        
        for i in tqdm(range(NUM_ITER)):
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
            t1a=CUDATimer(); t1b=CUDATimer(); t2a=CUDATimer(); t2b=CUDATimer(); t3=CUDATimer(); t4=CUDATimer()
            
            t1a.start(); txt_in = processor.tokenizer("USER: <image>\nDescribe.\nASSISTANT:", return_tensors="pt").to(device); t1a.stop()
            
            t1b.start(); img_in = processor(text="<image>", images=dummy_image, return_tensors="pt").to(device, torch.bfloat16)
            pixel_values = img_in.pixel_values
            if pixel_values.dim() == 5:
                b, num_patches, c, h, w = pixel_values.shape
                pixel_values = pixel_values.view(b * num_patches, c, h, w)
            t1b.stop()
            
            t2a.start(); v_tower = getattr(model, 'vision_tower', getattr(getattr(model, 'model', None), 'vision_tower', None)); v_out = v_tower(pixel_values, output_hidden_states=True); t2a.stop()
            t2b.start(); m_proj = getattr(model, 'multi_modal_projector', getattr(getattr(model, 'model', None), 'multi_modal_projector', None)); img_embs = m_proj(v_out.hidden_states[-2])
            if img_embs.dim() == 4: img_embs = img_embs.flatten(1, 2) 
            if img_embs.dim() == 3 and img_embs.shape[0] != 1: img_embs = img_embs.view(1, -1, img_embs.shape[-1])
            t2b.stop()
            
            t3.start(); in_embs = model.get_input_embeddings()(txt_in.input_ids); idx_img = torch.where(txt_in.input_ids == image_token_id)[1][0]
            f_embs = torch.cat([in_embs[:, :idx_img, :], img_embs, in_embs[:, idx_img+1:, :]], dim=1)
            m_img = torch.ones((1, img_embs.shape[1]), dtype=txt_in.attention_mask.dtype, device=device)
            f_mask = torch.cat([txt_in.attention_mask[:, :idx_img], m_img, txt_in.attention_mask[:, idx_img+1:]], dim=1)
            t3.stop()
            
            torch.cuda.synchronize(); t4.start(); hnd = TTFTLogitsProcessor()
            model.generate(inputs_embeds=f_embs, attention_mask=f_mask, max_new_tokens=10, logits_processor=LogitsProcessorList([hnd])); t4.stop(); torch.cuda.synchronize()

            if i >= 10:
                avg_preproc += (t1a.time() + t1b.time()) * 1000
                avg_encode += (t2a.time() + t2b.time() + t3.time()) * 1000
                avg_prefill += (t4.s.elapsed_time(hnd.evt))
                measure_count += 1
                
        results.append({"Seq_Length": label, "Image Preprocessing": avg_preproc / measure_count, "Image Encoding": avg_encode / measure_count, "LLM Prefill": avg_prefill / measure_count})

pd.DataFrame(results).to_csv("llava_seq_e2e.csv", index=False)
print("Saved llava_seq_e2e.csv")