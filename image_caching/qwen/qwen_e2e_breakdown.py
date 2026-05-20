import os, torch, time
import pandas as pd
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, LogitsProcessor, LogitsProcessorList
from tqdm import tqdm

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
device = "cuda"

RESOLUTIONS = {"448x448": (448, 448), "896x896": (896, 896), "1344x1344": (1344, 1344), "1792x1792": (1792, 1792), "2520x2520": (2520, 2520)}
NUM_ITER = 50

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

print("Loading Qwen E2E Model...")
model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map=device).eval()
processor = AutoProcessor.from_pretrained(MODEL_ID)

results = []
with torch.no_grad():
    for label, size in RESOLUTIONS.items():
        print(f"Testing Qwen E2E - Resolution: {label} | 100 Measure")
        dummy_image = Image.new('RGB', size, color='white')
        
        avg_preproc = 0.0; avg_encode = 0.0; avg_prefill = 0.0; measure_count = 0
        
        for i in tqdm(range(NUM_ITER)):
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
            t1a=CUDATimer(); t1b=CUDATimer(); t2a=CUDATimer(); t2b=CUDATimer(); t3=CUDATimer(); t4=CUDATimer()
            
            t1b.start()
            img_in = processor.image_processor(images=dummy_image, return_tensors="pt").to(device)
            p_val = img_in.pixel_values.to(torch.bfloat16); g_thw = img_in.image_grid_thw
            t1b.stop()
            
            vision_encoder = getattr(model, 'visual', getattr(getattr(model, 'model', None), 'visual', None))
            t2a.start(); v_out = vision_encoder(p_val, grid_thw=g_thw); img_embs = v_out.last_hidden_state if hasattr(v_out, 'last_hidden_state') else (v_out[0] if isinstance(v_out, tuple) else v_out); t2a.stop()
            
            t2b.start(); merger = getattr(vision_encoder, 'merger', None)
            if merger and img_embs.shape[-1] != model.get_input_embeddings().weight.shape[1]: img_embs = merger(img_embs)
            t2b.stop()

            t1a.start()
            prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|>{'<|image_pad|>' * img_embs.shape[0]}<|vision_end|>\nDescribe.<|im_end|>\n<|im_start|>assistant\n"
            txt_in = processor.tokenizer(prompt, return_tensors="pt").to(device)
            t1a.stop()
            
            t3.start(); in_embs = model.get_input_embeddings()(txt_in.input_ids); in_embs[txt_in.input_ids == processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")] = img_embs; t3.stop()
            
            torch.cuda.synchronize(); t4.start(); hnd = TTFTLogitsProcessor()
            model.generate(inputs_embeds=in_embs, attention_mask=txt_in.attention_mask, image_grid_thw=g_thw, max_new_tokens=10, logits_processor=LogitsProcessorList([hnd])); t4.stop(); torch.cuda.synchronize()

            if i >= 10:
                avg_preproc += (t1a.time() + t1b.time()) * 1000
                avg_encode += (t2a.time() + t2b.time() + t3.time()) * 1000
                avg_prefill += (t4.s.elapsed_time(hnd.evt))
                measure_count += 1
                
        results.append({"Resolution": label, "Image Preprocessing": avg_preproc / measure_count, "Image Encoding": avg_encode / measure_count, "LLM Prefill": avg_prefill / measure_count})

pd.DataFrame(results).to_csv("qwen_e2e_breakdown.csv", index=False)
print("Saved qwen_e2e.csv")