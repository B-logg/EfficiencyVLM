import os, torch, time
import pandas as pd
from datasets import load_dataset
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, LogitsProcessor, LogitsProcessorList
from tqdm import tqdm

MODEL_ID, EMBED_DIR, NUM_TEST_SAMPLES, WARMUP_SAMPLES, device = "Qwen/Qwen2-VL-2B-Instruct", "./qwen_embeddings", 1010, 10, "cuda"

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

model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map=device).eval()
processor = AutoProcessor.from_pretrained(MODEL_ID)
dataset = load_dataset("detection-datasets/coco", split=f"val[:{NUM_TEST_SAMPLES}]", trust_remote_code=True)

results = []
with torch.no_grad():
    for idx, data in enumerate(tqdm(dataset)):
        pt_path = os.path.join(EMBED_DIR, f"embed_{data['image_id']}.pt")
        if not os.path.exists(pt_path): continue
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        
        t1a=CUDATimer(); t2b_load=CUDATimer(); t3=CUDATimer(); t4=CUDATimer()
        
        t2b_load.start(); saved=torch.load(pt_path); img_embs=saved["embeds"].to(device, torch.bfloat16); g_thw=saved["grid_thw"].to(device); t2b_load.stop()
        t1a.start(); txt_in = processor.tokenizer(f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|>{'<|image_pad|>' * img_embs.shape[0]}<|vision_end|>Describe this image in detail.<|im_end|>\n<|im_start|>assistant\n", return_tensors="pt").to(device); t1a.stop()
        t3.start(); in_embs = model.get_input_embeddings()(txt_in.input_ids); in_embs[txt_in.input_ids == processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")] = img_embs; t3.stop()
        
        torch.cuda.synchronize()
        t4.start(); hnd = TTFTLogitsProcessor()
        outs = model.generate(inputs_embeds=in_embs, attention_mask=txt_in.attention_mask, image_grid_thw=g_thw, max_new_tokens=64, logits_processor=LogitsProcessorList([hnd])); t4.stop()
        torch.cuda.synchronize()

        if idx >= WARMUP_SAMPLES:
            r1a, r2b_load, r3 = t1a.time(), t2b_load.time(), t3.time()
            r4 = t4.s.elapsed_time(hnd.evt) / 1000.0
            true_ttft = r1a + r2b_load + r3 + r4
            decode = outs.shape[1] > 1 and (t4.time() - r4) or 0.0
            results.append([r1a, 0.0, 0.0, r2b_load, r3, r4, true_ttft, decode, true_ttft+decode, torch.cuda.max_memory_allocated()/(1024**3), outs.shape[1]])

pd.DataFrame(results, columns=['1a_text','1b_img','2a_vit','2b_mlp','3_fusion','4_gen','true_ttft','decode_time','total_latency','vram','tokens']).to_csv("qwen_cached.csv", index=False)