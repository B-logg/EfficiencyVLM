import os, torch, time
import pandas as pd
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, LogitsProcessor, LogitsProcessorList
from datasets import load_dataset
from tqdm import tqdm

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
device = "cuda"

TARGET_SEQS = {"256": 256, "1k": 1024, "2k": 2048, "4k": 4096, "8k": 8192}
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

print("Loading Qwen Cached Model & Data...")
model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map=device).eval()
processor = AutoProcessor.from_pretrained(MODEL_ID)

img_dataset = load_dataset("detection-datasets/coco", split="val[:1]", trust_remote_code=True)
original_image = img_dataset[0]['image'].convert("RGB")

print("Generating Natural Language Text Pool from Wikitext...")
wiki_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1000]")
NATURAL_TEXT_POOL = " ".join([doc['text'] for doc in wiki_data if doc['text'].strip()]) * 10

results = []
with torch.no_grad():
    # 이미지 고정이므로 캐시는 1번만 만듦
    img_in = processor.image_processor(images=original_image, return_tensors="pt").to(device)
    vision_encoder = getattr(model, 'visual', getattr(getattr(model, 'model', None), 'visual', None))
    v_out = vision_encoder(img_in.pixel_values.to(torch.bfloat16), grid_thw=img_in.image_grid_thw)
    img_embs = v_out.last_hidden_state if hasattr(v_out, 'last_hidden_state') else (v_out[0] if isinstance(v_out, tuple) else v_out)
    merger = getattr(vision_encoder, 'merger', None)
    if merger and img_embs.shape[-1] != model.get_input_embeddings().weight.shape[1]: img_embs = merger(img_embs)
    torch.save({"embeds": img_embs.cpu(), "grid_thw": img_in.image_grid_thw.cpu()}, "temp_qwen_fixed.pt")
    
    num_visual_tokens = img_embs.shape[0]

    for label, target_seq_len in TARGET_SEQS.items():
        print(f"Testing Qwen Cached - Target Seq: {label} | 100 Measure")
        
        needed_text_tokens = max(10, target_seq_len - num_visual_tokens - 20)
        padded_text = NATURAL_TEXT_POOL[:needed_text_tokens * 4]
        
        avg_preproc = 0.0; avg_encode = 0.0; avg_prefill = 0.0; measure_count = 0
        
        for i in tqdm(range(NUM_ITER)):
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
            t_db=CUDATimer(); t_text=CUDATimer(); t_fusion=CUDATimer(); t_gen=CUDATimer()
            
            t_db.start(); saved = torch.load("temp_qwen_fixed.pt"); embs = saved["embeds"].to(device, torch.bfloat16); g_thw = saved["grid_thw"].to(device); t_db.stop()
            
            t_text.start(); txt_in = processor.tokenizer(f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|>{'<|image_pad|>' * embs.shape[0]}<|vision_end|>\nContext: {padded_text}\nDescribe the image based on the context.<|im_end|>\n<|im_start|>assistant\n", return_tensors="pt").to(device); t_text.stop()
            
            t_fusion.start(); in_embs = model.get_input_embeddings()(txt_in.input_ids); in_embs[txt_in.input_ids == processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")] = embs; t_fusion.stop()
            
            torch.cuda.synchronize(); t_gen.start(); hnd = TTFTLogitsProcessor()
            model.generate(inputs_embeds=in_embs, attention_mask=txt_in.attention_mask, image_grid_thw=g_thw, max_new_tokens=10, logits_processor=LogitsProcessorList([hnd])); t_gen.stop(); torch.cuda.synchronize()

            if i >= 10:
                avg_preproc += (t_text.time()) * 1000
                avg_encode += (t_db.time() + t_fusion.time()) * 1000
                avg_prefill += (t_gen.s.elapsed_time(hnd.evt))
                measure_count += 1
                
        results.append({"Seq_Length": label, "Image Preprocessing": avg_preproc / measure_count, "Image Encoding (DB Load)": avg_encode / measure_count, "LLM Prefill": avg_prefill / measure_count})

pd.DataFrame(results).to_csv("qwen_seq_cached.csv", index=False)