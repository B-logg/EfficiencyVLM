import os, torch, time
import pandas as pd
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, LogitsProcessor, LogitsProcessorList
from datasets import load_dataset
from tqdm import tqdm

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
device = "cuda"

TARGET_SEQS = {"256": 256, "1k": 1024, "2k": 2048, "4k": 4096, "8k": 8192}
NUM_ITER = 110 # 10 Warmup + 100 Measure

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

print("Loading Qwen E2E Model & Data...")
model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map=device).eval()
processor = AutoProcessor.from_pretrained(MODEL_ID)

# 1. 원본 이미지 1장 로드
img_dataset = load_dataset("detection-datasets/coco", split="val[:1]", trust_remote_code=True)
original_image = img_dataset[0]['image'].convert("RGB")

# 2. [핵심] 위키피디아 자연어 텍스트 풀 생성 (약 5만 자 이상 넉넉하게)
print("Generating Natural Language Text Pool from Wikitext...")
wiki_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1000]")
NATURAL_TEXT_POOL = " ".join([doc['text'] for doc in wiki_data if doc['text'].strip()]) * 10

results = []
with torch.no_grad():
    for label, target_seq_len in TARGET_SEQS.items():
        print(f"Testing Qwen E2E - Target Seq: {label} | 100 Measure")
        
        avg_preproc = 0.0; avg_encode = 0.0; avg_prefill = 0.0; measure_count = 0
        
        for i in tqdm(range(NUM_ITER)):
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
            t1a=CUDATimer(); t1b=CUDATimer(); t2a=CUDATimer(); t2b=CUDATimer(); t3=CUDATimer(); t4=CUDATimer()
            
            t1b.start()
            img_in = processor.image_processor(images=original_image, return_tensors="pt").to(device)
            p_val = img_in.pixel_values.to(torch.bfloat16); g_thw = img_in.image_grid_thw
            t1b.stop()
            
            vision_encoder = getattr(model, 'visual', getattr(getattr(model, 'model', None), 'visual', None))
            t2a.start(); v_out = vision_encoder(p_val, grid_thw=g_thw); img_embs = v_out.last_hidden_state if hasattr(v_out, 'last_hidden_state') else (v_out[0] if isinstance(v_out, tuple) else v_out); t2a.stop()
            
            t2b.start(); merger = getattr(vision_encoder, 'merger', None)
            if merger and img_embs.shape[-1] != model.get_input_embeddings().weight.shape[1]: img_embs = merger(img_embs)
            t2b.stop()

            # [핵심] 자연어 패딩 슬라이싱 (1토큰 ≈ 4글자로 보수적 계산)
            num_visual_tokens = img_embs.shape[0]
            needed_text_tokens = max(10, target_seq_len - num_visual_tokens - 20)
            padded_text = NATURAL_TEXT_POOL[:needed_text_tokens * 4] 
            
            t1a.start()
            prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|>{'<|image_pad|>' * num_visual_tokens}<|vision_end|>\nContext: {padded_text}\nDescribe the image based on the context.<|im_end|>\n<|im_start|>assistant\n"
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
                
        results.append({"Seq_Length": label, "Image Preprocessing": avg_preproc / measure_count, "Image Encoding": avg_encode / measure_count, "LLM Prefill": avg_prefill / measure_count})

pd.DataFrame(results).to_csv("qwen_seq_e2e.csv", index=False)