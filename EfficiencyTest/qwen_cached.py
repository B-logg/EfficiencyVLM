import torch
import pandas as pd
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, LogitsProcessor, LogitsProcessorList
from tqdm import tqdm

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct" # 2B로 복구
EMBED_DIR = "./qwen_embeddings"
NUM_SAMPLES = 1010
WARMUP = 10

class TTFTLogitsProcessor(LogitsProcessor):
    def __init__(self):
        self.first_token_event = torch.cuda.Event(enable_timing=True)
        self.is_first = True
    def __call__(self, input_ids, scores):
        if self.is_first:
            self.first_token_event.record()
            self.is_first = False
        return scores

model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda").eval()
processor = AutoProcessor.from_pretrained(MODEL_ID)

results = []
with torch.no_grad():
    for i in tqdm(range(NUM_SAMPLES), desc="Qwen Cached"):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        
        # 1a. Text
        t1a_s = torch.cuda.Event(enable_timing=True); t1a_e = torch.cuda.Event(enable_timing=True)
        t1a_s.record()
        txt_in = processor.tokenizer("<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe.<|im_end|>\n<|im_start|>assistant\n", return_tensors="pt").to("cuda")
        t1a_e.record()
        
        r1b = 0.0; r2a = 0.0

        # 2b. DB Load (딕셔너리 언패킹)
        t2b_s = torch.cuda.Event(enable_timing=True); t2b_e = torch.cuda.Event(enable_timing=True)
        t2b_s.record()
        saved_data = torch.load(f"{EMBED_DIR}/embed_{i%1000}.pt")
        img_feats = saved_data["embeds"].to("cuda")
        grid_thw = saved_data["grid_thw"].to("cuda") # 꺼내오기!
        t2b_e.record()

        # 3. Fusion
        t3_s = torch.cuda.Event(enable_timing=True); t3_e = torch.cuda.Event(enable_timing=True)
        t3_s.record()
        inputs_embeds = model.get_input_embeddings()(txt_in.input_ids)
        image_mask = (txt_in.input_ids == processor.tokenizer.convert_tokens_to_ids("<|image_pad|>"))
        inputs_embeds[image_mask] = img_feats
        t3_e.record()

        # 4. Gen
        torch.cuda.synchronize()
        t4_start_ev = torch.cuda.Event(enable_timing=True); t4_start_ev.record()
        handler = TTFTLogitsProcessor()
        outputs = model.generate(inputs_embeds=inputs_embeds, attention_mask=txt_in.attention_mask, image_grid_thw=grid_thw, max_new_tokens=20, logits_processor=LogitsProcessorList([handler]), use_cache=True)
        t_total_end = torch.cuda.Event(enable_timing=True); t_total_end.record(); torch.cuda.synchronize()

        if i >= WARMUP:
            ms = 1000.0
            r1a=t1a_s.elapsed_time(t1a_e)/ms; r2b=t2b_s.elapsed_time(t2b_e)/ms; r3=t3_s.elapsed_time(t3_e)/ms; r4=t4_start_ev.elapsed_time(handler.first_token_event)/ms
            true_ttft = r1a + r1b + r2a + r2b + r3 + r4
            decode_time = handler.first_token_event.elapsed_time(t_total_end)/ms
            results.append([r1a, r1b, r2a, r2b, r3, r4, true_ttft, decode_time, true_ttft + decode_time, torch.cuda.max_memory_allocated()/1024**3, outputs.shape[1]])

pd.DataFrame(results, columns=['1a_text','1b_img','2a_vit','2b_mlp','3_fusion','4_gen','true_ttft','decode_time','total_latency','vram','tokens']).to_csv("qwen_cached_results.csv", index=False)