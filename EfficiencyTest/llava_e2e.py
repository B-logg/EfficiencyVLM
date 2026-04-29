import torch
import pandas as pd
from PIL import Image
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor, LogitsProcessor, LogitsProcessorList
from tqdm import tqdm

MODEL_ID = "llava-hf/llava-next-7b-hf"
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

model = LlavaNextForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda").eval()
processor = LlavaNextProcessor.from_pretrained(MODEL_ID)

results = []
with torch.no_grad():
    for i in tqdm(range(NUM_SAMPLES), desc="LLaVA E2E"):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        img = Image.new('RGB', (1344, 1344), color='white')
        
        # 1a. Text Pre
        t1a_s = torch.cuda.Event(enable_timing=True); t1a_e = torch.cuda.Event(enable_timing=True)
        t1a_s.record()
        txt_in = processor.tokenizer("USER: <image>\nDescribe. ASSISTANT:", return_tensors="pt").to("cuda")
        t1a_e.record()
        
        # 1b. Image Pre
        t1b_s = torch.cuda.Event(enable_timing=True); t1b_e = torch.cuda.Event(enable_timing=True)
        t1b_s.record()
        img_in = processor.image_processor(img, return_tensors="pt").to("cuda")
        t1b_e.record()

        # 2a. ViT
        t2a_s = torch.cuda.Event(enable_timing=True); t2a_e = torch.cuda.Event(enable_timing=True)
        t2a_s.record()
        v_out = model.vision_tower(img_in.pixel_values.to(torch.bfloat16))
        t2a_e.record()

        # 2b. Projector
        t2b_s = torch.cuda.Event(enable_timing=True); t2b_e = torch.cuda.Event(enable_timing=True)
        t2b_s.record()
        img_feats = model.multi_modal_projector(v_out.last_hidden_state)
        t2b_e.record()

        # 3. Fusion
        t3_s = torch.cuda.Event(enable_timing=True); t3_e = torch.cuda.Event(enable_timing=True)
        t3_s.record(); t3_e.record() # Simplified

        # 4. Gen
        torch.cuda.synchronize()
        t4_start_ev = torch.cuda.Event(enable_timing=True); t4_start_ev.record()
        handler = TTFTLogitsProcessor()
        outputs = model.generate(inputs_embeds=model.get_input_embeddings()(txt_in.input_ids), max_new_tokens=20, logits_processor=LogitsProcessorList([handler]))
        torch.cuda.synchronize()
        t_total_end = torch.cuda.Event(enable_timing=True); t_total_end.record(); torch.cuda.synchronize()

        if i >= WARMUP:
            ms = 1000.0
            r1a=t1a_s.elapsed_time(t1a_e)/ms; r1b=t1b_s.elapsed_time(t1b_e)/ms; r2a=t2a_s.elapsed_time(t2a_e)/ms
            r2b=t2b_s.elapsed_time(t2b_e)/ms; r3=t3_s.elapsed_time(t3_e)/ms; r4=t4_start_ev.elapsed_time(handler.first_token_event)/ms
            true_ttft = r1a + r1b + r2a + r2b + r3 + r4
            total_latency = true_ttft + (handler.first_token_event.elapsed_time(t_total_end)/ms)
            results.append([r1a, r1b, r2a, r2b, r3, r4, true_ttft, total_latency, torch.cuda.max_memory_allocated()/1024**3, outputs.shape[1]])

pd.DataFrame(results, columns=['1a_text','1b_img','2a_vit','2b_mlp','3_fusion','4_gen','true_ttft','total_latency','vram','tokens']).to_csv("llava_e2e_results.csv", index=False)