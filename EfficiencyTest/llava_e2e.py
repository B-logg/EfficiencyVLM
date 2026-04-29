import torch
import pandas as pd
from PIL import Image
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor, LogitsProcessor, LogitsProcessorList
from tqdm import tqdm

MODEL_ID = "llava-hf/llava-v1.6-vicuna-7b-hf"
NUM_SAMPLES = 1010
WARMUP = 10

class TTFTLogitsProcessor(LogitsProcessor):
    def __init__(self):
        self.first_token_event = torch.cuda.Event(enable_timing=True)
        self.is_first = True
    def __call__(self, input_ids, scores):
        if self.is_first: self.first_token_event.record(); self.is_first = False
        return scores

model = LlavaNextForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda").eval()
processor = LlavaNextProcessor.from_pretrained(MODEL_ID)
image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")

results = []
with torch.no_grad():
    for i in tqdm(range(NUM_SAMPLES), desc="LLaVA E2E"):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        img = Image.new('RGB', (1344, 1344), color='white')
        
        # 1a. Text
        t1a_s = torch.cuda.Event(enable_timing=True); t1a_e = torch.cuda.Event(enable_timing=True)
        t1a_s.record(); txt_in = processor.tokenizer("USER: <image>\nDescribe. ASSISTANT:", return_tensors="pt").to("cuda"); t1a_e.record()
        
        # 1b. Image
        t1b_s = torch.cuda.Event(enable_timing=True); t1b_e = torch.cuda.Event(enable_timing=True)
        t1b_s.record(); img_in = processor.image_processor(img, return_tensors="pt").to("cuda"); t1b_e.record()

        # 2a. ViT
        t2a_s = torch.cuda.Event(enable_timing=True); t2a_e = torch.cuda.Event(enable_timing=True)
        t2a_s.record(); v_out = model.model.vision_tower(img_in.pixel_values.to(torch.bfloat16), output_hidden_states=True); t2a_e.record()

        # 2b. Projector
        t2b_s = torch.cuda.Event(enable_timing=True); t2b_e = torch.cuda.Event(enable_timing=True)
        t2b_s.record(); img_feats = model.model.multi_modal_projector(v_out.last_hidden_state)
        img_feats = img_feats.view(1, -1, img_feats.shape[-1]) # 동적 토큰 이어붙이기
        t2b_e.record()

        # 3. Fusion
        t3_s = torch.cuda.Event(enable_timing=True); t3_e = torch.cuda.Event(enable_timing=True)
        t3_s.record()
        inputs_embeds = model.get_input_embeddings()(txt_in.input_ids)
        image_idx = torch.where(txt_in.input_ids == image_token_id)[1][0]
        final_embeds = torch.cat([inputs_embeds[:, :image_idx, :], img_feats, inputs_embeds[:, image_idx+1:, :]], dim=1)
        image_mask = torch.ones((1, img_feats.shape[1]), dtype=txt_in.attention_mask.dtype, device="cuda")
        final_mask = torch.cat([txt_in.attention_mask[:, :image_idx], image_mask, txt_in.attention_mask[:, image_idx+1:]], dim=1)
        t3_e.record()

        # 4. Gen
        torch.cuda.synchronize()
        t4_s = torch.cuda.Event(enable_timing=True); t4_s.record(); handler = TTFTLogitsProcessor()
        outputs = model.generate(inputs_embeds=final_embeds, attention_mask=final_mask, max_new_tokens=20, logits_processor=LogitsProcessorList([handler]))
        t_end = torch.cuda.Event(enable_timing=True); t_end.record(); torch.cuda.synchronize()

        if i >= WARMUP:
            r1a=t1a_s.elapsed_time(t1a_e)/1000; r1b=t1b_s.elapsed_time(t1b_e)/1000
            r2a=t2a_s.elapsed_time(t2a_e)/1000; r2b=t2b_s.elapsed_time(t2b_e)/1000; r3=t3_s.elapsed_time(t3_e)/1000
            r4 = t4_s.elapsed_time(handler.first_token_event)/1000
            ttft = r1a+r1b+r2a+r2b+r3+r4; decode = handler.first_token_event.elapsed_time(t_end)/1000
            results.append([r1a, r1b, r2a, r2b, r3, r4, ttft, decode, ttft+decode, torch.cuda.max_memory_allocated()/1024**3, outputs.shape[1]])

pd.DataFrame(results, columns=['1a_text','1b_img','2a_vit','2b_mlp','3_fusion','4_gen','true_ttft','decode_time','total_latency','vram','tokens']).to_csv("llava_e2e_results.csv", index=False)