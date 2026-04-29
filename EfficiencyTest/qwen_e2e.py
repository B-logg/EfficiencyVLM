import os
import torch
import pandas as pd
from datasets import load_dataset
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, LogitsProcessor, LogitsProcessorList
from tqdm import tqdm

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
NUM_TEST_SAMPLES = 1010
WARMUP_SAMPLES = 10
device = "cuda" if torch.cuda.is_available() else "cpu"

class CUDATimer:
    def __init__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
    def start(self): self.start_event.record()
    def stop(self): self.end_event.record()
    def get_time(self): return self.start_event.elapsed_time(self.end_event) / 1000.0

class TTFTLogitsProcessor(LogitsProcessor):
    def __init__(self):
        self.first_token_event = torch.cuda.Event(enable_timing=True)
        self.is_first = True
    def __call__(self, input_ids, scores):
        if self.is_first:
            self.first_token_event.record()
            self.is_first = False
        return scores

print("Loading Qwen E2E Model...")
model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map=device).eval()
processor = AutoProcessor.from_pretrained(MODEL_ID)
dataset = load_dataset("detection-datasets/coco", split=f"val[:{NUM_TEST_SAMPLES}]", trust_remote_code=True)

results = []
print("Starting Qwen E2E Inference...")
with torch.no_grad():
    for idx, data in enumerate(tqdm(dataset)):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        
        image = data['image'].convert("RGB") if data['image'].mode != "RGB" else data['image']
        timer_text = CUDATimer(); timer_img = CUDATimer(); timer_vit = CUDATimer(); timer_mlp = CUDATimer(); timer_fusion = CUDATimer(); timer_gen = CUDATimer()
        
        timer_img.start()
        img_in = processor.image_processor(images=image, return_tensors="pt").to(device)
        pixel_values = img_in.pixel_values.to(dtype=torch.bfloat16)
        grid_thw = img_in.image_grid_thw
        timer_img.stop()
        
        timer_vit.start()
        vision_encoder = getattr(model, 'visual', getattr(getattr(model, 'model', None), 'visual', None))
        vision_outputs = vision_encoder(pixel_values, grid_thw=grid_thw)
        if hasattr(vision_outputs, 'last_hidden_state'):
            image_embeds = vision_outputs.last_hidden_state
        elif isinstance(vision_outputs, tuple):
            image_embeds = vision_outputs[0]
        else:
            image_embeds = vision_outputs
        timer_vit.stop()
        
        timer_mlp.start()
        merger = getattr(vision_encoder, 'merger', None)
        if merger is not None and image_embeds.shape[-1] != model.get_input_embeddings().weight.shape[1]:
            image_embeds = merger(image_embeds)
        N_patches = image_embeds.shape[0]
        timer_mlp.stop()

        timer_text.start()
        image_token_str = "<|vision_start|>" + ("<|image_pad|>" * N_patches) + "<|vision_end|>"
        text_prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{image_token_str}Describe this image in detail.<|im_end|>\n<|im_start|>assistant\n"
        txt_in = processor.tokenizer(text_prompt, return_tensors="pt").to(device)
        timer_text.stop()
        
        timer_fusion.start()
        in_embs = model.get_input_embeddings()(txt_in.input_ids)
        image_mask = (txt_in.input_ids == processor.tokenizer.convert_tokens_to_ids("<|image_pad|>"))
        in_embs[image_mask] = image_embeds
        timer_fusion.stop()
        
        torch.cuda.synchronize()
        timer_gen.start()
        hnd = TTFTLogitsProcessor()
        outs = model.generate(inputs_embeds=in_embs, attention_mask=txt_in.attention_mask, image_grid_thw=grid_thw, max_new_tokens=64, logits_processor=LogitsProcessorList([hnd]))
        timer_gen.stop()
        torch.cuda.synchronize()

        if idx >= WARMUP_SAMPLES:
            r1a, r1b, r2a, r2b, r3 = timer_text.get_time(), timer_img.get_time(), timer_vit.get_time(), timer_mlp.get_time(), timer_fusion.get_time()
            r4 = timer_gen.start_event.elapsed_time(hnd.first_token_event) / 1000.0
            true_ttft = r1a + r1b + r2a + r2b + r3 + r4
            decode = (timer_gen.get_time() - r4) if outs.shape[1] > 1 else 0.0
            results.append([r1a, r1b, r2a, r2b, r3, r4, true_ttft, decode, true_ttft+decode, torch.cuda.max_memory_allocated()/(1024**3), outs.shape[1]])

df = pd.DataFrame(results, columns=['1a_text','1b_img','2a_vit','2b_mlp','3_fusion','4_gen','true_ttft','decode_time','total_latency','vram','tokens'])
df.to_csv("qwen_e2e.csv", index=False)