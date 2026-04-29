import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, LogitsProcessor, LogitsProcessorList
from tqdm import tqdm

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"
WARMUP_SAMPLES = 10
TEST_SAMPLES = 100

class CUDATimer:
    def __init__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
    def start(self): self.start_event.record()
    def stop(self): self.end_event.record()
    def get_time(self): return self.start_event.elapsed_time(self.end_event)

class TTFTLogitsProcessor(LogitsProcessor):
    def __init__(self):
        self.first_token_event = torch.cuda.Event(enable_timing=True)
        self.is_first = True
    def __call__(self, input_ids, scores):
        if self.is_first:
            self.first_token_event.record()
            self.is_first = False
        return scores

print("Loading Qwen Model & Processor...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map=device
).eval()
processor = AutoProcessor.from_pretrained(MODEL_ID)

seq_configs = [
    {"label": "256", "size": (448, 448)},
    {"label": "1k",  "size": (896, 896)},
    {"label": "2k",  "size": (1260, 1260)},
    {"label": "4k",  "size": (1792, 1792)},
    {"label": "8k",  "size": (2520, 2520)}
]

results = {"labels": [], "preproc": [], "encode": [], "prefill": []}

print("Starting Qwen E2E TTFT Breakdown Measurement...")
with torch.no_grad():
    for config in tqdm(seq_configs, desc="Sequence Lengths"):
        label = config["label"]
        w, h = config["size"]
        
        sum_preproc = 0.0; sum_encode = 0.0; sum_prefill = 0.0
        
        for i in range(WARMUP_SAMPLES + TEST_SAMPLES):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            dummy_image = Image.fromarray(np.random.randint(0, 255, (h, w, 3), dtype=np.uint8))
            text_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n"
            
            timer_preproc = CUDATimer(); timer_encode = CUDATimer(); timer_prefill = CUDATimer()

            # 1. Image Preprocessing
            timer_preproc.start()
            image_inputs = processor.image_processor(images=dummy_image, return_tensors="pt").to(device)
            pixel_values = image_inputs.pixel_values.to(dtype=torch.bfloat16)
            grid_thw = image_inputs.image_grid_thw
            timer_preproc.stop()

            # 2. Image Encoding (ViT + Merger)
            timer_encode.start()
            vision_encoder = model.visual if hasattr(model, 'visual') else model.model.visual
            vision_outputs = vision_encoder(pixel_values, grid_thw=grid_thw)
            image_embeds = vision_outputs.last_hidden_state if hasattr(vision_outputs, 'last_hidden_state') else (vision_outputs[0] if isinstance(vision_outputs, tuple) else vision_outputs)
            llm_hidden_size = model.get_input_embeddings().weight.shape[1]
            if image_embeds.shape[-1] != llm_hidden_size: 
                merger = getattr(vision_encoder, 'merger', None) or getattr(model.model.visual, 'merger', None)
                if merger: image_embeds = merger(image_embeds)
            timer_encode.stop()

            # 3. LLM Prefill (Fusion + Generate TTFT)
            text_inputs = processor.tokenizer(text_prompt, return_tensors="pt").to(device)
            inputs_embeds = model.get_input_embeddings()(text_inputs.input_ids)
            image_mask = (text_inputs.input_ids == processor.tokenizer.convert_tokens_to_ids("<|image_pad|>"))
            N_patches = image_embeds.shape[0]
            expanded_mask = torch.zeros(1, inputs_embeds.shape[1] + N_patches - 1, dtype=torch.bool, device=device)
            img_start_idx = torch.where(image_mask)[1][0]
            expanded_mask[0, img_start_idx:img_start_idx+N_patches] = True
            final_embeds = torch.cat([inputs_embeds[:, :img_start_idx, :], image_embeds.unsqueeze(0), inputs_embeds[:, img_start_idx+1:, :]], dim=1)
            final_attention_mask = torch.ones(1, final_embeds.shape[1], dtype=torch.long, device=device)

            ttft_processor = TTFTLogitsProcessor()
            processors = LogitsProcessorList([ttft_processor])
            
            timer_prefill.start()
            _ = model.generate(inputs_embeds=final_embeds, attention_mask=final_attention_mask, max_new_tokens=1, logits_processor=processors)
            timer_prefill.stop()
            torch.cuda.synchronize()

            if i >= WARMUP_SAMPLES:
                sum_preproc += timer_preproc.get_time()
                sum_encode += timer_encode.get_time()
                sum_prefill += timer_prefill.start_event.elapsed_time(ttft_processor.first_token_event)

        results["labels"].append(label)
        results["preproc"].append(sum_preproc / TEST_SAMPLES)
        results["encode"].append(sum_encode / TEST_SAMPLES)
        results["prefill"].append(sum_prefill / TEST_SAMPLES)

print("\nGenerating Chart...")
labels = results["labels"]
preproc = np.array(results["preproc"]); encode = np.array(results["encode"]); prefill = np.array(results["prefill"])

width = 0.55
plt.rcParams.update({'font.size': 12})
fig, ax = plt.subplots(figsize=(7, 6))

ax.bar(labels, preproc, width, label='Image Preprocessing', color='#6ebd6e', edgecolor='none')
ax.bar(labels, encode, width, bottom=preproc, label='Image Encoding', color='#f28e8e', hatch='////', edgecolor='#555555', linewidth=1)
ax.bar(labels, prefill, width, bottom=preproc + encode, label='LLM Prefill', color='#4f8bc6', hatch='\\\\\\\\', edgecolor='#555555', linewidth=1)

ax.set_ylabel('Time (ms)', fontsize=16, fontweight='bold')
ax.set_xlabel('Sequence Length', fontsize=16, fontweight='bold')
ax.set_title('TTFT Breakdown (Qwen E2E)', fontsize=18, fontweight='bold', pad=15)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.grid(axis='y', linestyle=':', alpha=0.4, color='gray')

max_y = max(preproc + encode + prefill)
for y_line in [200, 400, 600, 800, 1000, 1200]:
    if y_line < max_y * 1.1: ax.axhline(y_line, color='#f28e8e', linestyle='--', alpha=0.6, linewidth=1.2)

ax.legend(loc='upper left', fontsize=14, framealpha=1.0, edgecolor='lightgray')
plt.tight_layout()
plt.savefig("Chart_Qwen_E2E_Breakdown.png", dpi=300, bbox_inches='tight')
print("✅ Saved: Chart_Qwen_E2E_Breakdown.png")