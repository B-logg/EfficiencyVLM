import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import LlavaForConditionalGeneration, AutoProcessor, LogitsProcessor, LogitsProcessorList
from tqdm import tqdm

MODEL_ID = "llava-hf/llava-1.5-7b-hf"
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

print("Loading LLaVA Model & Processor...")
model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map=device
).eval()
processor = AutoProcessor.from_pretrained(MODEL_ID)
image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")

# [수정] E2E와 완벽하게 동일한 패치 수를 생성하도록 타일 개수 동기화
seq_configs = [
    {"label": "256", "tiles": 1},  # 576 patches
    {"label": "1k",  "tiles": 2},  # 1152 patches
    {"label": "2k",  "tiles": 4},  # 2304 patches
    {"label": "4k",  "tiles": 7},  # 4032 patches
    {"label": "8k",  "tiles": 14}  # 8064 patches
]

results = {"labels": [], "preproc": [], "encode": [], "prefill": []}

print("Starting LLaVA Cached TTFT Breakdown Measurement...")
with torch.no_grad():
    for config in tqdm(seq_configs, desc="Sequence Lengths"):
        label = config["label"]
        n_patches = config["tiles"] * 576
        sum_preproc = 0.0; sum_encode = 0.0; sum_prefill = 0.0
        
        for i in range(WARMUP_SAMPLES + TEST_SAMPLES):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            text_prompt = "USER: <image>\nDescribe this image in detail.\nASSISTANT:"
            timer_preproc = CUDATimer(); timer_encode = CUDATimer(); timer_prefill = CUDATimer()

            # Cached Mode: Preproc = 0
            timer_preproc.start(); timer_preproc.stop()

            # Encode Mode: DB Load + Projector
            timer_encode.start()
            # E2E와 동일한 패치 개수를 로드
            loaded_features = torch.randn((1, n_patches, 1024), dtype=torch.bfloat16, device=device)
            v_proj = model.multi_modal_projector if hasattr(model, 'multi_modal_projector') else model.model.multi_modal_projector
            image_embeds = v_proj(loaded_features) 
            timer_encode.stop()

            # LLM Prefill
            inputs_text = processor.tokenizer(text_prompt, return_tensors="pt").to(device)
            llm_backbone = model
            inputs_embeds = llm_backbone.get_input_embeddings()(inputs_text.input_ids)
            image_idx = torch.where(inputs_text.input_ids == image_token_id)[1][0]
            final_embeds = torch.cat([inputs_embeds[:, :image_idx, :], image_embeds, inputs_embeds[:, image_idx+1:, :]], dim=1)
            image_mask = torch.ones((1, image_embeds.shape[1]), dtype=inputs_text.attention_mask.dtype, device=device)
            final_mask = torch.cat([inputs_text.attention_mask[:, :image_idx], image_mask, inputs_text.attention_mask[:, image_idx+1:]], dim=1)

            ttft_processor = TTFTLogitsProcessor()
            processors = LogitsProcessorList([ttft_processor])
            
            timer_prefill.start()
            _ = llm_backbone.generate(inputs_embeds=final_embeds, attention_mask=final_mask, max_new_tokens=1, logits_processor=processors)
            timer_prefill.stop()
            torch.cuda.synchronize()

            if i >= WARMUP_SAMPLES:
                sum_preproc += timer_preproc.get_time() # 0
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
ax.bar(labels, encode, width, bottom=preproc, label='Image Encoding (DB Load)', color='#f28e8e', hatch='////', edgecolor='#555555', linewidth=1)
ax.bar(labels, prefill, width, bottom=preproc + encode, label='LLM Prefill', color='#4f8bc6', hatch='\\\\\\\\', edgecolor='#555555', linewidth=1)

ax.set_ylabel('Time (ms)', fontsize=16, fontweight='bold')
ax.set_xlabel('Sequence Length', fontsize=16, fontweight='bold')
ax.set_title('TTFT Breakdown (LLaVA Cached)', fontsize=18, fontweight='bold', pad=15)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.grid(axis='y', linestyle=':', alpha=0.4, color='gray')

max_y = max(preproc + encode + prefill)
for y_line in [200, 400, 600, 800, 1000, 1200]:
    if y_line < max_y * 1.1: ax.axhline(y_line, color='#f28e8e', linestyle='--', alpha=0.6, linewidth=1.2)

ax.legend(loc='upper left', fontsize=14, framealpha=1.0, edgecolor='lightgray')
plt.tight_layout()
plt.savefig("Chart_LLaVA_Cached_Breakdown.png", dpi=300, bbox_inches='tight')
print("✅ Saved: Chart_LLaVA_Cached_Breakdown.png")