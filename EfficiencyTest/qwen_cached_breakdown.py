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
# DB Load 시뮬레이션을 위한 더미 임베딩 경로 폴더 (실제 텐서 크기에 맞춰야 함)
EMBED_DIR = "./qwen_vision_embeddings" 
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

# 해상도 설정 (DB 로드 크기를 추정하기 위함)
seq_configs = [
    {"label": "256", "size": (448, 448)},
    {"label": "1k",  "size": (896, 896)},
    {"label": "2k",  "size": (1260, 1260)},
    {"label": "4k",  "size": (1792, 1792)},
    {"label": "8k",  "size": (2520, 2520)}
]

results = {"labels": [], "preproc": [], "encode": [], "prefill": []}

print("Starting Qwen Cached TTFT Breakdown Measurement...")
with torch.no_grad():
    for config in tqdm(seq_configs, desc="Sequence Lengths"):
        label = config["label"]
        w, h = config["size"]
        
        # 임의의 실제 pt 파일 하나를 계속 로드한다고 가정 (파일 크기가 중요하므로)
        # 만약 실제 pt 파일이 없다면, 메모리에서 직접 텐서를 생성하여 DB Load 시간을 시뮬레이션 합니다.
        
        sum_preproc = 0.0; sum_encode = 0.0; sum_prefill = 0.0
        
        for i in range(WARMUP_SAMPLES + TEST_SAMPLES):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            text_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n"
            
            timer_preproc = CUDATimer(); timer_encode = CUDATimer(); timer_prefill = CUDATimer()

            # Cached: Image Preproc (0초), Image Encoding (DB Load + Merger)
            # 여기서는 DB Load 시간을 Encode 단계에 합치거나 (또는 Preproc를 0으로 둠)
            timer_preproc.start()
            # 캐시 모드이므로 이미지 전처리는 생략
            timer_preproc.stop()

            timer_encode.start()
            # 1. DB Load 시뮬레이션 (해당 시퀀스 길이에 맞는 텐서 생성으로 대체, 실제 구현시 torch.load)
            N_patches_approx = (h // 28) * (w // 28) # Qwen 패치 계산
            loaded_embeds = torch.randn((N_patches_approx, 1280), dtype=torch.bfloat16, device=device)
            grid_thw = torch.tensor([[1, h//28, w//28]], device=device)
            
            # 2. Merger
            llm_hidden_size = model.get_input_embeddings().weight.shape[1]
            if loaded_embeds.shape[-1] != llm_hidden_size: 
                vision_encoder = model.visual if hasattr(model, 'visual') else model.model.visual
                merger = getattr(vision_encoder, 'merger', None) or getattr(model.model.visual, 'merger', None)
                if merger: loaded_embeds = merger(loaded_embeds)
            timer_encode.stop()

            # 3. LLM Prefill
            text_inputs = processor.tokenizer(text_prompt, return_tensors="pt").to(device)
            inputs_embeds = model.get_input_embeddings()(text_inputs.input_ids)
            image_mask = (text_inputs.input_ids == processor.tokenizer.convert_tokens_to_ids("<|image_pad|>"))
            N_patches = loaded_embeds.shape[0]
            expanded_mask = torch.zeros(1, inputs_embeds.shape[1] + N_patches - 1, dtype=torch.bool, device=device)
            img_start_idx = torch.where(image_mask)[1][0]
            expanded_mask[0, img_start_idx:img_start_idx+N_patches] = True
            final_embeds = torch.cat([inputs_embeds[:, :img_start_idx, :], loaded_embeds.unsqueeze(0), inputs_embeds[:, img_start_idx+1:, :]], dim=1)
            final_attention_mask = torch.ones(1, final_embeds.shape[1], dtype=torch.long, device=device)

            ttft_processor = TTFTLogitsProcessor()
            processors = LogitsProcessorList([ttft_processor])
            
            timer_prefill.start()
            _ = model.generate(inputs_embeds=final_embeds, attention_mask=final_attention_mask, max_new_tokens=1, logits_processor=processors)
            timer_prefill.stop()
            torch.cuda.synchronize()

            if i >= WARMUP_SAMPLES:
                sum_preproc += timer_preproc.get_time() # 0에 수렴
                sum_encode += timer_encode.get_time() # DB Load + Merger 시간
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
ax.set_title('TTFT Breakdown (Qwen Cached)', fontsize=18, fontweight='bold', pad=15)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.grid(axis='y', linestyle=':', alpha=0.4, color='gray')

max_y = max(preproc + encode + prefill)
for y_line in [200, 400, 600, 800, 1000, 1200]:
    if y_line < max_y * 1.1: ax.axhline(y_line, color='#f28e8e', linestyle='--', alpha=0.6, linewidth=1.2)

ax.legend(loc='upper left', fontsize=14, framealpha=1.0, edgecolor='lightgray')
plt.tight_layout()
plt.savefig("Chart_Qwen_Cached_Breakdown.png", dpi=300, bbox_inches='tight')
print("✅ Saved: Chart_Qwen_Cached_Breakdown.png")