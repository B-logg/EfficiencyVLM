import os
import time
import torch
from datasets import load_dataset
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# 1. 설정
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
LOG_DIR = "./runs/e2e_test"  # E2E 실험 로그
NUM_TEST_SAMPLES = 500      # 실험할 이미지 개수

writer = SummaryWriter(log_dir=LOG_DIR)
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading Model & Processor...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map=device
).eval()
processor = AutoProcessor.from_pretrained(MODEL_ID)

dataset = load_dataset("detection-datasets/coco", split=f"val[:{NUM_TEST_SAMPLES}]", trust_remote_code=True)

# 2. E2E 추론 실험 시작
print("Starting E2E Inference")

with torch.no_grad():
    for idx, data in enumerate(tqdm(dataset)):
        image = data['image'].convert("RGB") if data['image'].mode != "RGB" else data['image']
        
        # 1. 프롬프트 준비
        messages = [
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": "Describe this image in detail."}]}
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        torch.cuda.reset_peak_memory_stats()
        start_time = time.time()
        
        # 2. 프로세서 통과 (이미지 처리 + 텍스트 토큰화)
        inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)
        
        # 3. 모델 추론 (생성)
        outputs = model.generate(**inputs, max_new_tokens=64)
        
        # 소요 시간 및 생성된 텍스트 토큰 수 계산
        inference_time = time.time() - start_time
        generated_tokens = outputs.shape[1] - inputs.input_ids.shape[1]
        tokens_per_sec = generated_tokens / inference_time
        vram_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)

        # 4. 텐서보드 로깅
        writer.add_scalar('Metrics/1_Inference_Time_sec', inference_time, idx)
        writer.add_scalar('Metrics/2_Peak_VRAM_MB', vram_peak, idx)
        writer.add_scalar('Metrics/3_Tokens_Per_Sec', tokens_per_sec, idx)

writer.close()