import os
import time
import torch
from datasets import load_dataset
from transformers import LlavaForConditionalGeneration, AutoProcessor
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

MODEL_ID = "llava-hf/llava-1.5-7b-hf"
LOG_DIR = "./runs/llava_e2e_test"
NUM_TEST_SAMPLES = 500

writer = SummaryWriter(log_dir=LOG_DIR)
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading LLaVA Model & Processor...")
model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map=device
).eval()
processor = AutoProcessor.from_pretrained(MODEL_ID)

dataset = load_dataset("detection-datasets/coco", split=f"val[:{NUM_TEST_SAMPLES}]", trust_remote_code=True)

print("Starting LLaVA E2E Inference...")
total_pure_inference_time = 0.0

with torch.no_grad():
    for idx, data in enumerate(tqdm(dataset)):
        image = data['image'].convert("RGB") if data['image'].mode != "RGB" else data['image']
        
        # LLaVA 1.5 표준 프롬프트 형식
        prompt = "USER: <image>\nDescribe this image in detail.\nASSISTANT:"
        
        torch.cuda.reset_peak_memory_stats()
        start_time = time.time()
        
        # 1. 텍스트와 이미지 동시 처리
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        # pixel_values만 bfloat16으로 캐스팅 (input_ids는 int 유지)
        inputs["pixel_values"] = inputs["pixel_values"].to(device, dtype=torch.bfloat16)
        inputs["input_ids"] = inputs["input_ids"].to(device)
        inputs["attention_mask"] = inputs["attention_mask"].to(device)
        
        # 2. 모델 추론
        outputs = model.generate(**inputs, max_new_tokens=64)
        
        # 시간 및 성능 기록
        inference_time = time.time() - start_time
        total_pure_inference_time += inference_time
        
        # 입력 길이(프롬프트)를 제외한 순수 생성 토큰 수 계산
        input_len = inputs["input_ids"].shape[1]
        generated_tokens = outputs.shape[1] - input_len
        
        tokens_per_sec = generated_tokens / inference_time
        vram_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)

        writer.add_scalar('Metrics/1_Inference_Time_sec', inference_time, idx)
        writer.add_scalar('Metrics/2_Peak_VRAM_MB', vram_peak, idx)
        writer.add_scalar('Metrics/3_Tokens_Per_Sec', tokens_per_sec, idx)

print(f"LLaVA E2E 순수 모델 연산 총 소요 시간: {total_pure_inference_time:.2f}초")
writer.close()