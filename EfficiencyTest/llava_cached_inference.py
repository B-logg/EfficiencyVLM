import os
import time
import torch
from datasets import load_dataset
from transformers import LlavaForConditionalGeneration, AutoProcessor
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

MODEL_ID = "llava-hf/llava-1.5-7b-hf"
LOG_DIR = "./runs/llava_cached_test"
EMBED_DIR = "./llava_vision_embeddings"
NUM_TEST_SAMPLES = 100

writer = SummaryWriter(log_dir=LOG_DIR)
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading LLaVA LLM Backbone...")
model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map=device
).eval()
processor = AutoProcessor.from_pretrained(MODEL_ID)
tokenizer = processor.tokenizer
image_token_id = tokenizer.convert_tokens_to_ids("<image>")

dataset = load_dataset("detection-datasets/coco", split=f"val[:{NUM_TEST_SAMPLES}]", trust_remote_code=True)

print("Starting LLaVA Cached Inference...")
total_pure_inference_time = 0.0

with torch.no_grad():
    for idx, data in enumerate(tqdm(dataset)):
        image_id = data['image_id']
        pt_path = os.path.join(EMBED_DIR, f"embed_{image_id}.pt")
        
        if not os.path.exists(pt_path):
            continue
            
        # 1. 캐싱된 576개의 시각 토큰 임베딩 로드
        image_embeds = torch.load(pt_path).to(device, dtype=torch.bfloat16) # shape: [1, 576, 4096]
        
        torch.cuda.reset_peak_memory_stats()
        start_time = time.time()
        
        # 2. 텍스트 프롬프트 토큰화
        prompt = "USER: <image>\nDescribe this image in detail.\nASSISTANT:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        
        # 3. 텍스트를 기본 임베딩으로 변환
        inputs_embeds = model.language_model.get_input_embeddings()(input_ids)
        
        # 4. 모달리티 결합 (수동 Injection)
        # input_ids에서 <image> 토큰(32000)의 위치를 찾음
        image_idx = torch.where(input_ids == image_token_id)[1][0]
        
        # <image> 토큰을 기점으로 앞, 뒤 텍스트 임베딩을 분리하고 사이에 시각 임베딩을 끼워 넣음
        final_embeds = torch.cat([
            inputs_embeds[:, :image_idx, :],
            image_embeds,
            inputs_embeds[:, image_idx+1:, :]
        ], dim=1)
        
        # attention_mask도 늘어난 토큰 수(576개)에 맞게 확장
        image_mask = torch.ones((1, image_embeds.shape[1]), dtype=attention_mask.dtype, device=device)
        final_mask = torch.cat([
            attention_mask[:, :image_idx],
            image_mask,
            attention_mask[:, image_idx+1:]
        ], dim=1)
        
        # 5. 오직 언어 모델(LLaMA)만 구동하여 추론
        outputs = model.language_model.generate(
            inputs_embeds=final_embeds,
            attention_mask=final_mask,
            max_new_tokens=64
        )
        
        inference_time = time.time() - start_time
        total_pure_inference_time += inference_time
        
        generated_tokens = outputs.shape[1]
        tokens_per_sec = generated_tokens / inference_time
        vram_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)

        writer.add_scalar('Metrics/1_Inference_Time_sec', inference_time, idx)
        writer.add_scalar('Metrics/2_Peak_VRAM_MB', vram_peak, idx)
        writer.add_scalar('Metrics/3_Tokens_Per_Sec', tokens_per_sec, idx)

print(f"✅ LLaVA Cached 순수 모델 연산 총 소요 시간: {total_pure_inference_time:.2f}초")
writer.close()