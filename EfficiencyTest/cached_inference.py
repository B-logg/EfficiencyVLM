import os
import time
import torch
from datasets import load_dataset
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# 1. 설정
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
LOG_DIR = "./runs/cached_test"       # Cached 실험 로그
EMBED_DIR = "./vision_embeddings"    # pt 파일 경로
NUM_TEST_SAMPLES = 500               # 실험할 이미지 개수

writer = SummaryWriter(log_dir=LOG_DIR)
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading Model & Tokenizer...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map=device
).eval()
processor = AutoProcessor.from_pretrained(MODEL_ID)
tokenizer = processor.tokenizer
image_pad_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")

dataset = load_dataset("detection-datasets/coco", split=f"val[:{NUM_TEST_SAMPLES}]", trust_remote_code=True)

# 2. Cached 임베딩 주입 추론 실험 시작
print("Starting Cached Inference")

with torch.no_grad():
    for idx, data in enumerate(tqdm(dataset)):
        image_id = data['image_id']
        pt_path = os.path.join(EMBED_DIR, f"embed_{image_id}.pt")
        
        if not os.path.exists(pt_path):
            print(f"File missing: {pt_path}. Skipping...")
            continue
            
        # 1. 저장된 임베딩과 해상도 정보 로드
        saved_data = torch.load(pt_path)
        image_embeds = saved_data["embeds"].to(device, dtype=torch.bfloat16)
        grid_thw = saved_data["grid_thw"].to(device)
        
        N_patches = image_embeds.shape[0] # 비전 토큰 개수

        torch.cuda.reset_peak_memory_stats()
        start_time = time.time()
        
        # 2. 텍스트 프롬프트 수동 조립 - 이미지 토큰 자리 확보
        image_token_str = "<|vision_start|>" + ("<|image_pad|>" * N_patches) + "<|vision_end|>"
        text_prompt = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            f"{image_token_str}Describe this image in detail.<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        
        # 3. 텍스트 토큰화 및 임베딩 결합 (Injection)
        inputs = tokenizer(text_prompt, return_tensors="pt").to(device)
        input_ids = inputs.input_ids
        
        # 모델의 기본 임베딩 레이어를 통과시켜 텍스트 벡터 획득
        inputs_embeds = model.get_input_embeddings()(input_ids)
        
        # <|image_pad|> 위치를 찾아서, 우리가 로드한 비전 임베딩으로 교체
        image_mask = (input_ids == image_pad_id)
        inputs_embeds[image_mask] = image_embeds
        
        # 4. 모델 추론 (pixel_values 대신 inputs_embeds를 직접 전달)
        outputs = model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=inputs.attention_mask,
            image_grid_thw=grid_thw, # 2D RoPE 연산을 위해 필수
            max_new_tokens=64
        )
        
        # 소요 시간 및 생성된 텍스트 토큰 수 계산
        inference_time = time.time() - start_time
        generated_tokens = outputs.shape[1]
        tokens_per_sec = generated_tokens / inference_time
        vram_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)

        # 5. 텐서보드 로깅
        writer.add_scalar('Metrics/1_Inference_Time_sec', inference_time, idx)
        writer.add_scalar('Metrics/2_Peak_VRAM_MB', vram_peak, idx)
        writer.add_scalar('Metrics/3_Tokens_Per_Sec', tokens_per_sec, idx)

writer.close()