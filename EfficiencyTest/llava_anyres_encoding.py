import torch
import os
from PIL import Image
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
from tqdm import tqdm

# 모델 설정
MODEL_ID = "llava-hf/llava-v1.6-vicuna-7b-hf" # AnyRes 지원 모델
SAVE_DIR = "./llava_next_embeddings"
os.makedirs(SAVE_DIR, exist_ok=True)

processor = LlavaNextProcessor.from_pretrained(MODEL_ID)
model = LlavaNextForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda"
).eval()

def generate_caches(num_images=1000):
    with torch.no_grad():
        for i in tqdm(range(num_images)):
            # 다양한 해상도 시뮬레이션 (AnyRes 작동 확인용)
            w, h = (672, 672) if i % 2 == 0 else (1008, 672) 
            image = Image.new('RGB', (w, h), color=(i%255, i%255, i%255))
            
            inputs = processor(images=image, text="<image>", return_tensors="pt").to("cuda", torch.bfloat16)
            
            # 1. Vision Encoder + Projector 연산만 수행
            # Llava-Next의 경우 multi_modal_projector와 vision_tower를 직접 호출
            image_outputs = model.vision_tower(inputs.pixel_values, output_hidden_states=True)
            selected_features = image_outputs.last_hidden_state
            image_features = model.multi_modal_projector(selected_features)
            
            # 2. 결과 저장 (.pt)
            torch.save(image_features.cpu(), f"{SAVE_DIR}/embed_{i}.pt")

if __name__ == "__main__":
    generate_caches()