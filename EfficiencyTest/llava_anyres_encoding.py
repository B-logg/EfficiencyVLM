import torch
import os
from PIL import Image
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
from tqdm import tqdm

MODEL_ID = "llava-hf/llava-v1.6-vicuna-7b-hf"
SAVE_DIR = "./llava_embeddings"
os.makedirs(SAVE_DIR, exist_ok=True)

processor = LlavaNextProcessor.from_pretrained(MODEL_ID)
model = LlavaNextForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda"
).eval()

def generate_caches():
    with torch.no_grad():
        for i in tqdm(range(1000), desc="Generating LLaVA AnyRes Caches"):
            # AnyRes 체감을 위해 고해상도 생성 (1344x1344)
            image = Image.new('RGB', (1344, 1344), color=(i%255, i%255, i%255))
            inputs = processor(images=image, text="<image>", return_tensors="pt").to("cuda", torch.bfloat16)
            
            # LlavaNext 구조: model.model.vision_tower로 접근
            vision_tower = model.model.vision_tower
            multi_modal_projector = model.multi_modal_projector
            
            # Vision Tower 연산
            image_outputs = vision_tower(inputs.pixel_values, output_hidden_states=True)
            # Projector 연산
            image_features = multi_modal_projector(image_outputs.last_hidden_state)
            
            torch.save(image_features.cpu(), f"{SAVE_DIR}/embed_{i}.pt")

if __name__ == "__main__":
    generate_caches()