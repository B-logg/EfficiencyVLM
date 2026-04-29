import torch
import os
from PIL import Image
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
from tqdm import tqdm
from datasets import load_dataset

MODEL_ID = "llava-hf/llava-v1.6-vicuna-7b-hf"
SAVE_DIR = "./llava_embeddings"
NUM_TEST_SAMPLES = 1010
os.makedirs(SAVE_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = LlavaNextProcessor.from_pretrained(MODEL_ID)
model = LlavaNextForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map=device).eval()
dataset = load_dataset("detection-datasets/coco", split=f"val[:{NUM_TEST_SAMPLES}]", trust_remote_code=True)

print("Starting LLaVA Encoding...")
with torch.no_grad():
    for data in tqdm(dataset):
        image = data['image'].convert("RGB") if data['image'].mode != "RGB" else data['image']
        image_id = data['image_id']
        
        inputs = processor(images=image, text="<image>", return_tensors="pt").to(device, torch.bfloat16)
        pixel_values = inputs.pixel_values
        
        # [에러 해결] LLaVA-v1.6의 pixel_values는 (1, num_patches, 3, 336, 336) 입니다.
        # vision_tower에 넣을 때 image_sizes 정보가 필요할 수 있으므로, 모델 자체의 forward 경로를 일부 활용합니다.
        vision_tower = model.vision_tower
        multi_modal_projector = model.multi_modal_projector
        
        image_outputs = vision_tower(pixel_values, output_hidden_states=True)
        # AnyRes feature selection
        selected_image_feature = image_outputs.hidden_states[-2]
        image_features = multi_modal_projector(selected_image_feature)
        
        # 저장
        torch.save(image_features.cpu(), f"{SAVE_DIR}/embed_{image_id}.pt")