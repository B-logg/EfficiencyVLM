import torch
import os
from PIL import Image
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
from tqdm import tqdm
from datasets import load_dataset

MODEL_ID = "llava-hf/llava-v1.6-vicuna-7b-hf"
SAVE_DIR = "./llava_vision_embeddings"
NUM_TEST_SAMPLES = 3100
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
        
        # [에러 해결] LLaVA-v1.6 AnyRes의 5D 텐서를 ViT가 인식할 수 있는 4D로 풀어줍니다.
        if pixel_values.dim() == 5:
            b, num_patches, c, h, w = pixel_values.shape
            pixel_values = pixel_values.view(b * num_patches, c, h, w)
        
        vision_tower = getattr(model, 'vision_tower', getattr(getattr(model, 'model', None), 'vision_tower', None))
        multi_modal_projector = getattr(model, 'multi_modal_projector', getattr(getattr(model, 'model', None), 'multi_modal_projector', None))
        
        image_outputs = vision_tower(pixel_values, output_hidden_states=True)
        image_features = multi_modal_projector(image_outputs.hidden_states[-2])
        
        torch.save(image_features.cpu(), f"{SAVE_DIR}/embed_{image_id}.pt")