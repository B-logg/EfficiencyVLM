import torch
import os
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from tqdm import tqdm
from datasets import load_dataset

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
SAVE_DIR = "./qwen_vision_embeddings"
NUM_TEST_SAMPLES = 3100
os.makedirs(SAVE_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map=device).eval()
dataset = load_dataset("detection-datasets/coco", split=f"val[:{NUM_TEST_SAMPLES}]", trust_remote_code=True)

print("Starting Qwen Encoding...")
with torch.no_grad():
    for data in tqdm(dataset):
        image = data['image'].convert("RGB") if data['image'].mode != "RGB" else data['image']
        image_id = data['image_id']
        
        image_inputs = processor.image_processor(images=image, return_tensors="pt").to(device)
        pixel_values = image_inputs.pixel_values.to(dtype=torch.bfloat16)
        grid_thw = image_inputs.image_grid_thw
        
        vision_encoder = getattr(model, 'visual', getattr(getattr(model, 'model', None), 'visual', None))
        vision_outputs = vision_encoder(pixel_values, grid_thw=grid_thw)
        
        if hasattr(vision_outputs, 'last_hidden_state'):
            image_embeds = vision_outputs.last_hidden_state
        elif isinstance(vision_outputs, tuple):
            image_embeds = vision_outputs[0]
        else:
            image_embeds = vision_outputs
            
        merger = getattr(vision_encoder, 'merger', None)
        if merger is not None and image_embeds.shape[-1] != model.get_input_embeddings().weight.shape[1]:
            image_embeds = merger(image_embeds)
            
        torch.save({"embeds": image_embeds.cpu(), "grid_thw": grid_thw.cpu()}, f"{SAVE_DIR}/embed_{image_id}.pt")