import torch
import os
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from tqdm import tqdm

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct" 
SAVE_DIR = "./qwen_embeddings"
os.makedirs(SAVE_DIR, exist_ok=True)

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda"
).eval()

def generate_qwen_caches():
    with torch.no_grad():
        for i in tqdm(range(1000), desc="Generating Qwen Caches"):
            image = Image.new('RGB', (1344, 1344), color=(i%255, i%255, i%255))
            inputs = processor(images=[image], text=["<|image_pad|>"], return_tensors="pt").to("cuda")
            
            v_out = model.visual.patch_embed(inputs.pixel_values.to(torch.bfloat16))
            for block in model.visual.blocks:
                v_out = block(v_out)
            image_embeds = model.visual.merger(v_out)
            
            torch.save({
                "embeds": image_embeds.cpu(),
                "grid_thw": inputs.image_grid_thw.cpu()
            }, f"{SAVE_DIR}/embed_{i}.pt")

if __name__ == "__main__":
    generate_qwen_caches()