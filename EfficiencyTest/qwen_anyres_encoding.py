import torch
import os
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from tqdm import tqdm

# 선생님이 원래 쓰시던 2B 모델로 복구!
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
            # LLaVA 8K 부하와 동일한 조건 (1344x1344)
            image = Image.new('RGB', (1344, 1344), color=(i%255, i%255, i%255))
            
            inputs = processor(images=[image], text=["<|image_pad|>"], return_tensors="pt").to("cuda")
            pixel_values = inputs.pixel_values.to(torch.bfloat16)
            grid_thw = inputs.image_grid_thw
            
            # 2a. ViT (Backbone)
            v_out = model.visual.patch_embed(pixel_values)
            for block in model.visual.blocks:
                v_out = block(v_out)
            
            # 2b. MLP (Merger)
            image_embeds = model.visual.merger(v_out)
            
            # 중요: 선생님의 기존 코드 구조처럼 embeds와 grid_thw를 묶어서 저장
            torch.save({
                "embeds": image_embeds.cpu(),
                "grid_thw": grid_thw.cpu()
            }, f"{SAVE_DIR}/embed_{i}.pt")

if __name__ == "__main__":
    generate_qwen_caches()