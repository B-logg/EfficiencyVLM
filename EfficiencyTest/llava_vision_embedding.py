import os
import time
import torch
from datasets import load_dataset
from transformers import LlavaForConditionalGeneration, AutoProcessor
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# 1. 설정
MODEL_ID = "llava-hf/llava-1.5-7b-hf"
SAVE_DIR = "./llava_vision_embeddings"
LOG_DIR = "./runs/llava_encode_experiment"

os.makedirs(SAVE_DIR, exist_ok=True)
writer = SummaryWriter(log_dir=LOG_DIR)
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading Model & Processor: {MODEL_ID}")
model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map=device
).eval()
processor = AutoProcessor.from_pretrained(MODEL_ID)

dataset = load_dataset("detection-datasets/coco", split="val", trust_remote_code=True)

# 2. Vision Encoding (Projector까지 통과)
print("Starting LLaVA Vision Encoding Process")
start_total_time = time.time()

with torch.no_grad():
    for idx, data in enumerate(tqdm(dataset)):
        image = data['image'].convert("RGB") if data['image'].mode != "RGB" else data['image']
        image_id = data['image_id']
        
        start_time = time.time()
        
        # 프로세서로 이미지를 336x336 텐서로 변환
        inputs = processor.image_processor(images=image, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(device, dtype=torch.bfloat16)
        
        # LLaVA의 Vision Tower + MLP Projector를 한 번에 통과 (결과 shape: [1, 576, 4096])
        image_embeds = model.get_image_features(pixel_values)
        
        process_time = time.time() - start_time
        
        save_path = os.path.join(SAVE_DIR, f"embed_{image_id}.pt")
        torch.save(image_embeds.cpu(), save_path)
        
        # 로깅
        writer.add_scalar('Performance/Encoding_Time_per_Image', process_time, idx)
        if torch.cuda.is_available():
            vram_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
            writer.add_scalar('Memory/VRAM_Peak_MB', vram_peak, idx)
            torch.cuda.reset_peak_memory_stats()

total_time = time.time() - start_total_time
print(f"LLaVA 토큰화 완료! 총 소요 시간: {total_time:.2f}초")
writer.close()