import os
import time
import torch
from datasets import load_dataset
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# 설정 및 초기화
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
SAVE_DIR = "./vision_embeddings"       # 임베딩 파일 저장 경로
LOG_DIR = "./runs/qwen_vl_experiment"  # 텐서보드 로그

os.makedirs(SAVE_DIR, exist_ok=True)
writer = SummaryWriter(log_dir=LOG_DIR)
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading Model & Processor: {MODEL_ID}")

# bfloat16으로 모델 로드
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.bfloat16, 
    device_map=device
)
processor = AutoProcessor.from_pretrained(MODEL_ID)

# 모델 평가 모드 설정
model.eval()

# 데이터셋 로드 (MS COCO 2017 Val)
print("Loading COCO Validation Dataset")
# split="validation"으로 약 5000장 로드
dataset = load_dataset("detection-datasets/coco", split="val", trust_remote_code=True)

# Vision Encoding & TensorBoard 

print("Starting Vision Encoding Process...")

total_images = len(dataset)
start_total_time = time.time()

# 텐서 연산을 위해 그라디언트 계산 비활성화
with torch.no_grad():
    for idx, data in enumerate(tqdm(dataset)):
        image = data['image'] # PIL Image 객체
        image_id = data['image_id']
        
        # 흑백 이미지가 있을 경우 RGB로 변환
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        # 시간 측정 시작
        start_time = time.time()
        
        # 프로세서를 통해 이미지를 모델 입력 형태로 변환
        inputs = processor.image_processor(images=image, return_tensors="pt").to(device)
        
        # Qwen2-VL의 Vision Encoder만 호출
        # pixel_values와 grid_thw(해상도 정보)를 전달
        if hasattr(model, 'visual'):
            vision_encoder = model.visual
        elif hasattr(model, 'vision_tower'):
            vision_encoder = model.vision_tower
        elif hasattr(model, 'model') and hasattr(model.model, 'vision_tower'):
            vision_encoder = model.model.vision_tower
        else:
            # 만약 위 이름들로도 못 찾는다면 모델 구조를 전부 출력해서 직접 확인합니다.
            print("\n=== 현재 로드된 모델의 구조 ===")
            print(model)
            raise AttributeError("Vision Encoder의 정확한 변수명을 찾지 못했습니다. 터미널에 출력된 모델 구조를 확인해주세요!")

        # 2. Vision Encoder 통과 (버전에 따라 반환값이 튜플일 수 있으므로 안전하게 처리)
        vision_outputs = vision_encoder(
            inputs.pixel_values.to(torch.bfloat16), 
            grid_thw=inputs.image_grid_thw
        )
        
        # 만약 반환값이 튜플이나 객체라면 첫 번째 텐서(hidden_states)만 추출
        if isinstance(vision_outputs, tuple):
            image_embeds = vision_outputs[0]
        elif hasattr(vision_outputs, 'last_hidden_state'):
            image_embeds = vision_outputs.last_hidden_state
        else:
            image_embeds = vision_outputs
        
        # 시간 측정 종료
        process_time = time.time() - start_time
        
        # CPU로 옮겨서 저장 (.pt 포맷)
        save_path = os.path.join(SAVE_DIR, f"embed_{image_id}.pt")
        torch.save(image_embeds.cpu(), save_path)
        
        # 텐서보드 로깅
        # 1. 이미지 1장당 인코딩 소요 시간
        writer.add_scalar('Performance/Encoding_Time_per_Image', process_time, idx)
        
        # 2. 현재 VRAM 사용량 (MB)
        if torch.cuda.is_available():
            vram_allocated = torch.cuda.memory_allocated() / (1024 ** 2)
            writer.add_scalar('Memory/VRAM_Allocated_MB', vram_allocated, idx)

total_time = time.time() - start_total_time

print(f"토큰화 완료! 총 소요 시간: {total_time:.2f}초")
print(f"임베딩 저장 위치: {SAVE_DIR}")
print(f"텐서보드 확인: 터미널에서 'tensorboard --logdir={LOG_DIR}' 실행")

writer.close()