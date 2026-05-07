import torch, os
from PIL import Image
from transformers import AutoModel
from tqdm import tqdm
from datasets import load_dataset
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers.modeling_utils import PreTrainedModel # [추가된 부분]


PreTrainedModel.all_tied_weights_keys = []
MODEL_ID = "OpenGVLab/InternVL3_5-8B"
SAVE_DIR = "./internvl_vision_embeddings"
NUM_TEST_SAMPLES = 1010
os.makedirs(SAVE_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading InternVL Model for Encoding...")

original_linspace = torch.linspace
def patched_linspace(*args, **kwargs):
    # device가 명시되지 않아 강제로 meta로 끌려가는 것을 방지하고 cpu로 고정
    if kwargs.get('device') is None:
        kwargs['device'] = torch.device('cpu')
    return original_linspace(*args, **kwargs)

torch.linspace = patched_linspace # 패치 적용

model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True).eval().to(device)

torch.linspace = original_linspace # 로드가 끝나면 원상복구

transform = T.Compose([
    T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
    T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

dataset = load_dataset("detection-datasets/coco", split=f"val[:{NUM_TEST_SAMPLES}]", trust_remote_code=True)

print("Starting InternVL Stage 2 (ViT) Encoding...")
with torch.no_grad():
    for data in tqdm(dataset):
        pixel_values = transform(data['image']).unsqueeze(0).to(device, dtype=torch.bfloat16)
        
        # 오직 Stage 2(ViT)만 통과시킵니다.
        vit_embeds = model.vision_model(pixel_values).last_hidden_state[:, 1:, :] # CLS 토큰 제거
        
        # Unshuffle이나 MLP 없이 순수 ViT 아웃풋만 DB에 저장!
        torch.save({"vit_embeds": vit_embeds.cpu()}, f"{SAVE_DIR}/embed_{data['image_id']}.pt")