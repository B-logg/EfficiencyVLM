import torch, os
from PIL import Image
from transformers import AutoModel
from tqdm import tqdm
from datasets import load_dataset
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers.modeling_utils import PreTrainedModel

MODEL_ID = "OpenGVLab/InternVL3_5-8B" 
SAVE_DIR = "./internvl_vision_embeddings"
NUM_TEST_SAMPLES = 1050
os.makedirs(SAVE_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading InternVL Model for Encoding...")

PreTrainedModel.all_tied_weights_keys = {}

model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True).eval().to(device)


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
        
        # ViT forward
        vit_embeds = model.vision_model(pixel_values).last_hidden_state[:, 1:, :]  # [1, 256, D], CLS 제거

        # Pixel Shuffle (2x2 다운샘플, 256→64 토큰)
        b, s, c = vit_embeds.shape
        pixel_shuffled = (vit_embeds
            .reshape(b, int(s**0.5), int(s**0.5), c)
            .unfold(1, 2, 2).unfold(2, 2, 2)
            .reshape(b, int(s**0.5)//2, int(s**0.5)//2, 4, c)
            .reshape(b, int(s**0.5)//2, int(s**0.5)//2, c*4)
            .reshape(b, -1, c*4))  # [1, 64, D*4]

        torch.save({"pixel_shuffled": pixel_shuffled.cpu()}, f"{SAVE_DIR}/embed_{data['image_id']}.pt")