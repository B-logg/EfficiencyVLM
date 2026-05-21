"""
Pre-encode ViT + Pixel Shuffle outputs for cached pipelines.

Saves pixel_shuffled tensors as .pt files.
Must be run before run_experiment.py --pipeline cached or cached_sparse.

Usage:
    python encode_embeddings.py \
        --vqav2_n 550 \
        --pope_n_per_split 200 \
        --embed_dir embeddings/
"""
import argparse
import os
import sys
from pathlib import Path

import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from transformers import AutoModel
from transformers.modeling_utils import PreTrainedModel

ROOT = str(Path(__file__).resolve().parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dataset_utils import load_vqav2, load_pope

MODEL_ID = "OpenGVLab/InternVL3_5-8B"

TRANSFORM = T.Compose([
    T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
    T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


def pixel_shuffle_2x(vit_embs: torch.Tensor) -> torch.Tensor:
    """
    2×2 pixel shuffle downsampling: [B, 1024, D] → [B, 256, D*4].
    """
    b, s, c = vit_embs.shape
    h = w = int(s ** 0.5)
    return (
        vit_embs
        .reshape(b, h, w, c)
        .unfold(1, 2, 2).unfold(2, 2, 2)     # [b, h/2, w/2, c, 2, 2]
        .reshape(b, h // 2, w // 2, 4, c)
        .reshape(b, h // 2, w // 2, c * 4)
        .reshape(b, -1, c * 4)               # [b, 256, D*4]
    )


@torch.no_grad()
def encode_and_save(model, device, samples, embed_dir: str):
    os.makedirs(embed_dir, exist_ok=True)
    skipped = 0

    for item in tqdm(samples, desc="Encoding"):
        pt_path = os.path.join(embed_dir, f"{item['dataset']}_{item['id']}.pt")
        if os.path.exists(pt_path):
            continue

        try:
            pv = TRANSFORM(item["image"]).unsqueeze(0).to(device, dtype=torch.bfloat16)

            vit_out = model.vision_model(pv)
            hidden  = vit_out.last_hidden_state  # [1, 1025, D_vit]
            if hidden.shape[1] == 1025:
                hidden = hidden[:, 1:, :]        # remove CLS → [1, 1024, D_vit]

            pixel_shuffled = pixel_shuffle_2x(hidden)  # [1, 256, D_vit*4]

            torch.save({"pixel_shuffled": pixel_shuffled.cpu()}, pt_path)
        except Exception as e:
            print(f"[WARN] Skipping {item['id']}: {e}")
            skipped += 1

    print(f"Encoding done. Skipped: {skipped}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vqav2_n",        type=int, default=550)
    parser.add_argument("--pope_n_per_split", type=int, default=200)
    parser.add_argument("--embed_dir",      default="embeddings")
    parser.add_argument("--device",         default="cuda")
    args = parser.parse_args()

    # ── CUDA 가용성 확인 ──────────────────────────────────────────────────
    device = args.device
    if device == "cuda":
        if not torch.cuda.is_available():
            print("[WARN] CUDA를 사용할 수 없습니다. CPU로 전환합니다.")
            device = "cpu"
        else:
            try:
                n = torch.cuda.device_count()
                print(f"[INFO] 사용 가능한 GPU: {n}개")
                for i in range(n):
                    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            except RuntimeError as e:
                print(f"[ERROR] CUDA 초기화 실패: {e}")
                print("[HINT] nvidia-smi 로 드라이버 버전 확인 후 570+ 버전으로 업데이트 필요")
                print("[HINT] RTX 5090(Blackwell)은 CUDA 12.8+ / 드라이버 570+ 필요")
                raise

    print(f"Loading model: {MODEL_ID} (device={device})")
    PreTrainedModel.all_tied_weights_keys = {}
    model = AutoModel.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16,
        trust_remote_code=True, low_cpu_mem_usage=True,
    ).eval().to(device)
    args.device = device  # 이후 encode_and_save에 전달

    vqav2_samples = load_vqav2(n_total=args.vqav2_n)
    pope_samples  = load_pope(n_per_split=args.pope_n_per_split)

    print(f"\n--- Encoding VQAv2 ({len(vqav2_samples)} samples) ---")
    encode_and_save(model, args.device, vqav2_samples, args.embed_dir)

    print(f"\n--- Encoding POPE ({len(pope_samples)} samples) ---")
    encode_and_save(model, args.device, pope_samples, args.embed_dir)

    print(f"\nAll embeddings saved to: {args.embed_dir}/")


if __name__ == "__main__":
    main()
