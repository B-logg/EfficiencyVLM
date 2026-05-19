"""
InternViT-300M만 분리 호출.
ViT forward (CLS 제거, patch tokens만 반환) → pixel_shuffle 적용 전까지.
mlp1은 Stage B(query)에서 수행.
"""
from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn
from transformers import AutoModel

from src.ingest.pixel_shuffle import pixel_shuffle


def load_internvit(model_path: str, device: str = "cuda") -> nn.Module:
    """전체 InternVL 모델 로드 (ViT + mlp1 + LLM 포함)."""
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    return model


@torch.no_grad()
def encode_single_frame(
    model: nn.Module,
    pixel_values: torch.Tensor,
    downsample_ratio: float = 0.5,
) -> torch.Tensor:
    """
    ViT forward → pixel_shuffle 까지만 수행 (mlp1 제외).

    Args:
        model: 전체 InternVL 모델
        pixel_values: [N_tiles, 3, H, W] bfloat16, N_tiles=1 for video
        downsample_ratio: pixel_shuffle scale factor

    Returns:
        [N_tiles, 256, D_vit*4] bfloat16
    """
    vit_out = model.vision_model(pixel_values)
    # InternViT는 last_hidden_state에서 CLS 토큰(index 0)을 제거한 patch tokens 반환
    # shape: [N_tiles, H*W, D_vit] — 모델에 따라 CLS 포함 여부 다름
    hidden = vit_out.last_hidden_state  # [N_tiles, seq_len, D_vit]

    # CLS 토큰 제거: InternVL는 position 0이 CLS
    if hidden.shape[1] == 1025:  # 32*32 + 1 (CLS)
        hidden = hidden[:, 1:, :]   # [N_tiles, 1024, D_vit]

    shuffled = pixel_shuffle(hidden, scale_factor=downsample_ratio)  # [N_tiles, 256, D_vit*4]
    return shuffled
