"""
InternVL의 pixel_shuffle 함수를 그대로 차용.
scale_factor=0.5 → 공간 해상도 2× 압축, 채널 4× 확장.
입력: [B, H*W, C] (H*W=1024, C=D_vit) → 출력: [B, 256, D_vit*4]
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def pixel_shuffle(x: torch.Tensor, scale_factor: float = 0.5) -> torch.Tensor:
    """
    Args:
        x: [B, N, C] where N = h * w (e.g., 1024 = 32*32)
        scale_factor: 0.5 → 2× 공간 압축
    Returns:
        [B, N * scale_factor^2, C / scale_factor^2] e.g., [B, 256, C*4]
    """
    B, N, C = x.shape
    h = w = int(N ** 0.5)
    assert h * w == N, f"N={N}이 완전제곱수여야 함"

    x = x.view(B, h, w, C)

    new_h = int(h * scale_factor)
    new_w = int(w * scale_factor)
    new_c = int(C / (scale_factor ** 2))

    x = x.view(B, new_h, int(1 / scale_factor), new_w, int(1 / scale_factor), C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, new_h * new_w, new_c)
    return x
