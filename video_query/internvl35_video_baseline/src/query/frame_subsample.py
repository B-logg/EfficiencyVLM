"""
§1 Stage B: uniform sub-sampling.
indices = [round(i * (F-1) / (N-1)) for i in range(N)]
F < num_frames → SubsampleSkipError (§3 규칙).
"""
from __future__ import annotations
import torch


class SubsampleSkipError(Exception):
    """F < num_frames 조합 제외 규칙."""
    pass


def uniform_subsample(embeddings: torch.Tensor, num_frames: int) -> torch.Tensor:
    """
    Args:
        embeddings: [F, 256, D] cached pixel-shuffle output
        num_frames: 원하는 출력 프레임 수 N

    Returns:
        [num_frames, 256, D] sub-sampled embeddings

    Raises:
        SubsampleSkipError: F < num_frames
    """
    F = embeddings.shape[0]
    if F < num_frames:
        raise SubsampleSkipError(
            f"F={F} < num_frames={num_frames}: 이 조합은 N/A로 제외. §3 규칙."
        )

    if num_frames == 1:
        return embeddings[F // 2].unsqueeze(0)

    indices = [round(i * (F - 1) / (num_frames - 1)) for i in range(num_frames)]
    return embeddings[indices]
