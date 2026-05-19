"""
§3 규칙: F < num_frames 조합은 skip해야 한다.
"""
import sys
import os
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.query.frame_subsample import uniform_subsample, SubsampleSkipError


def make_fake_embeddings(F: int, D: int = 64) -> torch.Tensor:
    return torch.randn(F, 256, D)


@pytest.mark.parametrize("F,num_frames", [
    (10, 16),   # target_fps=1, num_frames=16 → skip
    (20, 32),   # target_fps=2, num_frames=32 → skip
    (5, 8),     # F < num_frames → skip
])
def test_skip_when_F_less_than_num_frames(F, num_frames):
    """F < num_frames이면 SubsampleSkipError가 발생해야 한다."""
    embeddings = make_fake_embeddings(F)
    with pytest.raises(SubsampleSkipError):
        uniform_subsample(embeddings, num_frames=num_frames)


@pytest.mark.parametrize("F,num_frames", [
    (50, 8),
    (50, 16),
    (50, 32),
    (16, 8),
    (8, 8),    # F == num_frames: OK
])
def test_ok_when_F_ge_num_frames(F, num_frames):
    """F >= num_frames이면 정확히 num_frames개의 임베딩을 반환해야 한다."""
    embeddings = make_fake_embeddings(F)
    result = uniform_subsample(embeddings, num_frames=num_frames)
    assert result.shape[0] == num_frames, (
        f"F={F}, num_frames={num_frames}: 반환 크기={result.shape[0]}"
    )
    assert result.shape[1] == 256
    assert result.shape[2] == embeddings.shape[2]


def test_uniform_boundary_indices():
    """uniform sub-sampling 인덱스가 경계 포함 균일 간격인지 확인."""
    F, num_frames = 50, 5
    embeddings = torch.arange(F).float().unsqueeze(-1).unsqueeze(-1).expand(F, 256, 64)
    result = uniform_subsample(embeddings, num_frames=num_frames)

    # indices = [round(i * (F-1) / (N-1)) for i in range(N)]
    expected_indices = [round(i * (F - 1) / (num_frames - 1)) for i in range(num_frames)]
    for i, exp_idx in enumerate(expected_indices):
        assert result[i, 0, 0].item() == exp_idx, (
            f"frame {i}: got {result[i,0,0].item()}, expected {exp_idx}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
