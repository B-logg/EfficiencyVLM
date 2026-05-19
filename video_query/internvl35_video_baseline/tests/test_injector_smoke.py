"""
Smoke test: embed_with_cache가 에러 없이 텍스트를 반환하는지 확인.
모델 없이는 mock으로 형상만 검증.
"""
import sys
import os
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.query.frame_subsample import uniform_subsample
from src.query.prompt import build_prompt


# ---------------------------------------------------------------------------
# Prompt 구성 테스트 (모델 불필요)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("num_frames", [4, 8, 16])
def test_prompt_contains_frame_tags(num_frames):
    """prompt에 num_frames개의 <image> 태그와 <frame>K</frame> 지시문이 있어야 한다."""
    prompt = build_prompt(question="What is happening?", num_frames=num_frames)
    assert prompt.count("<image>") == num_frames, "frame 수만큼 <image> 태그 필요"
    assert "<frame>K</frame>" in prompt or "frame>K<" in prompt.replace(" ", "")
    for i in range(1, num_frames + 1):
        assert f"Frame{i}:" in prompt


def test_prompt_k_range(num_frames=8):
    """prompt의 K 범위 지시문이 올바른지 확인."""
    prompt = build_prompt(question="test?", num_frames=num_frames)
    assert f"[1, {num_frames}]" in prompt


# ---------------------------------------------------------------------------
# subsample 형상 + injector import smoke (CUDA 없어도)
# ---------------------------------------------------------------------------

def test_subsample_shape_preserved():
    """uniform_subsample이 올바른 차원을 유지하는지 확인."""
    F, num_frames, D = 50, 8, 128
    embeddings = torch.randn(F, 256, D)
    result = uniform_subsample(embeddings, num_frames=num_frames)
    assert result.shape == (num_frames, 256, D)


def test_injector_importable():
    """injector 모듈이 import 가능해야 함."""
    try:
        from src.query.injector import embed_with_cache  # noqa: F401
    except ImportError as e:
        pytest.fail(f"injector import 실패: {e}")


def test_grounding_importable():
    """grounding 모듈이 import 가능해야 함."""
    try:
        from src.query.grounding import parse_frame_index, reextract_frame  # noqa: F401
    except ImportError as e:
        pytest.fail(f"grounding import 실패: {e}")


@pytest.mark.parametrize("text,expected", [
    ("The answer is rain. <frame>3</frame>", 3),
    ("Here is my answer.\n<frame>7</frame>", 7),
    ("<frame> 12 </frame>", 12),
    ("no frame tag here", None),
])
def test_parse_frame_index(text, expected):
    from src.query.grounding import parse_frame_index
    result = parse_frame_index(text)
    assert result == expected, f"text={text!r}: got {result}, expected {expected}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
