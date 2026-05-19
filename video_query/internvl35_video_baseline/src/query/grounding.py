"""
§1 Stage B: <frame>K</frame> 파싱 + decord로 원본 프레임 재추출.
K-1번째 원본 frame index = src_indices[uniform_indices[K-1]]
"""
from __future__ import annotations
import re
from typing import List, Optional
import numpy as np
from PIL import Image


FRAME_TAG_RE = re.compile(r"<frame>\s*(\d+)\s*</frame>", re.IGNORECASE)


def parse_frame_index(text: str) -> Optional[int]:
    """
    LLM 출력에서 <frame>K</frame>를 파싱하여 정수 K를 반환.
    없으면 None.
    """
    matches = FRAME_TAG_RE.findall(text)
    if not matches:
        return None
    return int(matches[-1])  # 마지막 매치 사용


def reextract_frame(
    video_path: str,
    src_indices: List[int],
    uniform_indices: List[int],
    k: int,
) -> Optional[Image.Image]:
    """
    LLM이 출력한 K (1-indexed)에 해당하는 원본 프레임을 decord로 재추출.

    Args:
        video_path: 원본 mp4 절대경로
        src_indices: Stage A에서 저장된 원본 프레임 번호 목록
        uniform_indices: Stage B sub-sampling 인덱스 [round(i*(F-1)/(N-1)) for i in range(N)]
        k: LLM이 출력한 1-indexed 프레임 번호

    Returns:
        PIL.Image (RGB) 또는 None (인덱스 범위 초과 시)
    """
    try:
        import decord
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"decord 로드 실패: {e}")
        return None

    k_zero = k - 1  # 0-indexed
    if k_zero < 0 or k_zero >= len(uniform_indices):
        return None

    orig_frame_idx = src_indices[uniform_indices[k_zero]]
    orig_frame_idx = min(orig_frame_idx, len(vr) - 1)

    frame_tensor = vr[orig_frame_idx]
    frame_np = frame_tensor.asnumpy() if hasattr(frame_tensor, "asnumpy") else np.array(frame_tensor)
    return Image.fromarray(frame_np.astype(np.uint8))
