"""
§2: decord 인덱싱 다운샘플링.
n_out = clip_duration * target_fps
timestamps = [(i + 0.5) / target_fps for i in range(n_out)]
src_indices = [min(int(t * fps_src), n_total - 1) for t in timestamps]
"""
from __future__ import annotations
from typing import Tuple, List


def compute_src_indices(
    fps_src: float,
    n_total: int,
    target_fps: int,
    clip_duration: int = 10,
) -> Tuple[List[int], List[float]]:
    """
    Returns:
        src_indices: 원본 영상에서 추출할 프레임 번호 리스트
        timestamps_sec: 각 프레임의 영상 내 timestamp (초)

    Raises:
        ValueError: 영상이 clip_duration보다 짧을 때
    """
    required_frames = int(fps_src * clip_duration)
    if n_total < required_frames:
        raise ValueError(
            f"영상 길이 부족: n_total={n_total} < clip_duration({clip_duration}s) × "
            f"fps_src({fps_src:.1f}) = {required_frames}. "
            f"10초 미만 영상은 제외 대상."
        )

    n_out = clip_duration * target_fps
    timestamps = [(i + 0.5) / target_fps for i in range(n_out)]
    src_indices = [min(int(t * fps_src), n_total - 1) for t in timestamps]

    return src_indices, timestamps
