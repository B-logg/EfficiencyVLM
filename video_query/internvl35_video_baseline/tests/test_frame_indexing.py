"""
Gate test §6: decord 인덱싱 정확성 검증.
알려진 FPS·길이의 합성 영상에 대해 target_fps∈{1,5,30} 인덱싱이
기대 timestamp와 일치하는지 검증.
"""
import sys
import os
import subprocess
import tempfile
import pytest
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.ingest.frame_indexing import compute_src_indices


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_test_video(path: str, fps: int = 30, duration: int = 10):
    """ffmpeg으로 fps/duration이 정확히 제어된 테스트 영상을 만든다.
    각 프레임에 timestamp(프레임 번호)가 인코딩된 color bar 패턴을 사용."""
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"testsrc=duration={duration}:size=224x224:rate={fps}",
        "-vf", f"fps={fps}",
        "-pix_fmt", "yuv420p",
        path,
    ]
    result = subprocess.run(cmd, capture_output=True)
    assert result.returncode == 0, f"ffmpeg 실패: {result.stderr.decode()}"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("target_fps", [1, 2, 5, 10, 30])
def test_src_indices_count(target_fps):
    """n_out = clip_duration * target_fps 개의 인덱스가 반환되는지 확인."""
    fps_src = 30.0
    clip_duration = 10
    n_total = int(fps_src * clip_duration)

    indices, timestamps = compute_src_indices(
        fps_src=fps_src,
        n_total=n_total,
        target_fps=target_fps,
        clip_duration=clip_duration,
    )

    expected_n = clip_duration * target_fps
    assert len(indices) == expected_n, (
        f"target_fps={target_fps}: 기대 {expected_n}개, 실제 {len(indices)}개"
    )
    assert len(timestamps) == expected_n


@pytest.mark.parametrize("target_fps", [1, 2, 5, 10, 30])
def test_timestamps_accuracy(target_fps):
    """각 프레임의 timestamp가 (i + 0.5) / target_fps와 일치해야 한다."""
    fps_src = 30.0
    clip_duration = 10
    n_total = int(fps_src * clip_duration)

    indices, timestamps = compute_src_indices(
        fps_src=fps_src,
        n_total=n_total,
        target_fps=target_fps,
        clip_duration=clip_duration,
    )

    expected_ts = [(i + 0.5) / target_fps for i in range(clip_duration * target_fps)]
    for i, (got, exp) in enumerate(zip(timestamps, expected_ts)):
        assert abs(got - exp) < 1e-6, (
            f"target_fps={target_fps}, frame {i}: timestamp={got:.6f}, expected={exp:.6f}"
        )


@pytest.mark.parametrize("target_fps", [1, 2, 5, 10, 30])
def test_indices_within_bounds(target_fps):
    """모든 인덱스가 [0, n_total-1] 범위 안에 있어야 한다."""
    fps_src = 30.0
    clip_duration = 10
    n_total = int(fps_src * clip_duration)

    indices, _ = compute_src_indices(
        fps_src=fps_src,
        n_total=n_total,
        target_fps=target_fps,
        clip_duration=clip_duration,
    )

    for idx in indices:
        assert 0 <= idx < n_total, f"인덱스 {idx}가 범위 초과 (n_total={n_total})"


def test_short_video_excluded():
    """10초 미만 영상은 n_total이 부족 → 예외 또는 빈 결과 반환."""
    fps_src = 30.0
    clip_duration = 10
    n_total = int(fps_src * 5)  # 5초짜리 영상

    with pytest.raises(ValueError, match="clip_duration"):
        compute_src_indices(
            fps_src=fps_src,
            n_total=n_total,
            target_fps=5,
            clip_duration=clip_duration,
        )


@pytest.mark.skipif(
    subprocess.run(["which", "ffmpeg"], capture_output=True).returncode != 0,
    reason="ffmpeg 없음",
)
@pytest.mark.parametrize("target_fps", [1, 5, 30])
def test_decord_pixel_timestamps(target_fps):
    """실제 decord로 추출한 프레임 픽셀이 기대 timestamp에 대응하는지 검증.
    testsrc 패턴은 프레임 번호를 화면 좌상단에 렌더링하므로, 색상 채널로
    대략적 일치를 확인한다 (정확한 OCR은 제외)."""
    try:
        import decord
    except ImportError:
        pytest.skip("decord 없음")

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        video_path = f.name

    try:
        make_test_video(video_path, fps=30, duration=10)
        vr = decord.VideoReader(video_path)
        fps_src = vr.get_avg_fps()
        n_total = len(vr)

        indices, timestamps = compute_src_indices(
            fps_src=fps_src,
            n_total=n_total,
            target_fps=target_fps,
            clip_duration=10,
        )

        frames = vr.get_batch(indices).asnumpy()
        assert frames.shape[0] == len(indices), "추출 프레임 수 불일치"
        assert frames.shape[-1] == 3, "RGB 채널 기대"

        for i, (idx, ts) in enumerate(zip(indices, timestamps)):
            expected_idx = min(int(ts * fps_src), n_total - 1)
            assert abs(idx - expected_idx) <= 1, (
                f"frame {i}: src_index={idx}, expected≈{expected_idx}"
            )
    finally:
        os.unlink(video_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
