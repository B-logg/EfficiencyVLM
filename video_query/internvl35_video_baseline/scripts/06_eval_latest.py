"""
각 조건(dataset × num_frames)별로 최신 비어있지 않은 JSONL 파일만 골라 평가.
빈 파일·중복 실행 자동 제외.
"""
import os
import sys
import glob
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RESPONSES_DIR = ROOT / "outputs" / "responses"
PATTERN = re.compile(r"^(sweep2_(?:msrvtt|mvbench)_fps\d+_nf\d+)_(\d{8}_\d{6})\.jsonl$")


def pick_latest(responses_dir: Path) -> dict[str, Path]:
    """조건별로 최신 비어있지 않은 파일 반환."""
    best: dict[str, tuple[str, Path]] = {}  # key → (timestamp, path)

    for path in responses_dir.glob("sweep2_*.jsonl"):
        if path.stat().st_size == 0:
            continue
        m = PATTERN.match(path.name)
        if not m:
            continue
        key, ts = m.group(1), m.group(2)
        if key not in best or ts > best[key][0]:
            best[key] = (ts, path)

    return {k: v[1] for k, v in best.items()}


def main():
    latest = pick_latest(RESPONSES_DIR)
    if not latest:
        print("평가할 파일이 없습니다.")
        return

    from src.eval.msrvtt_eval import evaluate_em, evaluate_frame_grounding_qualitative
    from src.eval.mvbench_eval import evaluate_mvbench

    for key in sorted(latest):
        path = latest[key]
        print(f"\n{'='*60}")
        print(f"  {key}")
        print(f"  파일: {path.name}")
        print(f"{'='*60}")
        if "msrvtt" in key:
            evaluate_em(str(path))
            evaluate_frame_grounding_qualitative(str(path))
        elif "mvbench" in key:
            evaluate_mvbench(str(path))


if __name__ == "__main__":
    main()
