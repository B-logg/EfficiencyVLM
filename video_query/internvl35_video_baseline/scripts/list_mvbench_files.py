"""
서버에 복사할 MVBench 파일 목록 출력.
- JSON: 20개 전부
- Video: 처음 N개 QA 아이템에 필요한 영상만

Usage:
    python scripts/list_mvbench_files.py --data_root data/mvbench --n 200
    python scripts/list_mvbench_files.py --data_root data/mvbench --n 200 --rsync
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data.mvbench import load_mvbench


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True, help="MVBench 루트 경로 (json/, video/ 포함)")
    parser.add_argument("--n",         type=int, default=200, help="수집할 QA 아이템 수")
    parser.add_argument("--rsync",     action="store_true",   help="rsync 명령어 형식으로 출력")
    parser.add_argument("--dest",      default="user@server:/data/mvbench", help="rsync 대상 경로")
    args = parser.parse_args()

    data_root = os.path.abspath(args.data_root)
    json_dir  = os.path.join(data_root, "json")
    video_dir = os.path.join(data_root, "video")

    # ── 필요한 영상 수집 ─────────────────────────────────────────────
    video_paths = []
    seen = set()
    for item in load_mvbench(data_root):
        if len(video_paths) >= args.n:
            break
        vp = item["video_path"]
        if vp not in seen:
            seen.add(vp)
            video_paths.append(vp)

    print(f"\n=== MVBench 파일 목록 (n={args.n}) ===")
    print(f"JSON 디렉토리: {json_dir}  (20개 task JSON 전부)")
    print(f"필요한 영상:   {len(video_paths)}개 (중복 제거)\n")

    # 소스 디렉토리별 분류
    subdir_counts: dict = {}
    for vp in video_paths:
        rel = os.path.relpath(vp, video_dir)
        subdir = rel.split(os.sep)[0]
        subdir_counts[subdir] = subdir_counts.get(subdir, 0) + 1

    print("소스별 영상 분포:")
    for sd, cnt in sorted(subdir_counts.items()):
        print(f"  video/{sd:<30} {cnt:>4}개")

    if args.rsync:
        print(f"\n# ── rsync 명령어 ────────────────────────────────────────────")
        print(f"# 1) JSON 전체")
        print(f"rsync -avz {json_dir}/ {args.dest}/json/\n")
        print(f"# 2) 영상 파일 목록 생성 후 rsync")
        list_file = "/tmp/mvbench_video_list.txt"
        print(f"python scripts/list_mvbench_files.py --data_root {args.data_root} --n {args.n} --write_list {list_file}")
        print(f"rsync -avz --files-from={list_file} {video_dir}/ {args.dest}/video/")
    else:
        print("\n영상 파일 목록:")
        for vp in video_paths:
            print(vp)

    # 용량 추정
    total_bytes = sum(os.path.getsize(vp) for vp in video_paths if os.path.exists(vp))
    print(f"\n추정 용량: {total_bytes / (1024**3):.2f} GB  ({len(video_paths)}개 영상)")


if __name__ == "__main__":
    main()
