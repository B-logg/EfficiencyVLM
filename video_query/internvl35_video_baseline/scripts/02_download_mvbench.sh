#!/usr/bin/env bash
# MVBench 다운로드 스크립트
# 공식 소스: https://huggingface.co/datasets/OpenGVLab/MVBench
set -euo pipefail

DATA_ROOT="${1:-data/mvbench}"
mkdir -p "$DATA_ROOT/json" "$DATA_ROOT/video"

echo "[MVBench] 데이터셋 다운로드 시작 → $DATA_ROOT"

if ! command -v huggingface-cli &>/dev/null; then
    echo "huggingface-cli 필요. 설치: pip install huggingface_hub[cli]"
    exit 1
fi

echo "=== HuggingFace에서 MVBench 다운로드 ==="
huggingface-cli download OpenGVLab/MVBench \
    --repo-type dataset \
    --local-dir "$DATA_ROOT" \
    --include "json/*" "video/*"

echo "MVBench 다운로드 완료: $DATA_ROOT"
echo ""
echo "=== 영상 목록 파일 생성 ==="
VIDEO_LIST="$DATA_ROOT/video_list.tsv"
> "$VIDEO_LIST"
find "$DATA_ROOT/video" -name "*.mp4" | while read -r mp4; do
    task=$(basename "$(dirname "$mp4")")
    clip=$(basename "$mp4" .mp4)
    vid_id="${task}/${clip}"
    echo -e "${vid_id}\t$(realpath "$mp4")" >> "$VIDEO_LIST"
done
echo "video_list.tsv 생성: $(wc -l < "$VIDEO_LIST")개 영상"
