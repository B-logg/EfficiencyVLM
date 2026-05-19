#!/usr/bin/env bash
# MSRVTT-QA 다운로드 스크립트
# 공식 소스: https://github.com/xudejing/video-question-answering
set -euo pipefail

DATA_ROOT="${1:-data/msrvtt}"
mkdir -p "$DATA_ROOT/videos/all"

echo "[MSRVTT-QA] 데이터셋 다운로드 시작 → $DATA_ROOT"

# 영상: MSRVTT 공식 배포 (약 6.5GB)
# 직접 링크는 라이선스 제한으로 수동 다운로드 필요.
# 참고: https://www.mediafire.com/folder/h14ebb/TestData  (test split)
echo "=== 영상 파일 ==="
echo "MSRVTT 영상은 공식 소스에서 수동 다운로드 필요:"
echo "  https://github.com/xudejing/video-question-answering"
echo "다운로드한 mp4 파일을 $DATA_ROOT/videos/all/ 에 배치하세요."
echo ""

# QA JSON
echo "=== QA 파일 다운로드 ==="
QA_URL="https://raw.githubusercontent.com/xudejing/video-question-answering/master/data/test_qa.json"
if command -v wget &>/dev/null; then
    wget -q -O "$DATA_ROOT/test_qa.json" "$QA_URL"
elif command -v curl &>/dev/null; then
    curl -sSL -o "$DATA_ROOT/test_qa.json" "$QA_URL"
else
    echo "wget 또는 curl이 필요합니다."
    exit 1
fi

echo "test_qa.json 다운로드 완료: $DATA_ROOT/test_qa.json"
echo ""
echo "=== 영상 목록 파일 생성 ==="
# video_id TAB video_path 형식의 TSV 생성
VIDEO_LIST="$DATA_ROOT/video_list.tsv"
> "$VIDEO_LIST"
for mp4 in "$DATA_ROOT/videos/all/"*.mp4; do
    [ -f "$mp4" ] || continue
    vid_id=$(basename "$mp4" .mp4)
    echo -e "${vid_id}\t$(realpath "$mp4")" >> "$VIDEO_LIST"
done
echo "video_list.tsv 생성: $(wc -l < "$VIDEO_LIST")개 영상"
