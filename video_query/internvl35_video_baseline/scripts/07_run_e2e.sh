#!/usr/bin/env bash
# E2E Video Query Experiment
# 쿼리가 들어올 때마다 처음부터 처리: decode → preprocess → ViT → pixel_shuffle → MLP → LLM
#
# 고정 설정:
#   FPS=10, clip=10s → 100 프레임 인코딩
#   LLM 입력: 16 프레임 서브샘플 (OOM 안전 범위)
#   QA 샘플: 100개

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-OpenGVLab/InternVL3_5-8B}"
MSRVTT_ROOT="${MSRVTT_ROOT:-data/msrvtt}"
MVBENCH_ROOT="${MVBENCH_ROOT:-data/mvbench}"
DEVICE="${DEVICE:-cuda}"
MAX_SAMPLES="${MAX_SAMPLES:-100}"
WARMUP="${WARMUP:-5}"
NF_LLM="${NF_LLM:-16}"    # LLM 입력 서브샘플 프레임 수
OUTPUT_DIR="outputs/e2e"

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

echo "================================================================"
echo "E2E Video Query Experiment"
echo "FPS=10 | clip=10s | 100 frames encoded | LLM input=${NF_LLM} frames"
echo "Max QA samples: ${MAX_SAMPLES} | Warmup: ${WARMUP}"
echo "================================================================"

run_e2e() {
    local dataset="$1"
    local data_root="$2"

    echo ""
    echo "--- E2E: dataset=${dataset} ---"

    if [ ! -d "$data_root" ]; then
        echo "[SKIP] data_root 없음: $data_root"
        return 0
    fi

    python -m src.query.run_e2e \
        --dataset       "$dataset" \
        --data_root     "$data_root" \
        --model_path    "$MODEL_PATH" \
        --device        "$DEVICE" \
        --max_samples   "$MAX_SAMPLES" \
        --warmup_samples "$WARMUP" \
        --num_frames_llm "$NF_LLM" \
        --output_dir    "$OUTPUT_DIR"

    echo "--- 완료: ${dataset} ---"
}

run_e2e "msrvtt"  "$MSRVTT_ROOT"
run_e2e "mvbench" "$MVBENCH_ROOT"

echo ""
echo "================================================================"
echo "E2E 실험 완료. 결과: ${OUTPUT_DIR}/"
echo "그래프 생성: python scripts/08_plot_e2e.py"
echo "================================================================"
