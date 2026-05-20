#!/usr/bin/env bash
# Cached Video Query Experiment (Stage A + Stage B)
#
# Stage A: decode → preprocess → ViT(×100f) → pixel_shuffle → .pt 저장 (1회)
# Stage B: .pt 로드 → MLP → LLM per query
#
# Fixed settings:
#   FPS=10, clip=10s → 100 frames encoded
#   LLM input: 100 frames (fits within 32k context)
#   QA samples: 100

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-OpenGVLab/InternVL3_5-8B}"
MVBENCH_ROOT="${MVBENCH_ROOT:-data/mvbench}"
DEVICE="${DEVICE:-cuda}"
MAX_SAMPLES="${MAX_SAMPLES:-100}"
WARMUP="${WARMUP:-5}"
EMBED_DIR="outputs/cached_video_embeddings"
OUTPUT_DIR="outputs/cached_video"

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

echo "================================================================"
echo "Cached Video Query Experiment (Stage A + Stage B)"
echo "FPS=10 | clip=10s | 100 frames encoded → all 100 to LLM"
echo "Max QA samples: ${MAX_SAMPLES} | Warmup: ${WARMUP}"
echo "Embed cache dir: ${EMBED_DIR}"
echo "================================================================"

if [ ! -d "$MVBENCH_ROOT" ]; then
    echo "[SKIP] data_root not found: $MVBENCH_ROOT"
    exit 1
fi

echo ""
echo "--- Cached: dataset=mvbench ---"

python -m src.query.run_cached_video \
    --dataset        mvbench \
    --data_root      "$MVBENCH_ROOT" \
    --model_path     "$MODEL_PATH" \
    --device         "$DEVICE" \
    --max_samples    "$MAX_SAMPLES" \
    --warmup_samples "$WARMUP" \
    --embed_dir      "$EMBED_DIR" \
    --output_dir     "$OUTPUT_DIR"

echo ""
echo "================================================================"
echo "Cached experiment done. Results: ${OUTPUT_DIR}/"
echo "Run comparison plot: python scripts/10_plot_comparison.py"
echo "================================================================"
