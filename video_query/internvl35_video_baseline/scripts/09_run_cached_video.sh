#!/usr/bin/env bash
# Cached Video Query Experiment (Stage A + Stage B)
#
# Stage A: decode → preprocess → ViT(×120f) → pixel_shuffle → .pt 저장 (1회)
# Stage B: .pt 로드 → MLP → LLM per query
#
# Fixed settings:
#   FPS=12, clip=10s → 120 frames encoded (context: 40960, all 120 frames safe)
#   LLM input: 120 frames (30720 img tokens + overhead < 40960)
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
echo "FPS=12 | clip=10s | 120 frames encoded → all 120 to LLM"
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
