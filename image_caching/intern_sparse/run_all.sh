#!/usr/bin/env bash
# ============================================================
# Full experiment runner: 4 pipelines × 2 datasets
#
# Steps:
#   1. Encode ViT embeddings (for cached pipelines)
#   2. Run all 4 pipelines on VQAv2 + POPE
#   3. Generate all plots
#
# Usage:
#   bash run_all.sh
#
# Environment overrides:
#   DEVICE=cuda bash run_all.sh
# ============================================================

set -euo pipefail

DEVICE="${DEVICE:-cuda}"
EMBED_DIR="embeddings"
RESULTS_DIR="results"
PLOTS_DIR="plots"

VQAV2_N=550          # 50 warmup + 500 measure
POPE_N_PER_SPLIT=200 # 200 × 3 splits = 600; up to 500 used for timing

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_DIR"

echo "============================================================"
echo "InternVL3.5 Sparse Experiment"
echo "  Device     : $DEVICE"
echo "  Embed dir  : $EMBED_DIR"
echo "  Results dir: $RESULTS_DIR"
echo "============================================================"

# ── Step 1: Encode embeddings (for cached pipelines) ─────────────────────────
echo ""
echo ">>> Step 1: Encoding ViT embeddings..."
python encode_embeddings.py \
    --vqav2_n          "$VQAV2_N" \
    --pope_n_per_split "$POPE_N_PER_SPLIT" \
    --embed_dir        "$EMBED_DIR" \
    --device           "$DEVICE"
echo ">>> Encoding done."

# ── Step 2: Run experiments ───────────────────────────────────────────────────
echo ""
echo ">>> Step 2: Running all 4 pipelines on both datasets..."
python run_experiment.py \
    --pipeline    all \
    --dataset     all \
    --embed_dir   "$EMBED_DIR" \
    --results_dir "$RESULTS_DIR" \
    --device      "$DEVICE"
echo ">>> Experiments done."

# ── Step 3: Plot ──────────────────────────────────────────────────────────────
echo ""
echo ">>> Step 3: Generating plots..."
python plot_all.py \
    --results_dir    "$RESULTS_DIR" \
    --plots_dir      "$PLOTS_DIR" \
    --timing_dataset vqav2
echo ">>> Plots saved to $PLOTS_DIR/"

echo ""
echo "============================================================"
echo "All done."
echo "  Results : $RESULTS_DIR/"
echo "  Plots   : $PLOTS_DIR/"
echo "============================================================"
