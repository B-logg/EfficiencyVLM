#!/usr/bin/env bash
# Sweep 1 — Stage A 부하 실험 (FPS sweep)
# §3: target_fps ∈ {1, 2, 5, 10, 30}, num_frames=8 고정 (권장 대안 A)
# §5: 각 sweep 전후 5분 cooldown
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-OpenGVLab/InternVL3_5-8B}"
MSRVTT_ROOT="${MSRVTT_ROOT:-data/msrvtt}"
MVBENCH_ROOT="${MVBENCH_ROOT:-data/mvbench}"
DEVICE="${DEVICE:-cuda}"
LOCK_CLOCKS="${LOCK_CLOCKS:-0}"

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

echo "================================================================"
echo "Sweep 1: Stage A FPS sweep"
echo "target_fps: 1 2 5 10 30 | num_frames: 8 (fixed)"
echo "================================================================"

# Gate test 먼저 실행
echo "[Gate] frame_indexing 테스트 실행..."
python -m pytest tests/test_frame_indexing.py -v --tb=short || {
    echo "GATE FAIL: frame_indexing 테스트 실패. 실험 중단."
    exit 1
}
echo "[Gate] PASS"

cooldown() {
    echo "=== Cooldown 5분 ==="
    sleep 300
}

run_stage_a() {
    local dataset="$1"
    local target_fps="$2"
    local data_root="$3"
    local config="configs/runs/sweep1_${dataset}_fps${target_fps}.yaml"
    local output_dir="outputs/embeddings/${dataset}/${target_fps}fps"
    local video_list="${data_root}/video_list.tsv"

    echo ""
    echo "--- Stage A: dataset=${dataset}, target_fps=${target_fps} ---"

    if [ ! -f "$video_list" ]; then
        echo "[SKIP] video_list 없음: $video_list"
        return 0
    fi

    local lock_flag=""
    [ "$LOCK_CLOCKS" = "1" ] && lock_flag="--lock_clocks"

    python -m src.ingest.run_ingestion \
        --config "$config" \
        --dataset "$dataset" \
        --target_fps "$target_fps" \
        --output_dir "$output_dir" \
        --video_list "$video_list" \
        --model_path "$MODEL_PATH" \
        --device "$DEVICE" \
        $lock_flag

    echo "--- 완료: ${dataset} fps=${target_fps} ---"
}

# ── MSRVTT sweeps ──────────────────────────────────────────
echo ""
echo "=== MSRVTT-QA ==="
for fps in 1 2 5 10 30; do
    run_stage_a "msrvtt" "$fps" "$MSRVTT_ROOT"
done

cooldown

# ── MVBench sweeps ─────────────────────────────────────────
echo ""
echo "=== MVBench ==="
for fps in 1 2 5 10; do
    run_stage_a "mvbench" "$fps" "$MVBENCH_ROOT"
done

# MVBench × 30fps: context check 후 진행
echo ""
echo "--- MVBench × 30fps: context 사전 검사 ---"
python -c "
from src.utils.context_check import check_context
ok = check_context(num_frames=8)  # num_frames=8 고정이므로 safe
print('Context check:', 'PASS' if ok else 'N/A → skip')
import sys; sys.exit(0 if ok else 2)
" && run_stage_a "mvbench" "30" "$MVBENCH_ROOT" || echo "[N/A] MVBench×30fps context overflow → skip"

cooldown

# ── Throughput sub-sweep (Sweep 1의 한 조건에서 batch_size 변동) ────
echo ""
echo "=== Throughput sub-sweep: msrvtt fps=5, batch={1,4,8} ==="
for bsz in 1 4 8; do
    echo "--- batch_size=${bsz} ---"
    python -m src.ingest.run_ingestion \
        --config "configs/runs/sweep1_msrvtt_fps5.yaml" \
        --dataset "msrvtt" \
        --target_fps 5 \
        --output_dir "outputs/embeddings/msrvtt/5fps_bsz${bsz}" \
        --video_list "${MSRVTT_ROOT}/video_list.tsv" \
        --model_path "$MODEL_PATH" \
        --device "$DEVICE" \
        --run_id "throughput_msrvtt_fps5_bsz${bsz}" || true
done

echo ""
echo "================================================================"
echo "Sweep 1 완료. timings: outputs/timings/"
echo "================================================================"
