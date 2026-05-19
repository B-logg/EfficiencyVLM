#!/usr/bin/env bash
# Sweep 2 — Stage B 정확도/latency 실험 (num_frames sweep)
# §3: target_fps=5 고정, num_frames ∈ {8, 16, 32}
# Stage A (fps=5) 임베딩이 먼저 생성되어 있어야 함 (03_run_sweep1.sh 실행 후)
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-OpenGVLab/InternVL3_5-8B}"
MSRVTT_ROOT="${MSRVTT_ROOT:-data/msrvtt}"
MVBENCH_ROOT="${MVBENCH_ROOT:-data/mvbench}"
DEVICE="${DEVICE:-cuda}"

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

echo "================================================================"
echo "Sweep 2: Stage B num_frames sweep"
echo "target_fps: 5 (fixed) | num_frames: 8 16 32"
echo "================================================================"

# Gate tests
echo "[Gate] pixel_shuffle 동등성 테스트..."
python -m pytest tests/test_pixel_shuffle_equivalence.py -v --tb=short \
    --model_path "$MODEL_PATH" 2>/dev/null || \
python -m pytest tests/test_pixel_shuffle_equivalence.py -v --tb=short || {
    echo "GATE FAIL: pixel_shuffle 동등성 불일치. 실험 중단."
    exit 1
}

echo "[Gate] subsample skip 테스트..."
python -m pytest tests/test_subsample_skip.py tests/test_injector_smoke.py -v --tb=short || {
    echo "GATE FAIL: subsample/injector 테스트 실패."
    exit 1
}
echo "[Gate] 모든 PASS"

cooldown() {
    echo "=== Cooldown 5분 ==="
    sleep 300
}

run_stage_b() {
    local dataset="$1"
    local num_frames="$2"
    local data_root="$3"
    local config="configs/runs/sweep2_${dataset}_nf${num_frames}.yaml"
    local embed_dir="outputs/embeddings/${dataset}/5fps"

    echo ""
    echo "--- Stage B: dataset=${dataset}, num_frames=${num_frames} ---"

    # context 사전 검사 (§6)
    python -c "
from src.utils.context_check import assert_context_safe
try:
    assert_context_safe(${num_frames})
    print('Context OK')
except RuntimeError as e:
    print(f'N/A: {e}')
    import sys; sys.exit(2)
" || { echo "[N/A] ${dataset}×nf${num_frames} context overflow"; return 0; }

    if [ ! -d "$embed_dir" ] || [ -z "$(ls -A "$embed_dir" 2>/dev/null)" ]; then
        echo "[SKIP] embedding 없음: $embed_dir (03_run_sweep1.sh 먼저 실행)"
        return 0
    fi

    python -m src.query.run_query \
        --config "$config" \
        --dataset "$dataset" \
        --target_fps 5 \
        --num_frames "$num_frames" \
        --embed_dir "$embed_dir" \
        --data_root "$data_root" \
        --model_path "$MODEL_PATH" \
        --device "$DEVICE" \
        --max_samples 200

    echo "--- 완료: ${dataset} nf=${num_frames} ---"
}

# ── MSRVTT-QA ─────────────────────────────────────────────
echo ""
echo "=== MSRVTT-QA ==="
for nf in 8 16 32; do
    run_stage_b "msrvtt" "$nf" "$MSRVTT_ROOT"
done

cooldown

# ── MVBench ───────────────────────────────────────────────
echo ""
echo "=== MVBench ==="
for nf in 8 16 32; do
    run_stage_b "mvbench" "$nf" "$MVBENCH_ROOT"
done

cooldown

# ── 평가 ──────────────────────────────────────────────────
echo ""
echo "=== 정확도 평가 ==="
for jsonl in outputs/responses/sweep2_*.jsonl; do
    [ -f "$jsonl" ] || continue
    fname=$(basename "$jsonl" .jsonl)
    if [[ "$fname" == *"msrvtt"* ]]; then
        python -m src.eval.msrvtt_eval --responses "$jsonl"
    elif [[ "$fname" == *"mvbench"* ]]; then
        python -m src.eval.mvbench_eval --responses "$jsonl"
    fi
done

echo ""
echo "================================================================"
echo "Sweep 2 완료. responses: outputs/responses/"
echo "================================================================"
