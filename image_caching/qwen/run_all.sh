#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

FAILED=()

run() {
    local script="$1"
    echo ""
    echo "================================================================"
    echo "Running: $script"
    echo "================================================================"
    if python "$script"; then
        echo "Done: $script"
    else
        echo "FAILED: $script (exit code $?)"
        FAILED+=("$script")
    fi
}

run qwen_anyres_encoding.py
run qwen_e2e_breakdown.py
run qwen_cached_breakdown.py
run qwen_cached.py
run qwen_e2e.py

echo ""
echo "================================================================"
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "All scripts completed successfully."
else
    echo "Completed with failures:"
    for s in "${FAILED[@]}"; do
        echo "  FAILED: $s"
    done
fi
echo "================================================================"
