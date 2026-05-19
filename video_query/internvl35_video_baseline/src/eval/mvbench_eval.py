"""
MVBench 평가: MCQA accuracy (A/B/C/D 선택지 일치).
task별 breakdown + 전체 accuracy 보고.
"""
from __future__ import annotations
import json
import re
from collections import defaultdict
from typing import Dict, Any, List


def extract_option(text: str) -> str | None:
    """응답 텍스트에서 선택지 문자(A/B/C/D)를 추출."""
    m = re.search(r"\b([ABCD])\b", text.strip().upper())
    return m.group(1) if m else None


def load_responses(jsonl_path: str) -> List[Dict[str, Any]]:
    results = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def evaluate_mvbench(jsonl_path: str) -> Dict[str, Any]:
    """전체 accuracy + task별 breakdown."""
    results = load_responses(jsonl_path)
    n_total = len(results)
    if n_total == 0:
        return {"accuracy": 0.0, "n": 0}

    task_correct: Dict[str, int] = defaultdict(int)
    task_total: Dict[str, int] = defaultdict(int)
    n_correct = 0

    for r in results:
        pred_option = extract_option(r.get("response", ""))
        gold = r.get("answer_gt", "").strip().upper()
        task = r.get("video_id", "unknown").split("/")[0] if "/" in r.get("video_id", "") else "unknown"

        task_total[task] += 1
        if pred_option == gold:
            n_correct += 1
            task_correct[task] += 1

    overall_acc = n_correct / n_total
    task_acc = {t: task_correct[t] / task_total[t] for t in task_total}

    print(f"MVBench Overall Accuracy: {overall_acc:.4f} ({n_correct}/{n_total})")
    for task, acc in sorted(task_acc.items()):
        print(f"  {task}: {acc:.4f} ({task_correct[task]}/{task_total[task]})")

    return {
        "accuracy": overall_acc,
        "n_correct": n_correct,
        "n_total": n_total,
        "task_breakdown": task_acc,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", required=True)
    args = parser.parse_args()
    evaluate_mvbench(args.responses)
