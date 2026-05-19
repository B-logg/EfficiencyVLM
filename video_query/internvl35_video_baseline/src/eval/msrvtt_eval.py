"""
MSRVTT-QA 평가: Exact Match (EM) + 선택적 GPT-4 judge.
§8 한계: ground truth frame 번호 라벨 없음, 텍스트 답변만 정량화.
"""
from __future__ import annotations
import json
import re
from typing import List, Dict, Any, Optional


def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def exact_match(pred: str, gold: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)


def load_responses(jsonl_path: str) -> List[Dict[str, Any]]:
    results = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def evaluate_em(jsonl_path: str) -> Dict[str, float]:
    """Exact Match 계산."""
    results = load_responses(jsonl_path)
    n_total = len(results)
    if n_total == 0:
        return {"em": 0.0, "n": 0}

    n_correct = sum(
        1 for r in results
        if exact_match(r.get("response", ""), r.get("answer_gt", ""))
    )
    em = n_correct / n_total
    print(f"MSRVTT-QA EM: {em:.4f} ({n_correct}/{n_total})")
    return {"em": em, "n_correct": n_correct, "n_total": n_total}


def evaluate_frame_grounding_qualitative(
    jsonl_path: str,
    n_samples: int = 50,
) -> List[Dict[str, Any]]:
    """
    §8: frame grounding은 dataset당 50개 qualitative sample로 검증.
    frame_ok 비율만 보고.
    """
    results = load_responses(jsonl_path)[:n_samples]
    frame_ok_rate = sum(1 for r in results if r.get("frame_ok", False)) / max(len(results), 1)
    print(f"Frame grounding parsed OK: {frame_ok_rate:.2%} ({len(results)} samples)")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", required=True)
    args = parser.parse_args()
    evaluate_em(args.responses)
    evaluate_frame_grounding_qualitative(args.responses)
