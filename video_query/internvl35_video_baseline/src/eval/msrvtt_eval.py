"""
MSRVTT-QA 평가: Exact Match (EM) + contains 기반 soft match.
모델 응답은 "answer text\n<frame>K</frame>" 형태이므로
<frame> 이전 텍스트를 answer로 추출 후 비교.
"""
from __future__ import annotations
import json
import re
from typing import List, Dict, Any


def extract_answer(response: str) -> str:
    """<frame>K</frame> 이전 텍스트를 답변으로 추출."""
    frame_pos = response.find("<frame>")
    if frame_pos != -1:
        response = response[:frame_pos]
    return response.strip()


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def exact_match(pred: str, gold: str) -> bool:
    return normalize(pred) == normalize(gold)


def contains_match(pred: str, gold: str) -> bool:
    """gold가 pred에 포함되면 정답."""
    return normalize(gold) in normalize(pred)


def load_responses(jsonl_path: str) -> List[Dict[str, Any]]:
    results = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def evaluate_em(jsonl_path: str) -> Dict[str, float]:
    results = load_responses(jsonl_path)
    n_total = len(results)
    if n_total == 0:
        print("MSRVTT-QA: 응답 없음 (빈 파일)")
        return {"em": 0.0, "contains": 0.0, "n": 0}

    n_em = n_contains = 0
    for r in results:
        pred = extract_answer(r.get("response", ""))
        gold = r.get("answer_gt", "")
        if exact_match(pred, gold):
            n_em += 1
        if contains_match(pred, gold):
            n_contains += 1

    em = n_em / n_total
    cont = n_contains / n_total
    print(f"MSRVTT-QA EM: {em:.4f} ({n_em}/{n_total})")
    print(f"MSRVTT-QA Contains: {cont:.4f} ({n_contains}/{n_total})")
    return {"em": em, "contains": cont, "n_em": n_em, "n_contains": n_contains, "n_total": n_total}


def evaluate_frame_grounding_qualitative(
    jsonl_path: str,
    n_samples: int = 50,
) -> List[Dict[str, Any]]:
    results = load_responses(jsonl_path)[:n_samples]
    if not results:
        print("Frame grounding: 응답 없음")
        return []
    frame_ok_rate = sum(1 for r in results if r.get("frame_ok", False)) / len(results)
    print(f"Frame grounding parsed OK: {frame_ok_rate:.2%} ({len(results)} samples)")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", required=True)
    args = parser.parse_args()
    evaluate_em(args.responses)
    evaluate_frame_grounding_qualitative(args.responses)
