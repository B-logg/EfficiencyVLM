"""
VQAv2 and POPE dataset loading + evaluation utilities.
"""
from __future__ import annotations
import re
import string
from typing import List, Dict, Tuple

# ── VQA answer normalization (official VQA eval protocol) ────────────────────

_PUNCT = set(string.punctuation) - {"'"}
_ARTICLES = {"a", "an", "the"}
_PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
_COMMA_STRIP  = re.compile(r"(?<=\d)(\,)(?=\d)")
_CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
    "couldnt": "couldn't", "didnt": "didn't", "doesnt": "doesn't",
    "dont": "don't", "hadnt": "hadn't", "hasnt": "hasn't", "havent": "haven't",
    "hed": "he'd", "hes": "he's", "howd": "how'd", "howll": "how'll",
    "hows": "how's", "ive": "i've", "im": "i'm", "isnt": "isn't",
    "itd": "it'd", "its": "it's", "lets": "let's", "mightve": "might've",
    "mustve": "must've", "neednt": "needn't", "notve": "not've",
    "shes": "she's", "shouldve": "should've", "shouldnt": "shouldn't",
    "somebodys": "somebody's", "someones": "someone's", "somethings": "something's",
    "thatll": "that'll", "thats": "that's", "thered": "there'd",
    "theres": "there's", "theyd": "they'd", "theyll": "they'll",
    "theyre": "they're", "theyve": "they've", "wasnt": "wasn't",
    "wed": "we'd", "were": "we're", "werent": "weren't",
    "whatll": "what'll", "whats": "what's", "whens": "when's",
    "wheres": "where's", "whos": "who's", "wholl": "who'll",
    "whos": "who's", "whys": "why's", "wont": "won't", "wouldve": "would've",
    "wouldnt": "wouldn't", "youd": "you'd", "youll": "you'll",
    "youre": "you're", "youve": "you've",
}


def _process_punctuation(s: str) -> str:
    out = ""
    for c in s:
        if c in _PUNCT:
            out += " "
        else:
            out += c
    return _PERIOD_STRIP.sub("", _COMMA_STRIP.sub("", out)).strip()


def normalize_vqa_answer(s: str) -> str:
    s = s.lower().strip()
    s = _process_punctuation(s)
    # contractions: apply word-by-word (not just whole-string match)
    tokens = [_CONTRACTIONS.get(t, t) for t in s.split()]
    tokens = [t for t in tokens if t not in _ARTICLES]
    return " ".join(tokens).strip()


def vqa_accuracy(pred: str, gt_answers: List[str]) -> float:
    """
    Soft VQA accuracy: min(#matching_human_answers / 3, 1.0).
    gt_answers: list of up to 10 human answer strings.
    """
    pred_norm = normalize_vqa_answer(pred)
    count = sum(normalize_vqa_answer(a) == pred_norm for a in gt_answers)
    return min(count / 3.0, 1.0)


# ── Dataset loaders ───────────────────────────────────────────────────────────

def load_vqav2(n_total: int = 550, seed: int = 42) -> List[Dict]:
    """
    Load VQAv2 validation samples with embedded PIL images.
    HuggingFaceM4/VQAv2 uses a loading script (not supported in datasets>=2.x).
    Uses lmms-lab/VQAv2 (Parquet format) instead.
    """
    from datasets import load_dataset

    # lmms-lab/VQAv2: Parquet format, no loading script required
    # split name may vary; try both "validation" and "val"
    ds = None
    for split_name in ("validation", "val"):
        try:
            print(f"[VQAv2] Trying lmms-lab/VQAv2 split={split_name}...")
            ds = load_dataset("lmms-lab/VQAv2", split=split_name)
            print(f"[VQAv2] OK: lmms-lab/VQAv2 ({split_name}), {len(ds)} samples")
            break
        except Exception as e:
            print(f"[VQAv2] {split_name} 실패: {e}")

    if ds is None:
        raise RuntimeError(
            "[VQAv2] 데이터셋 로드 실패.\n"
            "  확인: huggingface-cli login 또는\n"
            "  pip install -U datasets 후 재시도"
        )

    ds = ds.shuffle(seed=seed).select(range(min(n_total, len(ds))))

    samples = []
    for item in ds:
        # answers: list[str] | list[dict] | str — 포맷 무관하게 처리
        raw = item.get("answers", item.get("multiple_choice_answer", []))
        if isinstance(raw, list) and len(raw) > 0:
            if isinstance(raw[0], dict):
                answers = [a.get("answer", "") for a in raw]
            else:
                answers = [str(a) for a in raw]
        elif isinstance(raw, str) and raw:
            answers = [raw]
        else:
            answers = []

        image = item.get("image", item.get("img", None))
        if image is None:
            continue

        samples.append({
            "image":    image,
            "question": item["question"],
            "answers":  answers,
            "id":       str(item.get("question_id", item.get("id", len(samples)))),
            "dataset":  "vqav2",
        })

    print(f"[VQAv2] {len(samples)}개 샘플 로드 완료.")
    return samples


def _parse_pope_item(item: Dict, split_name: str, idx: int) -> Optional[Dict]:
    """POPE 단일 아이템을 표준 형식으로 변환. 이미지 없으면 None 반환."""
    image = item.get("image", None)
    if image is None:
        return None
    label = str(item.get("label", item.get("answer", ""))).lower().strip()
    if label in ("1", "true"):
        label = "yes"
    elif label in ("0", "false"):
        label = "no"
    return {
        "image":      image,
        "question":   item["question"],
        "label":      label,
        "pope_split": split_name,
        "id":         str(item.get("question_id", f"{split_name}_{idx}")),
        "dataset":    "pope",
    }


def load_pope(n_per_split: int = 200, seed: int = 42) -> List[Dict]:
    """
    Load POPE from lmms-lab/POPE (adversarial / popular / random).

    Strategy 1: named config 로딩 (adversarial, popular, random)
    Strategy 2: 'Full'/'default' 통합 config → category 필드로 분류
    """
    from datasets import load_dataset

    splits = ["adversarial", "popular", "random"]
    all_samples: List[Dict] = []

    # lmms-lab/POPE 실제 구조:
    #   config = "Full" (또는 "default")
    #   split  = "adversarial" | "popular" | "random"
    # → load_dataset("lmms-lab/POPE", "Full", split="adversarial") 형태로 로드

    for split_name in splits:
        print(f"[POPE] Loading {n_per_split} samples: config=Full, split={split_name}...")
        ds = None

        # 1순위: Full config + split 이름
        for cfg in ("Full", "default"):
            try:
                ds = load_dataset("lmms-lab/POPE", cfg, split=split_name)
                break
            except Exception:
                pass

        # 2순위: split을 config 이름으로 쓰는 구버전 형식
        if ds is None:
            try:
                ds = load_dataset("lmms-lab/POPE", split_name, split="test")
            except Exception as e:
                print(f"[POPE] '{split_name}' 로드 실패, 스킵: {e}")
                continue

        ds = ds.shuffle(seed=seed).select(range(min(n_per_split, len(ds))))
        for idx, item in enumerate(ds):
            parsed = _parse_pope_item(item, split_name, idx)
            if parsed:
                all_samples.append(parsed)
        print(f"[POPE] '{split_name}': {len(ds)} samples 로드 완료")

    print(f"[POPE] 총 {len(all_samples)}개 샘플 로드 완료.")
    return all_samples


# ── POPE metrics ──────────────────────────────────────────────────────────────

def extract_pope_answer(text: str) -> str:
    text = text.lower().strip()
    if text.startswith("yes"):
        return "yes"
    if text.startswith("no"):
        return "no"
    if re.search(r"\byes\b", text):
        return "yes"
    if re.search(r"\bno\b", text):
        return "no"
    return "unknown"


def compute_pope_metrics(preds: List[str], labels: List[str]) -> Dict:
    tp = sum(p == "yes" and l == "yes" for p, l in zip(preds, labels))
    tn = sum(p == "no"  and l == "no"  for p, l in zip(preds, labels))
    fp = sum(p == "yes" and l == "no"  for p, l in zip(preds, labels))
    fn = sum(p == "no"  and l == "yes" for p, l in zip(preds, labels))

    n = len(preds)
    acc  = (tp + tn) / n if n > 0 else 0.0
    prec = tp / (tp + fp)  if (tp + fp) > 0  else 0.0
    rec  = tp / (tp + fn)  if (tp + fn) > 0  else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
