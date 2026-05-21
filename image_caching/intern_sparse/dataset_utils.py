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
    s = _CONTRACTIONS.get(s, s)
    s = _process_punctuation(s)
    tokens = [t for t in s.split() if t not in _ARTICLES]
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
    Returns list of dicts with keys: image, question, answers, id.
    """
    from datasets import load_dataset
    print(f"[VQAv2] Loading {n_total} samples from HuggingFaceM4/VQAv2 (validation)...")
    ds = load_dataset(
        "HuggingFaceM4/VQAv2",
        split="validation",
        trust_remote_code=True,
    )
    ds = ds.shuffle(seed=seed).select(range(min(n_total, len(ds))))

    samples = []
    for item in ds:
        # answers field: list of {"answer_id": int, "answer": str, "answer_confidence": str}
        raw_answers = item.get("answers", [])
        if isinstance(raw_answers, list) and len(raw_answers) > 0:
            if isinstance(raw_answers[0], dict):
                answers = [a["answer"] for a in raw_answers]
            else:
                answers = list(raw_answers)
        else:
            answers = [item.get("multiple_choice_answer", "")]

        samples.append({
            "image":    item["image"],
            "question": item["question"],
            "answers":  answers,
            "id":       str(item.get("question_id", item.get("id", len(samples)))),
            "dataset":  "vqav2",
        })

    print(f"[VQAv2] Loaded {len(samples)} samples.")
    return samples


def load_pope(n_per_split: int = 200, seed: int = 42) -> List[Dict]:
    """
    Load POPE from lmms-lab/POPE, all 3 splits: adversarial, popular, random.
    Returns shuffled mix of all splits.
    """
    from datasets import load_dataset
    splits = ["adversarial", "popular", "random"]
    all_samples = []

    for split_name in splits:
        print(f"[POPE] Loading {n_per_split} samples from split={split_name}...")
        try:
            ds = load_dataset(
                "lmms-lab/POPE",
                split_name,
                split="test",
                trust_remote_code=True,
            )
        except Exception:
            # Fallback: try without config name
            ds = load_dataset("lmms-lab/POPE", split="test", trust_remote_code=True)

        ds = ds.shuffle(seed=seed).select(range(min(n_per_split, len(ds))))

        for idx, item in enumerate(ds):
            image = item.get("image", None)
            if image is None:
                continue

            # label field: "yes"/"no" or "1"/"0"
            label = str(item.get("label", item.get("answer", ""))).lower().strip()
            if label in ("1", "true"):
                label = "yes"
            elif label in ("0", "false"):
                label = "no"

            all_samples.append({
                "image":      image,
                "question":   item["question"],
                "label":      label,
                "pope_split": split_name,
                "id":         str(item.get("question_id", f"{split_name}_{idx}")),
                "dataset":    "pope",
            })

    print(f"[POPE] Loaded {len(all_samples)} samples total ({len(splits)} splits).")
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
