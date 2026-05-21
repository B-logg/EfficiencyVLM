"""
Unified experiment runner for 4 InternVL3.5 pipelines × 2 datasets.

Pipelines:
  baseline      : Preprocess → ViT → Unshuffle → MLP → Text → Fusion → Generate
  cached        : DB Load → MLP → Text → Fusion → Generate
  sparse        : Preprocess → ViT → Unshuffle → MLP → SparseVLM → Text → Fusion → Generate
  cached_sparse : DB Load → MLP → SparseVLM → Text → Fusion → Generate

Datasets:
  vqav2 : VQAv2 validation, metric = VQA Accuracy
  pope  : POPE (adversarial + popular + random averaged), metric = F1

Usage:
    python run_experiment.py --pipeline baseline --dataset vqav2
    python run_experiment.py --pipeline all --dataset all   # 8 runs in one execution
"""
from __future__ import annotations
import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, LogitsProcessor, LogitsProcessorList
from transformers.modeling_utils import PreTrainedModel

ROOT = str(Path(__file__).resolve().parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sparse_pruner import NormBasedTokenPruner
from dataset_utils import (
    load_vqav2, load_pope,
    vqa_accuracy, extract_pope_answer, compute_pope_metrics,
)

MODEL_ID    = "OpenGVLab/InternVL3_5-8B"
KEEP_RATIO  = 0.5          # 256 → 128 visual tokens
N_WARMUP    = 50
N_MEASURE   = 500
POPE_N_PER_SPLIT = 200     # 200 × 3 splits = 600 total; N_MEASURE=500 used for timing avg

TRANSFORM = T.Compose([
    T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
    T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

PIPELINES = ["baseline", "cached", "sparse", "cached_sparse"]
DATASETS  = ["vqav2", "pope"]


# ── Timing helpers ────────────────────────────────────────────────────────────

class CUDATimer:
    def __init__(self):
        self.s = torch.cuda.Event(enable_timing=True)
        self.e = torch.cuda.Event(enable_timing=True)

    def start(self): self.s.record()
    def stop(self):  self.e.record()

    def get_time(self) -> float:
        torch.cuda.synchronize()
        return self.s.elapsed_time(self.e) / 1000.0  # seconds


class TTFTLogitsProcessor(LogitsProcessor):
    def __init__(self):
        self.evt      = torch.cuda.Event(enable_timing=True)
        self.is_first = True

    def __call__(self, input_ids, scores):
        if self.is_first:
            self.evt.record()
            self.is_first = False
        return scores


# ── Pixel shuffle ─────────────────────────────────────────────────────────────

def pixel_shuffle_2x(vit_embs: torch.Tensor) -> torch.Tensor:
    b, s, c = vit_embs.shape
    h = w = int(s ** 0.5)
    return (
        vit_embs
        .reshape(b, h, w, c)
        .unfold(1, 2, 2).unfold(2, 2, 2)
        .reshape(b, h // 2, w // 2, 4, c)
        .reshape(b, h // 2, w // 2, c * 4)
        .reshape(b, -1, c * 4)
    )


# ── Prompt helpers ────────────────────────────────────────────────────────────

def make_prompt_tokens(tokenizer, question: str, dataset: str, device):
    """
    InternVL format: 'User: ' + [visual] + '\n{question}\n{instruction}\nAssistant:'
    Returns (tok_pre, tok_post) as token IDs.
    """
    if dataset == "vqav2":
        instruction = "Answer with a single word or short phrase."
    else:
        instruction = "Answer yes or no."

    tok_pre  = tokenizer(
        "User: ",
        return_tensors="pt", add_special_tokens=True,
    ).input_ids.to(device)
    tok_post = tokenizer(
        f"\n{question}\n{instruction}\nAssistant:",
        return_tensors="pt", add_special_tokens=False,
    ).input_ids.to(device)
    return tok_pre, tok_post


def max_new_tokens_for(dataset: str) -> int:
    return 16 if dataset == "vqav2" else 8


# ── Single-sample inference functions ────────────────────────────────────────

@torch.no_grad()
def run_baseline(item, model, tokenizer, device, dataset, pruner=None):
    """Preprocess → ViT → Unshuffle → MLP → [Sparse] → Text → Fusion → Generate."""
    question = item["question"]

    # 1. Preprocess (CPU + H2D)
    torch.cuda.synchronize()
    _t = time.perf_counter()
    pv = TRANSFORM(item["image"]).unsqueeze(0).to(device, dtype=torch.bfloat16)
    torch.cuda.synchronize()
    r_preproc = time.perf_counter() - _t

    # 2. ViT
    t_vit = CUDATimer(); t_vit.start()
    vit_out = model.vision_model(pv)
    hidden  = vit_out.last_hidden_state
    if hidden.shape[1] == 1025:
        hidden = hidden[:, 1:, :]
    t_vit.stop()
    torch.cuda.synchronize()
    r_vit = t_vit.get_time()

    # 3. Pixel Unshuffle
    t_unsh = CUDATimer(); t_unsh.start()
    shuffled = pixel_shuffle_2x(hidden)   # [1, 256, D*4]
    t_unsh.stop()
    torch.cuda.synchronize()
    r_unsh = t_unsh.get_time()

    # 4. MLP1
    t_mlp = CUDATimer(); t_mlp.start()
    img_embs = model.mlp1(shuffled).squeeze(0)   # [256, D_llm]
    t_mlp.stop()
    torch.cuda.synchronize()
    r_mlp = t_mlp.get_time()

    # 5. [Optional] Sparse pruning
    r_sparse = 0.0
    if pruner is not None:
        img_embs, r_sparse = pruner.prune(img_embs)
        r_sparse = r_sparse / 1000.0   # ms → s

    # 6. Text tokenize + embed
    torch.cuda.synchronize()
    _t = time.perf_counter()
    tok_pre, tok_post = make_prompt_tokens(tokenizer, question, dataset, device)
    emb_pre  = model.language_model.get_input_embeddings()(tok_pre)
    emb_post = model.language_model.get_input_embeddings()(tok_post)
    torch.cuda.synchronize()
    r_text = time.perf_counter() - _t

    # 7. Fusion
    t_fus = CUDATimer(); t_fus.start()
    n_vis  = img_embs.shape[0]
    f_embs = torch.cat([emb_pre, img_embs.unsqueeze(0), emb_post], dim=1)
    f_mask = torch.cat([
        torch.ones_like(tok_pre),
        torch.ones((1, n_vis), dtype=tok_pre.dtype, device=device),
        torch.ones_like(tok_post),
    ], dim=1)
    t_fus.stop()
    torch.cuda.synchronize()
    r_fus = t_fus.get_time()

    # 8. Generate
    hnd   = TTFTLogitsProcessor()
    t_gen = CUDATimer(); t_gen.start()
    outs  = model.language_model.generate(
        inputs_embeds=f_embs, attention_mask=f_mask,
        max_new_tokens=max_new_tokens_for(dataset),
        logits_processor=LogitsProcessorList([hnd]),
        do_sample=False,
    )
    t_gen.stop()
    torch.cuda.synchronize()

    r_ttft  = t_gen.s.elapsed_time(hnd.evt) / 1000.0
    r_decode = max(t_gen.get_time() - r_ttft, 0.0)
    true_ttft = r_preproc + r_vit + r_unsh + r_mlp + r_sparse + r_text + r_fus + r_ttft
    total_lat = true_ttft + r_decode
    n_tok     = max(outs.shape[1] - 1, 1)
    tpot      = r_decode / n_tok

    pred_text = tokenizer.decode(outs[0], skip_special_tokens=True).strip()

    return {
        "t_preprocess": r_preproc,
        "t_vit":        r_vit,
        "t_unshuffle":  r_unsh,
        "t_db_load":    0.0,
        "t_mlp":        r_mlp,
        "t_sparse":     r_sparse,
        "t_text":       r_text,
        "t_fusion":     r_fus,
        "t_gen_ttft":   r_ttft,
        "true_ttft":    true_ttft,
        "decode_time":  r_decode,
        "total_latency": total_lat,
        "n_tokens":     outs.shape[1],
        "tpot":         tpot,
        "prediction":   pred_text,
        "vram_gb":      torch.cuda.max_memory_allocated() / (1024 ** 3),
    }


@torch.no_grad()
def run_cached(item, model, tokenizer, device, dataset, embed_dir, pruner=None):
    """DB Load → MLP → [Sparse] → Text → Fusion → Generate."""
    question = item["question"]
    pt_path  = os.path.join(embed_dir, f"{item['dataset']}_{item['id']}.pt")

    if not os.path.exists(pt_path):
        return None  # embedding not found

    # 1. DB Load (CPU disk + H2D)
    torch.cuda.synchronize()
    _t = time.perf_counter()
    payload  = torch.load(pt_path, map_location="cpu")
    shuffled = payload["pixel_shuffled"].to(device, dtype=torch.bfloat16)
    torch.cuda.synchronize()
    r_db = time.perf_counter() - _t

    # 2. MLP1
    t_mlp = CUDATimer(); t_mlp.start()
    img_embs = model.mlp1(shuffled).squeeze(0)   # [256, D_llm]
    t_mlp.stop()
    torch.cuda.synchronize()
    r_mlp = t_mlp.get_time()

    # 3. [Optional] Sparse pruning
    r_sparse = 0.0
    if pruner is not None:
        img_embs, r_sparse = pruner.prune(img_embs)
        r_sparse = r_sparse / 1000.0

    # 4. Text tokenize + embed
    torch.cuda.synchronize()
    _t = time.perf_counter()
    tok_pre, tok_post = make_prompt_tokens(tokenizer, question, dataset, device)
    emb_pre  = model.language_model.get_input_embeddings()(tok_pre)
    emb_post = model.language_model.get_input_embeddings()(tok_post)
    torch.cuda.synchronize()
    r_text = time.perf_counter() - _t

    # 5. Fusion
    t_fus = CUDATimer(); t_fus.start()
    n_vis  = img_embs.shape[0]
    f_embs = torch.cat([emb_pre, img_embs.unsqueeze(0), emb_post], dim=1)
    f_mask = torch.cat([
        torch.ones_like(tok_pre),
        torch.ones((1, n_vis), dtype=tok_pre.dtype, device=device),
        torch.ones_like(tok_post),
    ], dim=1)
    t_fus.stop()
    torch.cuda.synchronize()
    r_fus = t_fus.get_time()

    # 6. Generate
    hnd   = TTFTLogitsProcessor()
    t_gen = CUDATimer(); t_gen.start()
    outs  = model.language_model.generate(
        inputs_embeds=f_embs, attention_mask=f_mask,
        max_new_tokens=max_new_tokens_for(dataset),
        logits_processor=LogitsProcessorList([hnd]),
        do_sample=False,
    )
    t_gen.stop()
    torch.cuda.synchronize()

    r_ttft   = t_gen.s.elapsed_time(hnd.evt) / 1000.0
    r_decode = max(t_gen.get_time() - r_ttft, 0.0)
    true_ttft = r_db + r_mlp + r_sparse + r_text + r_fus + r_ttft
    total_lat = true_ttft + r_decode
    n_tok     = max(outs.shape[1] - 1, 1)
    tpot      = r_decode / n_tok

    pred_text = tokenizer.decode(outs[0], skip_special_tokens=True).strip()

    return {
        "t_preprocess": 0.0,
        "t_vit":        0.0,
        "t_unshuffle":  0.0,
        "t_db_load":    r_db,
        "t_mlp":        r_mlp,
        "t_sparse":     r_sparse,
        "t_text":       r_text,
        "t_fusion":     r_fus,
        "t_gen_ttft":   r_ttft,
        "true_ttft":    true_ttft,
        "decode_time":  r_decode,
        "total_latency": total_lat,
        "n_tokens":     outs.shape[1],
        "tpot":         tpot,
        "prediction":   pred_text,
        "vram_gb":      torch.cuda.max_memory_allocated() / (1024 ** 3),
    }


# ── Main measurement loop ─────────────────────────────────────────────────────

def run_pipeline_on_dataset(
    pipeline: str,
    dataset_name: str,
    samples: List[Dict],
    model,
    tokenizer,
    device: str,
    embed_dir: str,
    results_dir: str,
):
    pruner    = NormBasedTokenPruner(KEEP_RATIO) if "sparse" in pipeline else None
    use_cache = "cached" in pipeline

    out_path = os.path.join(results_dir, f"{pipeline}_{dataset_name}.csv")
    print(f"\n{'='*60}")
    print(f"Pipeline: {pipeline} | Dataset: {dataset_name} | n={N_MEASURE}+{N_WARMUP}w")
    print(f"Output: {out_path}")
    print("=" * 60)

    timing_keys = [
        "t_preprocess", "t_vit", "t_unshuffle", "t_db_load",
        "t_mlp", "t_sparse", "t_text", "t_fusion", "t_gen_ttft",
        "true_ttft", "decode_time", "total_latency", "n_tokens", "tpot", "vram_gb",
    ]
    meta_keys   = ["pipeline", "dataset", "id", "question", "prediction"]
    acc_keys    = ["gt_answers", "vqa_score", "pope_label", "pope_split", "pope_correct"]
    fieldnames  = meta_keys + timing_keys + acc_keys

    rows         = []
    all_preds    = []
    all_labels   = []
    all_splits   = []
    vqa_scores   = []

    warmup_done  = 0
    measure_done = 0
    total_needed = N_WARMUP + N_MEASURE

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        pbar = tqdm(
            samples[:total_needed],
            desc=f"{pipeline}/{dataset_name}",
            unit="sample",
        )
        for item in pbar:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            try:
                if use_cache:
                    res = run_cached(item, model, tokenizer, device, dataset_name, embed_dir, pruner)
                else:
                    res = run_baseline(item, model, tokenizer, device, dataset_name, pruner)
            except Exception as e:
                print(f"[WARN] Sample {item['id']} failed: {e}")
                continue

            if res is None:
                print(f"[WARN] Embedding not found for {item['id']}, skipping.")
                continue

            is_warmup = warmup_done < N_WARMUP
            if is_warmup:
                warmup_done += 1
                continue

            # ── Accuracy ──────────────────────────────────────────────────
            pred = res["prediction"]
            if dataset_name == "vqav2":
                gt_answers = item.get("answers", [])
                score = vqa_accuracy(pred, gt_answers)
                vqa_scores.append(score)
                acc_data = {
                    "gt_answers":   str(gt_answers),
                    "vqa_score":    f"{score:.4f}",
                    "pope_label":   "",
                    "pope_split":   "",
                    "pope_correct": "",
                }
            else:
                pope_pred  = extract_pope_answer(pred)
                pope_label = item.get("label", "")
                correct    = int(pope_pred == pope_label)
                all_preds.append(pope_pred)
                all_labels.append(pope_label)
                all_splits.append(item.get("pope_split", "unknown"))
                acc_data = {
                    "gt_answers":   pope_label,
                    "vqa_score":    "",
                    "pope_label":   pope_label,
                    "pope_split":   item.get("pope_split", ""),
                    "pope_correct": str(correct),
                }

            row = {
                "pipeline":  pipeline,
                "dataset":   dataset_name,
                "id":        item["id"],
                "question":  item["question"],
                "prediction": pred,
            }
            row.update(res)
            row.update(acc_data)
            writer.writerow(row)
            rows.append(row)
            measure_done += 1

            pbar.set_postfix(measured=measure_done, done=warmup_done)
            if measure_done >= N_MEASURE:
                break

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n[{pipeline}/{dataset_name}] Measured: {measure_done} samples")
    if dataset_name == "vqav2" and vqa_scores:
        print(f"  VQA Accuracy: {sum(vqa_scores)/len(vqa_scores)*100:.2f}%")
    elif dataset_name == "pope" and all_preds:
        overall = compute_pope_metrics(all_preds, all_labels)
        print(f"  POPE Overall → Acc={overall['accuracy']*100:.2f}%  F1={overall['f1']*100:.2f}%")
        for sp in ["adversarial", "popular", "random"]:
            idx = [i for i, s in enumerate(all_splits) if s == sp]
            if idx:
                m = compute_pope_metrics(
                    [all_preds[i] for i in idx],
                    [all_labels[i] for i in idx],
                )
                print(f"    {sp:12s}: Acc={m['accuracy']*100:.2f}%  F1={m['f1']*100:.2f}%")

    return rows


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pipeline", default="all",
        choices=PIPELINES + ["all"],
        help="Which pipeline to run (or 'all' for all 4)",
    )
    parser.add_argument(
        "--dataset", default="all",
        choices=DATASETS + ["all"],
        help="Which dataset to run (or 'all' for both)",
    )
    parser.add_argument("--embed_dir",   default="embeddings")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--device",      default="cuda")
    args = parser.parse_args()

    pipelines_to_run = PIPELINES if args.pipeline == "all" else [args.pipeline]
    datasets_to_run  = DATASETS  if args.dataset  == "all" else [args.dataset]

    os.makedirs(args.results_dir, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────
    print(f"Loading model: {MODEL_ID}")
    PreTrainedModel.all_tied_weights_keys = {}
    model = AutoModel.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16,
        trust_remote_code=True, low_cpu_mem_usage=True,
    ).eval().to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    print("Model loaded.\n")

    # ── Load datasets (once, shared across pipelines) ─────────────────────
    samples_by_ds: Dict[str, List] = {}
    total_needed = N_WARMUP + N_MEASURE

    if "vqav2" in datasets_to_run:
        samples_by_ds["vqav2"] = load_vqav2(n_total=total_needed)

    if "pope" in datasets_to_run:
        # Load enough per split for warmup + measure
        # Warmup uses first N_WARMUP from combined list; rest for measure
        per_split = max(POPE_N_PER_SPLIT, (total_needed // 3) + 10)
        samples_by_ds["pope"] = load_pope(n_per_split=per_split)

    # ── Run experiments ───────────────────────────────────────────────────
    for ds_name in datasets_to_run:
        samples = samples_by_ds[ds_name]
        for pl in pipelines_to_run:
            # Check if cached pipeline has embeddings
            if "cached" in pl:
                test_id   = samples[0]["id"]
                test_path = os.path.join(
                    args.embed_dir, f"{samples[0]['dataset']}_{test_id}.pt"
                )
                if not os.path.exists(test_path):
                    print(
                        f"\n[ERROR] Embeddings not found for {pl}/{ds_name}.\n"
                        f"Please run: python encode_embeddings.py first.\n"
                        f"Expected: {test_path}"
                    )
                    continue

            run_pipeline_on_dataset(
                pipeline=pl,
                dataset_name=ds_name,
                samples=samples,
                model=model,
                tokenizer=tokenizer,
                device=args.device,
                embed_dir=args.embed_dir,
                results_dir=args.results_dir,
            )

    print(f"\nAll results saved to: {args.results_dir}/")
    print("Next step: python plot_all.py")


if __name__ == "__main__":
    main()
