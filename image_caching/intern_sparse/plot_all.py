"""
Generate all 11 graphs from experiment results.

Graph types:
  [1] Pipeline stage breakdown   — 4 separate figures (one per pipeline)
  [2] TTFT / Decode / Latency   — 4 separate figures (one per pipeline)
  [3] Throughput (Images/s)     — 1 combined figure (4 pipelines)
  [4] TPOT (s/token)            — 1 combined figure (4 pipelines)
  [5] Accuracy                  — 1 combined figure (VQA Acc + POPE F1)

Usage:
    python plot_all.py --results_dir results/ --plots_dir plots/
"""
from __future__ import annotations
import argparse
import os
from glob import glob
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PIPELINES = ["baseline", "cached", "sparse", "cached_sparse"]
PIPE_LABELS = {
    "baseline":      "Baseline",
    "cached":        "Cached",
    "sparse":        "Baseline\n+FastV",
    "cached_sparse": "Cached\n+FastV",
}
PIPE_COLORS = {
    "baseline":      "#4878d0",
    "cached":        "#e85d4a",
    "sparse":        "#6acc65",
    "cached_sparse": "#ff9f4a",
}

# Stage colors (consistent across all 4 pipeline graphs)
STAGE_COLORS = {
    "Preprocess":   "#4878d0",
    "ViT":          "#e85d4a",
    "Unshuffle":    "#6acc65",
    "DB Load":      "#ff9f4a",
    "MLP":          "#956cb4",
    "FastV":        "#17becf",
    "Text":         "#bcbd22",
    "Fusion":       "#8c564b",
    "Gen (TTFT)":   "#e377c2",
    "Decode":       "#7f7f7f",
}

# Stage definitions per pipeline: (label, csv_column)
STAGE_DEFS = {
    "baseline": [
        ("Preprocess", "t_preprocess"),
        ("ViT",        "t_vit"),
        ("Unshuffle",  "t_unshuffle"),
        ("MLP",        "t_mlp"),
        ("Text",       "t_text"),
        ("Fusion",     "t_fusion"),
        ("Gen (TTFT)", "t_gen_ttft"),
    ],
    "cached": [
        ("DB Load",    "t_db_load"),
        ("MLP",        "t_mlp"),
        ("Text",       "t_text"),
        ("Fusion",     "t_fusion"),
        ("Gen (TTFT)", "t_gen_ttft"),
    ],
    "sparse": [
        ("Preprocess", "t_preprocess"),
        ("ViT",        "t_vit"),
        ("Unshuffle",  "t_unshuffle"),
        ("MLP",        "t_mlp"),
        ("Text",       "t_text"),
        ("Fusion",     "t_fusion"),
        ("Gen (TTFT)", "t_gen_ttft"),  # FastV prefill+prune+first-token = LLM TTFT
    ],
    "cached_sparse": [
        ("DB Load",    "t_db_load"),
        ("MLP",        "t_mlp"),
        ("Text",       "t_text"),
        ("Fusion",     "t_fusion"),
        ("Gen (TTFT)", "t_gen_ttft"),  # FastV prefill+prune+first-token = LLM TTFT
    ],
}


def bar_labels(ax, bars, fmt="{:.4f}", fontsize=8):
    for bar in bars:
        h = bar.get_height()
        if pd.isna(h) or h < 1e-6:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + ax.get_ylim()[1] * 0.01,
            fmt.format(h),
            ha="center", va="bottom", fontsize=fontsize, fontweight="bold",
        )


def load_csv(results_dir: str, pipeline: str, dataset: str) -> Optional[pd.DataFrame]:
    path = os.path.join(results_dir, f"{pipeline}_{dataset}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    # Convert timing columns to float
    timing_cols = [
        "t_preprocess", "t_vit", "t_unshuffle", "t_db_load",
        "t_mlp", "t_sparse", "t_text", "t_fusion", "t_gen_ttft",
        "true_ttft", "decode_time", "total_latency", "n_tokens", "tpot", "vram_gb",
    ]
    for c in timing_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df


# ── Graph 1: Pipeline stage breakdown (4 separate figures) ───────────────────

def plot_stage_breakdown(pipeline: str, df: pd.DataFrame, plots_dir: str, dataset: str):
    stages = STAGE_DEFS[pipeline]
    labels = [s[0] for s in stages]
    cols   = [s[1] for s in stages]
    avgs   = [df[c].mean() if c in df.columns else 0.0 for c in cols]
    colors = [STAGE_COLORS.get(l, "#999999") for l in labels]
    n      = len(df)

    fig, ax = plt.subplots(figsize=(max(8, len(stages) * 1.4), 5))
    bars = ax.bar(labels, avgs, color=colors, edgecolor="white", width=0.6)
    bar_labels(ax, bars)

    ax.set_ylabel("Seconds (avg)")
    ax.set_title(
        f"[1] {PIPE_LABELS[pipeline].replace(chr(10), ' ')} Pipeline "
        f"Avg Time per Stage (n={n}, dataset={dataset.upper()})",
        fontweight="bold",
    )
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_ylim(0, max(avgs) * 1.18 if max(avgs) > 0 else 1)
    plt.tight_layout()

    fname = os.path.join(plots_dir, f"1_stages_{pipeline}_{dataset}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


# ── Graph 2: True TTFT / Decode / Total Latency (4 separate figures) ─────────

def plot_latency_breakdown(pipeline: str, df: pd.DataFrame, plots_dir: str, dataset: str):
    cols   = ["true_ttft", "decode_time", "total_latency"]
    labels = ["True TTFT", "Decode Time", "Total Latency"]
    colors = ["#4878d0", "#e85d4a", "#6acc65"]
    avgs   = [df[c].mean() if c in df.columns else 0.0 for c in cols]
    n      = len(df)

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, avgs, color=colors, edgecolor="white", width=0.5)
    bar_labels(ax, bars)

    ax.set_ylabel("Seconds (avg)")
    ax.set_title(
        f"[2] {PIPE_LABELS[pipeline].replace(chr(10), ' ')} "
        f"Inference Avg Time (n={n}, dataset={dataset.upper()})",
        fontweight="bold",
    )
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_ylim(0, max(avgs) * 1.18 if max(avgs) > 0 else 1)
    plt.tight_layout()

    fname = os.path.join(plots_dir, f"2_latency_{pipeline}_{dataset}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


# ── Graph 3: Throughput (1 combined figure) ───────────────────────────────────

def plot_throughput(data: Dict[str, pd.DataFrame], plots_dir: str, dataset: str):
    names, vals = [], []
    for pl in PIPELINES:
        df = data.get(pl)
        if df is None or df.empty:
            continue
        avg_lat = df["total_latency"].mean()
        if avg_lat > 0:
            names.append(PIPE_LABELS[pl])
            vals.append(1.0 / avg_lat)

    if not vals:
        return

    colors = [PIPE_COLORS[pl] for pl in PIPELINES if pl in data and data[pl] is not None and not data[pl].empty]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, vals, color=colors[:len(names)], edgecolor="white", width=0.5)
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + max(vals) * 0.02,
            f"{v:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold",
        )
    ax.set_ylabel("Images / second")
    ax.set_title(f"[3] Throughput (Images/s) — {dataset.upper()}", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_ylim(0, max(vals) * 1.2)
    plt.tight_layout()

    fname = os.path.join(plots_dir, f"3_throughput_{dataset}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


# ── Graph 4: TPOT (1 combined figure) ────────────────────────────────────────

def plot_tpot(data: Dict[str, pd.DataFrame], plots_dir: str, dataset: str):
    names, vals = [], []
    for pl in PIPELINES:
        df = data.get(pl)
        if df is None or df.empty or "tpot" not in df.columns:
            continue
        names.append(PIPE_LABELS[pl])
        vals.append(df["tpot"].mean())

    if not vals:
        return

    colors = [PIPE_COLORS[pl] for pl in PIPELINES if pl in data and data[pl] is not None and not data[pl].empty]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, vals, color=colors[:len(names)], edgecolor="white", width=0.5)
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + max(vals) * 0.02,
            f"{v:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold",
        )
    ax.set_ylabel("s / token")
    ax.set_title(f"[4] TPOT (s/token) — {dataset.upper()}", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_ylim(0, max(vals) * 1.2)
    plt.tight_layout()

    fname = os.path.join(plots_dir, f"4_tpot_{dataset}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


# ── Graph 5: Accuracy (1 combined figure) ────────────────────────────────────

def compute_accuracy_summary(results_dir: str) -> Dict[str, Dict[str, float]]:
    """Returns {pipeline: {vqa_acc, pope_f1}}."""
    summary: Dict[str, Dict] = {}

    for pl in PIPELINES:
        summary[pl] = {"vqa_acc": None, "pope_f1": None}

        # VQAv2
        df_vqa = load_csv(results_dir, pl, "vqav2")
        if df_vqa is not None and "vqa_score" in df_vqa.columns:
            vqa_scores = pd.to_numeric(df_vqa["vqa_score"], errors="coerce").dropna()
            if len(vqa_scores) > 0:
                summary[pl]["vqa_acc"] = vqa_scores.mean() * 100

        # POPE
        df_pope = load_csv(results_dir, pl, "pope")
        if df_pope is not None and "pope_label" in df_pope.columns:
            # extract_pope_answer와 동일한 로직으로 파싱 (word boundary 사용)
            # → 실험 런타임의 F1과 일치하도록 보장
            from dataset_utils import compute_pope_metrics, extract_pope_answer
            preds  = df_pope["prediction"].apply(
                lambda x: extract_pope_answer(str(x))
            ).tolist()
            labels = df_pope["pope_label"].tolist()
            if preds:
                m = compute_pope_metrics(preds, labels)
                summary[pl]["pope_f1"] = m["f1"] * 100

    return summary


def plot_accuracy(summary: Dict[str, Dict], plots_dir: str):
    pipe_labels = [PIPE_LABELS[pl] for pl in PIPELINES]
    vqa_vals    = [summary.get(pl, {}).get("vqa_acc") for pl in PIPELINES]
    pope_vals   = [summary.get(pl, {}).get("pope_f1") for pl in PIPELINES]

    has_vqa  = any(v is not None for v in vqa_vals)
    has_pope = any(v is not None for v in pope_vals)

    if not has_vqa and not has_pope:
        print("  [Skip] Graph 5: No accuracy data found.")
        return

    n_groups  = 2 if (has_vqa and has_pope) else 1
    fig, axes = plt.subplots(1, n_groups, figsize=(6 * n_groups, 5))
    if n_groups == 1:
        axes = [axes]

    plot_idx = 0

    if has_vqa:
        ax = axes[plot_idx]; plot_idx += 1
        bars_v = ax.bar(
            pipe_labels,
            [v if v is not None else 0.0 for v in vqa_vals],
            color=[PIPE_COLORS[pl] for pl in PIPELINES],
            edgecolor="white", width=0.5,
        )
        for bar, v in zip(bars_v, vqa_vals):
            if v is not None:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    v + 0.5,
                    f"{v:.1f}%", ha="center", va="bottom",
                    fontsize=10, fontweight="bold",
                )
        ax.set_ylabel("VQA Accuracy (%)")
        ax.set_ylim(0, 100)
        ax.set_title("[5a] VQA Accuracy (VQAv2)", fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    if has_pope:
        ax = axes[plot_idx]
        bars_p = ax.bar(
            pipe_labels,
            [v if v is not None else 0.0 for v in pope_vals],
            color=[PIPE_COLORS[pl] for pl in PIPELINES],
            edgecolor="white", width=0.5,
        )
        for bar, v in zip(bars_p, pope_vals):
            if v is not None:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    v + 0.5,
                    f"{v:.1f}%", ha="center", va="bottom",
                    fontsize=10, fontweight="bold",
                )
        ax.set_ylabel("POPE F1 (%)")
        ax.set_ylim(0, 100)
        ax.set_title("[5b] POPE F1 (avg 3 splits)", fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.suptitle(
        "[5] Accuracy — Baseline / Cached / Sparse / Cached+Sparse",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    fname = os.path.join(plots_dir, "5_accuracy.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--plots_dir",   default="plots")
    parser.add_argument(
        "--timing_dataset", default="vqav2",
        choices=["vqav2", "pope"],
        help="Which dataset's timing data to use for graphs 1-4",
    )
    args = parser.parse_args()

    os.makedirs(args.plots_dir, exist_ok=True)
    ds = args.timing_dataset

    print(f"\nLoading CSVs from: {args.results_dir}/  (timing dataset: {ds})")
    data = {pl: load_csv(args.results_dir, pl, ds) for pl in PIPELINES}

    print("\n--- Generating graphs 1 & 2 (per-pipeline stage + latency) ---")
    for pl in PIPELINES:
        df = data[pl]
        if df is None or df.empty:
            print(f"  [Skip] {pl}: no data")
            continue
        plot_stage_breakdown(pl, df, args.plots_dir, ds)
        plot_latency_breakdown(pl, df, args.plots_dir, ds)

    print("\n--- Generating graph 3 (throughput) ---")
    plot_throughput(data, args.plots_dir, ds)

    print("\n--- Generating graph 4 (TPOT) ---")
    plot_tpot(data, args.plots_dir, ds)

    print("\n--- Generating graph 5 (accuracy) ---")
    acc_summary = compute_accuracy_summary(args.results_dir)
    plot_accuracy(acc_summary, args.plots_dir)

    print(f"\nAll plots saved to: {args.plots_dir}/")
    print("\nAccuracy summary:")
    for pl, vals in acc_summary.items():
        vqa  = f"{vals['vqa_acc']:.1f}%" if vals.get("vqa_acc") is not None else "N/A"
        pope = f"{vals['pope_f1']:.1f}%"  if vals.get("pope_f1")  is not None else "N/A"
        print(f"  {pl:20s}  VQA={vqa:8s}  POPE F1={pope}")


if __name__ == "__main__":
    main()
