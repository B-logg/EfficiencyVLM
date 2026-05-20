"""
E2E vs Cached Video Query — Comparison Charts

E2E pipeline:    Decode | Preprocess | ViT(×100f) | PixelShuffle | MLP | LLM
Cached Stage A:  Decode | Preprocess | ViT(×100f) | PixelShuffle | Save  (amortized)
Cached Stage B:  Load .pt | MLP | LLM  (per-query latency)

Charts:
  1. Per-stage avg time (E2E full pipeline vs Cached Stage B)
  2. E2E total latency vs Cached Stage B latency — histogram overlay
  3. Grouped bar: Vision cost (E2E) vs Storage load cost (Cached) | MLP+LLM cost
  4. Stage A amortized cost per query (encode cost / N queries)
  5. Peak VRAM comparison
  6. MVBench accuracy (if available)
  7. Throughput (queries/s) — E2E TTFT vs Cached Stage B TTFT

Usage:
    python scripts/10_plot_comparison.py \
        --e2e_dir    outputs/e2e \
        --cached_dir outputs/cached_video \
        --out_dir    outputs/comparison_plots
"""
from __future__ import annotations
import argparse
import json
import os
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Color palette ─────────────────────────────────────────────────────────────
C_E2E    = "#56b4e9"   # blue
C_CACHED = "#e85d4a"   # red/orange
C_STAGE  = ["#4878d0", "#6acc65", "#ff9f4a", "#f6cf71", "#69b3a2", "#d65f5f"]

E2E_STAGE_COLS   = ["t_decode_ms", "t_preprocess_ms", "t_vit_ms",
                     "t_pixel_shuffle_ms", "t_mlp_ms", "t_llm_ms"]
E2E_STAGE_LABELS = ["Decode", "Preprocess", "ViT\n(×100f)", "Pixel\nShuffle",
                     "MLP", "LLM"]

CACHED_B_STAGE_COLS   = ["t_load_ms", "t_mlp_ms", "t_llm_ms"]
CACHED_B_STAGE_LABELS = ["Load .pt", "MLP", "LLM"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def bar_val(ax, bars, fmt="{:.1f}", fontsize=8):
    for bar in bars:
        h = bar.get_height()
        if pd.isna(h) or h < 1e-3:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + ax.get_ylim()[1] * 0.01,
            fmt.format(h),
            ha="center", va="bottom", fontsize=fontsize, fontweight="bold",
        )


def load_latest_e2e_csv(e2e_dir: str) -> pd.DataFrame:
    files = sorted(glob(os.path.join(e2e_dir, "e2e_*_timing.csv")))
    if not files:
        return pd.DataFrame()
    df = pd.read_csv(files[-1])
    print(f"  E2E CSV: {files[-1]}  (n={len(df)})")
    return df


def load_latest_cached_b_csv(cached_dir: str) -> pd.DataFrame:
    files = sorted(glob(os.path.join(cached_dir, "cached_*_stage_b.csv")))
    if not files:
        return pd.DataFrame()
    df = pd.read_csv(files[-1])
    df = df[df["status"] == "ok"].reset_index(drop=True)
    print(f"  Cached Stage B CSV: {files[-1]}  (n={len(df)})")
    return df


def load_latest_cached_a_csv(cached_dir: str) -> pd.DataFrame:
    files = sorted(glob(os.path.join(cached_dir, "cached_*_stage_a.csv")))
    if not files:
        return pd.DataFrame()
    df = pd.read_csv(files[-1])
    df = df[df["status"] == "encoded"].reset_index(drop=True)
    print(f"  Cached Stage A CSV: {files[-1]}  (n={len(df)})")
    return df


def load_accuracy(cached_dir: str) -> float | None:
    files = sorted(glob(os.path.join(cached_dir, "cached_*_accuracy.json")))
    if not files:
        return None
    with open(files[-1]) as f:
        data = json.load(f)
    return data.get("accuracy")


# ── Chart functions ───────────────────────────────────────────────────────────

def chart1_stage_avg(df_e2e, df_b, out_dir, prefix):
    """Per-stage avg: E2E all stages vs Cached Stage B stages (side by side)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: E2E
    ax = axes[0]
    avgs = [df_e2e[c].mean() for c in E2E_STAGE_COLS]
    bars = ax.bar(E2E_STAGE_LABELS, avgs, color=C_STAGE, edgecolor="white")
    bar_val(ax, bars)
    ax.set_ylabel("ms (avg)")
    ax.set_title("E2E Pipeline — Avg per Stage", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Right: Cached Stage B
    ax = axes[1]
    avgs_b = [df_b[c].mean() for c in CACHED_B_STAGE_COLS if c in df_b.columns]
    labels_b = [l for c, l in zip(CACHED_B_STAGE_COLS, CACHED_B_STAGE_LABELS)
                if c in df_b.columns]
    bars = ax.bar(labels_b, avgs_b, color=[C_STAGE[0], C_STAGE[4], C_STAGE[5]],
                  edgecolor="white")
    bar_val(ax, bars)
    ax.set_ylabel("ms (avg)")
    ax.set_title("Cached Stage B — Avg per Stage (per-query latency)", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.suptitle(f"[1] E2E vs Cached Stage B — Per-Stage Avg  "
                 f"(E2E n={len(df_e2e)}, Cached n={len(df_b)})",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    p = os.path.join(out_dir, f"{prefix}_1_stage_avg.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {p}")


def chart2_latency_hist(df_e2e, df_b, out_dir, prefix):
    """Histogram overlay: E2E total latency vs Cached Stage B total latency."""
    fig, ax = plt.subplots(figsize=(10, 5))

    e2e_lat = df_e2e["t_total_ms"]
    b_lat   = df_b["t_total_b_ms"] if "t_total_b_ms" in df_b.columns else None

    bins = np.linspace(
        min(e2e_lat.min(), b_lat.min() if b_lat is not None else e2e_lat.min()),
        max(e2e_lat.max(), b_lat.max() if b_lat is not None else e2e_lat.max()),
        35,
    )
    ax.hist(e2e_lat, bins=bins, alpha=0.65, color=C_E2E, label="E2E", edgecolor="white")
    if b_lat is not None:
        ax.hist(b_lat, bins=bins, alpha=0.65, color=C_CACHED,
                label="Cached Stage B", edgecolor="white")
        ax.axvline(b_lat.mean(), color=C_CACHED, linestyle="--",
                   label=f"Cached mean {b_lat.mean():.1f}ms")

    ax.axvline(e2e_lat.mean(), color=C_E2E, linestyle="--",
               label=f"E2E mean {e2e_lat.mean():.1f}ms")

    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Count")
    ax.set_title(f"[2] Latency Distribution — E2E vs Cached Stage B", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    p = os.path.join(out_dir, f"{prefix}_2_latency_hist.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {p}")


def chart3_functional_groups(df_e2e, df_b, out_dir, prefix):
    """Grouped bar: Vision cost vs MLP+LLM cost for E2E vs Cached."""
    e2e_vision  = (df_e2e["t_decode_ms"] + df_e2e["t_preprocess_ms"] +
                   df_e2e["t_vit_ms"] + df_e2e["t_pixel_shuffle_ms"]).mean()
    e2e_mlp_llm = (df_e2e["t_mlp_ms"] + df_e2e["t_llm_ms"]).mean()

    cached_load    = df_b["t_load_ms"].mean() if "t_load_ms" in df_b.columns else 0.0
    cached_mlp_llm = (df_b["t_mlp_ms"] + df_b["t_llm_ms"]).mean() \
                     if "t_mlp_ms" in df_b.columns else 0.0

    categories = ["Vision / Storage Load", "MLP + LLM"]
    e2e_vals    = [e2e_vision,   e2e_mlp_llm]
    cached_vals = [cached_load, cached_mlp_llm]

    x = np.arange(len(categories)); w = 0.35
    fig, ax = plt.subplots(figsize=(9, 6))
    b1 = ax.bar(x - w/2, e2e_vals,    w, label="E2E",           color=C_E2E)
    b2 = ax.bar(x + w/2, cached_vals, w, label="Cached Stage B", color=C_CACHED)
    for bars in [b1, b2]:
        bar_val(ax, bars, fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylabel("ms (avg per query)")
    ax.set_title("[3] Functional Group Avg — E2E vs Cached Stage B", fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    p = os.path.join(out_dir, f"{prefix}_3_functional_groups.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {p}")


def chart4_stage_a_amortized(df_a, df_b, out_dir, prefix):
    """Stage A amortized cost: avg encode time per video, and cost-per-query for N queries."""
    if df_a.empty:
        print("  [Skip] Chart 4: No Stage A data")
        return

    avg_a = df_a["t_total_a_ms"].mean()
    n_queries = len(df_b)

    # Amortized cost per query for different number of total queries served
    query_counts = [1, 5, 10, 20, 50, 100, 200, 500]
    amortized = [avg_a / n for n in query_counts]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Stage A breakdown avg
    ax = axes[0]
    a_cols   = ["t_decode_ms", "t_preprocess_ms", "t_vit_ms", "t_pixel_shuffle_ms", "t_save_ms"]
    a_labels = ["Decode", "Preprocess", "ViT\n(×100f)", "Pixel\nShuffle", "Save .pt"]
    a_avgs   = [df_a[c].mean() for c in a_cols if c in df_a.columns]
    a_lbls   = [l for c, l in zip(a_cols, a_labels) if c in df_a.columns]
    bars = ax.bar(a_lbls, a_avgs, color=C_STAGE, edgecolor="white")
    bar_val(ax, bars)
    ax.set_ylabel("ms (avg)")
    ax.set_title(f"Stage A: Encode Cost per Video (avg={avg_a:.0f}ms)", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Right: amortized cost vs N queries served
    ax = axes[1]
    ax.plot(query_counts, amortized, "o-", color=C_E2E, linewidth=2, markersize=8,
            label=f"Amortized Stage A (avg={avg_a:.0f}ms/video)")
    if "t_total_b_ms" in df_b.columns:
        avg_b = df_b["t_total_b_ms"].mean()
        ax.axhline(avg_b, color=C_CACHED, linestyle="--", linewidth=2,
                   label=f"Cached Stage B latency ({avg_b:.0f}ms/query)")
    ax.set_xscale("log")
    ax.set_xlabel("Number of queries served per video")
    ax.set_ylabel("Amortized Stage A cost (ms/query)")
    ax.set_title("Stage A Amortization Curve", fontweight="bold")
    ax.legend()
    ax.grid(linestyle="--", alpha=0.5)

    fig.suptitle("[4] Stage A (Encode) Cost and Amortization", fontsize=13, fontweight="bold")
    plt.tight_layout()
    p = os.path.join(out_dir, f"{prefix}_4_stage_a_amortized.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {p}")


def chart5_vram(df_e2e, df_b, out_dir, prefix):
    """Peak VRAM comparison."""
    vrams, labels, colors = [], [], []
    if "vram_gb" in df_e2e.columns:
        vrams.append(df_e2e["vram_gb"].max());  labels.append("E2E");           colors.append(C_E2E)
    if "vram_gb" in df_b.columns:
        vrams.append(df_b["vram_gb"].max());    labels.append("Cached Stage B"); colors.append(C_CACHED)

    if not vrams:
        print("  [Skip] Chart 5: No VRAM data")
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, vrams, color=colors, edgecolor="white")
    for bar, v in zip(bars, vrams):
        ax.text(bar.get_x() + bar.get_width()/2, v + max(vrams)*0.02,
                f"{v:.2f} GB", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("GB")
    ax.set_title("[5] Peak VRAM — E2E vs Cached Stage B", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    p = os.path.join(out_dir, f"{prefix}_5_vram.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {p}")


def chart6_accuracy(acc_cached: float | None, out_dir, prefix):
    """MVBench accuracy bar (Cached only; E2E accuracy not measured by default)."""
    if acc_cached is None:
        print("  [Skip] Chart 6: No accuracy data")
        return

    fig, ax = plt.subplots(figsize=(5, 5))
    bars = ax.bar(["Cached\n(Stage B)"], [acc_cached * 100], color=C_CACHED, edgecolor="white")
    ax.text(bars[0].get_x() + bars[0].get_width()/2,
            acc_cached * 100 + 1.5,
            f"{acc_cached*100:.1f}%", ha="center", va="bottom",
            fontsize=12, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("[6] MVBench MCQA Accuracy", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    p = os.path.join(out_dir, f"{prefix}_6_accuracy.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {p}")


def chart7_throughput(df_e2e, df_b, out_dir, prefix):
    """Throughput (queries/s) — E2E vs Cached Stage B."""
    vals, labels, colors = [], [], []
    if "t_total_ms" in df_e2e.columns:
        vals.append(1000 / df_e2e["t_total_ms"].mean())
        labels.append("E2E"); colors.append(C_E2E)
    if "t_total_b_ms" in df_b.columns:
        vals.append(1000 / df_b["t_total_b_ms"].mean())
        labels.append("Cached Stage B"); colors.append(C_CACHED)

    if not vals:
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, vals, color=colors, edgecolor="white")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + max(vals)*0.02,
                f"{v:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("queries / second")
    ax.set_title("[7] Throughput (TTFT-based) — E2E vs Cached Stage B", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    p = os.path.join(out_dir, f"{prefix}_7_throughput.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {p}")


# ── Summary print ─────────────────────────────────────────────────────────────

def print_summary(df_e2e, df_b, df_a, acc_cached):
    print("\n" + "=" * 60)
    print("  E2E vs Cached Comparison Summary")
    print("=" * 60)

    if not df_e2e.empty:
        print(f"\n  E2E (n={len(df_e2e)}):")
        print(f"    Avg total latency : {df_e2e['t_total_ms'].mean():8.1f} ms")
        print(f"    Median latency    : {df_e2e['t_total_ms'].median():8.1f} ms")
        print(f"    Throughput        : {1000/df_e2e['t_total_ms'].mean():8.3f} queries/s")
        for col, lbl in zip(E2E_STAGE_COLS, E2E_STAGE_LABELS):
            lbl_c = lbl.replace("\n", " ")
            print(f"    {lbl_c:<20}: {df_e2e[col].mean():8.1f} ms (avg)")

    if not df_a.empty:
        avg_a = df_a["t_total_a_ms"].mean()
        print(f"\n  Cached Stage A (n={len(df_a)} unique videos):")
        print(f"    Avg encode time   : {avg_a:8.1f} ms/video")

    if not df_b.empty:
        print(f"\n  Cached Stage B (n={len(df_b)}):")
        print(f"    Avg total latency : {df_b['t_total_b_ms'].mean():8.1f} ms")
        print(f"    Median latency    : {df_b['t_total_b_ms'].median():8.1f} ms")
        print(f"    Throughput        : {1000/df_b['t_total_b_ms'].mean():8.3f} queries/s")
        for col, lbl in zip(CACHED_B_STAGE_COLS, CACHED_B_STAGE_LABELS):
            if col in df_b.columns:
                print(f"    {lbl:<20}: {df_b[col].mean():8.1f} ms (avg)")

        if not df_e2e.empty and "t_total_ms" in df_e2e.columns:
            speedup = df_e2e["t_total_ms"].mean() / df_b["t_total_b_ms"].mean()
            print(f"\n  Stage B speedup vs E2E: {speedup:.2f}x faster per query")

    if acc_cached is not None:
        print(f"\n  MVBench Accuracy (Cached): {acc_cached*100:.1f}%")

    print("=" * 60 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="E2E vs Cached comparison plots")
    parser.add_argument("--e2e_dir",    default="outputs/e2e",           help="E2E output dir")
    parser.add_argument("--cached_dir", default="outputs/cached_video",  help="Cached output dir")
    parser.add_argument("--out_dir",    default="outputs/comparison_plots")
    parser.add_argument("--prefix",     default="mvbench_comparison",    help="Output filename prefix")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"\nLoading data from:\n  E2E:    {args.e2e_dir}\n  Cached: {args.cached_dir}\n")

    df_e2e = load_latest_e2e_csv(args.e2e_dir)
    df_b   = load_latest_cached_b_csv(args.cached_dir)
    df_a   = load_latest_cached_a_csv(args.cached_dir)
    acc    = load_accuracy(args.cached_dir)

    if df_e2e.empty and df_b.empty:
        print("[Error] No data found. Run 07_run_e2e.sh and 09_run_cached_video.sh first.")
        return

    prefix = args.prefix

    print("\nGenerating charts...")
    if not df_e2e.empty and not df_b.empty:
        chart1_stage_avg(df_e2e, df_b, args.out_dir, prefix)
        chart2_latency_hist(df_e2e, df_b, args.out_dir, prefix)
        chart3_functional_groups(df_e2e, df_b, args.out_dir, prefix)
    chart4_stage_a_amortized(df_a, df_b, args.out_dir, prefix)
    if not df_e2e.empty and not df_b.empty:
        chart5_vram(df_e2e, df_b, args.out_dir, prefix)
    chart6_accuracy(acc, args.out_dir, prefix)
    if not df_e2e.empty and not df_b.empty:
        chart7_throughput(df_e2e, df_b, args.out_dir, prefix)

    print_summary(df_e2e, df_b, df_a, acc)
    print(f"All charts saved to: {args.out_dir}/")


if __name__ == "__main__":
    main()
