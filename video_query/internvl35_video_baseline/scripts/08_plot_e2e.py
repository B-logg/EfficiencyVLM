"""
E2E Video Query 결과 시각화

Chart 1: 파이프라인 단계별 평균 시간 (avg per stage)
Chart 2: 파이프라인 단계별 총 시간 (sum, 100개 처리 총량)
Chart 3: 총 latency 분포 (histogram)
Chart 4: E2E vs Cached(Sweep2) 비교 - 단계별 avg
Chart 5: Peak VRAM

E2E 단계:
  decode | preprocess | ViT(×100frames) | pixel_shuffle | MLP(×N_llm) | LLM
Cached 단계 (Sweep2 nf=16):
  db_load | MLP(×16) | LLM
"""
import argparse
import os
import re
import json
from pathlib import Path
from glob import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 파이프라인 단계 컬럼
E2E_STAGE_COLS   = ["t_decode_ms", "t_preprocess_ms", "t_vit_ms",
                     "t_pixel_shuffle_ms", "t_mlp_ms", "t_llm_ms"]
E2E_STAGE_LABELS = ["Decode", "Preprocess", "ViT\n(×100f)", "Pixel\nShuffle", "MLP\n(×N_llm)", "LLM"]
E2E_COLORS       = ["#4878d0", "#6acc65", "#ff9f4a", "#f6cf71", "#69b3a2", "#d65f5f"]

CACHED_STAGE_COLS   = ["t_load_pt_ms", "t_query_total_ms"]
CACHED_STAGE_LABELS = ["DB Load\n(pt file)", "MLP+LLM"]
CACHED_COLORS       = ["#ff9f4a", "#d65f5f"]


def load_e2e_csv(e2e_dir: str, dataset: str) -> pd.DataFrame:
    """최신 E2E timing CSV 로드."""
    pattern = os.path.join(e2e_dir, f"e2e_{dataset}_*_timing.csv")
    files = sorted(glob(pattern))
    if not files:
        return pd.DataFrame()
    return pd.read_csv(files[-1])


def load_cached_jsonl(timing_dir: str, dataset: str, nf: int = 16) -> pd.DataFrame:
    """Sweep2 cached timing JSONL 로드 (nf 기준 최신 파일)."""
    pattern = os.path.join(timing_dir, f"sweep2_{dataset}_fps5_nf{nf}_*.jsonl")
    files = sorted(glob(pattern))
    if not files:
        return pd.DataFrame()
    rows = []
    with open(files[-1]) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                if r.get("stage") == "B":
                    rows.append(r)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def bar_val(ax, bars, fmt="{:.1f}"):
    for bar in bars:
        h = bar.get_height()
        if h < 1e-3:
            continue
        ax.text(bar.get_x() + bar.get_width()/2, h + max(b.get_height() for b in bars)*0.01,
                fmt.format(h), ha="center", va="bottom", fontsize=8, fontweight="bold")


def plot_e2e(dataset: str, e2e_dir: str, timing_dir: str, out_dir: str):
    df = load_e2e_csv(e2e_dir, dataset)
    if df.empty:
        print(f"[Skip] {dataset}: E2E CSV 없음 ({e2e_dir})")
        return

    os.makedirs(out_dir, exist_ok=True)
    n = len(df)
    prefix = f"{out_dir}/{dataset}_e2e"

    # ── Chart 1: Stage avg (ms) ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    avgs = [df[c].mean() for c in E2E_STAGE_COLS]
    bars = ax.bar(E2E_STAGE_LABELS, avgs, color=E2E_COLORS, edgecolor="white")
    bar_val(ax, bars)
    ax.set_ylabel("ms (avg)")
    ax.set_title(f"[1] {dataset.upper()} E2E Pipeline — Avg per Stage (n={n})", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{prefix}_1_stage_avg.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Chart 2: Stage total (sum, ms) ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    totals = [df[c].sum() / 1000 for c in E2E_STAGE_COLS]   # → seconds
    bars = ax.bar(E2E_STAGE_LABELS, totals, color=E2E_COLORS, edgecolor="white")
    bar_val(ax, bars, fmt="{:.2f}")
    ax.set_ylabel("seconds (total)")
    ax.set_title(f"[2] {dataset.upper()} E2E Pipeline — Total Time for {n} queries (s)", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{prefix}_2_stage_total.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Chart 3: Total latency distribution (histogram, ms) ──────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(df["t_total_ms"], bins=30, color="#4878d0", edgecolor="white", alpha=0.85)
    ax.axvline(df["t_total_ms"].mean(), color="red",    linestyle="--", label=f"Mean {df['t_total_ms'].mean():.1f}ms")
    ax.axvline(df["t_total_ms"].median(), color="orange", linestyle="--", label=f"Median {df['t_total_ms'].median():.1f}ms")
    ax.set_xlabel("Total E2E Latency (ms)")
    ax.set_ylabel("Count")
    ax.set_title(f"[3] {dataset.upper()} E2E Total Latency Distribution (n={n})", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{prefix}_3_latency_dist.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Chart 4: E2E vs Cached comparison (avg ms per functional group) ──
    nf_llm = int(df["n_frames_llm"].mode()[0]) if "n_frames_llm" in df.columns else 16
    df_cached = load_cached_jsonl(timing_dir, dataset, nf=nf_llm)

    if not df_cached.empty:
        # E2E 그룹: vision 처리 / MLP+LLM
        e2e_vision = (df["t_decode_ms"] + df["t_preprocess_ms"] +
                      df["t_vit_ms"] + df["t_pixel_shuffle_ms"]).mean()
        e2e_mlp_llm = (df["t_mlp_ms"] + df["t_llm_ms"]).mean()

        # Cached 그룹: db_load / mlp+llm (t_query_total_ms = mlp+llm)
        c_load = df_cached["t_load_pt_ms"].mean() if "t_load_pt_ms" in df_cached.columns else 0
        c_llm  = df_cached["t_query_total_ms"].mean() if "t_query_total_ms" in df_cached.columns else 0

        categories = ["Vision\n(decode+prep+ViT+shuffle)", "MLP + LLM"]
        e2e_vals    = [e2e_vision, e2e_mlp_llm]
        cached_vals = [c_load,    c_llm]

        x = np.arange(len(categories)); w = 0.35
        fig, ax = plt.subplots(figsize=(9, 6))
        b1 = ax.bar(x - w/2, e2e_vals,    w, label="E2E",    color="#56b4e9")
        b2 = ax.bar(x + w/2, cached_vals, w, label=f"Cached (nf={nf_llm})", color="#e85d4a")
        for bars in [b1, b2]:
            bar_val(ax, bars, fmt="{:.1f}")
        ax.set_xticks(x); ax.set_xticklabels(categories, fontsize=11)
        ax.set_ylabel("ms (avg per query)")
        ax.set_title(f"[4] {dataset.upper()} E2E vs Cached — Functional Group Avg", fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"{prefix}_4_e2e_vs_cached.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Chart 4 saved (Cached n={len(df_cached)})")
    else:
        print(f"  [Skip] Chart 4: Cached timing 데이터 없음 ({timing_dir})")

    # ── Chart 5: Peak VRAM ───────────────────────────────────────────────
    if "vram_gb" in df.columns:
        fig, ax = plt.subplots(figsize=(5, 5))
        vram_max = df["vram_gb"].max()
        ax.bar(["E2E"], [vram_max], color="#56b4e9")
        ax.text(0, vram_max + 0.1, f"{vram_max:.2f} GB", ha="center", va="bottom",
                fontsize=12, fontweight="bold")
        ax.set_ylabel("GB")
        ax.set_title(f"[5] {dataset.upper()} E2E Peak VRAM", fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"{prefix}_5_vram.png", dpi=150, bbox_inches="tight")
        plt.close()

    # ── Summary print ────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  {dataset.upper()} E2E Summary (n={n})")
    print(f"{'='*55}")
    print(f"  Frames encoded (ViT):  {df['n_frames_encoded'].mean():.0f} frames/query")
    print(f"  Frames to LLM:         {df['n_frames_llm'].mean():.0f} frames/query")
    for col, label in zip(E2E_STAGE_COLS, E2E_STAGE_LABELS):
        label_clean = label.replace("\n", " ")
        print(f"  {label_clean:<22}: {df[col].mean():7.1f} ms (avg)")
    print(f"  {'Total E2E':<22}: {df['t_total_ms'].mean():7.1f} ms (avg)")
    print(f"  {'Total E2E':<22}: {df['t_total_ms'].median():7.1f} ms (median)")
    print(f"  Throughput (TTFT):     {1000/df['t_total_ms'].mean():.2f} queries/s")
    print(f"{'='*55}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--e2e_dir",    default="outputs/e2e",    help="E2E 결과 디렉토리")
    parser.add_argument("--timing_dir", default="outputs/timings", help="Sweep2 timing JSONL 디렉토리")
    parser.add_argument("--out_dir",    default="outputs/e2e/plots")
    parser.add_argument("--datasets",   nargs="+", default=["msrvtt", "mvbench"])
    args = parser.parse_args()

    for ds in args.datasets:
        print(f"\n{'='*55}\n  Plotting: {ds}\n{'='*55}")
        plot_e2e(ds, args.e2e_dir, args.timing_dir, args.out_dir)

    print(f"\n모든 그래프 저장 완료: {args.out_dir}/")


if __name__ == "__main__":
    main()
