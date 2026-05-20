"""
LLaVA-1.6 Result Charts (7 charts)

Chart 1: Pipeline stage avg    (E2E vs Cached per stage, mean)
Chart 2: Inference metrics avg (mean)
Chart 3: Throughput
Chart 4: TPOT
Chart 5: Peak VRAM
Chart 6: Pipeline stage total  (sum over all samples)
Chart 7: Inference metrics total (sum)

CSV units: SECONDS
E2E    cols: 1a_text, 1b_img, 2a_vit, 2b_mlp, 3_fusion, 4_gen
Cached cols: 1a_text, 1b_img(0), 2a_vit(0), 2b_db_load, 3_fusion, 4_gen
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# (x-label, e2e_col, cached_col or None)
STAGES = [
    ("Text",            "1a_text",  "1a_text"),
    ("Img Preproc",     "1b_img",   "1b_img"),
    ("ViT / DB Load",   "2a_vit",   "2b_db_load"),  # E2E: ViT, Cached: DB Load
    ("Projector",       "2b_mlp",   "2b_mlp"),       # both: projector runs at query time
    ("Fusion",          "3_fusion", "3_fusion"),
    ("Gen (TTFT)",      "4_gen",    "4_gen"),
]
INF_METRICS = ["true_ttft", "decode_time", "total_latency"]
INF_LABELS  = ["True TTFT", "Decode Time", "Total Latency"]
COLORS = ("#56b4e9", "#e85d4a")


def bar_labels(ax, fmt="{:.4f}"):
    for rect in ax.patches:
        h = rect.get_height()
        if pd.isna(h) or h < 1e-9:
            continue
        ax.annotate(fmt.format(h),
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")


def stage_chart(ax, e2e, cached, use_sum, title):
    agg      = "sum" if use_sum else "mean"
    labels   = [s[0] for s in STAGES]
    e2e_vals = [getattr(e2e[s[1]], agg)()                    for s in STAGES]
    cac_vals = [getattr(cached[s[2]], agg)() if s[2] else 0.0 for s in STAGES]
    x = np.arange(len(labels)); w = 0.35
    ax.bar(x - w/2, e2e_vals, w, label="E2E",    color=COLORS[0])
    ax.bar(x + w/2, cac_vals, w, label="Cached", color=COLORS[1])
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Seconds"); ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(axis="y", linestyle="--", alpha=0.5)
    bar_labels(ax)


def inf_chart(ax, e2e, cached, use_sum, title):
    agg = "sum" if use_sum else "mean"
    e2e_vals = [getattr(e2e[m], agg)()    for m in INF_METRICS]
    cac_vals = [getattr(cached[m], agg)() for m in INF_METRICS]
    x = np.arange(len(INF_METRICS)); w = 0.35
    ax.bar(x - w/2, e2e_vals, w, label="E2E",    color=COLORS[0])
    ax.bar(x + w/2, cac_vals, w, label="Cached", color=COLORS[1])
    ax.set_xticks(x); ax.set_xticklabels(INF_LABELS, fontsize=10)
    ax.set_ylabel("Seconds"); ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(axis="y", linestyle="--", alpha=0.5)
    bar_labels(ax)


def plot_llava_result():
    e2e_path    = "llava_e2e.csv"
    cached_path = "llava_cached.csv"
    if not os.path.exists(e2e_path) or not os.path.exists(cached_path):
        print(f"[Skip] CSV not found: {e2e_path} or {cached_path}")
        return

    e2e    = pd.read_csv(e2e_path)
    cached = pd.read_csv(cached_path)
    n_e, n_c = len(e2e), len(cached)
    os.makedirs("plots", exist_ok=True)

    # Chart 1 ── Pipeline avg ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    stage_chart(ax, e2e, cached, use_sum=False,
                title=f"[1] LLaVA-1.6 Pipeline Avg Time (E2E vs Cached, n={n_e}/{n_c})")
    plt.tight_layout()
    plt.savefig("plots/llava_1_pipeline_avg.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Chart 2 ── Inference metrics avg ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    inf_chart(ax, e2e, cached, use_sum=False,
              title=f"[2] LLaVA-1.6 Inference Avg Time (n={n_e}/{n_c})")
    plt.tight_layout()
    plt.savefig("plots/llava_2_inference_avg.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Chart 3 ── Throughput ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 5))
    thrp = [1 / e2e["true_ttft"].mean(), 1 / cached["true_ttft"].mean()]
    bars = ax.bar(["E2E", "Cached"], thrp, color=list(COLORS))
    ax.set_ylabel("Images/s")
    ax.set_title("[3] LLaVA-1.6 Throughput (Images/s)", fontweight="bold")
    for bar, v in zip(bars, thrp):
        ax.text(bar.get_x() + bar.get_width()/2, v + max(thrp)*0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("plots/llava_3_throughput.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Chart 4 ── TPOT ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 5))
    tpot_e = e2e["decode_time"].mean()    / max(e2e["tokens"].mean()    - 1, 1)
    tpot_c = cached["decode_time"].mean() / max(cached["tokens"].mean() - 1, 1)
    bars = ax.bar(["E2E", "Cached"], [tpot_e, tpot_c], color=list(COLORS))
    ax.set_ylabel("s/token")
    ax.set_title("[4] LLaVA-1.6 TPOT (s/token)", fontweight="bold")
    for bar, v in zip(bars, [tpot_e, tpot_c]):
        ax.text(bar.get_x() + bar.get_width()/2, v + max(tpot_e, tpot_c)*0.01,
                f"{v:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("plots/llava_4_tpot.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Chart 5 ── Peak VRAM ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 5))
    vram = [e2e["vram"].max(), cached["vram"].max()]
    bars = ax.bar(["E2E", "Cached"], vram, color=list(COLORS))
    ax.set_ylabel("GB")
    ax.set_title("[5] LLaVA-1.6 Peak VRAM (GB)", fontweight="bold")
    for bar, v in zip(bars, vram):
        ax.text(bar.get_x() + bar.get_width()/2, v + max(vram)*0.01,
                f"{v:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("plots/llava_5_vram.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Chart 6 ── Pipeline total (sum) ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    stage_chart(ax, e2e, cached, use_sum=True,
                title=f"[6] LLaVA-1.6 Pipeline Total Time (E2E vs Cached, n={n_e}/{n_c})")
    plt.tight_layout()
    plt.savefig("plots/llava_6_pipeline_total.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Chart 7 ── Inference metrics total (sum) ────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    inf_chart(ax, e2e, cached, use_sum=True,
              title=f"[7] LLaVA-1.6 Inference Total Time (n={n_e}/{n_c})")
    plt.tight_layout()
    plt.savefig("plots/llava_7_inference_total.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved: plots/llava_1~7_*.png  (E2E n={n_e}, Cached n={n_c})")


if __name__ == "__main__":
    plot_llava_result()
