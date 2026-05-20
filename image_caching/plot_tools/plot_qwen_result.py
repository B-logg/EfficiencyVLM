"""
Qwen2-VL Result Charts (from qwen_e2e.csv / qwen_cached.csv)

CSV units: all timing columns are in SECONDS.

E2E columns   : 1a_text, 1b_img, 2a_vit, 2b_mlp(Merger), 3_fusion, 4_gen
Cached columns: 1a_text, 1b_img(0), 2a_vit(0), 2b_mlp(DB Load), 3_fusion, 4_gen
Shared        : true_ttft, decode_time, total_latency, vram, tokens

Note: In E2E, 2a_vit includes ViT+Merger forward (separated via hook),
      2b_mlp holds the Merger-only time.
      In Cached, 2b_mlp holds the DB Load time.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# E2E: text, img_preproc, ViT, Merger, Fusion, Generate
E2E_STAGES = ['1a_text', '1b_img', '2a_vit', '2b_mlp', '3_fusion', '4_gen']
E2E_LABELS = ['Text', 'Img Preproc', 'ViT', 'Merger', 'Fusion', 'Generate']
E2E_COLORS = ['#8172b2', '#4878d0', '#ff9f4a', '#f6cf71', '#d65f5f', '#6acc65']

# Cached: text, DB Load, Fusion, Generate  (img/vit=0 이므로 제외)
CACHED_STAGES = ['1a_text', '2b_mlp', '3_fusion', '4_gen']
CACHED_LABELS = ['Text',    'DB Load', 'Fusion',  'Generate']
CACHED_COLORS = ['#8172b2', '#ff9f4a', '#d65f5f', '#6acc65']

INF_METRICS = ['true_ttft', 'decode_time', 'total_latency']


def add_bar_labels(ax, fmt="{:.4f}"):
    for rect in ax.patches:
        h = rect.get_height()
        if pd.isna(h) or h < 1e-6:
            continue
        ax.annotate(fmt.format(h),
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)


def plot_qwen_result():
    e2e_path    = "qwen_e2e.csv"
    cached_path = "qwen_cached.csv"
    if not os.path.exists(e2e_path) or not os.path.exists(cached_path):
        print(f"[Skip] CSV not found: {e2e_path} or {cached_path}")
        return

    e2e    = pd.read_csv(e2e_path)
    cached = pd.read_csv(cached_path)
    os.makedirs('plots', exist_ok=True)

    # ── Chart 1: Stacked TTFT (E2E vs Cached) ──────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    e2e_means    = [e2e[c].mean()    for c in E2E_STAGES]
    cached_means = [cached[c].mean() for c in CACHED_STAGES]

    bottom_e, bottom_c = 0.0, 0.0
    for val, color, label in zip(e2e_means, E2E_COLORS, E2E_LABELS):
        ax.bar(0, val, bottom=bottom_e, color=color, label=label, width=0.4)
        bottom_e += val
    for val, color, label in zip(cached_means, CACHED_COLORS, CACHED_LABELS):
        ax.bar(1, val, bottom=bottom_c, color=color, width=0.4)
        bottom_c += val

    ax.set_xticks([0, 1]); ax.set_xticklabels(['E2E', 'Cached'], fontsize=12)
    ax.set_ylabel("Time (s)"); ax.set_title("Qwen2-VL [1] Avg TTFT Breakdown (Stacked)")
    ax.legend(loc='upper right', fontsize=9, bbox_to_anchor=(1.4, 1))
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("plots/qwen_1_ttft_stacked.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ── Chart 2: Shared stage comparison (Text, Fusion, Generate) ──────────
    common = {
        'Text':     ('1a_text',  '1a_text'),
        'Fusion':   ('3_fusion', '3_fusion'),
        'Generate': ('4_gen',    '4_gen'),
    }
    labels_c = list(common.keys())
    e2e_c    = [e2e[v[0]].mean()    for v in common.values()]
    cached_c = [cached[v[1]].mean() for v in common.values()]
    x = np.arange(len(labels_c)); w = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w/2, e2e_c,    w, label='E2E',    color='#4878d0')
    ax.bar(x + w/2, cached_c, w, label='Cached', color='#ee854a')
    ax.set_xticks(x); ax.set_xticklabels(labels_c)
    ax.set_ylabel("Time (s)")
    ax.set_title("Qwen2-VL [2] Shared Stage Comparison (E2E vs Cached)")
    ax.legend(); ax.grid(axis='y', linestyle='--', alpha=0.5)
    add_bar_labels(ax)
    plt.tight_layout()
    plt.savefig("plots/qwen_2_stage_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ── Chart 3: Inference metrics ─────────────────────────────────────────
    x_inf = np.arange(len(INF_METRICS)); w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x_inf - w/2, e2e[INF_METRICS].mean(),    w, label='E2E',    color='#4878d0')
    ax.bar(x_inf + w/2, cached[INF_METRICS].mean(), w, label='Cached', color='#ee854a')
    ax.set_xticks(x_inf); ax.set_xticklabels(['TTFT', 'Decode Time', 'Total Latency'])
    ax.set_ylabel("Time (s)")
    ax.set_title("Qwen2-VL [3] Inference Metrics (E2E vs Cached)")
    ax.legend(); ax.grid(axis='y', linestyle='--', alpha=0.5)
    add_bar_labels(ax)
    plt.tight_layout()
    plt.savefig("plots/qwen_3_inference_metrics.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ── Chart 4: Throughput ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 5))
    thrp = [1 / e2e['true_ttft'].mean(), 1 / cached['true_ttft'].mean()]
    bars = ax.bar(['E2E', 'Cached'], thrp, color=['#4878d0', '#ee854a'])
    ax.set_ylabel("Images/s"); ax.set_title("Qwen2-VL [4] Throughput")
    for bar, v in zip(bars, thrp):
        ax.text(bar.get_x() + bar.get_width()/2, v + max(thrp)*0.01,
                f"{v:.3f}", ha='center', va='bottom', fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("plots/qwen_4_throughput.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ── Chart 5: TPOT ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 5))
    tpot_e = e2e['decode_time'].mean()    / max(e2e['tokens'].mean()    - 1, 1)
    tpot_c = cached['decode_time'].mean() / max(cached['tokens'].mean() - 1, 1)
    bars = ax.bar(['E2E', 'Cached'], [tpot_e, tpot_c], color=['#4878d0', '#ee854a'])
    ax.set_ylabel("s/token"); ax.set_title("Qwen2-VL [5] TPOT")
    for bar, v in zip(bars, [tpot_e, tpot_c]):
        ax.text(bar.get_x() + bar.get_width()/2, v + max(tpot_e, tpot_c)*0.01,
                f"{v:.4f}", ha='center', va='bottom', fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("plots/qwen_5_tpot.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ── Chart 6: Peak VRAM ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 5))
    vram = [e2e['vram'].max(), cached['vram'].max()]
    bars = ax.bar(['E2E', 'Cached'], vram, color=['#4878d0', '#ee854a'])
    ax.set_ylabel("GB"); ax.set_title("Qwen2-VL [6] Peak VRAM")
    for bar, v in zip(bars, vram):
        ax.text(bar.get_x() + bar.get_width()/2, v + max(vram)*0.01,
                f"{v:.2f}", ha='center', va='bottom', fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("plots/qwen_6_vram.png", dpi=150, bbox_inches='tight')
    plt.close()

    print("Saved: plots/qwen_1~6_*.png")


if __name__ == "__main__":
    plot_qwen_result()
