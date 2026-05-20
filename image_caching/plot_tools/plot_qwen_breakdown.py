"""
Qwen2-VL Breakdown Chart
- E2E:    Image Preprocessing | Image Encoding (ViT + Merger 합산) | LLM Prefill
- Cached: DB Load             |                                     | LLM Prefill

Fusion (Text+Visual) 컬럼은 차트에 포함하지 않습니다.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.patches import Patch

COLORS = ['#4878d0', '#ff9f4a', '#6acc65']
LABELS = ['Image Preprocessing', 'Image Encoding / DB Load', 'LLM Prefill']


def add_total_labels(ax, x_pos, totals):
    ymax = max(totals) if max(totals) > 0 else 1
    for x, total in zip(x_pos, totals):
        if total == 0:
            ax.text(x, ymax * 0.05, "N/A", ha='center', va='bottom',
                    color='gray', fontsize=9, rotation=90)
        else:
            ax.text(x, total + ymax * 0.01 + 5, f"{total:.1f}",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')


def plot_qwen_breakdown():
    e2e_path    = "qwen_e2e_breakdown.csv"
    cached_path = "qwen_cached_breakdown.csv"
    if not os.path.exists(e2e_path) or not os.path.exists(cached_path):
        print(f"[Skip] CSV not found: {e2e_path} or {cached_path}")
        return

    e2e    = pd.read_csv(e2e_path)
    cached = pd.read_csv(cached_path)

    x_labels = e2e['Resolution'].tolist()
    x = np.arange(len(x_labels))
    width = 0.35

    # E2E: Preprocessing | ViT + Merger (합산) | Prefill  (Fusion 제외)
    e_prep = e2e['Image Preprocessing'].fillna(0).values
    e_enc  = (e2e['ViT'].fillna(0) + e2e['Merger'].fillna(0)).values  # ViT + Merger 합산
    e_pref = e2e['LLM Prefill (TTFT)'].fillna(0).values

    # Cached: 0 | DB Load | Prefill  (Fusion 제외)
    c_prep = np.zeros(len(cached))
    c_enc  = cached['DB Load'].fillna(0).values
    c_pref = cached['LLM Prefill (TTFT)'].fillna(0).values

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.bar(x - width/2, e_prep, width, color=COLORS[0], edgecolor='white')
    ax.bar(x - width/2, e_enc,  width, bottom=e_prep,         color=COLORS[1], edgecolor='white')
    ax.bar(x - width/2, e_pref, width, bottom=e_prep + e_enc, color=COLORS[2], edgecolor='white')

    ax.bar(x + width/2, c_prep, width, color=COLORS[0], edgecolor='white')
    ax.bar(x + width/2, c_enc,  width, bottom=c_prep,         color=COLORS[1], edgecolor='white')
    ax.bar(x + width/2, c_pref, width, bottom=c_prep + c_enc, color=COLORS[2], edgecolor='white')

    add_total_labels(ax, x - width/2, e_prep + e_enc + e_pref)
    add_total_labels(ax, x + width/2, c_prep + c_enc + c_pref)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=11)
    ax.set_xlabel("Image Resolution\n(Left: E2E  |  Right: Cached)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Latency (ms)", fontsize=12, fontweight='bold')
    ax.set_title("Qwen2-VL TTFT Breakdown\n(Image Encoding = ViT + Merger)", fontsize=15, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    legend_elements = [Patch(facecolor=c, edgecolor='white', label=l)
                       for c, l in zip(COLORS, LABELS)]
    ax.legend(handles=legend_elements, loc='upper left', ncol=1, fontsize=11)

    plt.margins(y=0.15)
    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    path = "plots/qwen_breakdown.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


if __name__ == "__main__":
    plot_qwen_breakdown()
