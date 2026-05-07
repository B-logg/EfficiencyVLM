import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.patches import Patch

def add_total_labels(ax, x_positions, totals, crashes):
    for x, total, is_crash in zip(x_positions, totals, crashes):
        if is_crash or total == 0:
            ax.text(x, 50, "CRASH", ha='center', va='bottom', color='red', fontweight='bold', fontsize=10, rotation=90)
        else:
            ax.text(x, total + total*0.02 + 10, f"{total:.1f}", ha='center', va='bottom', fontsize=10, fontweight='bold')

def plot_stacked_comparison(model_name):
    e2e_file = f"{model_name.lower()}_e2e_breakdown.csv"
    cached_file = f"{model_name.lower()}_cached_breakdown.csv"
    
    if not os.path.exists(e2e_file) or not os.path.exists(cached_file):
        print(f"Warning: {e2e_file} or {cached_file} not found. Skipping.")
        return

    e2e = pd.read_csv(e2e_file)
    cached = pd.read_csv(cached_file)
    
    token_labels = ["256\n(Tokens)", "1K\n(Tokens)", "2K\n(Tokens)", "4K\n(Tokens)", "8K\n(Tokens)"]
    data_len = len(e2e)
    x = np.arange(data_len)
    x_labels = token_labels[:data_len]
    width = 0.35 
    
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] 
    
    # ------------------ E2E (왼쪽 막대) ------------------
    e2e_prep = e2e['Image Preprocessing'].fillna(0)
    e2e_enc = e2e['Image Encoding'].fillna(0)
    e2e_pref = e2e['LLM Prefill'].fillna(0)
    
    ax.bar(x - width/2, e2e_prep, width, color=colors[0], edgecolor='white')
    ax.bar(x - width/2, e2e_enc, width, bottom=e2e_prep, color=colors[1], edgecolor='white')
    ax.bar(x - width/2, e2e_pref, width, bottom=e2e_prep+e2e_enc, color=colors[2], edgecolor='white')
    
    # ------------------ Cached (오른쪽 막대) ------------------
    cached_prep = cached['Image Preprocessing'].fillna(0)
    # 캐시 컬럼명 안전장치
    col_enc_cached = 'Image Encoding (DB Load)' if 'Image Encoding (DB Load)' in cached.columns else 'Image Encoding'
    cached_enc = cached[col_enc_cached].fillna(0)
    cached_pref = cached['LLM Prefill'].fillna(0)
    
    ax.bar(x + width/2, cached_prep, width, color=colors[0], edgecolor='white')
    ax.bar(x + width/2, cached_enc, width, bottom=cached_prep, color=colors[1], edgecolor='white')
    ax.bar(x + width/2, cached_pref, width, bottom=cached_prep+cached_enc, color=colors[2], edgecolor='white')

    # ------------------ 수치 및 CRASH 표기 ------------------
    e2e_totals = e2e_prep + e2e_enc + e2e_pref
    e2e_crashes = (e2e_pref == 0)
    add_total_labels(ax, x - width/2, e2e_totals, e2e_crashes)
    
    cached_totals = cached_prep + cached_enc + cached_pref
    cached_crashes = (cached_pref == 0)
    add_total_labels(ax, x + width/2, cached_totals, cached_crashes)

    # ------------------ 차트 디자인 ------------------
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=11, fontweight='bold')
    ax.set_xlabel("Image Resolution\n(Left Bar: E2E  |  Right Bar: Cached)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Latency (ms)", fontsize=12, fontweight='bold')
    ax.set_title(f"{model_name.upper()} TTFT Breakdown", fontsize=16, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    legend_elements = [
        Patch(facecolor=colors[0], edgecolor='white', label='Image Preprocessing'),
        Patch(facecolor=colors[1], edgecolor='white', label='Image Encoding / DB Load'),
        Patch(facecolor=colors[2], edgecolor='white', label='LLM Prefill')
    ]
    ax.legend(handles=legend_elements, loc='upper center', ncol=3, fontsize=11, bbox_to_anchor=(0.5, 1.12))
    
    plt.margins(y=0.15)
    plt.tight_layout()
    
    os.makedirs('plots', exist_ok=True)
    save_path = f"plots/{model_name.lower()}_breakdown_stacked.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

if __name__ == "__main__":
    plot_stacked_comparison("InternVL")
    print("✅ InternVL 맞춤형 Breakdown 그래프 생성 완료!")