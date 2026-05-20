import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def add_value_labels(ax, spacing=3):
    for rect in ax.patches:
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2
        if pd.isna(y_value) or y_value == 0:
            continue
        label = f"{y_value:.3f}"
        ax.annotate(label, (x_value, y_value), xytext=(0, spacing), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

def create_internvl_charts():
    model_name = "internvl"
    if not os.path.exists(f"{model_name}_e2e.csv") or not os.path.exists(f"{model_name}_cached.csv"):
        print(f"Warning: {model_name} CSV files not found. Skipping.")
        return

    e2e = pd.read_csv(f"{model_name}_e2e.csv")
    cached = pd.read_csv(f"{model_name}_cached.csv")
    
    # [핵심] InternVL 전용 7단계 컬럼 매핑
    e2e_stages = ['1_img_preproc', '2_vit', '3_unshuffle', '4_mlp', '5_text', '6_fusion', '7_gen']
    cached_stages = ['1_img_preproc', '2_vit_or_db', '3_unshuffle', '4_mlp', '5_text', '6_fusion', '7_gen']
    display_labels = ['Preproc', 'ViT / DB Load', 'Unshuffle', 'MLP', 'Text', 'Fusion', 'Gen']
    
    inf_metrics = ['true_ttft', 'decode_time', 'total_latency']
    os.makedirs('plots', exist_ok=True)

    x_stages = np.arange(len(display_labels))
    x_inf = np.arange(len(inf_metrics))
    width = 0.35

    # 1. Pipeline Average (E2E vs Cached)
    fig, ax = plt.subplots(figsize=(12, 6)) # 항목이 7개로 늘어나서 가로 길이를 살짝 늘림
    ax.bar(x_stages - width/2, e2e[e2e_stages].mean(), width, label='E2E', color='skyblue')
    ax.bar(x_stages + width/2, cached[cached_stages].mean(), width, label='Cached', color='salmon')
    ax.set_xticks(x_stages); ax.set_xticklabels(display_labels)
    ax.set_title(f"[1] INTERNVL Pipeline Avg Time (E2E vs Cached)")
    ax.set_ylabel('Seconds'); ax.legend()
    add_value_labels(ax)
    plt.margins(y=0.15); plt.tight_layout()
    plt.savefig(f"plots/{model_name}_1_Pipeline_Avg.png"); plt.close()

    # 2. LLM Inference Metrics Average (E2E vs Cached)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x_inf - width/2, e2e[inf_metrics].mean(), width, label='E2E', color='skyblue')
    ax.bar(x_inf + width/2, cached[inf_metrics].mean(), width, label='Cached', color='salmon')
    ax.set_xticks(x_inf); ax.set_xticklabels(inf_metrics)
    ax.set_title(f"[2] INTERNVL Inference Avg Time")
    ax.set_ylabel('Seconds'); ax.legend()
    add_value_labels(ax)
    plt.margins(y=0.15); plt.tight_layout()
    plt.savefig(f"plots/{model_name}_2_Inference_Avg.png"); plt.close()

    # 3. Pipeline Total Time (E2E vs Cached)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x_stages - width/2, e2e[e2e_stages].sum(), width, label='E2E', color='skyblue')
    ax.bar(x_stages + width/2, cached[cached_stages].sum(), width, label='Cached', color='salmon')
    ax.set_xticks(x_stages); ax.set_xticklabels(display_labels)
    ax.set_title(f"[3] INTERNVL Pipeline Total Time (E2E vs Cached)")
    ax.set_ylabel('Seconds'); ax.legend()
    add_value_labels(ax)
    plt.margins(y=0.15); plt.tight_layout()
    plt.savefig(f"plots/{model_name}_3_Pipeline_Total.png"); plt.close()

    # 4. LLM Inference Total Time (E2E vs Cached)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x_inf - width/2, e2e[inf_metrics].sum(), width, label='E2E', color='skyblue')
    ax.bar(x_inf + width/2, cached[inf_metrics].sum(), width, label='Cached', color='salmon')
    ax.set_xticks(x_inf); ax.set_xticklabels(inf_metrics)
    ax.set_title(f"[4] INTERNVL Inference Total Time")
    ax.set_ylabel('Seconds'); ax.legend()
    add_value_labels(ax)
    plt.margins(y=0.15); plt.tight_layout()
    plt.savefig(f"plots/{model_name}_4_Inference_Total.png"); plt.close()

    # 5. Throughput
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.bar(['E2E', 'Cached'], [1/e2e['true_ttft'].mean(), 1/cached['true_ttft'].mean()], color=['skyblue', 'salmon'])
    ax.set_title(f"[5] INTERNVL Throughput (Images/s)")
    ax.set_ylabel('Img/s')
    add_value_labels(ax)
    plt.margins(y=0.15); plt.tight_layout()
    plt.savefig(f"plots/{model_name}_5_Throughput.png"); plt.close()

    # 6. TPOT
    fig, ax = plt.subplots(figsize=(6, 6))
    tpot_e = e2e['decode_time'].mean() / max((e2e['tokens'].mean() - 1), 1)
    tpot_c = cached['decode_time'].mean() / max((cached['tokens'].mean() - 1), 1)
    ax.bar(['E2E', 'Cached'], [tpot_e, tpot_c], color=['skyblue', 'salmon'])
    ax.set_title(f"[6] INTERNVL TPOT (s/token)")
    ax.set_ylabel('s/token')
    add_value_labels(ax)
    plt.margins(y=0.15); plt.tight_layout()
    plt.savefig(f"plots/{model_name}_6_TPOT.png"); plt.close()

    # 7. VRAM
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.bar(['E2E', 'Cached'], [e2e['vram'].max(), cached['vram'].max()], color=['skyblue', 'salmon'])
    ax.set_title(f"[7] INTERNVL Peak VRAM (GB)")
    ax.set_ylabel('GB')
    add_value_labels(ax)
    plt.margins(y=0.15); plt.tight_layout()
    plt.savefig(f"plots/{model_name}_7_VRAM.png"); plt.close()

if __name__ == "__main__":
    create_internvl_charts()
    print("✅ InternVL 맞춤형 파이프라인 그래프(1~7번) 생성 완료!")
