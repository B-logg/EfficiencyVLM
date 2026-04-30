import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 막대 위에 값을 텍스트로 달아주는 헬퍼 함수
def add_value_labels(ax, spacing=3):
    for rect in ax.patches:
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2
        
        # 값이 너무 작거나 0인 경우 렌더링 최적화를 위해 패스하거나 0으로 표시
        if pd.isna(y_value):
            continue
            
        label = f"{y_value:.3f}"
        
        ax.annotate(
            label,
            (x_value, y_value),
            xytext=(0, spacing),
            textcoords="offset points",
            ha='center',
            va='bottom',
            fontsize=9,
            fontweight='bold'
        )

def create_charts(model_name):
    # 파일이 존재하는지 확인 (예외 처리)
    if not os.path.exists(f"{model_name}_e2e.csv") or not os.path.exists(f"{model_name}_cached.csv"):
        print(f"Warning: Data files for {model_name} not found. Skipping.")
        return

    e2e = pd.read_csv(f"{model_name}_e2e.csv")
    cached = pd.read_csv(f"{model_name}_cached.csv")
    
    stages = ['1a_text', '1b_img', '2a_vit', '2b_mlp', '3_fusion', '4_gen']
    inf_metrics = ['true_ttft', 'decode_time', 'total_latency']
    os.makedirs('plots', exist_ok=True)

    # x축 위치 및 막대 너비 설정
    x_stages = np.arange(len(stages))
    x_inf = np.arange(len(inf_metrics))
    width = 0.35

    # 1. Pipeline Average (E2E vs Cached)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x_stages - width/2, e2e[stages].mean(), width, label='E2E', color='skyblue')
    ax.bar(x_stages + width/2, cached[stages].mean(), width, label='Cached', color='salmon')
    ax.set_xticks(x_stages)
    ax.set_xticklabels(stages)
    ax.set_title(f"[1] {model_name} Pipeline Avg Time (E2E vs Cached)")
    ax.set_ylabel('Seconds')
    ax.legend()
    add_value_labels(ax)
    plt.margins(y=0.15) # 텍스트가 잘리지 않도록 y축 여백 추가
    plt.tight_layout()
    plt.savefig(f"plots/{model_name}_1_Pipeline_Avg.png")
    plt.close()

    # 2. LLM Inference Metrics Average (E2E vs Cached)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x_inf - width/2, e2e[inf_metrics].mean(), width, label='E2E', color='skyblue')
    ax.bar(x_inf + width/2, cached[inf_metrics].mean(), width, label='Cached', color='salmon')
    ax.set_xticks(x_inf)
    ax.set_xticklabels(inf_metrics)
    ax.set_title(f"[2] {model_name} Inference Avg Time (E2E vs Cached)")
    ax.set_ylabel('Seconds')
    ax.legend()
    add_value_labels(ax)
    plt.margins(y=0.15)
    plt.tight_layout()
    plt.savefig(f"plots/{model_name}_2_Inference_Avg.png")
    plt.close()

    # 3. Pipeline Total Time (E2E vs Cached)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x_stages - width/2, e2e[stages].sum(), width, label='E2E', color='skyblue')
    ax.bar(x_stages + width/2, cached[stages].sum(), width, label='Cached', color='salmon')
    ax.set_xticks(x_stages)
    ax.set_xticklabels(stages)
    ax.set_title(f"[3] {model_name} Pipeline Total Time (E2E vs Cached)")
    ax.set_ylabel('Seconds')
    ax.legend()
    add_value_labels(ax)
    plt.margins(y=0.15)
    plt.tight_layout()
    plt.savefig(f"plots/{model_name}_3_Pipeline_Total.png")
    plt.close()

    # 4. LLM Total Processing Time (E2E vs Cached)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x_inf - width/2, e2e[inf_metrics].sum(), width, label='E2E', color='skyblue')
    ax.bar(x_inf + width/2, cached[inf_metrics].sum(), width, label='Cached', color='salmon')
    ax.set_xticks(x_inf)
    ax.set_xticklabels(inf_metrics)
    ax.set_title(f"[4] {model_name} Inference Total Time (E2E vs Cached)")
    ax.set_ylabel('Seconds')
    ax.legend()
    add_value_labels(ax)
    plt.margins(y=0.15)
    plt.tight_layout()
    plt.savefig(f"plots/{model_name}_4_Inference_Total.png")
    plt.close()

    # 5. Throughput (기존 유지 + 라벨 추가)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.bar(['E2E', 'Cached'], [1/e2e['true_ttft'].mean(), 1/cached['true_ttft'].mean()], color=['skyblue', 'salmon'])
    ax.set_title(f"[5] {model_name} Throughput (Images/s)")
    ax.set_ylabel('Img/s')
    add_value_labels(ax)
    plt.margins(y=0.15)
    plt.tight_layout()
    plt.savefig(f"plots/{model_name}_5_Throughput.png")
    plt.close()

    # 6. TPOT (기존 유지 + 라벨 추가)
    fig, ax = plt.subplots(figsize=(6, 6))
    tpot_e = e2e['decode_time'].mean() / max((e2e['tokens'].mean() - 1), 1)
    tpot_c = cached['decode_time'].mean() / max((cached['tokens'].mean() - 1), 1)
    ax.bar(['E2E', 'Cached'], [tpot_e, tpot_c], color=['skyblue', 'salmon'])
    ax.set_title(f"[6] {model_name} TPOT (s/token)")
    ax.set_ylabel('s/token')
    add_value_labels(ax)
    plt.margins(y=0.15)
    plt.tight_layout()
    plt.savefig(f"plots/{model_name}_6_TPOT.png")
    plt.close()

    # 7. VRAM (기존 유지 + 라벨 추가)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.bar(['E2E', 'Cached'], [e2e['vram'].max(), cached['vram'].max()], color=['skyblue', 'salmon'])
    ax.set_title(f"[7] {model_name} Peak VRAM (GB)")
    ax.set_ylabel('GB')
    add_value_labels(ax)
    plt.margins(y=0.15)
    plt.tight_layout()
    plt.savefig(f"plots/{model_name}_7_VRAM.png")
    plt.close()

if __name__ == "__main__":
    create_charts("qwen")
    create_charts("llava")
    print("그래프 생성이 완료되었습니다.")