import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def generate_graphs(model_name):
    # CSV 로드
    e2e = pd.read_csv(f"{model_name.lower()}_e2e_results.csv")
    cached = pd.read_csv(f"{model_name.lower()}_cached_results.csv")
    
    stages = ['1a_text', '1b_img', '2a_vit', '2b_mlp', '3_fusion', '4_gen']
    inf_metrics = ['true_ttft', 'decode_time', 'total_latency']

    # ==========================================
    # 1. Pipeline Average Time (평균 막대그래프)
    # ==========================================
    plt.figure(figsize=(12, 6))
    x = np.arange(len(stages))
    plt.bar(x - 0.2, e2e[stages].mean(), 0.4, label='E2E', color='skyblue')
    plt.bar(x + 0.2, cached[stages].mean(), 0.4, label='Cached', color='salmon')
    plt.xticks(x, stages); plt.title(f"[1] {model_name} Pipeline Average Time"); plt.ylabel('Seconds'); plt.legend()
    plt.savefig(f"{model_name}_1_Pipeline_Average.png")

    # ==========================================
    # 2. LLM Inference Metrics (평균 막대그래프)
    # ==========================================
    plt.figure(figsize=(10, 6))
    x_inf = np.arange(len(inf_metrics))
    plt.bar(x_inf - 0.2, e2e[inf_metrics].mean(), 0.4, label='E2E', color='cornflowerblue')
    plt.bar(x_inf + 0.2, cached[inf_metrics].mean(), 0.4, label='Cached', color='orange')
    plt.xticks(x_inf, ['True TTFT', 'Decode Time', 'Total Latency']); plt.title(f"[2] {model_name} LLM Inference Metrics (Avg)"); plt.ylabel('Seconds'); plt.legend()
    plt.savefig(f"{model_name}_2_Inference_Average.png")

    # ==========================================
    # 3. Pipeline Total Time (1000장 처리 총 시간)
    # ==========================================
    plt.figure(figsize=(12, 6))
    plt.bar(x - 0.2, e2e[stages].sum(), 0.4, label='E2E', color='skyblue')
    plt.bar(x + 0.2, cached[stages].sum(), 0.4, label='Cached', color='salmon')
    plt.xticks(x, stages); plt.title(f"[3] {model_name} Pipeline Total Time (1000 Images)"); plt.ylabel('Seconds'); plt.legend()
    plt.savefig(f"{model_name}_3_Pipeline_Total.png")

    # ==========================================
    # 4. LLM Total Processing Time (1000장 처리 총 시간)
    # ==========================================
    plt.figure(figsize=(10, 6))
    plt.bar(x_inf - 0.2, e2e[inf_metrics].sum(), 0.4, label='E2E', color='cornflowerblue')
    plt.bar(x_inf + 0.2, cached[inf_metrics].sum(), 0.4, label='Cached', color='orange')
    plt.xticks(x_inf, ['True TTFT', 'Decode Time', 'Total Latency']); plt.title(f"[4] {model_name} LLM Total Time (1000 Images)"); plt.ylabel('Seconds'); plt.legend()
    plt.savefig(f"{model_name}_4_Inference_Total.png")

    # ==========================================
    # 5, 6, 7. Throughput / TPOT / VRAM (한 PNG에 통합)
    # ==========================================
    # 5. Throughput: 입력부터 첫 토큰까지 기준 (1 / True TTFT)
    thr_e = 1 / e2e['true_ttft'].mean()
    thr_c = 1 / cached['true_ttft'].mean()
    
    # 6. TPOT: Decode Time / (Tokens - 1)
    tpot_e = e2e['decode_time'].mean() / max((e2e['tokens'].mean() - 1), 1)
    tpot_c = cached['decode_time'].mean() / max((cached['tokens'].mean() - 1), 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].bar(['E2E', 'Cached'], [thr_e, thr_c], color=['blue', 'orange'])
    axes[0].set_title("[5] Throughput (1/True TTFT)"); axes[0].set_ylabel("Img/s")
    
    axes[1].bar(['E2E', 'Cached'], [tpot_e, tpot_c], color=['blue', 'orange'])
    axes[1].set_title("[6] TPOT (Decode Time / Tokens)"); axes[1].set_ylabel("s/token")
    
    axes[2].bar(['E2E', 'Cached'], [e2e['vram'].max(), cached['vram'].max()], color=['blue', 'orange'])
    axes[2].set_title("[7] Peak VRAM Usage"); axes[2].set_ylabel("GB")
    
    plt.tight_layout(); plt.savefig(f"{model_name}_5_6_7_Metrics.png")

    # ==========================================
    # 8. TTFT Breakdown (누적 스택 막대그래프)
    # ==========================================
    plt.figure(figsize=(8, 6))
    bottom_e, bottom_c = 0, 0
    colors = ['#4f8bc6', '#ff9999', '#6ebd6e', '#ffcc99', '#99ccff', '#ffff99']
    
    for i, s in enumerate(stages):
        plt.bar('E2E', e2e[s].mean(), bottom=bottom_e, color=colors[i], label=s)
        plt.bar('Cached', cached[s].mean(), bottom=bottom_c, color=colors[i])
        bottom_e += e2e[s].mean()
        bottom_c += cached[s].mean()
        
    plt.title(f"[8] {model_name} TTFT Breakdown (Stacked)"); plt.ylabel('Seconds'); plt.legend(loc='upper right')
    plt.savefig(f"{model_name}_8_TTFT_Breakdown.png")

# 실행
generate_graphs("Qwen")
generate_graphs("Llava")