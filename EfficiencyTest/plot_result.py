import pandas as pd
import matplotlib.pyplot as plt
import os

def create_charts(model_name):
    e2e = pd.read_csv(f"{model_name}_e2e.csv")
    cached = pd.read_csv(f"{model_name}_cached.csv")
    
    stages = ['1a_text', '1b_img', '2a_vit', '2b_mlp', '3_fusion', '4_gen']
    inf_metrics = ['true_ttft', 'decode_time', 'total_latency']
    os.makedirs('plots', exist_ok=True)

    # 1. Pipeline Average (1~4번 중 4개)
    for data, mode in zip([e2e, cached], ['E2E', 'Cached']):
        plt.figure(figsize=(10,6)); plt.bar(stages, data[stages].mean(), color='skyblue' if mode=='E2E' else 'salmon')
        plt.title(f"[1] {model_name} {mode} - Pipeline Avg Time"); plt.ylabel('Seconds')
        plt.savefig(f"plots/{model_name}_{mode}_1_Pipeline_Avg.png")

    # 2. LLM Inference Metrics Average (1~4번 중 4개)
    for data, mode in zip([e2e, cached], ['E2E', 'Cached']):
        plt.figure(figsize=(8,6)); plt.bar(inf_metrics, data[inf_metrics].mean(), color='cornflowerblue' if mode=='E2E' else 'orange')
        plt.title(f"[2] {model_name} {mode} - Inference Avg Time"); plt.ylabel('Seconds')
        plt.savefig(f"plots/{model_name}_{mode}_2_Inference_Avg.png")

    # 3. Pipeline Total Time (1~4번 중 4개)
    for data, mode in zip([e2e, cached], ['E2E', 'Cached']):
        plt.figure(figsize=(10,6)); plt.bar(stages, data[stages].sum(), color='skyblue' if mode=='E2E' else 'salmon')
        plt.title(f"[3] {model_name} {mode} - Pipeline Total Time"); plt.ylabel('Seconds')
        plt.savefig(f"plots/{model_name}_{mode}_3_Pipeline_Total.png")

    # 4. LLM Total Processing Time (1~4번 중 4개)
    for data, mode in zip([e2e, cached], ['E2E', 'Cached']):
        plt.figure(figsize=(8,6)); plt.bar(inf_metrics, data[inf_metrics].sum(), color='cornflowerblue' if mode=='E2E' else 'orange')
        plt.title(f"[4] {model_name} {mode} - Inference Total Time"); plt.ylabel('Seconds')
        plt.savefig(f"plots/{model_name}_{mode}_4_Inference_Total.png")

    # 5. Throughput (5, 6, 7번 중 2개)
    plt.figure(figsize=(6,6))
    plt.bar(['E2E', 'Cached'], [1/e2e['true_ttft'].mean(), 1/cached['true_ttft'].mean()], color=['blue', 'orange'])
    plt.title(f"[5] {model_name} Throughput (Images/s)"); plt.ylabel('Img/s')
    plt.savefig(f"plots/{model_name}_5_Throughput.png")

    # 6. TPOT (5, 6, 7번 중 2개)
    plt.figure(figsize=(6,6))
    tpot_e = e2e['decode_time'].mean() / max((e2e['tokens'].mean() - 1), 1)
    tpot_c = cached['decode_time'].mean() / max((cached['tokens'].mean() - 1), 1)
    plt.bar(['E2E', 'Cached'], [tpot_e, tpot_c], color=['blue', 'orange'])
    plt.title(f"[6] {model_name} TPOT (s/token)"); plt.ylabel('s/token')
    plt.savefig(f"plots/{model_name}_6_TPOT.png")

    # 7. VRAM (5, 6, 7번 중 2개)
    plt.figure(figsize=(6,6))
    plt.bar(['E2E', 'Cached'], [e2e['vram'].max(), cached['vram'].max()], color=['blue', 'orange'])
    plt.title(f"[7] {model_name} Peak VRAM (GB)"); plt.ylabel('GB')
    plt.savefig(f"plots/{model_name}_7_VRAM.png")

    # 8. TTFT Breakdown (8번 중 4개)
    colors = ['#4f8bc6', '#ff9999', '#6ebd6e', '#ffcc99', '#99ccff', '#ffff99']
    for data, mode in zip([e2e, cached], ['E2E', 'Cached']):
        plt.figure(figsize=(6, 8)); bottom = 0
        for i, s in enumerate(stages):
            plt.bar(f"{mode} TTFT", data[s].mean(), bottom=bottom, color=colors[i], label=s)
            bottom += data[s].mean()
        plt.legend(); plt.title(f"[8] {model_name} {mode} TTFT Breakdown"); plt.ylabel('Seconds')
        plt.savefig(f"plots/{model_name}_{mode}_8_TTFT_Breakdown.png")

if __name__ == "__main__":
    create_charts("qwen")
    create_charts("llava")