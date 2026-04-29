import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def make_plots(name):
    e2e = pd.read_csv(f"{name.lower()}_e2e_results.csv")
    cached = pd.read_csv(f"{name.lower()}_cached_results.csv")
    stages = ['1a_text', '1b_img', '2a_vit', '2b_mlp', '3_fusion', '4_gen']
    inf_metrics = ['true_ttft', 'decode_time', 'total_latency']
    
    e2e['decode_time'] = e2e['total_latency'] - e2e['true_ttft']
    cached['decode_time'] = cached['total_latency'] - cached['true_ttft']

    # 1. Pipeline Average Time (막대그래프)
    plt.figure(figsize=(10,6))
    plt.bar(np.arange(6)-0.2, e2e[stages].mean(), 0.4, label='E2E')
    plt.bar(np.arange(6)+0.2, cached[stages].mean(), 0.4, label='Cached')
    plt.xticks(np.arange(6), stages); plt.title(f"{name} Pipeline Average"); plt.legend(); plt.savefig(f"{name}_1_avg.png")

    # 2. LLM inference Metrics Average (막대그래프)
    plt.figure(figsize=(10,6))
    plt.bar(np.arange(3)-0.2, e2e[inf_metrics].mean(), 0.4, label='E2E')
    plt.bar(np.arange(3)+0.2, cached[inf_metrics].mean(), 0.4, label='Cached')
    plt.xticks(np.arange(3), inf_labels); plt.title(f"{name} Inf Metrics Average"); plt.legend(); plt.savefig(f"{name}_2_inf_avg.png")

    # 3. Pipeline Total Time (막대그래프)
    plt.figure(figsize=(10,6))
    plt.bar(np.arange(6)-0.2, e2e[stages].sum(), 0.4, label='E2E')
    plt.bar(np.arange(6)+0.2, cached[stages].sum(), 0.4, label='Cached')
    plt.xticks(np.arange(6), stages); plt.title(f"{name} Pipeline Total"); plt.legend(); plt.savefig(f"{name}_3_total.png")

    # 4. LLM Total Processing Time (막대그래프)
    plt.figure(figsize=(10,6))
    plt.bar(np.arange(3)-0.2, e2e[inf_metrics].sum(), 0.4, label='E2E')
    plt.bar(np.arange(3)+0.2, cached[inf_metrics].sum(), 0.4, label='Cached')
    plt.xticks(np.arange(3), inf_labels); plt.title(f"{name} Inf Metrics Total"); plt.legend(); plt.savefig(f"{name}_4_inf_total.png")

    # 5, 6, 7. Throughput, TPOT, VRAM (통합 막대그래프)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].bar(['E2E', 'Cached'], [1/e2e['true_ttft'].mean(), 1/cached['true_ttft'].mean()])
    axes[0].set_title("Throughput (img/s)")
    axes[1].bar(['E2E', 'Cached'], [e2e['decode_time'].mean()/19, cached['decode_time'].mean()/19])
    axes[1].set_title("TPOT (s/t)")
    axes[2].bar(['E2E', 'Cached'], [e2e['vram'].max(), cached['vram'].max()])
    axes[2].set_title("VRAM (GB)")
    plt.savefig(f"{name}_567_metrics.png")

    # 8. TTFT breakdown (Stacked Bar)
    plt.figure(figsize=(8,6))
    bottom_e = 0; bottom_c = 0
    for s in stages:
        plt.bar('E2E', e2e[s].mean(), bottom=bottom_e, label=s)
        plt.bar('Cached', cached[s].mean(), bottom=bottom_c)
        bottom_e += e2e[s].mean(); bottom_c += cached[s].mean()
    plt.legend(); plt.title(f"{name} TTFT Breakdown"); plt.savefig(f"{name}_8_breakdown.png")

make_plots("Qwen")
make_plots("Llava")