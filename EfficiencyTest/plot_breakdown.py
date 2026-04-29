import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib

matplotlib.use('Agg') # 서버 환경 오류 방지

def plot_breakdown(csv_file, title):
    if not os.path.exists(csv_file):
        print(f"File {csv_file} not found. Skipping plot.")
        return
        
    df = pd.read_csv(csv_file)
    labels = df['Seq_Length']
    
    preproc = df['Image Preprocessing']
    encode_col = 'Image Encoding (DB Load)' if 'Cached' in title else 'Image Encoding'
    encode = df[encode_col]
    prefill = df['LLM Prefill']

    # 사진과 완벽하게 동일한 색상 및 빗금 설정
    colors = ['#6ebd6e', '#e07a7a', '#4f8bc6'] 
    edgecolors = ['#4ca04c', '#555555', '#555555']
    hatches = ['', '///', '\\\\\\']
    
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    width = 0.55

    # 스택 바 차트 그리기
    p1 = ax.bar(labels, preproc, width, label='Image Preprocessing', color=colors[0], edgecolor=edgecolors[0], linewidth=1.2)
    p2 = ax.bar(labels, encode, width, bottom=preproc, label=encode_col, color=colors[1], edgecolor=edgecolors[1], hatch=hatches[1], linewidth=1.2)
    p3 = ax.bar(labels, prefill, width, bottom=preproc + encode, label='LLM Prefill', color=colors[2], edgecolor=edgecolors[2], hatch=hatches[2], linewidth=1.2)

    # 그래프 텍스트 및 레이아웃 (선생님 사진 스타일 완벽 복사)
    ax.set_ylabel('Time (ms)', fontsize=20, fontweight='bold')
    ax.set_xlabel('Sequence Length', fontsize=20, fontweight='bold')
    ax.set_title(title, fontsize=22, fontweight='bold', pad=15)
    
    ax.tick_params(axis='both', which='major', labelsize=16)
    
    # y축 가로 점선 그리드 (연한 빨간색 점선)
    ax.yaxis.grid(True, linestyle='--', color='lightcoral', alpha=0.6, linewidth=1.2)
    ax.set_axisbelow(True) # 그리드를 바 뒤로 보내기

    # 범례 설정
    ax.legend(loc='upper left', fontsize=16, framealpha=1.0)
    
    plt.tight_layout()
    
    save_name = title.replace(" ", "_").replace("(", "").replace(")", "") + ".png"
    plt.savefig(save_name, bbox_inches='tight')
    print(f"Saved plot: {save_name}")

if __name__ == "__main__":
    plot_breakdown("qwen_seq_e2e.csv", "TTFT Breakdown (Qwen E2E)")
    plot_breakdown("qwen_seq_cached.csv", "TTFT Breakdown (Qwen Cached)")
    plot_breakdown("llava_seq_e2e.csv", "TTFT Breakdown (LLaVA E2E)")
    plot_breakdown("llava_seq_cached.csv", "TTFT Breakdown (LLaVA Cached)")