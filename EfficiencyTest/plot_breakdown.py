import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# CSV 파일 로드
files = {
    "Qwen (E2E)": "qwen_e2e.csv",
    "Qwen (Cached)": "qwen_cached.csv",
    "LLaVA (E2E)": "llava_e2e.csv",
    "LLaVA (Cached)": "llava_cached.csv"
}

dataframes = {}
for name, file in files.items():
    try:
        dataframes[name] = pd.read_csv(file)
        print(f"Loaded {file}")
    except FileNotFoundError:
        print(f"Warning: {file} not found. Skipping.")

if not dataframes:
    print("No CSV files found to plot.")
    exit()

# X축 레이블을 CSV에서 바로 가져옴
resolutions = list(dataframes.values())[0]['Resolution'].tolist()

fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharey=True)
axes = axes.flatten()
colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] 

for idx, (model_name, df) in enumerate(dataframes.items()):
    ax = axes[idx]
    
    # 에러 방지 (길이 맞추기)
    if len(df) < len(resolutions):
        missing = len(resolutions) - len(df)
        df_dummy = pd.DataFrame([{"Resolution": resolutions[len(df)+i], "Image Preprocessing": 0, "Image Encoding (DB Load)" if "Cached" in model_name else "Image Encoding": 0, "LLM Prefill": 0} for i in range(missing)])
        df = pd.concat([df, df_dummy], ignore_index=True)

    if "Cached" in model_name:
        bottom = np.zeros(len(df))
        ax.bar(df['Resolution'], df['Image Preprocessing'], label='Image Preprocessing', color=colors[0])
        bottom += df['Image Preprocessing'].fillna(0)
        
        # 컬럼 이름이 혹시 다를까봐 안전하게 가져오기
        col_encode = 'Image Encoding (DB Load)' if 'Image Encoding (DB Load)' in df.columns else 'Image Encoding'
        ax.bar(df['Resolution'], df[col_encode], bottom=bottom, label='DB Load + Fusion', color=colors[1])
        bottom += df[col_encode].fillna(0)
        
        ax.bar(df['Resolution'], df['LLM Prefill'], bottom=bottom, label='LLM Prefill', color=colors[2])
    else:
        bottom = np.zeros(len(df))
        ax.bar(df['Resolution'], df['Image Preprocessing'], label='Image Preprocessing', color=colors[0])
        bottom += df['Image Preprocessing'].fillna(0)
        ax.bar(df['Resolution'], df['Image Encoding'], bottom=bottom, label='Vision Encoding + Fusion', color=colors[1])
        bottom += df['Image Encoding'].fillna(0)
        ax.bar(df['Resolution'], df['LLM Prefill'], bottom=bottom, label='LLM Prefill', color=colors[2])

    ax.set_title(f"{model_name} TTFT Breakdown by Resolution", fontsize=14, fontweight='bold')
    ax.set_xlabel("Image Resolution (Native Input)", fontsize=12)
    ax.set_ylabel("Latency (ms)", fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 0으로 기록된 Crash 구간 표시
    for i, val in enumerate(bottom + df['LLM Prefill'].fillna(0)):
        if val == 0:
            ax.text(i, 50, "CRASH\n(Context Limit)", ha='center', va='bottom', color='red', fontweight='bold')
        else:
            ax.text(i, val + max(bottom.max(), 1)*0.02, f"{val:.1f}ms", ha='center', va='bottom', fontsize=10)

# 범례 추가
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=12, bbox_to_anchor=(0.5, 1.05))

plt.tight_layout()
plt.savefig("resolution_breakdown_plot.png", dpi=300, bbox_inches='tight')
print("Plot saved as resolution_breakdown_plot.png")