import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 막대 최상단에 총합 수치 또는 CRASH를 적어주는 헬퍼 함수
def add_total_labels(ax, x_positions, totals, crashes):
    for x, total, is_crash in zip(x_positions, totals, crashes):
        if is_crash or total == 0:
            ax.text(x, 50, "CRASH", ha='center', va='bottom', color='red', fontweight='bold', fontsize=10, rotation=90)
        else:
            # 총합(Total TTFT)을 소수점 1자리까지 표시
            ax.text(x, total + total*0.02, f"{total:.1f}", ha='center', va='bottom', fontsize=10, fontweight='bold')

def plot_stacked_comparison(model_name):
    # CSV 파일명 (대소문자 맞춰서 로드, 사용하신 파일명 그대로 유지)
    e2e_file = f"{model_name.lower()}_e2e_breakdown.csv"
    cached_file = f"{model_name.lower()}_cached_breakdown.csv"
    
    if not os.path.exists(e2e_file) or not os.path.exists(cached_file):
        print(f"Warning: {e2e_file} or {cached_file} not found. Skipping {model_name}.")
        return

    e2e = pd.read_csv(e2e_file)
    cached = pd.read_csv(cached_file)
    
    # X축 토큰 수 매핑 (기존 448x448, 896x896... 대신 표시)
    token_labels = ["256\n(Tokens)", "1K\n(Tokens)", "2K\n(Tokens)", "4K\n(Tokens)", "8K\n(Tokens)"]
    
    # 실제 데이터 길이에 맞춰 X축 생성 (안전장치)
    data_len = len(e2e)
    x = np.arange(data_len)
    x_labels = token_labels[:data_len]
    width = 0.35 # 막대 두께
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 색상 팔레트 (E2E는 진한 색, Cached는 파스텔톤 연한 색상으로 대비를 줌)
    colors_e2e = ['#1f77b4', '#ff7f0e', '#2ca02c']     # 파랑, 주황, 초록
    colors_cached = ['#aec7e8', '#ffbb78', '#98df8a']  # 연파랑, 연주황, 연초록
    
    # ------------------ E2E (왼쪽 막대) ------------------
    e2e_prep = e2e['Image Preprocessing'].fillna(0)
    e2e_enc = e2e['Image Encoding'].fillna(0)
    e2e_pref = e2e['LLM Prefill'].fillna(0)
    
    # 누적 막대 그리기
    ax.bar(x - width/2, e2e_prep, width, label='Image Preproc (E2E)', color=colors_e2e[0], edgecolor='white')
    ax.bar(x - width/2, e2e_enc, width, bottom=e2e_prep, label='Vision Encoding (E2E)', color=colors_e2e[1], edgecolor='white')
    ax.bar(x - width/2, e2e_pref, width, bottom=e2e_prep+e2e_enc, label='LLM Prefill (E2E)', color=colors_e2e[2], edgecolor='white')
    
    # ------------------ Cached (오른쪽 막대) ------------------
    cached_prep = cached['Image Preprocessing'].fillna(0)
    # 캐시 데이터의 인코딩 컬럼명 확인
    col_enc_cached = 'Image Encoding (DB Load)' if 'Image Encoding (DB Load)' in cached.columns else 'Image Encoding'
    cached_enc = cached[col_enc_cached].fillna(0)
    cached_pref = cached['LLM Prefill'].fillna(0)
    
    # 누적 막대 그리기
    ax.bar(x + width/2, cached_prep, width, label='Image Preproc (Cached)', color=colors_cached[0], edgecolor='white')
    ax.bar(x + width/2, cached_enc, width, bottom=cached_prep, label='DB Load (Cached)', color=colors_cached[1], edgecolor='white')
    ax.bar(x + width/2, cached_pref, width, bottom=cached_prep+cached_enc, label='LLM Prefill (Cached)', color=colors_cached[2], edgecolor='white')

    # ------------------ 수치 및 CRASH 표기 ------------------
    # E2E 총합 및 에러 판별 (LLM Prefill이 0이면 CRASH로 간주)
    e2e_totals = e2e_prep + e2e_enc + e2e_pref
    e2e_crashes = (e2e_pref == 0)
    add_total_labels(ax, x - width/2, e2e_totals, e2e_crashes)
    
    # Cached 총합 및 에러 판별
    cached_totals = cached_prep + cached_enc + cached_pref
    cached_crashes = (cached_pref == 0)
    add_total_labels(ax, x + width/2, cached_totals, cached_crashes)

    # ------------------ 차트 디자인 설정 ------------------
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=11, fontweight='bold')
    ax.set_xlabel("Image Resolution", fontsize=12, fontweight='bold')
    ax.set_ylabel("Latency (ms)", fontsize=12, fontweight='bold')
    ax.set_title(f"{model_name.upper()} TTFT Breakdown (E2E vs Cached)", fontsize=16, fontweight='bold')
    
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 범례 설정 (위쪽에 가로로 길게 배치)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper center', ncol=3, fontsize=10, bbox_to_anchor=(0.5, 1.15))
    
    plt.margins(y=0.15) # 텍스트 안 잘리게 여백 추가
    plt.tight_layout()
    
    # 저장
    os.makedirs('plots', exist_ok=True)
    save_path = f"plots/{model_name.lower()}_breakdown_stacked.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

if __name__ == "__main__":
    # Qwen과 LLaVA에 대해 각각 하나의 통합 그래프를 생성
    plot_stacked_comparison("Qwen")
    plot_stacked_comparison("LLaVA")
    print("✅ Breakdown E2E vs Cached 그래프 렌더링 완료!")