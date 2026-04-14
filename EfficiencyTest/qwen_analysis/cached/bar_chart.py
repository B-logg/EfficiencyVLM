import pandas as pd
import matplotlib.pyplot as plt
import os

def load_csv_data(filename, method='sum'):
    """CSV 파일을 읽어서 합계(sum), 평균(mean), 또는 최대(max)를 반환"""
    if not os.path.exists(filename):
        print(f" ⚠️ '{filename}' 파일이 없습니다. 0.0으로 처리합니다.")
        return 0.0
    
    try:
        df = pd.read_csv(filename)
        if 'Value' not in df.columns:
            print(f" ⚠️ '{filename}'에 'Value' 컬럼이 없습니다.")
            return 0.0
            
        if method == 'sum': return df['Value'].sum()
        elif method == 'mean': return df['Value'].mean()
        elif method == 'max': return df['Value'].max()
    except Exception as e:
        print(f" ⚠️ '{filename}' 읽기 오류: {e}")
        return 0.0

def main():
    print("📊 [Qwen CSV 데이터 분석 및 4종 차트 생성을 시작합니다...]\n")

    # =====================================================================
    # 🔍 모드 자동 감지 (image나 vit 파일이 있으면 E2E, 없으면 Cached)
    # =====================================================================
    is_e2e = os.path.exists('qwen_image.csv') or os.path.exists('qwen_vit.csv')
    mode = "E2E" if is_e2e else "Cached"
    print(f"✅ 감지된 폴더 모드: [{mode}] 방식을 기준으로 분석합니다.\n")

    f_ttft = 'qwen_TTFT.csv' if os.path.exists('qwen_TTFT.csv') else 'qwen_TTPF.csv'

    # =====================================================================
    # 1. 지표 데이터 로드
    # =====================================================================
    
    # [차트 1] 파이프라인 누적 시간 (Sum)
    if mode == "Cached":
        pipe_times = {
            '0. DB Load': load_csv_data('qwen_db_load.csv', 'sum'),
            '1a. Text': load_csv_data('qwen_text.csv', 'sum'),
            '1b. Image': 0.0,
            '2a. ViT': 0.0,
            '2b. MLP': load_csv_data('qwen_mlp.csv', 'sum'),
            '3. Fusion': load_csv_data('qwen_fusion.csv', 'sum'),
            '4. TTFT': load_csv_data(f_ttft, 'sum')
        }
    else: # E2E
        pipe_times = {
            '1a. Text': load_csv_data('qwen_text.csv', 'sum'),
            '1b. Image': load_csv_data('qwen_image.csv', 'sum'),
            '2a. ViT': load_csv_data('qwen_vit.csv', 'sum'),
            '2b. MLP': load_csv_data('qwen_mlp.csv', 'sum'),
            '3. Fusion': load_csv_data('qwen_fusion.csv', 'sum'),
            '4. TTFT': load_csv_data(f_ttft, 'sum')
        }

    # [차트 2] LLM 누적 시간 (Sum)
    llm_times = {
        'TTFT\n(Sum)': load_csv_data(f_ttft, 'sum'),
        'Decode\n(Sum)': load_csv_data('qwen_decode_time.csv', 'sum'),
        'Latency\n(Sum)': load_csv_data('qwen_total_latency.csv', 'sum')
    }

    # [차트 3] 속도 지표 평균 (Mean)
    val_throughput = load_csv_data('qwen_Throughput.csv', 'mean')
    val_tpot = load_csv_data('qwen_TPOT.csv', 'mean')

    # [차트 4] 최고 VRAM (Max)
    val_vram = load_csv_data('qwen_vram.csv', 'max')

    # 저장 경로
    curr_dir = os.getcwd()
    paths = {
        'pipe': os.path.join(curr_dir, f'Chart1_Qwen_{mode}_Pipeline.png'),
        'llm': os.path.join(curr_dir, f'Chart2_Qwen_{mode}_LLM_Time.png'),
        'speed': os.path.join(curr_dir, f'Chart3_Qwen_{mode}_Speed_Metrics.png'),
        'vram': os.path.join(curr_dir, f'Chart4_Qwen_{mode}_VRAM.png')
    }

    # =====================================================================
    # 🎨 1. 파이프라인 (Total Sum)
    # =====================================================================
    plt.figure(figsize=(10, 6))
    bars1 = plt.bar(list(pipe_times.keys()), list(pipe_times.values()), color=['#B39EB5', '#AEC6CF', '#FFD1DC', '#FFB347', '#FF6961', '#77DD77', '#CFCFC4'][:len(pipe_times)], edgecolor='black', linewidth=1)
    max_val1 = max(list(pipe_times.values())) or 1
    for bar in bars1:
        yval = bar.get_height()
        if yval > 0.01:
            plt.text(bar.get_x() + bar.get_width()/2, yval + (max_val1 * 0.02), f'{yval:.1f}s', ha='center', va='bottom', fontweight='bold')
        else:
            plt.text(bar.get_x() + bar.get_width()/2, max_val1 * 0.02, '0.0s', ha='center', va='bottom', color='#555')
    plt.title(f'Qwen {mode}: Pipeline Total Time (1000 Images)', fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('Total Accumulated Time (s)', fontweight='bold')
    plt.gca().spines['top'].set_visible(False); plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(paths['pipe'], dpi=300)
    plt.close()

    # =====================================================================
    # 🎨 2. LLM 지표 (Total Sum)
    # =====================================================================
    plt.figure(figsize=(8, 6))
    bars2 = plt.bar(list(llm_times.keys()), list(llm_times.values()), color=['#FF9999', '#66B2FF', '#99FF99'], edgecolor='black', linewidth=1, width=0.5)
    max_val2 = max(list(llm_times.values())) or 1
    for bar in bars2:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (max_val2 * 0.02), f'{yval:.1f}s', ha='center', va='bottom', fontweight='bold', fontsize=11)
    plt.title(f'Qwen {mode}: LLM Total Processing Time', fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('Total Accumulated Time (s)', fontweight='bold')
    plt.gca().spines['top'].set_visible(False); plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(paths['llm'], dpi=300)
    plt.close()

    # =====================================================================
    # 🎨 3. 생성 속도 (Throughput & TPOT) -> 이중 Y축 적용
    # =====================================================================
    fig, ax1 = plt.subplots(figsize=(7, 6))
    ax2 = ax1.twinx() # 이중 Y축 생성

    # 각각 독립된 축에 막대 그리기
    bar_t  = ax1.bar([0], [val_throughput], color='#FFD700', edgecolor='black', width=0.4, label='Throughput')
    bar_tp = ax2.bar([1], [val_tpot], color='#DDA0DD', edgecolor='black', width=0.4, label='TPOT')

    # 축 설정
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(['Throughput\n(Mean)', 'TPOT\n(Mean)'], fontweight='bold', fontsize=11)
    ax1.set_ylabel('Throughput (Tokens / Sec)', fontweight='bold', color='#B8860B')
    ax2.set_ylabel('TPOT (Sec / Token)', fontweight='bold', color='#8B008B')
    
    # 윗 테두리 제거
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    # 수치 텍스트 표기
    max_t = val_throughput or 1
    max_tp = val_tpot or 1
    ax1.text(0, val_throughput + (max_t * 0.02), f'{val_throughput:.1f} t/s', ha='center', va='bottom', fontweight='bold', fontsize=11)
    ax2.text(1, val_tpot + (max_tp * 0.02), f'{val_tpot:.4f} s/t', ha='center', va='bottom', fontweight='bold', fontsize=11)

    plt.title(f'Qwen {mode}: Generation Speed Metrics', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(paths['speed'], dpi=300)
    plt.close()

    # =====================================================================
    # 🎨 4. Peak VRAM
    # =====================================================================
    plt.figure(figsize=(5, 6))
    bars4 = plt.bar(['Peak VRAM'], [val_vram], color=['#20B2AA'], edgecolor='black', linewidth=1, width=0.4)
    max_val4 = val_vram or 1
    for bar in bars4:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (max_val4 * 0.02), f'{yval:.1f} MB', ha='center', va='bottom', fontweight='bold', fontsize=12)
    plt.title(f'Qwen {mode}: Maximum VRAM Usage', fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('Memory (MB)', fontweight='bold')
    plt.gca().spines['top'].set_visible(False); plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(paths['vram'], dpi=300)
    plt.close()

    # 요약 출력
    print("\n🎉 총 4개의 통합 막대그래프가 성공적으로 생성되었습니다!")
    for idx, (key, path) in enumerate(paths.items(), 1):
        print(f" 📍 {idx}번 차트: {path}")

if __name__ == "__main__":
    main()