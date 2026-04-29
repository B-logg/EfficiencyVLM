import pandas as pd
import numpy as np

def analyze_experiment_results():
    print("Qwen2-VL-2B-Instruct E2E vs Cached 벤치마크 결과 분석")

    try:
        # 1. 데이터 로드 (Value 컬럼만 추출)
        time_e2e = pd.read_csv("e2e_inference.csv")['Value']
        time_cached = pd.read_csv("cached_inference.csv")['Value']

        token_e2e = pd.read_csv("e2e_token.csv")['Value']
        token_cached = pd.read_csv("cached_token.csv")['Value']

        vram_e2e = pd.read_csv("e2e_vram.csv")['Value']
        vram_cached = pd.read_csv("cached_vram.csv")['Value']

        num_samples = len(time_e2e)
        
        # 1. 추론 소요 시간 분석 (Inference Time)
        print(f"\n[1] 이미지당 추론 소요 시간 (초)")
        print(f" - E2E 평균    : {time_e2e.mean():.4f} 초 (최대: {time_e2e.max():.4f}, 최소: {time_e2e.min():.4f})")
        print(f" - Cached 평균 : {time_cached.mean():.4f} 초 (최대: {time_cached.max():.4f}, 최소: {time_cached.min():.4f})")
        
        time_diff = time_e2e.mean() - time_cached.mean()
        time_percent = (time_diff / time_e2e.mean()) * 100
        
        print(f" 결과: Cached 방식이 장당 평균 {time_diff:.4f}초 빠름 (▼ {time_percent:.2f}% 시간 감소)")
        print(f" 전체 {num_samples}장 총 소요시간: E2E({time_e2e.sum():.1f}초) vs Cached({time_cached.sum():.1f}초) -> 총 {time_e2e.sum() - time_cached.sum():.1f}초 단축")

        # 2. 초당 토큰 생성 속도 분석 (Tokens per Second)
        print(f"\n[2] 초당 토큰 생성 속도 (Tokens/Sec)")
        print(f" - E2E 평균    : {token_e2e.mean():.2f} t/s (최대: {token_e2e.max():.2f}, 최소: {token_e2e.min():.2f})")
        print(f" - Cached 평균 : {token_cached.mean():.2f} t/s (최대: {token_cached.max():.2f}, 최소: {token_cached.min():.2f})")
        
        token_diff = token_cached.mean() - token_e2e.mean()
        token_percent = (token_diff / token_e2e.mean()) * 100
        
        print(f" 결과: Cached 방식이 초당 {token_diff:.2f}개의 토큰을 더 생성함 (▲ {token_percent:.2f}% 처리량 증가)")

        # 3. 최대 VRAM 사용량 분석 (Peak VRAM MB)
        print(f"\n[3] 최대 VRAM 사용량 (MB)")
        print(f" - E2E 평균 Peak    : {vram_e2e.mean():.2f} MB (절대 최고치: {vram_e2e.max():.2f})")
        print(f" - Cached 평균 Peak : {vram_cached.mean():.2f} MB (절대 최고치: {vram_cached.max():.2f})")
        
        vram_mean_diff = vram_e2e.mean() - vram_cached.mean()
        vram_max_diff = vram_e2e.max() - vram_cached.max()
        
        print(f" 결과: Cached 방식이 평균적으로 {vram_mean_diff:.2f} MB 덜 사용함.")
        print(f" 방어: VRAM이 가장 치솟았던 구간 기준으로는 {vram_max_diff:.2f} MB 감소 효과.")

    except FileNotFoundError as e:
        print(f"에러: CSV 파일을 찾을 수 없습니다. 파일 이름을 확인해주세요.\n({e})")
    except KeyError as e:
        print("에러: CSV 파일 내에 'Value' 컬럼이 없습니다. TensorBoard에서 다운받은 원본 포맷인지 확인해주세요.")

if __name__ == "__main__":
    analyze_experiment_results()