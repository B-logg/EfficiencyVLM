"""
모든 모델의 Result 차트를 한 번에 생성합니다.
개별 파일: plot_llava_result.py / plot_qwen_result.py
"""
from plot_llava_result import plot_llava_result
from plot_qwen_result import plot_qwen_result

if __name__ == "__main__":
    plot_llava_result()
    plot_qwen_result()
    print("All result charts saved to plots/")
