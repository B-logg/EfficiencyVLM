"""
모든 모델의 Breakdown 차트를 한 번에 생성합니다.
개별 파일: plot_llava_breakdown.py / plot_qwen_breakdown.py
"""
from plot_llava_breakdown import plot_llava_breakdown
from plot_qwen_breakdown import plot_qwen_breakdown

if __name__ == "__main__":
    plot_llava_breakdown()
    plot_qwen_breakdown()
    print("All breakdown charts saved to plots/")
