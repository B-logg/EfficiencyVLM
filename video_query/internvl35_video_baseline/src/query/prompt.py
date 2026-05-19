"""
§1 Stage B: InternVL chat template + <frame>K</frame> 지시문 구성.
MVBench MCQA 모드와 MSRVTT-QA 개방형 모드 모두 지원.
"""
from __future__ import annotations
from typing import List


IMG_TOKEN = "<image>"


def build_prompt(
    question: str,
    num_frames: int,
    options: List[str] | None = None,
) -> str:
    """
    Args:
        question: 질문 텍스트
        num_frames: 프레임 수 N
        options: MVBench MCQA 선택지 ["A. ...", "B. ...", ...]

    Returns:
        완성된 프롬프트 문자열
    """
    frame_tags = "".join(f"Frame{i+1}: {IMG_TOKEN}\n" for i in range(num_frames))

    if options:
        options_str = "\n".join(options)
        content = (
            f"{frame_tags}"
            f"Question: {question}\n"
            f"Options:\n{options_str}\n\n"
            f"Answer with the option letter (A/B/C/D). "
            f"Then on a new line, output the single most relevant frame index "
            f"in the exact format <frame>K</frame> where K is an integer in [1, {num_frames}]."
        )
    else:
        content = (
            f"{frame_tags}"
            f"Question: {question}\n"
            f"Answer the question. "
            f"Then on a new line, output the single most relevant frame index "
            f"in the exact format <frame>K</frame> where K is an integer in [1, {num_frames}]."
        )

    return content


def build_internvl_messages(
    question: str,
    num_frames: int,
    options: List[str] | None = None,
) -> List[dict]:
    """InternVL의 conversation 형식으로 반환 (model.chat에 직접 전달 가능)."""
    return [{"role": "user", "content": build_prompt(question, num_frames, options)}]
