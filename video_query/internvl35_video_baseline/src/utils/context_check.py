"""
§6: 각 sweep run 시작 전 LLM context 사용량 사전 검사.
(num_frames × 256 + prompt_tokens) < model.config.max_position_embeddings 를 확인.
초과 시 해당 run을 N/A로 마킹.
"""
from __future__ import annotations
import logging

logger = logging.getLogger(__name__)

# InternVL3.5-8B의 LLM context 길이
INTERNVL35_MAX_CONTEXT = 40960  # 실측값 (max_position_embeddings)
# <image> 토큰 1개 = 256 visual tokens (pixel_shuffle 후)
VISUAL_TOKENS_PER_FRAME = 256
# 프롬프트 오버헤드 추정치 (system prompt + question + 지시문)
PROMPT_OVERHEAD_TOKENS = 512


def check_context(
    num_frames: int,
    max_new_tokens: int = 256,
    max_context: int = INTERNVL35_MAX_CONTEXT,
    prompt_tokens: int = PROMPT_OVERHEAD_TOKENS,
) -> bool:
    """
    Returns:
        True if safe, False if context overflow 위험.
    """
    total = num_frames * VISUAL_TOKENS_PER_FRAME + prompt_tokens + max_new_tokens
    if total >= max_context:
        logger.warning(
            f"Context overflow 위험: {total} >= {max_context}. "
            f"(num_frames={num_frames}, visual={num_frames*VISUAL_TOKENS_PER_FRAME}, "
            f"prompt={prompt_tokens}, max_new={max_new_tokens}). "
            f"이 run은 N/A로 마킹."
        )
        return False
    logger.info(f"Context 사전 검사 통과: {total}/{max_context} tokens")
    return True


def assert_context_safe(num_frames: int, **kwargs):
    """통과 못하면 RuntimeError."""
    if not check_context(num_frames, **kwargs):
        raise RuntimeError(
            f"num_frames={num_frames}은 LLM context를 초과합니다. "
            "이 조합은 N/A로 건너뜁니다."
        )
