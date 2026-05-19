"""
§1 Stage B 구현 노트:
cached pixel-shuffle embeddings → mlp1 → model.extract_feature monkey-patch → model.chat().

InternVL의 extract_feature (ViT → pixel_shuffle → mlp1) 중
우리는 ViT+pixel_shuffle 결과를 캐시로 갖고 있으므로,
mlp1만 적용한 뒤 extract_feature를 monkey-patch해서 model.chat()이 그대로 동작하게 함.
InternVL 코드를 fork/수정하지 않고 공식 모듈을 호출 순서만 바꿔서 사용. "변형 없음" 원칙 준수.
"""
from __future__ import annotations
from typing import List, Optional
import torch
import torch.nn as nn

from src.query.prompt import build_prompt


def embed_with_cache(
    model_full: nn.Module,
    tokenizer,
    embeddings: torch.Tensor,
    question: str,
    device: str = "cuda",
    options: Optional[List[str]] = None,
    max_new_tokens: int = 256,
    num_frames: Optional[int] = None,
) -> str:
    """
    cached pixel-shuffle embeddings → mlp1 → model.chat() (monkey-patched).

    Args:
        model_full: 전체 InternVL 모델
        tokenizer: InternVL tokenizer
        embeddings: [N, 256, D_vit*4] pixel-shuffle output (bfloat16)
        question: 질문 텍스트
        device: "cuda" or "cpu"
        options: MVBench MCQA 선택지
        max_new_tokens: 최대 생성 토큰 수
        num_frames: 명시적 프레임 수 (None이면 embeddings.shape[0])
    """
    N = num_frames or embeddings.shape[0]
    embeddings = embeddings.to(device, dtype=torch.bfloat16)

    # mlp1: pixel_shuffle output → LLM hidden dim
    # extract_feature()의 마지막 단계만 수행 (ViT+pixel_shuffle은 이미 캐시됨)
    with torch.no_grad():
        visual_embeds = model_full.mlp1(embeddings)  # [N, 256, LLM_dim]

    # extract_feature monkey-patch:
    # model.chat() → model.generate() → self.extract_feature(pixel_values) 호출을
    # 우리 캐시 임베딩 반환으로 대체. dummy pixel_values는 무시됨.
    _orig_extract = model_full.extract_feature
    _cached = visual_embeds

    def _patched_extract(pixel_values):
        return _cached

    model_full.extract_feature = _patched_extract

    try:
        prompt = build_prompt(question=question, num_frames=N, options=options)
        # dummy pixel_values: shape만 맞추면 됨 (extract_feature에서 사용하지 않음)
        dummy_pv = torch.zeros(N, 3, 448, 448, device=device, dtype=torch.bfloat16)

        gen_config = dict(do_sample=False, max_new_tokens=max_new_tokens)
        response, _ = model_full.chat(
            tokenizer,
            dummy_pv,
            prompt,
            generation_config=gen_config,
            num_patches_list=[1] * N,  # 프레임당 타일 1개 → 256 IMG_CONTEXT 토큰
            return_history=True,
        )
    finally:
        model_full.extract_feature = _orig_extract

    return response
