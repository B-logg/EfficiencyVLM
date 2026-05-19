"""
§1 Stage B 구현 노트:
cached pixel-shuffle embeddings → mlp1 → LLM embedding 주입 → greedy decode.

model.chat()을 그대로 호출하면 ViT부터 다시 돌므로, mlp1부터 시작하는 커스텀 forward를 만든다.
InternVL 코드를 fork/수정하지 않고 공식 모듈을 호출 순서만 바꿔서 사용.
"""
from __future__ import annotations
from typing import List, Optional
import re
import torch
import torch.nn as nn

from src.query.prompt import build_prompt


IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"


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
    cached embeddings [F, 256, D_vit*4] → mlp1 → LLM → 텍스트 생성.

    Args:
        model_full: 전체 InternVL 모델 (mlp1 + LLM 포함)
        tokenizer: InternVL tokenizer
        embeddings: [N, 256, D_vit*4] pixel-shuffle output (bfloat16)
        question: 질문 텍스트
        device: "cuda" or "cpu"
        options: MVBench MCQA 선택지 (없으면 개방형)
        max_new_tokens: 최대 생성 토큰 수
        num_frames: 명시적 프레임 수 (None이면 embeddings.shape[0])
    """
    N = num_frames or embeddings.shape[0]
    embeddings = embeddings.to(device, dtype=torch.bfloat16)

    # mlp1: pixel_shuffle output → LLM hidden dim
    with torch.no_grad():
        visual_embeds = model_full.mlp1(embeddings)  # [N, 256, LLM_dim]

    visual_embeds_flat = visual_embeds.view(N * 256, -1)  # [N*256, LLM_dim]

    # 프롬프트 구성 (N개 <image> 태그 포함)
    prompt_text = build_prompt(question=question, num_frames=N, options=options)

    # InternVL 토크나이저: <image>는 내부적으로 num_image_token개의 <IMG_CONTEXT>로 치환
    num_image_token = 256  # pixel_shuffle 후 고정
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

    # <image>를 <img><IMG_CONTEXT>×256</img>로 치환
    placeholder = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * num_image_token + IMG_END_TOKEN
    expanded_prompt = prompt_text.replace("<image>", placeholder)

    input_ids = tokenizer(
        expanded_prompt,
        return_tensors="pt",
        add_special_tokens=True,
    ).input_ids.to(device)

    # text embedding
    text_embeds = model_full.language_model.get_input_embeddings()(input_ids)  # [1, L, D]

    # <IMG_CONTEXT> 위치에 visual embedding scatter 주입
    img_mask = (input_ids == img_context_token_id).squeeze(0)  # [L]
    n_img_tokens = img_mask.sum().item()
    assert n_img_tokens == N * 256, (
        f"<IMG_CONTEXT> 토큰 수 불일치: {n_img_tokens} ≠ {N * 256}. "
        f"tokenizer IMG_CONTEXT 설정 확인 필요."
    )

    text_embeds_flat = text_embeds.squeeze(0)  # [L, D]
    text_embeds_flat[img_mask] = visual_embeds_flat
    inputs_embeds = text_embeds_flat.unsqueeze(0)  # [1, L, D]

    # greedy decode
    with torch.no_grad():
        output_ids = model_full.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=torch.ones(1, inputs_embeds.shape[1], device=device),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response
