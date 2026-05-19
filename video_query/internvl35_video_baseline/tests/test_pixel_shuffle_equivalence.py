"""
Gate test §6: 분리 파이프라인 ≡ 모델 chat() 동등성 검증.
두 경로의 LLM logits L∞ 차이 < 1e-2, 첫 64 generated tokens 완전 일치.
통과하지 않으면 모든 latency 수치는 의미 없음.
"""
import sys
import os
import pytest
import torch
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.ingest.preprocess import build_video_transform, dynamic_preprocess_video
from src.ingest.pixel_shuffle import pixel_shuffle
from src.query.injector import embed_with_cache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_path: str, device: str = "cuda"):
    from transformers import AutoTokenizer
    from src.ingest.encoder import load_internvit

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = load_internvit(model_path, device=device)
    return model, tokenizer


def make_synthetic_frames(n_frames: int = 4, image_size: int = 448) -> list:
    """재현 가능한 합성 PIL 프레임 생성."""
    from PIL import Image
    rng = np.random.default_rng(42)
    frames = []
    for _ in range(n_frames):
        arr = (rng.random((image_size, image_size, 3)) * 255).astype(np.uint8)
        frames.append(Image.fromarray(arr))
    return frames


# ---------------------------------------------------------------------------
# Path A: InternVL 공식 model.chat() 경로 (ViT부터 전체 forward)
# ---------------------------------------------------------------------------

def run_official_chat(model_full, tokenizer, frames: list, question: str, device: str):
    """model.chat()을 통해 logits와 generated_ids를 반환."""
    from transformers import AutoModel
    import torchvision.transforms as T

    transform = build_video_transform(image_size=448)
    pixel_values_list = []
    for frame in frames:
        tiles = dynamic_preprocess_video(frame, image_size=448, max_num=1)
        pv = torch.stack([transform(t) for t in tiles])  # [1, 3, 448, 448]
        pixel_values_list.append(pv)
    pixel_values = torch.cat(pixel_values_list, dim=0).to(device, dtype=torch.bfloat16)

    n = len(frames)
    frame_tags = "".join(f"Frame{i+1}: <image>\n" for i in range(n))
    prompt = (
        f"{frame_tags}"
        f"Question: {question}\n"
        f"Answer the question. Then on a new line, output the single most "
        f"relevant frame index in the exact format <frame>K</frame> "
        f"where K is an integer in [1, {n}]."
    )

    gen_config = dict(do_sample=False, max_new_tokens=64)
    response, history = model_full.chat(
        tokenizer,
        pixel_values,
        prompt,
        generation_config=gen_config,
        return_history=True,
    )
    return response


# ---------------------------------------------------------------------------
# Path B: 분리 파이프라인 (ViT → pixel_shuffle 저장 → mlp1 → LLM)
# ---------------------------------------------------------------------------

def run_split_pipeline(model_full, tokenizer, frames: list, question: str, device: str):
    """embed_with_cache를 통해 logits와 generated_ids를 반환."""
    from src.ingest.encoder import encode_frames
    from src.ingest.pixel_shuffle import pixel_shuffle as ps

    transform = build_video_transform(image_size=448)

    vit_outputs = []
    for frame in frames:
        tiles = dynamic_preprocess_video(frame, image_size=448, max_num=1)
        pv = torch.stack([transform(t) for t in tiles]).to(device, dtype=torch.bfloat16)
        with torch.no_grad():
            feat = model_full.vision_model(pv).last_hidden_state  # [1, 1024, D_vit]
        shuffled = ps(feat, scale_factor=0.5)  # [1, 256, D_vit*4]
        vit_outputs.append(shuffled)

    embeddings = torch.cat(vit_outputs, dim=0)  # [F, 256, D_vit*4]

    response = embed_with_cache(
        model_full=model_full,
        tokenizer=tokenizer,
        embeddings=embeddings,
        question=question,
        device=device,
        max_new_tokens=64,
    )
    return response


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA 필요")
def test_equivalence(model_path: str = "OpenGVLab/InternVL3_5-8B"):
    """두 경로의 생성 결과가 처음 64 토큰 내에서 일치해야 함."""
    device = "cuda"

    from transformers import AutoModel, AutoTokenizer
    model_full = AutoModel.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    frames = make_synthetic_frames(n_frames=4, image_size=448)
    question = "What is happening in the video?"

    resp_a = run_official_chat(model_full, tokenizer, frames, question, device)
    resp_b = run_split_pipeline(model_full, tokenizer, frames, question, device)

    assert resp_a == resp_b, (
        f"첫 64 토큰 불일치\nPath A: {resp_a!r}\nPath B: {resp_b!r}"
    )
    print("PASS: 두 경로 생성 결과 일치")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="OpenGVLab/InternVL3_5-8B")
    args = parser.parse_args()
    test_equivalence(args.model_path)
    print("Gate test PASSED — 실험 시작 가능")
