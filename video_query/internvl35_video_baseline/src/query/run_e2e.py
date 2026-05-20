"""
E2E Video Query Runner (Stage A + B 통합, 쿼리 시 전체 처리)

기존 캐시 방식과 달리, 쿼리가 들어올 때마다:
  video decode → preprocess → ViT(×F) → pixel_shuffle → subsample → mlp1 → LLM

FPS=10, clip_duration=10s → F=100 프레임 인코딩
LLM 입력은 그 중 num_frames_llm 개를 균등 서브샘플링

사용법:
    python -m src.query.run_e2e \
        --dataset msrvtt \
        --data_root data/msrvtt \
        --model_path OpenGVLab/InternVL3_5-8B \
        --max_samples 100 \
        --warmup_samples 5 \
        --num_frames_llm 16 \
        --output_dir outputs/e2e
"""
from __future__ import annotations
import argparse
import csv
import datetime
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Iterator

import numpy as np
import torch
from tqdm import tqdm

ROOT = str(Path(__file__).resolve().parents[2])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.ingest.frame_indexing import compute_src_indices
from src.ingest.preprocess import build_video_transform
from src.ingest.pixel_shuffle import pixel_shuffle
from src.query.prompt import build_prompt
from src.utils.seed import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TARGET_FPS     = 12
CLIP_DURATION  = 10   # seconds → 120 frames
N_TOTAL_FRAMES = TARGET_FPS * CLIP_DURATION   # 120

# InternVL3.5-8B: 40960 토큰 컨텍스트 (실측값)
# 100프레임 × 256 + prompt(512) + gen(256) = 26368 < 40960 → 전부 입력 가능
MAX_SAFE_FRAMES = (40960 - 512 - 256) // 256  # = 156


def iter_qa(dataset: str, data_root: str) -> Iterator[Dict[str, Any]]:
    if dataset == "msrvtt":
        from src.data.msrvtt_qa import load_msrvtt_qa
        yield from load_msrvtt_qa(data_root)
    elif dataset == "mvbench":
        from src.data.mvbench import load_mvbench
        yield from load_mvbench(data_root)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def uniform_subsample_indices(total: int, n: int) -> List[int]:
    """total 프레임 중 n개를 균등 간격으로 추출하는 인덱스 반환."""
    if n >= total:
        return list(range(total))
    return [round(i * (total - 1) / (n - 1)) for i in range(n)]


@torch.no_grad()
def e2e_query_single(
    item: Dict[str, Any],
    model,
    tokenizer,
    transform,
    num_frames_llm: int,
    device: str,
    max_new_tokens: int = 128,
) -> Optional[Dict[str, Any]]:
    """
    단일 쿼리에 대해 E2E 처리 + 단계별 시간 측정.

    Returns:
        timing + response dict, 또는 None (비디오 로드 실패 시)
    """
    import decord
    from PIL import Image

    video_path = item.get("video_path", "")
    video_id   = item["video_id"]
    question   = item["question"]
    options    = item.get("options", None)

    # ── 1. Video decode + frame extraction (CPU wall clock) ──────────────
    torch.cuda.synchronize()
    _t = time.perf_counter()
    try:
        decord.bridge.set_bridge("torch")
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
        fps_src = vr.get_avg_fps()
        n_total = len(vr)
        src_indices, _ = compute_src_indices(
            fps_src=fps_src, n_total=n_total,
            target_fps=TARGET_FPS, clip_duration=CLIP_DURATION,
        )
        frames_tensor = vr.get_batch(src_indices)  # [F, H, W, C]
    except Exception as e:
        logger.warning(f"[SKIP] {video_id}: {e}")
        return None
    torch.cuda.synchronize()
    t_decode_ms = (time.perf_counter() - _t) * 1000

    F = len(src_indices)   # should be 100

    # ── 2. Preprocess: PIL → tensor + H2D (CPU wall clock) ──────────────
    torch.cuda.synchronize()
    _t = time.perf_counter()
    pil_frames = [Image.fromarray(frames_tensor[i].numpy().astype(np.uint8)) for i in range(F)]
    pixel_tensors = [transform(f) for f in pil_frames]
    pixel_batch = torch.stack(pixel_tensors).to(device, dtype=torch.bfloat16)  # [F, 3, 448, 448]
    torch.cuda.synchronize()   # H2D 완료 대기
    t_preprocess_ms = (time.perf_counter() - _t) * 1000

    # ── 3. ViT encoding (GPU, 100프레임 배치) ──────────────────────────
    ev_vs, ev_ve = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ev_vs.record()
    vit_out = model.vision_model(pixel_batch)          # [F, seq, D_vit]
    hidden  = vit_out.last_hidden_state
    if hidden.shape[1] == 1025:                        # CLS 토큰 제거
        hidden = hidden[:, 1:, :]
    ev_ve.record()
    torch.cuda.synchronize()
    t_vit_ms = ev_vs.elapsed_time(ev_ve)

    # ── 4. Pixel shuffle (GPU) ─────────────────────────────────────────
    ev_ss, ev_se = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ev_ss.record()
    shuffled = pixel_shuffle(hidden, scale_factor=0.5)  # [F, 256, D_vit*4]
    ev_se.record()
    torch.cuda.synchronize()
    t_shuffle_ms = ev_ss.elapsed_time(ev_se)

    # ── 5. Uniform subsample → LLM에 입력할 N 프레임 선택 ─────────────
    sub_idxs   = uniform_subsample_indices(F, num_frames_llm)
    shuffled_n = shuffled[sub_idxs]   # [N, 256, D_vit*4]  (인덱싱, GPU)

    # ── 6. MLP1 projection (GPU) ───────────────────────────────────────
    ev_ms, ev_me = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ev_ms.record()
    visual_embeds = model.mlp1(shuffled_n)   # [N, 256, LLM_dim]
    ev_me.record()
    torch.cuda.synchronize()
    t_mlp_ms = ev_ms.elapsed_time(ev_me)

    # ── 7. LLM inference (GPU) ─────────────────────────────────────────
    # extract_feature monkey-patch: model.chat() 내부 ViT 호출을 캐시로 대체
    N = num_frames_llm
    _orig_extract = model.extract_feature
    _cached_embeds = visual_embeds

    def _patched_extract(pixel_values):
        return _cached_embeds

    model.extract_feature = _patched_extract
    try:
        prompt    = build_prompt(question=question, num_frames=N, options=options)
        dummy_pv  = torch.zeros(N, 3, 448, 448, device=device, dtype=torch.bfloat16)
        gen_cfg   = dict(do_sample=False, max_new_tokens=max_new_tokens)

        ev_ls, ev_le = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        ev_ls.record()
        response, _ = model.chat(
            tokenizer, dummy_pv, prompt,
            generation_config=gen_cfg,
            num_patches_list=[1] * N,
            return_history=True,
        )
        ev_le.record()
        torch.cuda.synchronize()
        t_llm_ms = ev_ls.elapsed_time(ev_le)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        logger.warning(f"[OOM] {video_id}: num_frames_llm={num_frames_llm}")
        return {"video_id": video_id, "status": "oom"}
    finally:
        model.extract_feature = _orig_extract

    t_total_ms = t_decode_ms + t_preprocess_ms + t_vit_ms + t_shuffle_ms + t_mlp_ms + t_llm_ms

    return {
        "video_id":          video_id,
        "question":          question,
        "answer_gt":         item.get("answer", ""),
        "response":          response,
        "status":            "ok",
        "n_frames_encoded":  F,
        "n_frames_llm":      N,
        "t_decode_ms":       t_decode_ms,
        "t_preprocess_ms":   t_preprocess_ms,
        "t_vit_ms":          t_vit_ms,
        "t_pixel_shuffle_ms": t_shuffle_ms,
        "t_mlp_ms":          t_mlp_ms,
        "t_llm_ms":          t_llm_ms,
        "t_total_ms":        t_total_ms,
    }


def main():
    parser = argparse.ArgumentParser(description="E2E video query (no pre-caching)")
    parser.add_argument("--dataset",        required=True, choices=["msrvtt", "mvbench"])
    parser.add_argument("--data_root",      required=True)
    parser.add_argument("--model_path",     default="OpenGVLab/InternVL3_5-8B")
    parser.add_argument("--device",         default="cuda")
    parser.add_argument("--max_samples",    type=int, default=100)
    parser.add_argument("--warmup_samples", type=int, default=5)
    parser.add_argument("--num_frames_llm", type=int, default=None,
                        help="LLM에 입력할 프레임 수. None이면 컨텍스트 한도 내 최대(기본 100프레임 전부)")
    parser.add_argument("--output_dir",     default="outputs/e2e")
    parser.add_argument("--run_id",         default=None)
    args = parser.parse_args()

    set_seed(42)
    os.makedirs(args.output_dir, exist_ok=True)

    # LLM 입력 프레임 수: None이면 컨텍스트 한도 내 최대 (100프레임 전부)
    num_frames_llm = args.num_frames_llm or min(N_TOTAL_FRAMES, MAX_SAFE_FRAMES)
    logger.info(f"LLM 입력 프레임: {num_frames_llm} / {N_TOTAL_FRAMES} (context limit: {MAX_SAFE_FRAMES})")

    run_id = args.run_id or (
        f"e2e_{args.dataset}_fps{TARGET_FPS}_dur{CLIP_DURATION}s_nf{num_frames_llm}_"
        f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    logger.info(f"모델 로딩: {args.model_path}")
    from transformers import AutoModel, AutoTokenizer
    model = AutoModel.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        trust_remote_code=True, low_cpu_mem_usage=True,
    ).to(args.device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    transform = build_video_transform(image_size=448)

    # ── Warmup ─────────────────────────────────────────────────────────
    logger.info(f"Warmup {args.warmup_samples} 샘플...")
    warmup_done = 0
    for item in iter_qa(args.dataset, args.data_root):
        if warmup_done >= args.warmup_samples:
            break
        if not item.get("video_path") or not os.path.exists(item["video_path"]):
            continue
        e2e_query_single(item, model, tokenizer, transform,
                         num_frames_llm, args.device)
        warmup_done += 1

    # ── Main measurement loop ───────────────────────────────────────────
    logger.info(f"측정 시작 (run_id={run_id}, max_samples={args.max_samples})")
    results: list = []
    n_ok = n_skip = n_oom = 0
    n_found = 0

    out_path = os.path.join(args.output_dir, f"{run_id}.jsonl")
    with open(out_path, "w") as f_out:
        pbar = tqdm(
            total=args.max_samples,
            desc=f"E2E [{args.dataset} fps={TARGET_FPS} nf_llm={num_frames_llm}]",
            unit="video",
        )
        for item in iter_qa(args.dataset, args.data_root):
            if n_found >= args.max_samples:
                break
            if not item.get("video_path") or not os.path.exists(item["video_path"]):
                continue

            n_found += 1
            pbar.update(1)

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            result = e2e_query_single(
                item, model, tokenizer, transform,
                args.num_frames_llm, args.device,
            )

            if result is None:
                n_skip += 1
            elif result.get("status") == "oom":
                n_oom += 1
            else:
                n_ok += 1
                result["run_id"] = run_id
                result["vram_gb"] = torch.cuda.max_memory_allocated() / (1024**3)
                results.append(result)
                f_out.write(json.dumps(result, ensure_ascii=False) + "\n")

            pbar.set_postfix(ok=n_ok, skip=n_skip, oom=n_oom)

    logger.info(f"완료: ok={n_ok}, skip={n_skip}, oom={n_oom}")
    logger.info(f"결과 저장: {out_path}")

    # ── CSV 저장 (plot 용) ──────────────────────────────────────────────
    if results:
        timing_cols = [
            "video_id", "n_frames_encoded", "n_frames_llm",
            "t_decode_ms", "t_preprocess_ms", "t_vit_ms",
            "t_pixel_shuffle_ms", "t_mlp_ms", "t_llm_ms", "t_total_ms",
            "vram_gb",
        ]
        csv_path = os.path.join(args.output_dir, f"{run_id}_timing.csv")
        with open(csv_path, "w", newline="") as cf:
            writer = csv.DictWriter(cf, fieldnames=timing_cols, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
        logger.info(f"타이밍 CSV 저장: {csv_path}")


if __name__ == "__main__":
    main()
