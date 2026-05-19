"""
Stage B entry point: .pt embedding 로드 → sub-sample → mlp1 주입 → LLM → 파싱 → 저장.

Usage:
    python -m src.query.run_query \
        --config configs/runs/sweep2_msrvtt_nf16.yaml \
        --dataset msrvtt \
        --target_fps 5 \
        --num_frames 16 \
        --embed_dir outputs/embeddings/msrvtt/5fps \
        --qa_list path/to/test_qa.json \
        --output_dir outputs/responses
"""
from __future__ import annotations
import argparse
import os
import sys
import time
import json
import logging
import datetime
from pathlib import Path
from typing import Dict, Any, List, Iterator

import torch
import yaml
from tqdm import tqdm

ROOT = str(Path(__file__).resolve().parents[2])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.query.frame_subsample import uniform_subsample, SubsampleSkipError
from src.query.injector import embed_with_cache
from src.query.grounding import parse_frame_index, reextract_frame
from src.timing.cuda_timer import CudaTimer
from src.timing.jsonl_logger import JsonlLogger
from src.utils.seed import set_seed
from src.utils.context_check import assert_context_safe

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

WARMUP_ITERS = 3
MIN_MEASURE_SAMPLES = 30


def iter_qa(dataset: str, data_root: str) -> Iterator[Dict[str, Any]]:
    if dataset == "msrvtt":
        from src.data.msrvtt_qa import load_msrvtt_qa
        yield from load_msrvtt_qa(data_root)
    elif dataset == "mvbench":
        from src.data.mvbench import load_mvbench
        yield from load_mvbench(data_root)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def query_single(
    item: Dict[str, Any],
    model_full,
    tokenizer,
    embed_dir: str,
    num_frames: int,
    device: str,
    cuda_timer: CudaTimer,
    run_id: str,
    timing_logger: JsonlLogger,
) -> Dict[str, Any] | None:
    video_id = item["video_id"].replace("/", "_")
    # Stage A는 video_list.tsv의 video_id를 그대로 씀 (task prefix 없을 수 있음)
    # → basename으로 fallback 조회
    embed_fname = os.path.basename(item["video_id"])
    embed_path = os.path.join(embed_dir, f"{embed_fname}.pt")

    if not os.path.exists(embed_path):
        logger.debug(f"[SKIP] embedding 없음: {embed_path}")
        return None

    # 1. .pt 로드
    t0_load = time.perf_counter()
    payload = torch.load(embed_path, map_location="cpu")
    embeddings = payload["embeddings"]  # [F, 256, D]
    src_indices = payload["src_indices"]
    t_load = (time.perf_counter() - t0_load) * 1000

    # 2. uniform sub-sample
    t0_sub = time.perf_counter()
    try:
        sub_emb = uniform_subsample(embeddings, num_frames=num_frames)
    except SubsampleSkipError as e:
        logger.info(f"[N/A] {video_id}: {e}")
        return {"video_id": video_id, "status": "skipped_F_lt_N", "reason": str(e)}
    F = embeddings.shape[0]
    uniform_indices = [round(i * (F - 1) / (num_frames - 1)) for i in range(num_frames)]
    t_sub = (time.perf_counter() - t0_sub) * 1000

    # 3. mlp1 + LLM
    question = item["question"]
    options = item.get("options", None)

    cuda_timer.start()
    try:
        response = embed_with_cache(
            model_full=model_full,
            tokenizer=tokenizer,
            embeddings=sub_emb,
            question=question,
            device=device,
            options=options,
            max_new_tokens=256,
            num_frames=num_frames,
        )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        logger.warning(f"[OOM] {video_id}: num_frames={num_frames} — GPU 메모리 부족, skip")
        return {"video_id": video_id, "status": "skipped_oom", "reason": "CUDA OOM"}
    t_query = cuda_timer.stop()

    # 4. <frame>K</frame> 파싱
    t0_parse = time.perf_counter()
    k = parse_frame_index(response)
    frame_ok = k is not None and 1 <= k <= num_frames
    t_parse = (time.perf_counter() - t0_parse) * 1000

    # 5. 원본 프레임 재추출 (grounding)
    t_reextract = 0.0
    if frame_ok and "video_path" in payload:
        t0_re = time.perf_counter()
        pil_frame = reextract_frame(
            video_path=payload["source_path"],
            src_indices=src_indices,
            uniform_indices=uniform_indices,
            k=k,
        )
        t_reextract = (time.perf_counter() - t0_re) * 1000

    timing_logger.log({
        "run_id": run_id,
        "stage": "B",
        "video_id": video_id,
        "t_load_pt_ms": t_load,
        "t_subsample_ms": t_sub,
        "t_query_total_ms": t_query,
        "t_grounding_parse_ms": t_parse,
        "t_frame_reextract_ms": t_reextract,
        "frame_index_parsed_ok": frame_ok,
        "frame_k": k,
        "num_frames": num_frames,
        "F": F,
    })

    return {
        "video_id": video_id,
        "question": question,
        "answer_gt": item.get("answer", ""),
        "response": response,
        "frame_k": k,
        "frame_ok": frame_ok,
        "status": "ok",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True, choices=["msrvtt", "mvbench"])
    parser.add_argument("--target_fps", type=int, required=True)
    parser.add_argument("--num_frames", type=int, required=True)
    parser.add_argument("--embed_dir", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_dir", default="outputs/responses")
    parser.add_argument("--model_path", default="OpenGVLab/InternVL3_5-8B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--max_samples", type=int, default=None, help="디버그용 샘플 제한")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(42)

    # context 사전 검사 (§6)
    assert_context_safe(args.num_frames)

    run_id = args.run_id or (
        f"sweep2_{args.dataset}_fps{args.target_fps}_nf{args.num_frames}_"
        f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("outputs/timings", exist_ok=True)
    response_path = os.path.join(args.output_dir, f"{run_id}.jsonl")
    timing_logger = JsonlLogger(f"outputs/timings/{run_id}.jsonl")
    cuda_timer = CudaTimer(device=args.device)

    logger.info(f"모델 로딩: {args.model_path}")
    from transformers import AutoModel, AutoTokenizer
    model_full = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(args.device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # Warmup (§5)
    logger.info(f"Warmup {WARMUP_ITERS} iterations...")
    qa_iter = iter_qa(args.dataset, args.data_root)
    warmup_items = []
    for item in qa_iter:
        embed_fname = os.path.basename(item["video_id"])
        if os.path.exists(os.path.join(args.embed_dir, f"{embed_fname}.pt")):
            warmup_items.append(item)
        if len(warmup_items) >= WARMUP_ITERS:
            break

    for item in warmup_items:
        query_single(
            item=item, model_full=model_full, tokenizer=tokenizer,
            embed_dir=args.embed_dir, num_frames=args.num_frames,
            device=args.device, cuda_timer=cuda_timer,
            run_id="warmup", timing_logger=JsonlLogger("/dev/null"),
        )

    logger.info(f"측정 시작 (run_id={run_id})")
    results = []
    n_ok = n_skip = n_na = n_oom = 0
    # max_samples = .pt 파일을 찾아서 실제 처리한 수 기준 (QA 순회 수 아님)
    n_found = 0

    with open(response_path, "w") as resp_f:
        pbar = tqdm(
            total=args.max_samples,
            desc=f"Stage B [{args.dataset} nf={args.num_frames}]",
            unit="video",
            dynamic_ncols=True,
        )
        for item in iter_qa(args.dataset, args.data_root):
            if args.max_samples and n_found >= args.max_samples:
                break

            # .pt 없으면 순회만 하고 max_samples 카운트 안 함
            embed_fname = os.path.basename(item["video_id"])
            embed_path = os.path.join(args.embed_dir, f"{embed_fname}.pt")
            if not os.path.exists(embed_path):
                continue

            n_found += 1
            pbar.update(1)

            result = query_single(
                item=item, model_full=model_full, tokenizer=tokenizer,
                embed_dir=args.embed_dir, num_frames=args.num_frames,
                device=args.device, cuda_timer=cuda_timer,
                run_id=run_id, timing_logger=timing_logger,
            )

            if result is None:
                n_skip += 1
                pbar.set_postfix(ok=n_ok, skip=n_skip, na=n_na, oom=n_oom)
                continue
            if result.get("status") == "skipped_F_lt_N":
                n_na += 1
                pbar.set_postfix(ok=n_ok, skip=n_skip, na=n_na, oom=n_oom)
                continue
            if result.get("status") == "skipped_oom":
                n_oom += 1
                pbar.set_postfix(ok=n_ok, skip=n_skip, na=n_na, oom=n_oom)
                continue

            n_ok += 1
            pbar.set_postfix(ok=n_ok, skip=n_skip, na=n_na, oom=n_oom)
            resp_f.write(json.dumps(result, ensure_ascii=False) + "\n")

    timing_logger.log({
        "run_id": run_id,
        "event": "query_done",
        "n_ok": n_ok, "n_skip": n_skip, "n_na": n_na, "n_oom": n_oom,
    })

    logger.info(f"완료: ok={n_ok}, skip={n_skip}, n/a(F<N)={n_na}, oom={n_oom}")
    logger.info(f"응답 저장: {response_path}")


if __name__ == "__main__":
    main()
