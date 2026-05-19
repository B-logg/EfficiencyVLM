"""
Stage A entry point: 영상 목록을 읽어 .pt embedding 파일 저장.

Usage:
    python -m src.ingest.run_ingestion \
        --config configs/runs/sweep1_msrvtt_fps5.yaml \
        --dataset msrvtt \
        --target_fps 5 \
        --output_dir outputs/embeddings/msrvtt/5fps \
        --video_list path/to/video_ids.txt
"""
from __future__ import annotations
import argparse
import os
import sys
import time
import logging
from pathlib import Path
from typing import List, Dict, Any

import torch
import yaml
import numpy as np
from PIL import Image

ROOT = str(Path(__file__).resolve().parents[2])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.ingest.frame_indexing import compute_src_indices
from src.ingest.preprocess import build_video_transform, dynamic_preprocess_video
from src.ingest.encoder import load_internvit, encode_single_frame
from src.data.stream_simulator import StreamSimulator
from src.timing.cuda_timer import CudaTimer
from src.timing.jsonl_logger import JsonlLogger
from src.utils.seed import set_seed
from src.utils.gpu_lock import lock_gpu_clocks, unlock_gpu_clocks

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)


def ingest_video(
    video_path: str,
    video_id: str,
    model: torch.nn.Module,
    transform,
    target_fps: int,
    clip_duration: int,
    output_path: str,
    timing_logger: JsonlLogger,
    cuda_timer: CudaTimer,
    run_id: str,
) -> Dict[str, Any] | None:
    """단일 영상 처리: decord 로드 → FPS 다운샘플 → ViT → pixel_shuffle → 저장."""
    try:
        import decord
        decord.bridge.set_bridge("torch")
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    except Exception as e:
        logger.warning(f"[SKIP] {video_id}: decord 로드 실패 — {e}")
        return None

    fps_src = vr.get_avg_fps()
    n_total = len(vr)

    try:
        src_indices, timestamps = compute_src_indices(
            fps_src=fps_src,
            n_total=n_total,
            target_fps=target_fps,
            clip_duration=clip_duration,
        )
    except ValueError as e:
        logger.info(f"[SKIP] {video_id}: {e}")
        return None

    # 배치 디코드
    t0_decode = time.perf_counter()
    frames_tensor = vr.get_batch(src_indices)  # [F, H, W, C] torch.uint8
    t_decode = (time.perf_counter() - t0_decode) * 1000  # ms

    F = len(src_indices)
    embeddings_list = []
    stream_sim = StreamSimulator(target_fps=target_fps)
    n_late_frames = 0

    for i in range(F):
        frame_np = frames_tensor[i].numpy()
        pil_frame = Image.fromarray(frame_np.astype(np.uint8))

        # preprocess
        t0_pre = time.perf_counter()
        tiles = dynamic_preprocess_video(pil_frame, image_size=448, max_num=1)
        pv = torch.stack([transform(t) for t in tiles]).to(
            next(model.parameters()).device, dtype=torch.bfloat16
        )  # [1, 3, 448, 448]
        t_pre = (time.perf_counter() - t0_pre) * 1000

        # ViT forward
        cuda_timer.start()
        emb = encode_single_frame(model, pv)  # [1, 256, D_vit*4]
        t_vit = cuda_timer.stop()

        embeddings_list.append(emb.cpu())

        # 스트림 부하 시뮬레이션
        late = stream_sim.sleep_until_next(start_time=time.perf_counter())
        if late:
            n_late_frames += 1

        timing_logger.log({
            "run_id": run_id,
            "stage": "A",
            "video_id": video_id,
            "frame_idx": i,
            "t_decode_ms": t_decode / F,  # 프레임당 평균
            "t_preprocess_ms": t_pre,
            "t_vit_forward_ms": t_vit,
            "n_late": int(late),
        })

    embeddings = torch.cat(embeddings_list, dim=0)  # [F, 256, D_vit*4]

    # 저장
    t0_save = time.perf_counter()
    payload = {
        "video_id": video_id,
        "source_path": os.path.abspath(video_path),
        "fps_src": float(fps_src),
        "fps_target": int(target_fps),
        "clip_duration_sec": float(clip_duration),
        "src_indices": src_indices,
        "timestamps_sec": timestamps,
        "embeddings": embeddings.to(torch.bfloat16),
        "config": {
            "model": "OpenGVLab/InternVL3_5-8B",
            "image_size": 448,
            "downsample_ratio": 0.5,
            "max_num_tiles": 1,
            "normalize": "imagenet",
        },
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(payload, output_path)
    t_save = (time.perf_counter() - t0_save) * 1000

    timing_logger.log({
        "run_id": run_id,
        "stage": "A",
        "video_id": video_id,
        "event": "video_done",
        "F": F,
        "t_save_pt_ms": t_save,
        "n_late_frames": n_late_frames,
        "embedding_bytes": os.path.getsize(output_path),
    })

    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True, choices=["msrvtt", "mvbench"])
    parser.add_argument("--target_fps", type=int, required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--video_list", required=True, help="video id와 경로 목록 TSV (id\\tpath)")
    parser.add_argument("--model_path", default="OpenGVLab/InternVL3_5-8B")
    parser.add_argument("--clip_duration", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--lock_clocks", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(42)

    import datetime
    run_id = args.run_id or (
        f"sweep1_{args.dataset}_fps{args.target_fps}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    timing_log_path = f"outputs/timings/{run_id}.jsonl"
    os.makedirs("outputs/timings", exist_ok=True)
    timing_logger = JsonlLogger(timing_log_path)
    cuda_timer = CudaTimer(device=args.device)

    if args.lock_clocks:
        lock_gpu_clocks()

    logger.info(f"모델 로딩: {args.model_path}")
    model = load_internvit(args.model_path, device=args.device)
    transform = build_video_transform(image_size=448)

    with open(args.video_list) as f:
        entries = [line.strip().split("\t") for line in f if line.strip()]

    logger.info(f"총 {len(entries)}개 영상 처리 시작 (target_fps={args.target_fps})")
    t_dataset_start = time.perf_counter()

    n_ok = n_skip = 0
    for video_id, video_path in entries:
        output_path = os.path.join(args.output_dir, f"{video_id}.pt")
        if os.path.exists(output_path):
            logger.debug(f"[CACHE] {video_id} 이미 존재, skip")
            n_ok += 1
            continue

        result = ingest_video(
            video_path=video_path,
            video_id=video_id,
            model=model,
            transform=transform,
            target_fps=args.target_fps,
            clip_duration=args.clip_duration,
            output_path=output_path,
            timing_logger=timing_logger,
            cuda_timer=cuda_timer,
            run_id=run_id,
        )
        if result is None:
            n_skip += 1
        else:
            n_ok += 1

    t_dataset_total = (time.perf_counter() - t_dataset_start) / 60
    timing_logger.log({
        "run_id": run_id,
        "event": "dataset_done",
        "n_ok": n_ok,
        "n_skip": n_skip,
        "t_dataset_total_min": t_dataset_total,
    })

    if args.lock_clocks:
        unlock_gpu_clocks()

    logger.info(f"완료: ok={n_ok}, skip={n_skip}, 총 {t_dataset_total:.1f}분")


if __name__ == "__main__":
    main()
