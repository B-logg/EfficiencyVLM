"""
Cached Video Query Runner (Stage A + Stage B 분리 측정)

Stage A (사전 처리, 1회):
    video decode → preprocess → ViT(×100frames) → pixel_shuffle → .pt 저장

Stage B (쿼리당, per-query):
    .pt 로드 → mlp1 → LLM → 응답 + 정확도

FPS=10, clip=10s → 100프레임
LLM 입력: 100프레임 전부 (32k 컨텍스트 내 충분)

사용법:
    python -m src.query.run_cached_video \
        --dataset mvbench \
        --data_root data/mvbench \
        --model_path OpenGVLab/InternVL3_5-8B \
        --max_samples 100 \
        --warmup_samples 5 \
        --embed_dir outputs/cached_video_embeddings \
        --output_dir outputs/cached_video
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
from typing import Dict, Any, List, Optional, Iterator

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

TARGET_FPS    = 12
CLIP_DURATION = 10      # seconds → 120 frames
N_FRAMES      = TARGET_FPS * CLIP_DURATION   # 120
MAX_SAFE_FRAMES = (40960 - 512 - 256) // 256  # = 156  (40960 실측 context 기준)


def iter_qa(dataset: str, data_root: str) -> Iterator[Dict[str, Any]]:
    if dataset == "mvbench":
        from src.data.mvbench import load_mvbench
        yield from load_mvbench(data_root)
    elif dataset == "msrvtt":
        from src.data.msrvtt_qa import load_msrvtt_qa
        yield from load_msrvtt_qa(data_root)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def collect_sorted_by_duration(
    dataset: str,
    data_root: str,
    min_duration: float = CLIP_DURATION,
    max_samples: int = 200,
) -> List[Dict[str, Any]]:
    """
    전체 QA 아이템을 스캔해서:
      1) duration >= min_duration 인 것만 수집
      2) duration 오름차순 정렬 (10초에 가장 가까운 영상 우선)
      3) 상위 max_samples개 반환
    """
    import decord

    logger.info(f"영상 길이 스캔 중 (>= {min_duration}s)...")
    candidates = []
    for item in iter_qa(dataset, data_root):
        vp = item.get("video_path", "")
        if not vp or not os.path.exists(vp):
            continue
        try:
            vr = decord.VideoReader(vp, ctx=decord.cpu(0))
            duration = len(vr) / vr.get_avg_fps()
            if duration >= min_duration:
                item["_duration"] = duration
                candidates.append(item)
        except Exception:
            continue

    candidates.sort(key=lambda x: x["_duration"])
    logger.info(f"유효 샘플: {len(candidates)}개 (>= {min_duration}s) → 상위 {min(max_samples, len(candidates))}개 사용")
    return candidates[:max_samples]


# ─────────────────────────────────────────────────────────────────────────────
# Stage A: 비디오 인코딩 (1회 처리)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def stage_a_encode(
    video_path: str,
    video_id: str,
    model,
    transform,
    embed_dir: str,
    device: str,
) -> Optional[Dict[str, Any]]:
    """
    decode → preprocess → ViT → pixel_shuffle → .pt 저장.
    Returns timing dict or None on failure.
    """
    import decord
    from PIL import Image

    safe_id  = video_id.replace("/", "_")
    # FPS를 파일명에 포함 → FPS 변경 시 자동으로 새 파일 생성 (stale cache 방지)
    pt_path  = os.path.join(embed_dir, f"{safe_id}_fps{TARGET_FPS}.pt")

    # 이미 처리된 파일은 스킵 (Stage A 재시작 지원)
    if os.path.exists(pt_path):
        logger.debug(f"[CACHE HIT] {video_id}")
        payload = torch.load(pt_path, map_location="cpu")
        n_frames = payload["pixel_shuffled"].shape[0]
        return {"video_id": video_id, "status": "cached",
                "n_frames": n_frames, "t_total_a_ms": 0.0}

    # 1. Decode (CPU wall clock)
    torch.cuda.synchronize()
    _t = time.perf_counter()
    try:
        decord.bridge.set_bridge("torch")
        vr       = decord.VideoReader(video_path, ctx=decord.cpu(0))
        fps_src  = vr.get_avg_fps()
        n_total  = len(vr)
        src_indices, _ = compute_src_indices(
            fps_src=fps_src, n_total=n_total,
            target_fps=TARGET_FPS, clip_duration=CLIP_DURATION,
        )
        frames_tensor = vr.get_batch(src_indices)
    except Exception as e:
        logger.warning(f"[SKIP A] {video_id}: {e}")
        return None
    torch.cuda.synchronize()
    t_decode_ms = (time.perf_counter() - _t) * 1000

    F = len(src_indices)   # 100

    # 2. Preprocess + H2D (CPU wall clock)
    torch.cuda.synchronize()
    _t = time.perf_counter()
    pil_frames = [Image.fromarray(frames_tensor[i].numpy().astype(np.uint8)) for i in range(F)]
    pixel_tensors = [transform(f) for f in pil_frames]
    pixel_batch = torch.stack(pixel_tensors).to(device, dtype=torch.bfloat16)
    torch.cuda.synchronize()
    t_preprocess_ms = (time.perf_counter() - _t) * 1000

    # 3. ViT (GPU, 100프레임 배치)
    ev_vs, ev_ve = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ev_vs.record()
    vit_out = model.vision_model(pixel_batch)
    hidden  = vit_out.last_hidden_state
    if hidden.shape[1] == 1025:
        hidden = hidden[:, 1:, :]
    ev_ve.record()
    torch.cuda.synchronize()
    t_vit_ms = ev_vs.elapsed_time(ev_ve)

    # 4. Pixel shuffle (GPU)
    ev_ss, ev_se = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ev_ss.record()
    shuffled = pixel_shuffle(hidden, scale_factor=0.5)   # [F, 256, D*4]
    ev_se.record()
    torch.cuda.synchronize()
    t_shuffle_ms = ev_ss.elapsed_time(ev_se)

    # 5. Save .pt
    _t_save = time.perf_counter()
    os.makedirs(embed_dir, exist_ok=True)
    torch.save({"pixel_shuffled": shuffled.cpu(), "video_id": video_id}, pt_path)
    t_save_ms = (time.perf_counter() - _t_save) * 1000

    t_total_a = t_decode_ms + t_preprocess_ms + t_vit_ms + t_shuffle_ms + t_save_ms

    return {
        "video_id":       video_id,
        "status":         "encoded",
        "n_frames":       F,
        "t_decode_ms":    t_decode_ms,
        "t_preprocess_ms": t_preprocess_ms,
        "t_vit_ms":       t_vit_ms,
        "t_pixel_shuffle_ms": t_shuffle_ms,
        "t_save_ms":      t_save_ms,
        "t_total_a_ms":   t_total_a,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Stage B: 캐시 로드 + MLP + LLM (쿼리당)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def stage_b_query(
    item: Dict[str, Any],
    model,
    tokenizer,
    embed_dir: str,
    num_frames_llm: int,
    device: str,
    max_new_tokens: int = 128,
) -> Optional[Dict[str, Any]]:
    """
    .pt 로드 → (서브샘플 없이 전체 사용) → mlp1 → LLM.
    Returns timing + response dict or None.
    """
    video_id = item["video_id"]
    safe_id  = video_id.replace("/", "_")
    pt_path  = os.path.join(embed_dir, f"{safe_id}_fps{TARGET_FPS}.pt")

    if not os.path.exists(pt_path):
        logger.debug(f"[SKIP B] .pt 없음: {pt_path}")
        return None

    question = item["question"]
    options  = item.get("options", None)

    # 1. Load .pt + H2D (CPU wall clock)
    torch.cuda.synchronize()
    _t = time.perf_counter()
    payload  = torch.load(pt_path, map_location="cpu")
    shuffled = payload["pixel_shuffled"].to(device, dtype=torch.bfloat16)  # [F, 256, D*4]
    torch.cuda.synchronize()
    t_load_ms = (time.perf_counter() - _t) * 1000

    F = shuffled.shape[0]
    # 서브샘플: num_frames_llm이 F보다 작으면 균등 추출, 같으면 전체 사용
    if num_frames_llm < F:
        idxs = [round(i * (F - 1) / (num_frames_llm - 1)) for i in range(num_frames_llm)]
        shuffled = shuffled[idxs]
    N = shuffled.shape[0]

    # 2. MLP1 (GPU)
    ev_ms, ev_me = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ev_ms.record()
    visual_embeds = model.mlp1(shuffled)   # [N, 256, LLM_dim]
    ev_me.record()
    torch.cuda.synchronize()
    t_mlp_ms = ev_ms.elapsed_time(ev_me)

    # 3. LLM inference (GPU, monkey-patch)
    _orig = model.extract_feature
    _embs = visual_embeds

    def _patched(_pv):   # noqa: ARG001  (dummy pixel values 무시)
        return _embs

    model.extract_feature = _patched
    try:
        prompt   = build_prompt(question=question, num_frames=N, options=options)
        dummy_pv = torch.zeros(N, 3, 448, 448, device=device, dtype=torch.bfloat16)
        gen_cfg  = dict(do_sample=False, max_new_tokens=max_new_tokens)

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
        logger.warning(f"[OOM B] {video_id}: N={N}")
        return {"video_id": video_id, "status": "oom"}
    finally:
        model.extract_feature = _orig

    t_total_b = t_load_ms + t_mlp_ms + t_llm_ms

    return {
        "video_id":       video_id,
        "task":           item.get("task", ""),
        "question":       question,
        "answer_gt":      item.get("answer", ""),
        "response":       response,
        "status":         "ok",
        "n_frames_loaded": F,
        "n_frames_llm":   N,
        "t_load_ms":      t_load_ms,
        "t_mlp_ms":       t_mlp_ms,
        "t_llm_ms":       t_llm_ms,
        "t_total_b_ms":   t_total_b,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cached video query (Stage A + B)")
    parser.add_argument("--dataset",        required=True, choices=["msrvtt", "mvbench"])
    parser.add_argument("--data_root",      required=True)
    parser.add_argument("--model_path",     default="OpenGVLab/InternVL3_5-8B")
    parser.add_argument("--device",         default="cuda")
    parser.add_argument("--max_samples",    type=int, default=100)
    parser.add_argument("--warmup_samples", type=int, default=5)
    parser.add_argument("--num_frames_llm", type=int, default=None,
                        help="LLM 입력 프레임 수. None=전체(컨텍스트 한도 내)")
    parser.add_argument("--embed_dir",      default="outputs/cached_video_embeddings")
    parser.add_argument("--output_dir",     default="outputs/cached_video")
    parser.add_argument("--run_id",         default=None)
    parser.add_argument("--stage_a_only",   action="store_true",
                        help="Stage A(인코딩)만 실행하고 Stage B(LLM)는 건너뜀")
    args = parser.parse_args()

    set_seed(42)
    os.makedirs(args.embed_dir,  exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    num_frames_llm = args.num_frames_llm or min(N_FRAMES, MAX_SAFE_FRAMES)
    logger.info(f"LLM 입력 프레임: {num_frames_llm} (context limit: {MAX_SAFE_FRAMES})")

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = args.run_id or f"cached_{args.dataset}_fps{TARGET_FPS}_dur{CLIP_DURATION}s_nf{num_frames_llm}_{ts}"

    logger.info(f"모델 로딩: {args.model_path}")
    from transformers import AutoModel, AutoTokenizer
    model = AutoModel.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        trust_remote_code=True, low_cpu_mem_usage=True,
    ).to(args.device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    transform = build_video_transform(image_size=448)

    # ── QA 샘플 수집: duration >= 10s, 오름차순 정렬 ───────────────────
    total_needed = args.max_samples + args.warmup_samples
    qa_items: List[Dict[str, Any]] = collect_sorted_by_duration(
        args.dataset, args.data_root,
        min_duration=CLIP_DURATION,
        max_samples=total_needed,
    )
    if len(qa_items) < total_needed:
        logger.warning(f"유효 샘플 부족: {len(qa_items)} < {total_needed}")
    logger.info(f"QA 샘플: {len(qa_items)}개 (unique 비디오: {len(set(it['video_path'] for it in qa_items))}개)")

    # ── Warmup (앞쪽 warmup_samples개 사용) ────────────────────────────
    warmup_items  = qa_items[:args.warmup_samples]
    measure_items = qa_items[args.warmup_samples:args.warmup_samples + args.max_samples]

    logger.info(f"Warmup {len(warmup_items)}회 (Stage B)...")
    for item in warmup_items:
        stage_a_encode(item["video_path"], item["video_id"],
                       model, transform, args.embed_dir, args.device)
        stage_b_query(item, model, tokenizer, args.embed_dir,
                      num_frames_llm, args.device)

    # ── Stage A: 전체 비디오 인코딩 ──────────────────────────────────────
    # unique 비디오 처리 (동일 비디오에 여러 QA가 있을 수 있음)
    seen_videos: set = set()
    stage_a_results: List[Dict] = []

    logger.info("Stage A: 비디오 인코딩 시작...")
    pbar_a = tqdm(total=len(measure_items), desc=f"Stage A [{args.dataset}]", unit="video")
    for item in measure_items:
        vid = item["video_id"]
        if vid in seen_videos:
            pbar_a.update(1)
            continue
        seen_videos.add(vid)

        torch.cuda.empty_cache()
        result_a = stage_a_encode(
            item["video_path"], vid, model, transform, args.embed_dir, args.device
        )
        if result_a:
            stage_a_results.append(result_a)
        pbar_a.update(1)
    pbar_a.close()

    # Stage A timing CSV 저장
    a_csv_path = os.path.join(args.output_dir, f"{run_id}_stage_a.csv")
    a_cols = ["video_id", "status", "n_frames", "t_decode_ms", "t_preprocess_ms",
              "t_vit_ms", "t_pixel_shuffle_ms", "t_save_ms", "t_total_a_ms"]
    with open(a_csv_path, "w", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=a_cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(stage_a_results)
    logger.info(f"Stage A 완료 ({len(stage_a_results)}개 비디오) → {a_csv_path}")

    if args.stage_a_only:
        logger.info("--stage_a_only: Stage B 건너뜀. 종료.")
        return

    # ── Stage B: 쿼리 처리 ────────────────────────────────────────────────
    logger.info("Stage B: 쿼리 처리 시작...")
    b_results: List[Dict] = []
    n_ok = n_skip = n_oom = 0

    b_jsonl_path = os.path.join(args.output_dir, f"{run_id}.jsonl")
    with open(b_jsonl_path, "w") as jf:
        pbar_b = tqdm(total=len(measure_items), desc=f"Stage B [{args.dataset}]", unit="query")
        for item in measure_items:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            result_b = stage_b_query(
                item, model, tokenizer, args.embed_dir,
                num_frames_llm, args.device,
            )

            if result_b is None:
                n_skip += 1
            elif result_b.get("status") == "oom":
                n_oom += 1
            else:
                n_ok += 1
                result_b["run_id"]  = run_id
                result_b["vram_gb"] = torch.cuda.max_memory_allocated() / (1024**3)
                b_results.append(result_b)
                jf.write(json.dumps(result_b, ensure_ascii=False) + "\n")

            pbar_b.set_postfix(ok=n_ok, skip=n_skip, oom=n_oom)
            pbar_b.update(1)
        pbar_b.close()

    logger.info(f"Stage B 완료: ok={n_ok}, skip={n_skip}, oom={n_oom}")

    # Stage B timing CSV 저장
    b_csv_path = os.path.join(args.output_dir, f"{run_id}_stage_b.csv")
    b_cols = ["video_id", "task", "answer_gt", "response", "status",
              "n_frames_loaded", "n_frames_llm",
              "t_load_ms", "t_mlp_ms", "t_llm_ms", "t_total_b_ms", "vram_gb"]
    with open(b_csv_path, "w", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=b_cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(b_results)
    logger.info(f"Stage B timing CSV → {b_csv_path}")

    # ── Accuracy 평가 (MVBench MCQA) ──────────────────────────────────────
    if args.dataset == "mvbench" and b_results:
        import re
        from collections import defaultdict
        n_correct = 0
        task_correct: Dict[str, int] = defaultdict(int)
        task_total:   Dict[str, int] = defaultdict(int)
        for r in b_results:
            m = re.search(r"\b([ABCD])\b", r.get("response", "").upper())
            pred = m.group(1) if m else None
            gold = r.get("answer_gt", "").strip().upper()
            task = r.get("task", "unknown")
            task_total[task] += 1
            if pred == gold:
                n_correct += 1
                task_correct[task] += 1
        acc = n_correct / len(b_results)
        logger.info(f"\nMVBench Accuracy: {acc:.4f} ({n_correct}/{len(b_results)})")
        for t in sorted(task_total):
            logger.info(f"  {t}: {task_correct[t]/task_total[t]:.4f} ({task_correct[t]}/{task_total[t]})")

        # 정확도를 JSONL에 추가 저장
        acc_path = os.path.join(args.output_dir, f"{run_id}_accuracy.json")
        with open(acc_path, "w") as af:
            json.dump({
                "accuracy": acc, "n_correct": n_correct, "n_total": len(b_results),
                "task_breakdown": {t: task_correct[t] / task_total[t] for t in task_total},
            }, af, indent=2)
        logger.info(f"정확도 저장 → {acc_path}")

    # ── Summary ────────────────────────────────────────────────────────────
    if stage_a_results:
        real_a = [r for r in stage_a_results if r.get("status") == "encoded"]
        if real_a:
            avg_a = sum(r["t_total_a_ms"] for r in real_a) / len(real_a)
            logger.info(f"Stage A avg per video: {avg_a:.1f} ms")
    if b_results:
        avg_b = sum(r["t_total_b_ms"] for r in b_results) / len(b_results)
        logger.info(f"Stage B avg per query:  {avg_b:.1f} ms")


if __name__ == "__main__":
    main()
