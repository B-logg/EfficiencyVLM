"""
§7 산출물 자동 생성: JSONL → Table 1~5 + Figure 1~6.

Usage:
    python scripts/05_aggregate.py \
        --timing_dir outputs/timings \
        --response_dir outputs/responses \
        --embed_dir outputs/embeddings \
        --output_dir outputs/report
"""
from __future__ import annotations
import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── JSONL 로드 ─────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


_TIMING_COND_RE = re.compile(r"^(sweep\d+_(?:msrvtt|mvbench)_fps\d+(?:_nf\d+)?)_(\d{8}_\d{6})$")

def _pick_latest_timings(timing_dir: str) -> set:
    """sweep 조건별 최신 파일만 반환 (non-sweep 파일은 전부 포함)."""
    best: Dict[str, tuple] = {}
    non_sweep = []
    for p in Path(timing_dir).glob("*.jsonl"):
        if p.stat().st_size == 0:
            continue
        m = _TIMING_COND_RE.match(p.stem)
        if not m:
            non_sweep.append(p)
            continue
        key, ts = m.group(1), m.group(2)
        if key not in best or ts > best[key][0]:
            best[key] = (ts, p)
    return set(v[1] for v in best.values()) | set(non_sweep)


def load_all_timings(timing_dir: str) -> pd.DataFrame:
    records = []
    for p in _pick_latest_timings(timing_dir):
        for row in load_jsonl(str(p)):
            row["_source"] = p.stem
            records.append(row)
    return pd.DataFrame(records) if records else pd.DataFrame()


def _stage_filter(df: pd.DataFrame, stage: str) -> pd.DataFrame:
    """stage 컬럼으로 필터링 (NaN 안전)."""
    if "stage" not in df.columns:
        return df.iloc[0:0]
    return df[df["stage"].fillna("") == stage]


_COND_RE = re.compile(r"^(sweep\d+_(?:msrvtt|mvbench)_fps\d+(?:_nf\d+)?)_(\d{8}_\d{6})$")

def _pick_latest_responses(response_dir: str) -> Dict[str, Path]:
    """조건별 최신 비어있지 않은 JSONL 파일만 반환."""
    best: Dict[str, tuple] = {}
    for p in Path(response_dir).glob("sweep*.jsonl"):
        if p.stat().st_size == 0:
            continue
        m = _COND_RE.match(p.stem)
        if not m:
            continue
        key, ts = m.group(1), m.group(2)
        if key not in best or ts > best[key][0]:
            best[key] = (ts, p)
    return {k: v[1] for k, v in best.items()}


def load_all_responses(response_dir: str) -> pd.DataFrame:
    records = []
    for key, p in _pick_latest_responses(response_dir).items():
        for row in load_jsonl(str(p)):
            row["_source"] = p.stem
            records.append(row)
    return pd.DataFrame(records) if records else pd.DataFrame()


# ── 메타 파싱 ──────────────────────────────────────────────────────────────

def parse_run_meta(run_id: str) -> Dict[str, Any]:
    """run_id에서 sweep/dataset/fps/nf 추출."""
    meta: Dict[str, Any] = {"run_id": run_id}
    m = re.search(r"sweep(\d+)", run_id)
    if m:
        meta["sweep"] = int(m.group(1))
    m = re.search(r"(msrvtt|mvbench)", run_id)
    if m:
        meta["dataset"] = m.group(1)
    m = re.search(r"fps(\d+)", run_id)
    if m:
        meta["target_fps"] = int(m.group(1))
    m = re.search(r"nf(\d+)", run_id)
    if m:
        meta["num_frames"] = int(m.group(1))
    return meta


# ── 통계 집계 ──────────────────────────────────────────────────────────────

def stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": float("nan"), "std": float("nan"), "p50": float("nan"), "p95": float("nan")}
    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }


# ── Table 1: Sweep 1 Stage A 지표 ─────────────────────────────────────────

def make_table1(timings: pd.DataFrame, output_dir: str):
    if timings.empty:
        print("[Table 1] 데이터 없음")
        return

    sweep1 = _stage_filter(timings[timings["_source"].str.contains("sweep1", na=False)], "A")

    rows = []
    for src, grp in sweep1.groupby("_source"):
        meta = parse_run_meta(src)
        frame_rows = grp[grp.get("event", pd.Series()).isna()]

        t_vit = frame_rows["t_vit_forward_ms"].dropna().tolist()
        t_pre = frame_rows["t_preprocess_ms"].dropna().tolist()
        t_dec = frame_rows["t_decode_ms"].dropna().tolist()

        video_rows = grp[grp.get("event", pd.Series()) == "video_done"]
        n_late = video_rows["n_late_frames"].sum() if "n_late_frames" in video_rows else 0
        fps_actual = None
        if "t_dataset_total_min" in grp.columns:
            dataset_done = grp[grp.get("event", pd.Series()) == "dataset_done"]
            if not dataset_done.empty:
                total_min = dataset_done["t_dataset_total_min"].iloc[-1]
                n_frames = len(frame_rows)
                fps_actual = n_frames / (total_min * 60) if total_min > 0 else None

        row = {
            "dataset": meta.get("dataset", "?"),
            "target_fps": meta.get("target_fps", "?"),
            "t_vit_mean_ms": stats(t_vit)["mean"],
            "t_vit_p95_ms": stats(t_vit)["p95"],
            "t_preprocess_mean_ms": stats(t_pre)["mean"],
            "t_decode_mean_ms": stats(t_dec)["mean"],
            "t_total_per_frame_ms": stats([a+b+c for a,b,c in zip(t_vit,t_pre,t_dec)])["mean"],
            "n_late_frames": int(n_late),
            "fps_actual": fps_actual,
        }
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(["dataset", "target_fps"])
    path = os.path.join(output_dir, "table1_sweep1_stageA.csv")
    df.to_csv(path, index=False)
    print(f"[Table 1] 저장: {path}")
    print(df.to_string(index=False))


# ── Table 3: Sweep 2 Stage B 지표 ─────────────────────────────────────────

def make_table3(timings: pd.DataFrame, output_dir: str):
    if timings.empty:
        print("[Table 3] 데이터 없음")
        return

    sweep2 = timings[timings["_source"].str.contains("sweep2", na=False)]
    rows = []
    for src, grp in sweep2.groupby("_source"):
        meta = parse_run_meta(src)
        b_rows = grp[grp.get("stage", pd.Series()) == "B"]

        t_query = b_rows["t_query_total_ms"].dropna().tolist()
        t_load = b_rows["t_load_pt_ms"].dropna().tolist()
        t_sub = b_rows["t_subsample_ms"].dropna().tolist()
        t_parse = b_rows["t_grounding_parse_ms"].dropna().tolist()

        row = {
            "dataset": meta.get("dataset", "?"),
            "num_frames": meta.get("num_frames", "?"),
            "t_query_mean_ms": stats(t_query)["mean"],
            "t_query_p50_ms": stats(t_query)["p50"],
            "t_query_p95_ms": stats(t_query)["p95"],
            "t_load_mean_ms": stats(t_load)["mean"],
            "t_subsample_mean_ms": stats(t_sub)["mean"],
            "t_parse_mean_ms": stats(t_parse)["mean"],
        }
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(["dataset", "num_frames"])
    path = os.path.join(output_dir, "table3_sweep2_stageB.csv")
    df.to_csv(path, index=False)
    print(f"[Table 3] 저장: {path}")
    print(df.to_string(index=False))


# ── Table 4: 정확도 ────────────────────────────────────────────────────────

def make_table4(responses: pd.DataFrame, output_dir: str):
    if responses.empty:
        print("[Table 4] 데이터 없음")
        return

    import re as _re

    def extract_option(text):
        if not isinstance(text, str):
            return None
        m = _re.search(r"\b([ABCD])\b", text.strip().upper())
        return m.group(1) if m else None

    rows = []
    for src, grp in responses.groupby("_source"):
        meta = parse_run_meta(src)
        dataset = meta.get("dataset", "?")

        if dataset == "msrvtt":
            def norm(t):
                # <frame>K</frame> 이전 텍스트만 추출 후 정규화
                t = str(t)
                frame_pos = t.find("<frame>")
                if frame_pos != -1:
                    t = t[:frame_pos]
                return _re.sub(r"\s+", " ", _re.sub(r"[^\w\s]", "", t.lower().strip()))
            n_correct = sum(norm(g) in norm(r) for r, g in zip(grp["response"], grp["answer_gt"]))
            acc = n_correct / len(grp)
            metric = "Contains"
        else:
            pred_opts = grp["response"].apply(extract_option)
            n_correct = (pred_opts == grp["answer_gt"].str.strip().str.upper()).sum()
            acc = n_correct / len(grp)
            metric = "ACC"

        frame_ok = grp["frame_ok"].mean() if "frame_ok" in grp.columns else float("nan")
        rows.append({
            "dataset": dataset,
            "sweep": meta.get("sweep", "?"),
            "target_fps": meta.get("target_fps", "-"),
            "num_frames": meta.get("num_frames", "-"),
            "metric": metric,
            "score": acc,
            "frame_grounding_ok_rate": frame_ok,
            "n": len(grp),
        })

    df = pd.DataFrame(rows).sort_values(["dataset", "sweep", "target_fps", "num_frames"])
    path = os.path.join(output_dir, "table4_accuracy.csv")
    df.to_csv(path, index=False)
    print(f"[Table 4] 저장: {path}")
    print(df.to_string(index=False))


# ── Table 5: 저장 용량 ─────────────────────────────────────────────────────

def make_table5(embed_dir: str, output_dir: str):
    rows = []
    for dataset in ["msrvtt", "mvbench"]:
        for fps_dir in sorted(Path(embed_dir).glob(f"{dataset}/*fps")):
            pt_files = list(fps_dir.glob("*.pt"))
            if not pt_files:
                continue
            sizes = [f.stat().st_size for f in pt_files]
            fps_label = fps_dir.name
            rows.append({
                "dataset": dataset,
                "fps_dir": fps_label,
                "n_videos": len(pt_files),
                "mb_per_video_mean": np.mean(sizes) / 1e6,
                "gb_total": sum(sizes) / 1e9,
            })

    if not rows:
        print("[Table 5] embedding 파일 없음")
        return

    df = pd.DataFrame(rows)
    path = os.path.join(output_dir, "table5_storage.csv")
    df.to_csv(path, index=False)
    print(f"[Table 5] 저장: {path}")
    print(df.to_string(index=False))


# ── Figure 1: target_fps vs fps_actual ────────────────────────────────────

def make_figure1(timings: pd.DataFrame, output_dir: str):
    sweep1_done = timings[
        (timings.get("event", pd.Series()) == "dataset_done") &
        timings["_source"].str.contains("sweep1", na=False)
    ] if not timings.empty else pd.DataFrame()

    if sweep1_done.empty:
        print("[Figure 1] No data")
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    for dataset in ["msrvtt", "mvbench"]:
        sub = sweep1_done[sweep1_done["_source"].str.contains(dataset, na=False)]
        if sub.empty:
            continue
        fps_list, actual_list = [], []
        for _, row in sub.iterrows():
            meta = parse_run_meta(row["_source"])
            if "target_fps" in meta and "t_dataset_total_min" in row:
                n_frames = row.get("n_ok", 0) * meta["target_fps"] * 10
                fps_actual = n_frames / (row["t_dataset_total_min"] * 60) if row["t_dataset_total_min"] > 0 else 0
                fps_list.append(meta["target_fps"])
                actual_list.append(fps_actual)

        if fps_list:
            ax.plot(fps_list, actual_list, "o-", label=dataset)

    max_fps = 32
    ax.plot([0, max_fps], [0, max_fps], "--", color="gray", label="ideal")
    ax.set_xlabel("target_fps")
    ax.set_ylabel("fps_actual")
    ax.set_title("Figure 1: Actual FPS vs Target FPS (diagonal = ideal)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = os.path.join(output_dir, "figure1_fps_actual.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Figure 1] 저장: {path}")


# ── Figure 2: target_fps vs t_total stacked bar ───────────────────────────

def make_figure2(timings: pd.DataFrame, output_dir: str):
    if timings.empty:
        print("[Figure 2] No data")
        return

    sweep1_frames = _stage_filter(timings[timings["_source"].str.contains("sweep1", na=False)], "A")

    agg = defaultdict(lambda: defaultdict(list))
    for _, row in sweep1_frames.iterrows():
        meta = parse_run_meta(row.get("_source", ""))
        fps = meta.get("target_fps")
        if fps is None:
            continue
        for col in ["t_decode_ms", "t_preprocess_ms", "t_vit_forward_ms"]:
            if col in row and pd.notna(row[col]):
                agg[fps][col].append(float(row[col]))

    fps_vals = sorted(agg.keys())
    if not fps_vals:
        print("[Figure 2] No data")
        return

    cols = ["t_decode_ms", "t_preprocess_ms", "t_vit_forward_ms"]
    labels = ["decode", "preprocess", "ViT"]
    means = {c: [np.mean(agg[f][c]) if agg[f][c] else 0 for f in fps_vals] for c in cols}

    fig, ax = plt.subplots(figsize=(7, 5))
    bottom = np.zeros(len(fps_vals))
    for c, label in zip(cols, labels):
        vals = np.array(means[c])
        ax.bar([str(f) for f in fps_vals], vals, bottom=bottom, label=label)
        bottom += vals

    ax.set_xlabel("target_fps")
    ax.set_ylabel("ms/frame")
    ax.set_title("Figure 2: Stage A per-frame latency breakdown")
    ax.legend()
    path = os.path.join(output_dir, "figure2_stageA_breakdown.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Figure 2] 저장: {path}")


# ── Figure 3: num_frames vs t_query_total ─────────────────────────────────

def make_figure3(timings: pd.DataFrame, output_dir: str):
    if timings.empty:
        print("[Figure 3] No data")
        return

    sweep2 = _stage_filter(timings[timings["_source"].str.contains("sweep2", na=False)], "B")

    fig, ax = plt.subplots(figsize=(6, 5))
    for dataset in ["msrvtt", "mvbench"]:
        sub = sweep2[sweep2["_source"].str.contains(dataset, na=False)]
        nf_vals, p50s, p95s = [], [], []
        for src, grp in sub.groupby("_source"):
            meta = parse_run_meta(src)
            nf = meta.get("num_frames")
            if nf is None:
                continue
            vals = grp["t_query_total_ms"].dropna().tolist()
            if vals:
                nf_vals.append(nf)
                p50s.append(np.percentile(vals, 50))
                p95s.append(np.percentile(vals, 95))

        if nf_vals:
            order = np.argsort(nf_vals)
            nf_sorted = [nf_vals[i] for i in order]
            p50_sorted = [p50s[i] for i in order]
            p95_sorted = [p95s[i] for i in order]
            color = ax._get_lines.get_next_color()
            ax.plot(nf_sorted, p50_sorted, "o-", color=color, label=f"{dataset} p50")
            ax.plot(nf_sorted, p95_sorted, "s--", color=color, alpha=0.6, label=f"{dataset} p95")

    ax.set_xlabel("num_frames (N)")
    ax.set_ylabel("t_query_total (ms)")
    ax.set_title("Figure 3: num_frames vs query latency (p50/p95)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = os.path.join(output_dir, "figure3_query_latency.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Figure 3] 저장: {path}")


# ── Figure 4: num_frames vs accuracy (Pareto) ─────────────────────────────

def make_figure4(timings: pd.DataFrame, responses: pd.DataFrame, output_dir: str):
    if responses.empty or timings.empty:
        print("[Figure 4] No data")
        return

    import re as _re

    def get_acc(grp, dataset):
        if dataset == "msrvtt":
            def extract_ans(t):
                t = str(t)
                pos = t.find("<frame>")
                if pos != -1:
                    t = t[:pos]
                return _re.sub(r"\s+", " ", _re.sub(r"[^\w\s]", "", t.lower().strip()))
            def norm_gold(t):
                return _re.sub(r"\s+", " ", _re.sub(r"[^\w\s]", "", str(t).lower().strip()))
            n = sum(norm_gold(g) in extract_ans(r) for r, g in zip(grp["response"], grp["answer_gt"]))
            return n / len(grp)
        else:
            def extract_opt(t):
                m = _re.search(r"\b([ABCD])\b", str(t).strip().upper())
                return m.group(1) if m else None
            preds = grp["response"].apply(extract_opt)
            return (preds == grp["answer_gt"].str.strip().str.upper()).mean()

    sweep2_timing = _stage_filter(timings[timings["_source"].str.contains("sweep2", na=False)], "B")

    fig, ax = plt.subplots(figsize=(6, 5))
    for dataset in ["msrvtt", "mvbench"]:
        nf_list, acc_list, lat_list = [], [], []
        for src, grp_t in sweep2_timing.groupby("_source"):
            if dataset not in src:
                continue
            meta = parse_run_meta(src)
            nf = meta.get("num_frames")
            if nf is None:
                continue
            grp_r = responses[responses["_source"] == src]
            if grp_r.empty:
                continue
            acc = get_acc(grp_r, dataset)
            lat = grp_t["t_query_total_ms"].median()
            nf_list.append(nf)
            acc_list.append(acc)
            lat_list.append(lat)

        if nf_list:
            sc = ax.scatter(lat_list, acc_list, label=dataset, zorder=3)
            for nf, lat, acc in zip(nf_list, lat_list, acc_list):
                ax.annotate(f"N={nf}", (lat, acc), textcoords="offset points", xytext=(4, 4))

    ax.set_xlabel("t_query_total p50 (ms)")
    ax.set_ylabel("Accuracy / EM")
    ax.set_title("Figure 4: Latency-Accuracy Pareto (Sweep 2)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = os.path.join(output_dir, "figure4_pareto.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Figure 4] 저장: {path}")


# ── Figure 5: 저장 용량 vs target_fps ─────────────────────────────────────

def make_figure5(embed_dir: str, output_dir: str):
    data = defaultdict(lambda: defaultdict(list))
    for dataset in ["msrvtt", "mvbench"]:
        for fps_dir in sorted(Path(embed_dir).glob(f"{dataset}/*fps")):
            fps_match = re.search(r"(\d+)fps$", fps_dir.name)
            if not fps_match:
                continue
            fps = int(fps_match.group(1))
            sizes = [f.stat().st_size for f in fps_dir.rglob("*.pt")]
            if sizes:
                data[dataset][fps] = sizes

    fig, ax = plt.subplots(figsize=(6, 5))
    for dataset, fps_dict in data.items():
        fps_vals = sorted(fps_dict.keys())
        mb_means = [np.mean(fps_dict[f]) / 1e6 for f in fps_vals]
        if len(fps_vals) == 1:
            ax.scatter(fps_vals, mb_means, label=dataset, zorder=3, s=80)
        else:
            ax.plot(fps_vals, mb_means, "o-", label=dataset)

    ax.set_xlabel("target_fps")
    ax.set_ylabel("MB/video (mean)")
    ax.set_title("Figure 5: Storage Size vs target_fps")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = os.path.join(output_dir, "figure5_storage_vs_fps.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Figure 5] 저장: {path}")


# ── Figure 6: Stage B latency 분해 ────────────────────────────────────────

def make_figure6(timings: pd.DataFrame, output_dir: str):
    if timings.empty:
        print("[Figure 6] No data")
        return

    sweep2 = _stage_filter(timings[timings["_source"].str.contains("sweep2", na=False)], "B")

    cols_map = {
        "t_load_pt_ms": "load_pt",
        "t_subsample_ms": "subsample",
        "t_query_total_ms": "mlp1+prefill+decode",
        "t_grounding_parse_ms": "parse",
        "t_frame_reextract_ms": "reextract",
    }
    existing = [c for c in cols_map if c in sweep2.columns]
    if not existing:
        print("[Figure 6] No columns")
        return

    means = {cols_map[c]: sweep2[c].dropna().mean() for c in existing}
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(list(means.keys()), list(means.values()))
    ax.set_xlabel("ms (mean)")
    ax.set_title("Figure 6: Stage B Latency Breakdown")
    ax.grid(True, alpha=0.3, axis="x")
    path = os.path.join(output_dir, "figure6_stageB_breakdown.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Figure 6] 저장: {path}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timing_dir", default="outputs/timings")
    parser.add_argument("--response_dir", default="outputs/responses")
    parser.add_argument("--embed_dir", default="outputs/embeddings")
    parser.add_argument("--output_dir", default="outputs/report")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=== JSONL 로딩 ===")
    timings = load_all_timings(args.timing_dir)
    responses = load_all_responses(args.response_dir)
    print(f"timing rows: {len(timings)}, response rows: {len(responses)}")

    print("\n=== 표 생성 ===")
    make_table1(timings, args.output_dir)
    make_table3(timings, args.output_dir)
    make_table4(responses, args.output_dir)
    make_table5(args.embed_dir, args.output_dir)

    print("\n=== 그림 생성 ===")
    make_figure1(timings, args.output_dir)
    make_figure2(timings, args.output_dir)
    make_figure3(timings, args.output_dir)
    make_figure4(timings, responses, args.output_dir)
    make_figure5(args.embed_dir, args.output_dir)
    make_figure6(timings, args.output_dir)

    print(f"\n모든 산출물 저장 완료: {args.output_dir}/")


if __name__ == "__main__":
    main()
