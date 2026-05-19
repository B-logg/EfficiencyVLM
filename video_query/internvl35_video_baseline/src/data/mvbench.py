"""
MVBench 20 sub-task 통합 로더.
실제 구조:
  json/<task>.json  →  {"video": "166583.webm", "candidates": [...], "answer": "A"}
  video/            →  ssv2_video/, clevrer/, FunQA_test/, ... (source별 하위폴더)

video 필드는 파일명만 있고 경로 정보가 없으므로,
시작 시 video/ 전체를 스캔해서 filename→path 인덱스를 구축한다.
"""
from __future__ import annotations
import json
import os
import logging
from typing import Iterator, Dict, Any, List, Optional

logger = logging.getLogger(__name__)

MVBENCH_TASKS = [
    "action_antonym",
    "action_count",
    "action_localization",
    "action_prediction",
    "action_sequence",
    "character_order",
    "counterfactual_inference",
    "egocentric_navigation",
    "episodic_reasoning",
    "fine_grained_action",
    "fine_grained_pose",
    "moving_attribute",
    "moving_count",
    "moving_direction",
    "object_existence",
    "object_interaction",
    "object_shuffle",
    "scene_transition",
    "state_change",
    "unexpected_action",
]


def build_video_index(video_root: str) -> Dict[str, str]:
    """
    video/ 하위 모든 파일을 재귀 스캔해서 {filename: absolute_path} 인덱스 생성.
    동일 파일명이 여러 곳에 있으면 마지막 발견 경로로 덮어씀 (경고 출력).
    """
    index: Dict[str, str] = {}
    duplicates = 0
    for root, _, files in os.walk(video_root):
        for fname in files:
            if fname.startswith("."):
                continue
            if fname in index:
                duplicates += 1
            index[fname] = os.path.join(root, fname)

    logger.info(f"MVBench 영상 인덱스: {len(index)}개 파일 ({duplicates}개 중복 파일명)")
    return index


def load_mvbench(
    data_root: str,
    tasks: Optional[List[str]] = None,
    video_index: Optional[Dict[str, str]] = None,
) -> Iterator[Dict[str, Any]]:
    """
    Yields:
        {
            "video_id": str,       # "<task>/<파일명 확장자 제외>"
            "video_path": str,     # 절대경로
            "task": str,
            "question": str,
            "options": List[str],  # ["A. ...", "B. ...", ...]
            "answer": str,         # "A" / "B" / "C" (선택지 인덱스→문자 변환)
            "qid": str,
        }
    """
    tasks = tasks or MVBENCH_TASKS
    json_dir = os.path.join(data_root, "json")
    video_root = os.path.join(data_root, "video")

    # 영상 인덱스 (한 번만 구축)
    if video_index is None:
        video_index = build_video_index(video_root)

    skipped = 0
    for task in tasks:
        json_path = os.path.join(json_dir, f"{task}.json")
        if not os.path.exists(json_path):
            logger.warning(f"MVBench: {json_path} 없음, skip")
            continue

        with open(json_path) as f:
            items = json.load(f)

        for idx, item in enumerate(items):
            video_fname = item.get("video", "")
            video_path = video_index.get(video_fname)

            if video_path is None:
                skipped += 1
                if skipped <= 5:
                    logger.debug(f"MVBench: '{video_fname}' 인덱스에 없음 (task={task})")
                continue

            # candidates → A/B/C/D 선택지
            candidates = item.get("candidates", item.get("options", []))
            options = [f"{chr(65+i)}. {opt}" for i, opt in enumerate(candidates)]

            # answer: 문자열 "A"/"B"/... 또는 후보 텍스트인 경우 변환
            raw_answer = item.get("answer", "")
            answer = _normalize_answer(raw_answer, candidates)

            video_id = f"{task}/{os.path.splitext(video_fname)[0]}"

            yield {
                "video_id": video_id,
                "video_path": video_path,
                "task": task,
                "question": item["question"],
                "options": options,
                "answer": answer,
                "qid": f"{task}_{idx}",
            }

    if skipped:
        logger.warning(f"MVBench: 총 {skipped}개 영상 파일 인덱스에서 찾지 못해 skip")


def _normalize_answer(raw: str, candidates: List[str]) -> str:
    """
    answer가 이미 'A'/'B'/... 이면 그대로 반환.
    candidates 텍스트 자체인 경우 인덱스→문자로 변환.
    """
    raw = raw.strip()
    if len(raw) == 1 and raw.upper() in "ABCDE":
        return raw.upper()
    # 후보 텍스트와 비교
    for i, cand in enumerate(candidates):
        if raw == cand:
            return chr(65 + i)
    return raw  # 변환 불가 시 원본 반환
