"""
MVBench 20 sub-task 통합 로더.
공식 구조: <root>/json/<task>.json + <root>/video/<task>/<clip>.mp4
답변은 MCQA 형식 (A/B/C/D 선택지).
"""
from __future__ import annotations
import json
import os
from typing import Iterator, Dict, Any, List


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


def load_mvbench(
    data_root: str,
    tasks: List[str] | None = None,
) -> Iterator[Dict[str, Any]]:
    """
    Yields:
        {
            "video_id": str,       # "<task>/<clip_id>" unique key
            "video_path": str,
            "task": str,
            "question": str,
            "options": List[str],  # ["A. ...", "B. ...", ...]
            "answer": str,         # "A" / "B" / "C" / "D"
            "qid": str,
        }
    """
    tasks = tasks or MVBENCH_TASKS
    json_dir = os.path.join(data_root, "json")
    video_dir = os.path.join(data_root, "video")

    skipped = 0
    for task in tasks:
        json_path = os.path.join(json_dir, f"{task}.json")
        if not os.path.exists(json_path):
            import logging
            logging.getLogger(__name__).warning(f"MVBench: {json_path} 없음, skip")
            continue

        with open(json_path) as f:
            items = json.load(f)

        for item in items:
            video_file = item.get("video", "")
            video_path = os.path.join(video_dir, task, video_file)
            if not os.path.exists(video_path):
                skipped += 1
                continue

            video_id = f"{task}/{os.path.splitext(video_file)[0]}"
            options_raw = item.get("candidates", item.get("options", []))
            options = [f"{chr(65+i)}. {opt}" for i, opt in enumerate(options_raw)]

            yield {
                "video_id": video_id,
                "video_path": video_path,
                "task": task,
                "question": item["question"],
                "options": options,
                "answer": item["answer"],
                "qid": f"{task}_{item.get('id', video_file)}",
            }

    if skipped:
        import logging
        logging.getLogger(__name__).warning(f"MVBench: {skipped}개 mp4 파일 없음")
