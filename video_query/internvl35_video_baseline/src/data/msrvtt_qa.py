"""
MSRVTT-QA test split 로더.
공식 데이터셋 경로: <root>/videos/all/*.mp4 + test_qa.json
"""
from __future__ import annotations
import json
import os
from typing import Iterator, Dict, Any


def load_msrvtt_qa(
    data_root: str,
    split: str = "test",
) -> Iterator[Dict[str, Any]]:
    """
    Yields:
        {
            "video_id": str,      # "video1234"
            "video_path": str,    # 절대경로 mp4
            "question": str,
            "answer": str,        # ground-truth 텍스트 답변
            "qid": int,
        }
    """
    qa_path = os.path.join(data_root, f"{split}_qa.json")
    video_dir = os.path.join(data_root, "videos")

    with open(qa_path) as f:
        qa_list = json.load(f)

    skipped = 0
    for item in qa_list:
        video_id = f"video{item['video_id']}"
        video_path = os.path.join(video_dir, f"{video_id}.mp4")
        if not os.path.exists(video_path):
            skipped += 1
            continue
        yield {
            "video_id": video_id,
            "video_path": video_path,
            "question": item["question"],
            "answer": item["answer"],
            "qid": item.get("id", item.get("qid", -1)),
        }

    if skipped:
        import logging
        logging.getLogger(__name__).warning(f"MSRVTT-QA: {skipped}개 mp4 파일 없음")
