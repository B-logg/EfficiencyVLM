"""
§5: 한 줄 = 한 이벤트 JSONL 로거.
"""
from __future__ import annotations
import json
import os
import time
from typing import Any, Dict


class JsonlLogger:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._path = path
        self._f = open(path, "a", buffering=1)

    def log(self, event: Dict[str, Any]):
        event.setdefault("wall_time", time.time())
        self._f.write(json.dumps(event) + "\n")

    def close(self):
        self._f.close()

    def __del__(self):
        try:
            self._f.close()
        except Exception:
            pass
