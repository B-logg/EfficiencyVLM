"""
§5: GPU 시간 측정 — torch.cuda.Event 기반.
time.time()은 async kernel 때문에 부정확하므로 사용 금지.
"""
from __future__ import annotations
import torch


class CudaTimer:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self._start_event: torch.cuda.Event | None = None

    def start(self):
        if "cuda" not in self.device:
            self._cpu_start = __import__("time").perf_counter()
            return
        self._start_event = torch.cuda.Event(enable_timing=True)
        self._end_event = torch.cuda.Event(enable_timing=True)
        self._start_event.record()

    def stop(self) -> float:
        """Returns elapsed milliseconds."""
        if "cuda" not in self.device:
            return (__import__("time").perf_counter() - self._cpu_start) * 1000
        self._end_event.record()
        torch.cuda.synchronize()
        return self._start_event.elapsed_time(self._end_event)
