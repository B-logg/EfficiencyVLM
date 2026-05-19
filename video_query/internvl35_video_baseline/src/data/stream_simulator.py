"""
§2: 스트림 부하 시뮬레이션.
각 프레임 처리 후 time.sleep(max(0, 1/target_fps - elapsed))로 다음 프레임 yield 간격 강제.
"""
from __future__ import annotations
import time


class StreamSimulator:
    def __init__(self, target_fps: int):
        self.interval = 1.0 / target_fps
        self._last_time: float | None = None

    def start(self):
        self._last_time = time.perf_counter()

    def sleep_until_next(self, start_time: float | None = None) -> bool:
        """
        다음 프레임까지 남은 시간만큼 sleep.
        Returns:
            True if late (elapsed > interval), False otherwise
        """
        if self._last_time is None:
            self._last_time = time.perf_counter()
            return False

        now = time.perf_counter()
        elapsed = now - self._last_time
        sleep_time = self.interval - elapsed

        late = sleep_time < 0
        if sleep_time > 0:
            time.sleep(sleep_time)

        self._last_time = time.perf_counter()
        return late
