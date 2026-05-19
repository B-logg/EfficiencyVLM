"""
§5: GPU clock 고정 — 실험 재현성을 위해 nvidia-smi로 고정.
sudo 권한 필요.
"""
from __future__ import annotations
import subprocess
import logging

logger = logging.getLogger(__name__)


def _get_max_clock() -> int:
    """nvidia-smi에서 현재 GPU의 최대 그래픽 클럭(MHz)을 조회."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=clocks.max.graphics", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi 실패: {result.stderr}")
    return int(result.stdout.strip().split("\n")[0])


def lock_gpu_clocks(clock_mhz: int | None = None):
    """GPU 클럭을 최대값으로 고정."""
    if clock_mhz is None:
        clock_mhz = _get_max_clock()
    result = subprocess.run(
        ["sudo", "nvidia-smi", "-lgc", str(clock_mhz)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        logger.warning(f"GPU 클럭 고정 실패 (sudo 권한 확인): {result.stderr}")
    else:
        logger.info(f"GPU 클럭 고정 완료: {clock_mhz} MHz")


def unlock_gpu_clocks():
    """GPU 클럭 제한 해제."""
    result = subprocess.run(
        ["sudo", "nvidia-smi", "-rgc"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        logger.warning(f"GPU 클럭 해제 실패: {result.stderr}")
    else:
        logger.info("GPU 클럭 해제 완료")


def assert_exclusive_gpu():
    """다른 프로세스가 GPU를 점유하고 있지 않은지 확인."""
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    procs = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
    import os
    own_pid = str(os.getpid())
    others = [p for p in procs if not p.startswith(own_pid)]
    if others:
        logger.warning(f"GPU를 점유하는 다른 프로세스 감지: {others}")
    else:
        logger.info("GPU 단독 사용 확인됨")
