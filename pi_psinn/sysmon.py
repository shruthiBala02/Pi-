"""Memory, temperature and environment readings of the Pi. Linux only: a missing source stops with a message, nothing is
substituted. The parsers take text so they can be tested on recorded samples.

Units: memory in MB = 10^6 bytes (the kernel reports kB = 1024 bytes); clock in Hz.
Throttle word (`vcgencmd get_throttled`): bits 0-3 are the current state (under-voltage, frequency capped, throttled, soft
temperature limit); bits 16-19 are the same events having happened since boot and stay set.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import threading
import time
from pathlib import Path

CURRENT_BITS = 0xF
HISTORY_BITS = 0xF0000


def parse_status_kb(text: str, key: str) -> float:
    """Value of a `Key:   123 kB` line of /proc/<pid>/status, in MB (10^6 bytes)."""
    for line in text.splitlines():
        if line.startswith(key + ":"):
            return float(line.split()[1]) * 1024.0 / 1e6
    raise ValueError(f"{key} not found in the process status text")


def _status_text() -> str:
    p = Path("/proc/self/status")
    if not p.exists():
        raise RuntimeError("/proc/self/status is missing: memory is read on Linux (the Pi) only")
    return p.read_text()


def rss_mb() -> float:
    return parse_status_kb(_status_text(), "VmRSS")


def peak_rss_mb() -> float:
    return parse_status_kb(_status_text(), "VmHWM")


def parse_millicelsius(text: str) -> float:
    return int(text.strip()) / 1000.0


def parse_khz_to_hz(text: str) -> int:
    """/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq holds kHz."""
    return int(text.strip()) * 1000


def parse_vcgencmd_throttled(text: str) -> str:
    """`throttled=0x0` -> `0x0`"""
    return text.strip().split("=")[1]


def _vcgencmd(*args: str) -> str:
    exe = shutil.which("vcgencmd")
    if exe is None:
        raise RuntimeError("vcgencmd is missing: the throttle flags are read on the Pi (Raspberry Pi OS)")
    return subprocess.run([exe, *args], capture_output=True, text=True, check=True).stdout


def _sys_text(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise RuntimeError(f"{p} is missing: temperature and clock are read on the Pi")
    return p.read_text()


def read_temp_c() -> float:
    return parse_millicelsius(_sys_text("/sys/class/thermal/thermal_zone0/temp"))


def read_clock_hz() -> int:
    return parse_khz_to_hz(_sys_text("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"))


def read_throttled() -> str:
    return parse_vcgencmd_throttled(_vcgencmd("get_throttled"))


def throttled_during(start: str, end: str) -> bool:
    """True when the run saw a throttle or under-voltage: a current bit set at either end, or a history bit that is new
    at the end (the history bits stay set once an event has happened, so only a change counts)."""
    s, e = int(start, 16), int(end, 16)
    return bool((s & CURRENT_BITS) or (e & CURRENT_BITS) or ((e & HISTORY_BITS) & ~(s & HISTORY_BITS)))


class ThermalLog:
    """Samples (time_s, temp_c, clock_hz) every `interval_s` in a thread between start() and stop() from sysfs files (cheap,
    no extra process), and reads the throttle word with vcgencmd once at start and once at stop."""

    def __init__(self, interval_s: float, readers: dict | None = None):
        r = readers or {}
        self.interval = interval_s
        self.read_temp = r.get("temp", read_temp_c)
        self.read_clock = r.get("clock", read_clock_hz)
        self.read_throttled = r.get("throttled", read_throttled)
        self.rows: list[tuple] = []
        self.throttled_start = ""
        self._stop = threading.Event()
        self._thread = None
        self._t0 = 0.0
        self.error: Exception | None = None

    def _sample(self) -> None:
        self.rows.append((round(time.perf_counter() - self._t0, 3), self.read_temp(), self.read_clock()))

    def _loop(self) -> None:
        try:
            while not self._stop.wait(self.interval):
                self._sample()
        except Exception as exc:    # noqa: BLE001  re-raised in stop()
            self.error = exc

    def start(self) -> None:
        self._t0 = time.perf_counter()
        self.throttled_start = self.read_throttled()
        self._sample()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        self._stop.set()
        self._thread.join()
        if self.error is not None:
            raise self.error
        self._sample()
        end = self.read_throttled()
        temps = [r[1] for r in self.rows]
        clocks = [r[2] for r in self.rows]
        return {"temp_c_start": temps[0], "temp_c_max": max(temps), "temp_c_end": temps[-1],
                "throttled_start": self.throttled_start, "throttled_end": end,
                "throttled": end, "throttled_any": throttled_during(self.throttled_start, end),
                "arm_clock_hz_start": clocks[0], "arm_clock_hz_end": clocks[-1], "arm_clock_hz_min": min(clocks),
                "arm_clock_hz": min(clocks)}


def env_info() -> dict:
    import numpy
    import torch
    info = {"machine": platform.machine(), "platform": platform.platform(), "python": platform.python_version(),
            "torch": torch.__version__, "numpy": numpy.__version__, "torch_threads": torch.get_num_threads()}
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.exists():
        raise RuntimeError("/proc/cpuinfo is missing: the environment is recorded on Linux (the Pi)")
    lines = cpuinfo.read_text().splitlines()
    feats = next((ln.split(":", 1)[1].split() for ln in lines if ln.startswith("Features")), None)
    model = next((ln.split(":", 1)[1].strip() for ln in lines if ln.startswith("Model")), None)
    if feats is None or model is None:
        raise RuntimeError("/proc/cpuinfo has no Features or Model line: this is not the Raspberry Pi (arm64 Linux); "
                           "the environment record states the CPU features and is not guessed")
    info["cpu_model"] = model
    info["asimddp"] = "asimddp" in feats
    info["asimdhp"] = "asimdhp" in feats
    return info
