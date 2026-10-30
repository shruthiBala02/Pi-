"""Memory, temperature and environment readings of the Pi (code/pi_ablations/sysmon.py without the power log: no energy
is measured here, so nothing piles up in memory while a row runs). Linux only: a missing source stops with a message,
nothing is substituted. The parsers take text so they can be tested on samples.

Inference memory (`inference_peak_bytes`): the most memory one inference adds on top of what the process holds before
it starts, read from the kernel: the resident set before the pass, and its peak (VmHWM) during the pass after resetting
the peak (`/proc/self/clear_refs` = 5, Linux 4.0+). Two settings keep the reading to the pass itself:
  - glibc gives every block of at least `MMAP_THRESHOLD` bytes its own mapping and returns it on free (mallopt), and
    `malloc_trim` hands freed memory back before each reading, so memory freed earlier is not reused unseen;
  - the warm-up passes run first, so PyTorch's one-time set-up (thread pool, kernels) is not counted.
The weights (loaded before) and the batch-size sweep (none here) are not in it.

Throttle word (`vcgencmd get_throttled`): bits 0-3 are the current state (under-voltage, frequency capped, throttled, soft
temperature limit); bits 16-19 are the same events having happened since boot and stay set.
"""

from __future__ import annotations

import ctypes
import gc
import platform
import shutil
import statistics
import subprocess
import threading
import time
from pathlib import Path

CURRENT_BITS = 0xF
HISTORY_BITS = 0xF0000
MMAP_THRESHOLD = 64 * 1024
_M_TRIM_THRESHOLD, _M_MMAP_THRESHOLD = -1, -3


def parse_status_bytes(text: str, key: str) -> int:
    """Value of a `Key:   123 kB` line of /proc/<pid>/status, in bytes (the kernel's kB are 1024 bytes)."""
    for line in text.splitlines():
        if line.startswith(key + ":"):
            return int(line.split()[1]) * 1024
    raise ValueError(f"{key} not found in the process status text")


def _status_text() -> str:
    p = Path("/proc/self/status")
    if not p.exists():
        raise RuntimeError("/proc/self/status is missing: memory is read on Linux (the Pi) only")
    return p.read_text()


def rss_bytes() -> int:
    return parse_status_bytes(_status_text(), "VmRSS")


def peak_rss_bytes() -> int:
    return parse_status_bytes(_status_text(), "VmHWM")


def _libc():
    try:
        return ctypes.CDLL("libc.so.6")
    except OSError as e:
        raise RuntimeError(f"glibc (libc.so.6) is not loadable: the memory reading needs Linux with glibc ({e})") from e


def setup_allocator() -> None:
    """Large blocks get their own mapping, returned to the kernel on free (call once, before any tensor is made)."""
    libc = _libc()
    if libc.mallopt(_M_MMAP_THRESHOLD, MMAP_THRESHOLD) != 1 or libc.mallopt(_M_TRIM_THRESHOLD, MMAP_THRESHOLD) != 1:
        raise RuntimeError("mallopt refused the mmap / trim thresholds: the memory reading would not be the pass's own")


def reset_peak() -> None:
    p = Path("/proc/self/clear_refs")
    if not p.exists():
        raise RuntimeError("/proc/self/clear_refs is missing: the peak memory cannot be reset on this kernel")
    p.write_text("5")


def inference_peak_bytes(fn) -> int:
    """The peak resident memory fn() adds on top of the memory before it starts (its result included, as it is made
    during the pass)."""
    gc.collect()
    _libc().malloc_trim(0)
    before = rss_bytes()
    reset_peak()
    out = fn()
    peak = peak_rss_bytes()
    del out
    return peak - before


def peak_reset_works() -> tuple[bool, str]:
    """Whether resetting the peak works here: a 64 MB block raises the peak, and after freeing it and resetting, the
    peak is back near the resident set."""
    import numpy as np
    setup_allocator()
    big = np.ones(64 * 2 ** 20 // 8)
    high = peak_rss_bytes()
    del big
    gc.collect()
    _libc().malloc_trim(0)
    try:
        reset_peak()
    except (RuntimeError, OSError) as e:
        return False, str(e)
    after, now = peak_rss_bytes(), rss_bytes()
    ok = after - now < 8 * 2 ** 20 and high - after > 48 * 2 ** 20
    return ok, f"peak with a 64 MB block {high / 1e6:.1f} MB, after free + reset {after / 1e6:.1f} MB (resident {now / 1e6:.1f} MB)"


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


def cool_down(limit_c: float, max_wait_s: float, read=read_temp_c) -> tuple[float, float]:
    """Wait until the CPU is at most limit_c degrees, at most max_wait_s seconds. Returns (seconds waited, temperature
    at the end); a CPU still too hot is recorded, not hidden."""
    t0 = time.perf_counter()
    temp = read()
    while temp > limit_c and time.perf_counter() - t0 < max_wait_s:
        time.sleep(2.0)
        temp = read()
    return time.perf_counter() - t0, temp


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
                "throttled_any": throttled_during(self.throttled_start, end),
                "arm_clock_hz_min": min(clocks), "arm_clock_hz_median": statistics.median(clocks)}


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
    for flag in ("asimdhp", "fphp"):                                    # fp16 vector / scalar arithmetic
        info[flag] = flag in feats
    return info
