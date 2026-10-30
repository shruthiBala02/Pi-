"""Memory, temperature, power and environment readings of the Pi (exp17's code/pi/sysmon.py, plus the power log of exp25).
Linux only: a missing source stops with a message, nothing is substituted. The parsers take text so they can be tested
on samples.

Power: the Pi 5's power chip, `vcgencmd pmic_read_adc`, gives a current (`<RAIL>_A current(n)=...A`) and a voltage
(`<RAIL>_V volt(n)=...V`) per supply line; the board power is the sum of V x A over the lines that have both. The 5 V
input lines (EXT5V, USB, HATs) have no current reading, so the sum misses them and reads below a USB meter
(jfikar/RPi5-power); stated in the tables, no correction applied.

Units: memory in MB = 10^6 bytes (the kernel reports kB = 1024 bytes); clock in Hz.
Throttle word (`vcgencmd get_throttled`): bits 0-3 are the current state (under-voltage, frequency capped, throttled, soft
temperature limit); bits 16-19 are the same events having happened since boot and stay set.
"""

from __future__ import annotations

import platform
import re
import shutil
import statistics
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


_PMIC = re.compile(r"^\s*(\S+)_([AV])\s+(?:current|volt)\(\d+\)=([-+0-9.eE]+)([AV])\s*$")


def parse_pmic(text: str) -> dict[str, tuple[float, float]]:
    """{rail: (volts, amps)} for every supply line of a `vcgencmd pmic_read_adc` text that has both a voltage and a
    current. Stops when no line has both (the chip was not read, or its output changed)."""
    volts: dict[str, float] = {}
    amps: dict[str, float] = {}
    for line in text.splitlines():
        m = _PMIC.match(line)
        if not m:
            continue
        rail, kind, value, unit = m.groups()
        if kind != unit:
            raise ValueError(f"pmic line with a mismatched unit: {line.strip()!r}")
        (amps if kind == "A" else volts)[rail] = float(value)
    both = {r: (volts[r], amps[r]) for r in sorted(set(volts) & set(amps))}
    if not both:
        raise ValueError("no supply line with both a voltage and a current in the pmic_read_adc text")
    return both


def read_pmic() -> str:
    return _vcgencmd("pmic_read_adc")


def total_watts(rails: dict[str, tuple[float, float]]) -> float:
    return sum(v * a for v, a in rails.values())


class PowerLog:
    """Reads the power chip in a thread between start() and stop(), every `interval_s` (0: back to back), each reading
    tagged with the phase the main thread set (mark()): rows (time_s, phase, {rail: (V, A)}, total W). Every reading
    must name the same rails as the first; a reading that fails stops the run at stop()."""

    def __init__(self, interval_s: float, reader=None):
        self.interval = interval_s
        self.read = reader or read_pmic
        self.rows: list[tuple] = []
        self.rails: list[str] = []
        self.phase = "start"
        self._stop = threading.Event()
        self._running = threading.Event()          # cleared while paused: the thread blocks on it (no spinning)
        self._running.set()
        self._lock = threading.Lock()              # held while a reading is under way
        self._thread = None
        self._t0 = 0.0
        self.error: Exception | None = None

    def mark(self, phase: str) -> None:
        self.phase = phase

    def pause(self) -> None:
        """No readings until resume() (the latency passes are timed without the sampler, as in exp17). Returns once a
        reading under way has finished; the thread then sleeps on an event, using no CPU."""
        self._running.clear()
        with self._lock:
            pass

    def resume(self) -> None:
        self._running.set()

    def now(self) -> float:
        return time.perf_counter() - self._t0

    def _sample(self) -> None:
        phase = self.phase
        r = parse_pmic(self.read())
        if not self.rails:
            self.rails = list(r)
        elif list(r) != self.rails:
            raise ValueError(f"the power chip's lines changed during the run: {list(r)} vs {self.rails}")
        self.rows.append((round(self.now(), 4), phase, r, total_watts(r)))

    def _loop(self) -> None:
        try:
            while not self._stop.wait(self.interval):
                self._running.wait()                   # blocks while paused
                with self._lock:
                    if self._stop.is_set() or not self._running.is_set():
                        continue
                    self._sample()
        except Exception as exc:    # noqa: BLE001  re-raised in stop()
            self.error = exc

    def start(self) -> None:
        self._t0 = time.perf_counter()
        self._sample()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._running.set()                            # wake a paused thread so it can end
        self._thread.join()
        if self.error is not None:
            raise self.error

    def check(self) -> None:
        """Raise the error a reading in the thread met, if any (else a later step would stop on its symptom)."""
        if self.error is not None:
            raise self.error

    def watts(self, phase: str) -> list[float]:
        """The readings of a phase; raises the thread's reading error first, if there was one."""
        self.check()
        return [w for _, p, _, w in self.rows if p == phase]


def window_energy(watts: list[float], seconds: float, passes: int, idle_w: float) -> dict:
    """One energy window: the mean power of its readings x its length / passes, with and without the idle power taken
    away. A window with no reading stops (no energy is guessed)."""
    if not watts:
        raise ValueError("an energy window got no power reading: the window is shorter than the chip's reading interval")
    if passes <= 0 or seconds <= 0:
        raise ValueError(f"an energy window needs passes and time ({passes} passes, {seconds} s)")
    mean_w = statistics.fmean(watts)
    return {"mean_w": mean_w, "energy_j_per_pass_total": mean_w * seconds / passes,
            "energy_j_per_pass": (mean_w - idle_w) * seconds / passes, "readings": len(watts)}


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
    for flag in ("asimddp", "i8mm", "asimdhp", "fphp", "bf16"):        # int8 dot / matrix, fp16 vector / scalar, bf16
        info[flag] = flag in feats
    return info
