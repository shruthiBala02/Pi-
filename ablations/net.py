"""The rounded INT8 network as the Pi holds it: files written by the desktop export, nothing rounded or trained here.

One checkpoint file = int8 weights + one float32 scale per output row, float32 biases, one float32 activation scale per
hidden layer. `storage="fp32"` (arms B, C) dequantizes the weights once at load (the simulation: FP32 tensors on the INT8 grid);
`storage="int8"` (arms D, E) keeps them as int8 and dequantizes one layer at a time.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F


def round_int8(x: torch.Tensor, int8_max: float) -> torch.Tensor:
    """pinnbench.quantize.formats.round_int8: integers -127..127, ties to even."""
    return torch.clamp(torch.round(x), -int8_max, int8_max)


def fake_quant(y: torch.Tensor, scale: torch.Tensor, int8_max: float) -> torch.Tensor:
    """pinnbench quant_scaled: x / scale rounded onto the integers, times scale (scale is a float32 scalar)."""
    return round_int8(y / scale, int8_max) * scale


TRACK_DTYPES = {"FP16": torch.float16, "BF16": torch.bfloat16}


class Net:
    def __init__(self, files: dict, settings: dict, storage: str, int8_max: float):
        if storage not in ("fp32", "int8"):
            raise ValueError(f"storage must be fp32 or int8, got {storage!r}")
        if settings["track_cast"] not in TRACK_DTYPES:
            raise ValueError(f"unknown track_cast {settings['track_cast']!r}")
        self.settings, self.storage, self.int8_max = settings, storage, float(int8_max)
        self.track_dtype = TRACK_DTYPES[settings["track_cast"]]
        self.tf = settings["input_transform"]
        self.bias = [b.float() for b in files["bias"]]
        self.act_scale = [s.float() for s in files["act_scale"]]
        self.w_scale = [s.float() for s in files["w_scale"]]
        self.n_layers = len(self.bias)
        self.meta = files["meta"]
        if storage == "int8":
            self.w_int8 = [w for w in files["w_int8"]]
            self.w_f32 = None
        else:
            self.w_f32 = [w.float() * s for w, s in zip(files["w_int8"], self.w_scale)]
            self.w_int8 = None
            self.w_scale = []    # the scales are folded into the weights and no longer held
        self._n_params = sum(w.numel() for w in files["w_int8"]) + sum(b.numel() for b in files["bias"])
        self._n_weights = sum(w.numel() for w in files["w_int8"])
        self._n_rows = sum(w.shape[0] for w in files["w_int8"])

    # ---- sizes (bytes) ------------------------------------------------------------------------------------
    def held_weight_bytes(self) -> int:
        """Bytes of the weight tensors (and their scales) this arm keeps in RAM."""
        if self.storage == "int8":
            return sum(w.numel() * w.element_size() for w in self.w_int8) + sum(s.numel() * 4 for s in self.w_scale)
        return sum(w.numel() * 4 for w in self.w_f32)

    def fp32_equivalent_bytes(self) -> int:
        return self._n_params * 4

    def int8_network_bytes(self) -> int:
        """The network as INT8 storage: 1 byte per weight, float32 row scales, biases and activation scales."""
        return self._n_weights + self._n_rows * 4 + sum(b.numel() for b in self.bias) * 4 + len(self.act_scale) * 4

    # ---- pieces -------------------------------------------------------------------------------------------
    def weight(self, i: int) -> torch.Tensor:
        """Layer i's weight as a float32 tensor: the held one (fp32 storage) or a temporary copy dequantized from int8."""
        if self.storage == "fp32":
            return self.w_f32[i]
        return self.w_int8[i].to(torch.float32) * self.w_scale[i]

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """pinnbench nets.InputTransform for the kinds that are affine per coordinate."""
        kind = self.tf["kind"]
        if kind == "none":
            return x
        if kind == "scale_pm1":
            lb, ub = _t(self.tf["lb"]), _t(self.tf["ub"])
            return 2.0 * (x - lb) / (ub - lb + self.tf["eps"]) - 1.0
        if kind == "minmax":
            lb, ub = _t(self.tf["lb"]), _t(self.tf["ub"])
            return (x - lb) / (ub - lb)
        if kind == "zscore":
            return (x - _t(self.tf["mu"])) / _t(self.tf["sigma"])
        raise ValueError(f"input transform {kind!r} is not supported (affine kinds only)")

    def act_q(self, y: torch.Tensor, i: int) -> torch.Tensor:
        return fake_quant(y, self.act_scale[i], self.int8_max)

    def forward_ste(self, x: torch.Tensor) -> torch.Tensor:
        """Arm B's forward: F.linear with the rounded weights, tanh, activation rounded with the straight-through form
        q + (y - y.detach()) of pinnbench QAct, so autograd differentiates through the rounding as the identity."""
        h = self.transform(x)
        for i in range(self.n_layers - 1):
            y = torch.tanh(F.linear(h, self.weight(i), self.bias[i]))
            with torch.no_grad():
                q = self.act_q(y, i)
            h = q + (y - y.detach())
        return F.linear(h, self.weight(self.n_layers - 1), self.bias[self.n_layers - 1])


def _t(v) -> torch.Tensor:
    return torch.tensor(v, dtype=torch.float32)


def load_settings(equation_dir: Path) -> dict:
    return json.loads((equation_dir / "settings.json").read_text(encoding="utf-8"))


def ckpt_path(equation_dir: Path, seed: int, rounding: str) -> Path:
    return equation_dir / f"seed{seed:02d}_{rounding}.pt"


def load(equation_dir: Path, seed: int, rounding: str, storage: str, int8_max: float) -> Net:
    path = ckpt_path(equation_dir, seed, rounding)
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing (nothing is rounded or trained on the Pi; copy the export's file here)")
    files = torch.load(path, map_location="cpu", weights_only=True)
    settings = load_settings(equation_dir)
    meta = files["meta"]
    if (meta["seed"], meta["rounding"], meta["equation"]) != (seed, rounding, settings["equation"]):
        raise ValueError(f"{path}: the file holds {meta['equation']} seed {meta['seed']} {meta['rounding']}, "
                         f"asked for {settings['equation']} seed {seed} {rounding}")
    return Net(files, settings, storage, int8_max)
