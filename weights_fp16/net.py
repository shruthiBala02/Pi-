"""One row of the exp25 Pi folder: the network as the desktop export wrote it; nothing rounded or trained here.

A row file holds the weights packed in the row's weight format (`formats.pack`, with their scales; the FP32 row: plain
float32), float32 biases, one activation scale per hidden layer (scaled value formats only, else none), and one track
scale per (direction, hidden layer, order) (scaled track formats only, keys "coord/layer/k"). `meta` names the
equation, seed, part and the three formats.

  fp32      the FP32 row: nothing rounded
  simulate  every weight decoded to float32 once at load; values and tracks rounded and kept as float32 tensors
  real      the weights stay packed and one layer is decoded at a time; between layers the value c0 and the tracks are
            held packed in their formats (jets.py); `matmul_path` says how the value matmul is done
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

import formats as FM

MODES = ("fp32", "simulate", "real")
MATMUL_PATHS = ("fp32", "int", "int4_unpack_int8", "int16", "fp16", "decode_per_layer")


def path_fits(path: str, weight_format: str, act_format: str) -> bool:
    """Whether the value matmul `path` can run on these weight and value formats."""
    pair = {weight_format, act_format}
    if path == "fp32":
        return pair == {"FP32"}
    if path == "decode_per_layer":
        return "FP32" not in pair
    if path == "int":
        return pair == {"INT8"}
    if path == "int4_unpack_int8":
        return "INT4" in pair and pair <= {"INT4", "INT8"}
    if path == "int16":
        return "INT16" in pair and pair <= {"INT4", "INT8", "INT16"}
    if path == "fp16":
        return pair == {"FP16"}
    raise ValueError(f"unknown matmul_path {path!r}")


def choose_path(weight_format: str, act_format: str, native: dict) -> str:
    """The value matmul path of a row from the probe's verdict `native` ({"int": bool, "int16": bool, "fp16": bool}):
    the first native path that fits the formats and the probe found working, else decode_per_layer; FP32: fp32."""
    if {weight_format, act_format} == {"FP32"}:
        return "fp32"
    for path, need in (("int", "int"), ("int4_unpack_int8", "int"), ("int16", "int16"), ("fp16", "fp16")):
        if native[need] and path_fits(path, weight_format, act_format):
            return path
    return "decode_per_layer"


class Net:
    def __init__(self, files: dict, settings: dict, mode: str, cfg: dict, matmul_path: str | None = None):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        meta = files["meta"]
        self.meta, self.settings, self.mode, self.cfg = meta, settings, mode, cfg
        self.weight_format, self.act_format, self.track_format = (meta["weight_format"], meta["act_format"],
                                                                  meta["track_format"])
        is_fp32 = (self.weight_format, self.act_format, self.track_format) == ("FP32", "FP32", "FP32")
        if (mode == "fp32") != is_fp32:
            raise ValueError(f"mode {mode!r} does not fit a row of formats {self.weight_format} / {self.act_format} / "
                             f"{self.track_format} (the FP32 row runs in mode fp32, every other row in simulate or real)")
        if mode == "real" and matmul_path not in MATMUL_PATHS:
            raise ValueError(f"real mode needs a matmul_path from {MATMUL_PATHS}, got {matmul_path!r}")
        if mode == "real" and not path_fits(matmul_path, self.weight_format, self.act_format):
            raise ValueError(f"matmul_path {matmul_path!r} cannot run on {self.weight_format} weights with "
                             f"{self.act_format} values")
        if mode == "fp32" and matmul_path not in (None, "fp32"):
            raise ValueError(f"the FP32 row has matmul_path fp32, got {matmul_path!r}")
        if mode == "simulate" and matmul_path is not None:
            raise ValueError("simulate mode has no matmul_path (every matmul is float32)")
        self.matmul_path = "fp32" if mode == "fp32" else matmul_path
        self.tf = settings["input_transform"]
        self.packed = list(files["weights"])
        self.bias = [b.float() for b in files["bias"]]
        self.n_layers = len(self.bias)
        self.act_scale = list(files["act_scale"])
        self.track_scale = {tuple(int(v) for v in k.split("/")): s for k, s in files["track_scale"].items()}
        if kind(self.act_format) == "scaled" and len(self.act_scale) != self.n_layers - 1:
            raise ValueError(f"{self.act_format} values need one scale per hidden layer, the file has {len(self.act_scale)}")
        if kind(self.track_format) == "scaled" and not self.track_scale:
            raise ValueError(f"{self.track_format} tracks need their scales, the file has none")
        self.w_f32 = [self._decode(i) for i in range(self.n_layers)] if mode != "real" else None
        self._shapes = [tuple(p["shape"]) for p in self.packed]

    # ---- pieces -------------------------------------------------------------------------------------------
    def _decode(self, i: int) -> torch.Tensor:
        p = self.packed[i]
        return p["codes"].float() if p["format"] == "FP32" else FM.decode(p)

    def weight(self, i: int) -> torch.Tensor:
        """Layer i's weight as float32: the one decoded at load, or (real) a copy decoded now."""
        return self.w_f32[i] if self.w_f32 is not None else self._decode(i)

    def weight_int(self, i: int) -> torch.Tensor:
        """Layer i's integer codes (real mode, integer weight formats): int8 for INT4 / INT8, int16 for INT16."""
        return FM.int_codes(self.packed[i])

    def weight_row_scale(self, i: int) -> torch.Tensor:
        return self.packed[i]["scale"]

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

    # ---- sizes (bytes) ------------------------------------------------------------------------------------
    def weights_bytes(self) -> int:
        """The weights at the format's real storage, scales included (biases not counted)."""
        return sum(p["codes"].numel() * 4 if p["format"] == "FP32" else FM.stored_bytes(p) for p in self.packed)

    def held_weight_bytes(self) -> int:
        """Bytes of the weight tensors (and scales) this mode keeps in RAM."""
        if self.w_f32 is not None:
            return sum(w.numel() * 4 for w in self.w_f32)
        return self.weights_bytes()

    def fp32_equivalent_bytes(self) -> int:
        return sum(r * c * 4 for r, c in self._shapes) + sum(b.numel() * 4 for b in self.bias)


def kind(fmt: str) -> str:
    """The format's kind (formats.FORMATS), or 'fp32'; an unknown name stops."""
    if fmt == "FP32":
        return "fp32"
    if fmt not in FM.FORMATS:
        raise ValueError(f"unknown format {fmt!r}")
    return FM.FORMATS[fmt][0]


def _t(v) -> torch.Tensor:
    return torch.tensor(v, dtype=torch.float32)


def load_settings(equation_dir: Path) -> dict:
    return json.loads((equation_dir / "settings.json").read_text(encoding="utf-8"))


def row_path(equation_dir: Path, part: str, fmt: str) -> Path:
    return equation_dir / f"{equation_dir.name}__{part}__{fmt}.pt"


def load(equation_dir: Path, part: str, fmt: str, mode: str, cfg: dict, matmul_path: str | None = None) -> Net:
    path = row_path(equation_dir, part, fmt)
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing (nothing is rounded or trained on the Pi; copy the export's file here)")
    files = torch.load(path, map_location="cpu", weights_only=True)
    settings = load_settings(equation_dir)
    meta = files["meta"]
    if (meta["equation"], meta["part"], meta["format"]) != (settings["equation"], part, fmt):
        raise ValueError(f"{path}: the file holds {meta['equation']} {meta['part']} {meta['format']}, "
                         f"asked for {settings['equation']} {part} {fmt}")
    return Net(files, settings, mode, cfg, matmul_path)
