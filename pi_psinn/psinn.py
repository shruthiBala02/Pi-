"""Jurj's Ψ-NN as the Pi holds it: the INT8 weights of an exp20 checkpoint (weights only), nothing rounded or trained here.

One checkpoint file (exp20 `run.save_quantized`) = int8 weights and their scales per matrix (Jurj's rounding: one scale
per matrix; DA-AdaRound: one per row), FP32 biases, `act_scales` None (the activations stay FP32). The weights stay int8 in
RAM; `weight(name)` turns one matrix back to FP32 when a stage needs it.

The four stages are exp20 `graph._p1.._p4` (Ψ-NN's forward, input (t, x)): the + and − branches share their matrices.
    forward(x)          F.linear, for autograd (arm J)
    jets(x, coord, K)   GR-Jets as exp20 `jets.stage_coefficients` mode fp16_tracks: h @ Wᵀ (+ b on c0), tanh_jet per
                        branch, c1..cK cast to the track format after every hidden stage, c0 FP32 (arm G)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

TRACK_DTYPES = {"FP16": torch.float16, "BF16": torch.bfloat16}


def matmul(h: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None) -> torch.Tensor:
    """exp20 graph.matmul: h @ Wᵀ + b (what the shared jets compute)."""
    return h @ w.T + b if b is not None else h @ w.T


def _p1(h, W, b, lin):
    t, x = h[0][:, 0:1], h[0][:, 1:2]
    a = lin(t, W["fc1_1"], b["fc1_1"])
    c = lin(x, W["fc1_3"], b["fc1_3"])
    return [a + c, a - c]


def _p2(h, W, b, lin):
    u11, u12 = h
    return [lin(u11, W["fc2_1"], b["fc2_1"]) + lin(u12, W["fc2_3"], b["fc2_3"]),
            lin(u11, W["fc2_3"], b["fc2_3"]) + lin(u12, W["fc2_1"], b["fc2_1"]),
            lin(u11, W["fc2_2"], b["fc2_2"]) - lin(u12, W["fc2_2"], b["fc2_2"])]


def _p3(h, W, b, lin):
    u21, u22, u23 = h
    return [lin(u21, W["fc3_1"], b["fc3_1"]) - lin(u22, W["fc3_1"], b["fc3_1"]) + lin(u23, W["fc3_2"], b["fc3_2"])]


def _p4(h, W, b, lin):
    return [lin(h[0], W["fc4_1"], b["fc4_1"])]


# (pre-activation function, the matrices it uses with their number of uses, activation) — exp20 graph._PSINN
STAGES = [(_p1, {"fc1_1": 2, "fc1_3": 2}, "tanh"),
          (_p2, {"fc2_1": 2, "fc2_3": 2, "fc2_2": 2}, "tanh"),
          (_p3, {"fc3_1": 2, "fc3_2": 1}, "tanh"),
          (_p4, {"fc4_1": 1}, None)]


def _cauchy(a: list[torch.Tensor], b: list[torch.Tensor], k: int) -> torch.Tensor:
    return sum(a[i] * b[k - i] for i in range(k + 1))


def tanh_jet(z: list[torch.Tensor]) -> list[torch.Tensor]:
    """Taylor coefficients of tanh(z) from those of z (code/pi/arms.py tanh_jet, = pinnbench dmethods.jets.tanh_jet)."""
    y = [torch.tanh(z[0])]
    y2 = [y[0] * y[0]]
    for k in range(1, len(z)):
        s = sum(j * z[j] * ((1.0 if k == j else 0.0) - y2[k - j]) for j in range(1, k + 1))
        y.append(s / k)
        y2.append(_cauchy(y, y, k))
    return y


class Net:
    def __init__(self, files: dict, settings: dict):
        if files["act_scales"] is not None:
            raise ValueError("the checkpoint has activation scales: this Pi folder runs weights-only checkpoints only")
        names = {n: u for _, uses, _ in STAGES for n, u in uses.items()}
        if settings["matrices"] != names:
            raise ValueError(f"settings.json matrices {settings['matrices']} differ from Ψ-NN's {names}")
        if sorted(files["int8"]) != sorted(names) or sorted(files["scale"]) != sorted(names):
            raise ValueError(f"the checkpoint holds matrices {sorted(files['int8'])}, Ψ-NN has {sorted(names)}")
        if settings["track_cast"] not in TRACK_DTYPES:
            raise ValueError(f"unknown track_cast {settings['track_cast']!r}")
        for n, q in files["int8"].items():
            if q.dtype != torch.int8:
                raise ValueError(f"{n}: weights are {q.dtype}, not int8")
        self.int8 = dict(files["int8"])
        self.scale = {n: s.float() for n, s in files["scale"].items()}
        self.bias = {n: (files["bias"][n].float() if n in files["bias"] else None) for n in names}
        self.track_dtype = TRACK_DTYPES[settings["track_cast"]]
        self.meta = files["meta"]

    # ---- sizes (bytes) ------------------------------------------------------------------------------------
    def held_weight_bytes(self) -> int:
        """Bytes of the weight tensors and their scales as held in RAM (int8 + FP32 scales)."""
        return sum(q.numel() * q.element_size() for q in self.int8.values()) + sum(s.numel() * 4 for s in self.scale.values())

    def int8_network_bytes(self) -> int:
        """The network as INT8 storage: 1 byte per weight, FP32 scales and biases."""
        return (sum(q.numel() for q in self.int8.values()) + sum(s.numel() * 4 for s in self.scale.values())
                + sum(b.numel() * 4 for b in self.bias.values() if b is not None))

    def fp32_equivalent_bytes(self) -> int:
        return 4 * (sum(q.numel() for q in self.int8.values()) + sum(b.numel() for b in self.bias.values() if b is not None))

    # ---- pieces -------------------------------------------------------------------------------------------
    def weight(self, name: str) -> torch.Tensor:
        """One matrix as FP32: a temporary copy dequantized from int8 (int8 × scale)."""
        return self.int8[name].to(torch.float32) * self.scale[name]

    def stage_weights(self, i: int) -> dict[str, torch.Tensor]:
        return {n: self.weight(n) for n in STAGES[i][1]}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """u [N, 1] at (t, x) points: F.linear, tanh on every branch (exp20 graph.forward)."""
        h = [x]
        for i, (pre, _, act) in enumerate(STAGES):
            z = pre(h, self.stage_weights(i), self.bias, F.linear)
            if act is None:
                return z[0]
            h = [torch.tanh(v) for v in z]
        raise ValueError("the last stage must have no activation")

    def jets(self, x: torch.Tensor, coord: int, K: int) -> list[torch.Tensor]:
        """d^k u / d x_coord^k, k = 0..K, each [N]. Input jet: c0 = x, c1 = the unit vector along coord (exact for
        Ψ-NN's identity input), c2.. = 0."""
        e = torch.zeros_like(x)
        e[:, coord] = 1.0
        c = ([[x], [e]] + [[torch.zeros_like(x)] for _ in range(K - 1)])[:K + 1]
        nobias = {n: None for n in self.bias}
        for i, (pre, _, act) in enumerate(STAGES):
            W = self.stage_weights(i)
            z = [pre(c[k], W, self.bias if k == 0 else nobias, matmul) for k in range(K + 1)]     # z[k][branch]
            del W
            if act is None:
                return [z[k][0][:, 0] * math.factorial(k) for k in range(K + 1)]
            nb = len(z[0])
            ys = [tanh_jet([z[k][b] for k in range(K + 1)]) for b in range(nb)]                     # ys[branch][k]
            for b in range(nb):
                for k in range(1, K + 1):
                    ys[b][k] = ys[b][k].to(self.track_dtype).to(torch.float32)
            c = [[ys[b][k] for b in range(nb)] for k in range(K + 1)]
        raise ValueError("the last stage must have no activation")


def load_settings(ckpt_dir: Path) -> dict:
    return json.loads((ckpt_dir / "settings.json").read_text(encoding="utf-8"))


def load(path: Path, settings: dict) -> Net:
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing (nothing is rounded or trained on the Pi; copy the collected files here)")
    files = torch.load(path, map_location="cpu", weights_only=True)
    return Net(files, settings)
