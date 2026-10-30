"""Per-order jets on one row (from code/pi/arms.py, arms C, D, E), with any value format and any track format.

With every value the network carries its Taylor coefficients along one input coordinate, c_0..c_K (c_k = d^k / k!);
the derivative is read off as k! c_k. After each hidden layer's tanh:
  c0 (the value)       rounded onto the row's value format with the layer's activation scale (block formats: their
                       own blocks, computed on this pass, so a pass over another batch can give other block scales)
  c1..cK (the tracks)  rounded onto the row's track format with the (direction, layer, order) scale
The input and the output layer are not rounded (as exp17 and pinnbench's weights + activations arm).

  fp32      nothing rounded; `record_amax` collects max |c_k| per (direction, layer, k): the track scales' source
  simulate  the rounded values are float32 tensors (formats.round_to)
  real      between layers c0 and the tracks are held packed (formats.pack) and decoded at the next layer; the weights
            are decoded one layer at a time. The value matmul c0 W^T of the hidden layers after the first follows
            net.matmul_path:
              decode_per_layer   float32 on the decoded values and weights
              int                INT8 codes x INT8 codes, torch._int_mm (exact int32), then the two scales (exp17 arm E)
              int4_unpack_int8   as int, the INT4 codes unpacked to int8 first
              int16              the integer codes as int64, integer matmul (exact), then the two scales
              fp16               FP16 values x FP16 weights, float16 matmul
            The tracks always take the float32 matmul on the decoded weights, as in exp17.
"""

from __future__ import annotations

import math

import torch

import formats as FM
from net import Net, kind


def _cauchy(a: list[torch.Tensor], b: list[torch.Tensor], k: int) -> torch.Tensor:
    return sum(a[i] * b[k - i] for i in range(k + 1))


def tanh_jet(z: list[torch.Tensor]) -> list[torch.Tensor]:
    y = [torch.tanh(z[0])]
    y2 = [y[0] * y[0]]
    for k in range(1, len(z)):
        s = sum(j * z[j] * ((1.0 if k == j else 0.0) - y2[k - j]) for j in range(1, k + 1))
        y.append(s / k)
        y2.append(_cauchy(y, y, k))
    return y


def _value_matmul(net: Net, li: int, c0: torch.Tensor, held0: dict | None) -> torch.Tensor:
    """z0 = c0 W^T + b of hidden layer li."""
    path = net.matmul_path
    if held0 is None or path in ("fp32", "decode_per_layer"):
        return c0 @ net.weight(li).T + net.bias[li]
    if path in ("int", "int4_unpack_int8"):
        acc = torch._int_mm(FM.int_codes(held0).to(torch.int8), net.weight_int(li).to(torch.int8).T.contiguous())
        return acc.to(torch.float32) * (held0["scale"] * net.weight_row_scale(li).T) + net.bias[li]
    if path == "int16":
        acc = FM.int_codes(held0).to(torch.int64) @ net.weight_int(li).to(torch.int64).T
        return acc.to(torch.float32) * (held0["scale"] * net.weight_row_scale(li).T) + net.bias[li]
    if path == "fp16":
        return (held0["codes"] @ net.packed[li]["codes"].T).to(torch.float32) + net.bias[li]
    raise ValueError(f"unknown matmul_path {path!r}")


def _scale(fmt: str, s):
    return s if kind(fmt) == "scaled" else None


def jet_pass(net: Net, x: torch.Tensor, coord: int, K: int, record_amax: dict | None = None,
             stats: dict | None = None) -> list[torch.Tensor]:
    """Taylor coefficients c_0..c_K of the whole output vector along `coord`: K+1 tensors [N, n_out].
    record_amax (fp32 mode only): {(coord, layer, k): max |c_k|} updated in place. stats (simulate, real): counts of
    the rounding, {"act": [values, saturated, underflow], "track": [...]}, and in real and fp32 mode the largest bytes
    held between layers ("act_bytes_peak", "track_bytes_peak"; fp32: float32 tensors)."""
    if record_amax is not None and net.mode != "fp32":
        raise ValueError("record_amax runs on the FP32 row only (the track scales come from the FP32 network)")
    e = torch.zeros_like(x)
    e[:, coord] = 1.0
    with torch.no_grad():
        t0 = net.transform(x)
        t1 = net.transform(x + e) - t0
    c: list[torch.Tensor] = ([t0, t1] + [torch.zeros_like(t0) for _ in range(K - 1)])[:K + 1]
    held: tuple[dict, list[dict]] | None = None          # (c0, [c1..cK]) packed after a hidden layer, real mode only
    for li in range(net.n_layers - 1):
        held0 = None
        if held is not None:
            held0, held_t = held
            c = [FM.decode(held0)] + [FM.decode(p) for p in held_t]
        W = net.weight(li)
        z = [_value_matmul(net, li, c[0], held0)] + [c[k] @ W.T for k in range(1, K + 1)]
        del W
        c = tanh_jet(z)
        if net.mode == "fp32":
            if stats is not None:
                stats["act_bytes_peak"] = max(stats.get("act_bytes_peak", 0), c[0].numel() * 4)
                stats["track_bytes_peak"] = max(stats.get("track_bytes_peak", 0), sum(t.numel() * 4 for t in c[1:]))
            if record_amax is not None:
                for k in range(K + 1):
                    key = (coord, li, k)
                    record_amax[key] = max(record_amax.get(key, 0.0), float(c[k].abs().max()))
            continue
        s0 = _scale(net.act_format, net.act_scale[li] if net.act_scale else None)
        st = [_scale(net.track_format, net.track_scale.get((coord, li, k))) for k in range(1, K + 1)]
        if kind(net.track_format) == "scaled" and any(s is None for s in st):
            raise ValueError(f"no track scale for direction {coord}, layer {li} in the row file")
        if stats is not None:
            _count(stats, "act", FM.round_stats(c[0], net.act_format, net.cfg, s0))
            for k in range(1, K + 1):
                _count(stats, "track", FM.round_stats(c[k], net.track_format, net.cfg, st[k - 1]))
        if net.mode == "simulate":
            c = [FM.round_to(c[0], net.act_format, net.cfg, s0)] + [
                FM.round_to(c[k], net.track_format, net.cfg, st[k - 1]) for k in range(1, K + 1)]
        else:
            held = (FM.pack(c[0], net.act_format, net.cfg, s0),
                    [FM.pack(c[k], net.track_format, net.cfg, st[k - 1]) for k in range(1, K + 1)])
            c = None
            if stats is not None:
                stats["act_bytes_peak"] = max(stats.get("act_bytes_peak", 0), FM.stored_bytes(held[0]))
                stats["track_bytes_peak"] = max(stats.get("track_bytes_peak", 0),
                                                sum(FM.stored_bytes(p) for p in held[1]))
    if held is not None:
        c = [FM.decode(held[0])] + [FM.decode(p) for p in held[1]]
    Wo = net.weight(net.n_layers - 1)
    return [c[k] @ Wo.T + (net.bias[-1] if k == 0 else 0.0) for k in range(K + 1)]


def _count(stats: dict, key: str, got: tuple[int, int, int]) -> None:
    cur = stats.setdefault(key, [0, 0, 0])
    for i in range(3):
        cur[i] += got[i]


def score(net: Net, x: torch.Tensor, cols: list[dict], stats: dict | None = None) -> dict[str, torch.Tensor]:
    """{column name: float32 [N]}: u (and every output) and every scored pure-direction derivative, one jet pass per
    direction with K = the direction's highest order (arm C's reading)."""
    out: dict[str, torch.Tensor] = {}
    coords = sorted({c["coord"] for c in cols if c["order"] > 0})
    with torch.no_grad():
        for i, coord in enumerate(coords):
            K = max(c["order"] for c in cols if c["coord"] == coord)
            u = jet_pass(net, x, coord, K, stats=stats)
            for c in cols:
                if c["order"] > 0 and c["coord"] == coord:
                    out[c["name"]] = u[c["order"]][:, c["out"]] * math.factorial(c["order"])
                elif c["order"] == 0 and i == 0:
                    out[c["name"]] = u[0][:, c["out"]]
        if not coords:
            u = jet_pass(net, x, 0, 1, stats=stats)
            for c in cols:
                out[c["name"]] = u[0][:, c["out"]]
    return out


def track_amax(net: Net, x: torch.Tensor, cols: list[dict], chunk: int) -> dict:
    """max |c_k| of the FP32 row on the points x, per (direction, hidden layer, k), every direction with its K."""
    amax: dict = {}
    with torch.no_grad():
        for coord in sorted({c["coord"] for c in cols if c["order"] > 0}):
            K = max(c["order"] for c in cols if c["coord"] == coord)
            for i in range(0, x.shape[0], chunk):
                jet_pass(net, x[i:i + chunk], coord, K, record_amax=amax)
    return amax
