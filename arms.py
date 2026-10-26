"""The four arms. Each takes a float32 point batch x [N, d] and the settings' column list and returns
{column name: float32 tensor [N]} (u and every scored pure-direction derivative).

B  usual route: autograd on the round-to-nearest network, activations rounded with the straight-through form.
C  ours, simulation: per-order jets, c0 rounded onto the INT8 grid with the layer's activation scale, c1..cK cast to FP16
   (T5), everything held as float32 tensors.
D  ours, real storage: as C, but the weights stay int8 and one layer is dequantized at a time; between layers c0 is held as
   int8 and c1..cK as float16 (the conversions back are exact, so the values equal C's).
E  as D, with the value path c0 @ W^T of the hidden layers after the first done as an integer matmul (torch._int_mm).
   The derivative tracks are float16 values and keep the dequantized float32 matmul.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from net import Net, fake_quant, round_int8


def _d(y: torch.Tensor, x: torch.Tensor, col: int) -> torch.Tensor:
    g = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]
    return g[:, col:col + 1]


def arm_b(net: Net, x: torch.Tensor, cols: list[dict]) -> dict[str, torch.Tensor]:
    xg = x.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        y = net.forward_ste(xg)
        out: dict[str, torch.Tensor] = {}
        chains: dict[tuple[int, int], list[torch.Tensor]] = {}
        for c in sorted(cols, key=lambda c: c["order"]):
            key = (c["out"], c["coord"])
            if c["order"] == 0:
                out[c["name"]] = y[:, c["out"]].detach()
                continue
            chain = chains.setdefault(key, [y[:, c["out"]:c["out"] + 1]])
            while len(chain) <= c["order"]:
                chain.append(_d(chain[-1], xg, c["coord"]))
            out[c["name"]] = chain[c["order"]][:, 0].detach()
    return out


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


def _int_mm_value(q: torch.Tensor, W_int8: torch.Tensor) -> torch.Tensor:
    """q [N, in] int8 @ W_int8^T [in, out] -> int32 [N, out] (exact)."""
    return torch._int_mm(q, W_int8.T.contiguous())


def jet_pass(net: Net, x: torch.Tensor, coord: int, K: int, real: bool, integer: bool) -> list[torch.Tensor]:
    """Taylor coefficients c_0..c_K of the whole output vector along `coord`: list of K+1 tensors [N, n_out].
    real: between hidden layers c0 is held as int8 (its grid integers) and c1..cK in the track_cast dtype; the conversions back are exact."""
    e = torch.zeros_like(x)
    e[:, coord] = 1.0
    with torch.no_grad():
        t0 = net.transform(x)
        t1 = net.transform(x + e) - t0
    c: list[torch.Tensor] = ([t0, t1] + [torch.zeros_like(t0) for _ in range(K - 1)])[:K + 1]
    held = None                                  # (q0 int8, [c1..cK] float16) after a hidden layer, real storage only
    for li in range(net.n_layers - 1):
        if held is not None:
            q0, tracks = held
            c = [q0.to(torch.float32) * net.act_scale[li - 1]] + [t.to(torch.float32) for t in tracks]
        else:
            q0 = None
        W = net.weight(li)
        z = [None] * (K + 1)
        for k in range(K + 1):
            if k == 0 and integer and q0 is not None:
                acc = _int_mm_value(q0, net.w_int8[li])        # exact int32 of the grid integers
                z[0] = acc.to(torch.float32) * (net.act_scale[li - 1] * net.w_scale[li].T) + net.bias[li]
            else:
                z[k] = c[k] @ W.T + (net.bias[li] if k == 0 else 0.0)
        del W
        c = tanh_jet(z)
        c[0] = fake_quant(c[0], net.act_scale[li], net.int8_max)                 # c0 on the INT8 grid
        for k in range(1, K + 1):
            c[k] = c[k].to(net.track_dtype).to(torch.float32)                         # T5: derivative tracks in the track_cast format
        if real:
            if not torch.isfinite(c[0]).all():
                raise ValueError("a non-finite c0 cannot be held as int8: arms D and E stop (arm C keeps the NaN)")
            held = (round_int8(c[0] / net.act_scale[li], net.int8_max).to(torch.int8),
                    [t.to(net.track_dtype) for t in c[1:]])
            c = None
    if held is not None:
        q0, tracks = held
        c = [q0.to(torch.float32) * net.act_scale[net.n_layers - 2]] + [t.to(torch.float32) for t in tracks]
    Wo = net.weight(net.n_layers - 1)
    return [c[k] @ Wo.T + (net.bias[-1] if k == 0 else 0.0) for k in range(K + 1)]


def _jets(net: Net, x: torch.Tensor, cols: list[dict], real: bool, integer: bool) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    coords = sorted({c["coord"] for c in cols if c["order"] > 0})
    first = True
    with torch.no_grad():
        for coord in coords:
            K = max(c["order"] for c in cols if c["coord"] == coord)
            u = jet_pass(net, x, coord, K, real, integer)
            for c in cols:
                if c["order"] > 0 and c["coord"] == coord:
                    out[c["name"]] = u[c["order"]][:, c["out"]] * math.factorial(c["order"])
            if first:
                for c in cols:
                    if c["order"] == 0:
                        out[c["name"]] = u[0][:, c["out"]]
                first = False
        if not coords:
            u = jet_pass(net, x, 0, 1, real, integer)
            for c in cols:
                out[c["name"]] = u[0][:, c["out"]]
    return out


def arm_c(net: Net, x: torch.Tensor, cols: list[dict]) -> dict[str, torch.Tensor]:
    return _jets(net, x, cols, real=False, integer=False)


def arm_d(net: Net, x: torch.Tensor, cols: list[dict]) -> dict[str, torch.Tensor]:
    return _jets(net, x, cols, real=True, integer=False)


def arm_e(net: Net, x: torch.Tensor, cols: list[dict]) -> dict[str, torch.Tensor]:
    return _jets(net, x, cols, real=True, integer=True)


ARMS = {"B": arm_b, "C": arm_c, "D": arm_d, "E": arm_e}
ROUNDING = {"B": "nearest", "C": "da_adaround", "D": "da_adaround", "E": "da_adaround"}
STORAGE = {"B": "fp32", "C": "fp32", "D": "int8", "E": "int8"}
