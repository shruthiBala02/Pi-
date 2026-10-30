"""The baseline: FP32 autograd on the unrounded FP32 network (from code/pinnbench/exp26/autograd.py, the desktop's r0).

    columns     every scored column: u (each output), and each pure-direction derivative by nested autograd chains per
                (output, direction), the chain logic of exp17's `code/pi/arms.py` `arm_b` without its rounding
    net_forward the FP32 network as a function, from the Net of the FP32 row file (input transform, tanh layers,
                linear output)
"""

from __future__ import annotations

from typing import Callable

import torch


def _d(y: torch.Tensor, x: torch.Tensor, col: int) -> torch.Tensor:
    g = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]
    return g[:, col:col + 1]


def columns(model: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, cols: list[dict]) -> dict[str, torch.Tensor]:
    """{column name: float32 [N]} by FP32 autograd on `model` at the points x [N, d]."""
    xg = x.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        y = model(xg)
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


def net_forward(net) -> Callable[[torch.Tensor], torch.Tensor]:
    """x -> the FP32 network's outputs [N, n_out], on the weights and biases of the Net (fp32 mode)."""
    def f(x: torch.Tensor) -> torch.Tensor:
        h = net.transform(x)
        for li in range(net.n_layers - 1):
            h = torch.tanh(h @ net.weight(li).T + net.bias[li])
        return h @ net.weight(net.n_layers - 1).T + net.bias[-1]
    return f
