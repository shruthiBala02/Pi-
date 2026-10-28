"""The two arms on Ψ-NN, weights only. Each takes float32 (t, x) points [N, 2] and returns {u, u_t, u_x, u_xx: [N]}.

J  Jurj's rounding (n1_q1.pt) + autograd: as exp20 `arms.derivs(.., "autograd")`
G  GR-ADA-Jets (n1_q2.pt, DA-AdaRound): jets along t (K = orders t) and x (K = orders x), FP32 value, FP16 derivative
   tracks; u from the x pass; as exp20 `arms.derivs(.., "jets_fp16")` on a weights-only arm
"""

from __future__ import annotations

import torch

from psinn import Net

IT, IX = 0, 1                     # Ψ-NN's input order (t, x)


def arm_j(net: Net, x: torch.Tensor, orders: dict) -> dict[str, torch.Tensor]:
    if {k: int(v) for k, v in orders.items()} != {"t": 1, "x": 2}:
        raise ValueError(f"arm J computes u_t, u_x, u_xx only (orders t 1, x 2), got {orders}")
    xg = x.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        u = net.forward(xg)
        d1 = torch.autograd.grad(u.sum(), xg, create_graph=True)[0]
        u_xx = torch.autograd.grad(d1[:, IX].sum(), xg)[0][:, IX]
    return {"u": u.detach()[:, 0], "u_t": d1.detach()[:, IT], "u_x": d1.detach()[:, IX], "u_xx": u_xx.detach()}


def arm_g(net: Net, x: torch.Tensor, orders: dict) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        jt = net.jets(x, IT, int(orders["t"]))
        jx = net.jets(x, IX, int(orders["x"]))
    return {"u": jx[0], "u_t": jt[1], "u_x": jx[1], "u_xx": jx[2]}


ARMS = {"J": arm_j, "G": arm_g}
