"""The 15 number formats on the Pi (no pinnbench): rounding, packing at the real width, decoding to float32.

`round_to` is pinnbench.quantize.formats.fake_quant with the same expressions (the block-scale lines are shared with
`pack` here); the tests check they give the same bits.
`pack` stores a rounded tensor as the format holds it; `decode(pack(x))` equals `round_to(x)` exactly (as values: a
-0.0 of an integer format decodes as +0.0, the integers have no sign of zero).

  cast    FP16, BF16                          2 bytes per value (the dtype itself)
  scaled  INT32 / INT16 / INT8                4 / 2 / 1 byte per value, signed; one float32 scale (per row, per tensor)
          INT4                                2 values per byte (4-bit two's complement, -7..7)
          FP8-E4M3 / E5M2, FP6-*, FP4-E2M1    sign | exponent | mantissa codes: 1 byte, 4 in 3 bytes, 2 per byte
  block   MXFP8 / MXFP6 / MXFP4               element codes as above + one E8M0 byte per block (2^(code - 127))
  nvfp4   NVFP4                               FP4 codes + one FP8-E4M3 byte per block + one float32 tensor scale

INT32: the desktop rounds in float32, where 2^31 - 1 is held as 2^31, so a row's largest weight rounds to 2^31. That is
stored as 2^31 - 1, the largest int32, which turns back into 2^31 in float32: the decoded value is the desktop's
(researcher, 2026-10-30).
"""

from __future__ import annotations

import math

import torch

ELEMENTS: dict[str, dict] = {
    "fp8_e4m3": {"E": 4, "M": 3, "bias": 7, "max": 448.0, "drop": {(15, 7)}},
    "fp8_e5m2": {"E": 5, "M": 2, "bias": 15, "max": 57344.0, "drop_exp": {31}},
    "fp6_e2m3": {"E": 2, "M": 3, "bias": 1, "max": 7.5},
    "fp6_e3m2": {"E": 3, "M": 2, "bias": 3, "max": 28.0},
    "fp4_e2m1": {"E": 2, "M": 1, "bias": 1, "max": 6.0},
}
FORMATS: dict[str, tuple[str, str | None]] = {
    "INT32": ("scaled", None), "INT16": ("scaled", None),
    "FP16": ("cast", None), "BF16": ("cast", None),
    "INT8": ("scaled", None), "FP8-E4M3": ("scaled", "fp8_e4m3"), "FP8-E5M2": ("scaled", "fp8_e5m2"),
    "MXFP8": ("block", "fp8_e4m3"),
    "FP6-E2M3": ("scaled", "fp6_e2m3"), "FP6-E3M2": ("scaled", "fp6_e3m2"), "MXFP6": ("block", "fp6_e2m3"),
    "INT4": ("scaled", None),
    "FP4-E2M1": ("scaled", "fp4_e2m1"), "MXFP4": ("block", "fp4_e2m1"), "NVFP4": ("nvfp4", "fp4_e2m1"),
}
INT_MAX: dict[str, float] = {"INT4": 7.0, "INT8": 127.0, "INT16": 32767.0, "INT32": 2147483647.0}
CAST_DTYPES = {"FP16": torch.float16, "BF16": torch.bfloat16}
INT_DTYPES = {"INT32": torch.int32, "INT16": torch.int16, "INT8": torch.int8}


# ---- rounding (pinnbench.quantize.formats, copied) ----------------------------------------------------------------

_GRIDS: dict[str, torch.Tensor] = {}


def element_grid(name: str) -> torch.Tensor:
    """Sorted non-negative values the element format can represent (0 first); index i is the magnitude code i."""
    if name not in _GRIDS:
        s = ELEMENTS[name]
        vals = []
        for e in range(2 ** s["E"]):
            for m in range(2 ** s["M"]):
                if (e, m) in s.get("drop", ()) or e in s.get("drop_exp", ()):
                    continue
                vals.append(m / 2 ** s["M"] * 2.0 ** (1 - s["bias"]) if e == 0
                            else (1 + m / 2 ** s["M"]) * 2.0 ** (e - s["bias"]))
        grid = torch.tensor(sorted(set(vals)), dtype=torch.float32)
        if float(grid[-1]) != s["max"]:
            raise AssertionError(f"{name}: largest value {float(grid[-1])} != {s['max']}")
        _GRIDS[name] = grid
    return _GRIDS[name]


def format_max(fmt: str) -> float:
    kind, elem = FORMATS[fmt]
    if kind == "cast":
        return INT_MAX["INT8"]
    return INT_MAX[fmt] if elem is None else ELEMENTS[elem]["max"]


def round_element(x: torch.Tensor, name: str) -> torch.Tensor:
    s = ELEMENTS[name]
    x32 = x if x.dtype == torch.float32 else x.to(torch.float32)
    a = x32.abs().clamp(max=s["max"])
    e = ((a.view(torch.int32) >> 23) - 127).clamp(min=1 - s["bias"])
    spacing = ((e - s["M"] + 127) << 23).view(torch.float32)
    q = torch.sign(x32) * (torch.round(a / spacing) * spacing)
    return q if x.dtype == torch.float32 else q.to(x.dtype)


def round_int(x: torch.Tensor, fmt: str) -> torch.Tensor:
    return torch.clamp(torch.round(x), -INT_MAX[fmt], INT_MAX[fmt])


def _blocks(x: torch.Tensor, block: int) -> tuple[torch.Tensor, int]:
    pad = (-x.shape[-1]) % block
    xp = torch.nn.functional.pad(x, (0, pad)) if pad else x
    return xp.reshape(*x.shape[:-1], -1, block), pad


def _unblock(xb: torch.Tensor, pad: int) -> torch.Tensor:
    flat = xb.reshape(*xb.shape[:-2], -1)
    return flat[..., :flat.shape[-1] - pad] if pad else flat


def _mx_scale(xb: torch.Tensor, elem: str) -> torch.Tensor:
    emax = math.floor(math.log2(ELEMENTS[elem]["max"]))
    amax = xb.abs().amax(dim=-1, keepdim=True)
    e = torch.floor(torch.log2(amax.clamp(min=2.0 ** -126))) - emax
    return torch.where(amax > 0, torch.exp2(e.clamp(-127, 127)), torch.ones_like(amax))


def _nvfp4_scales(xb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(block scale on the FP8-E4M3 grid, tensor scale, the scale used per block)."""
    fp4_max, fp8_max = ELEMENTS["fp4_e2m1"]["max"], ELEMENTS["fp8_e4m3"]["max"]
    tensor_amax = xb.abs().amax()
    ts = torch.where(tensor_amax > 0, tensor_amax / (fp4_max * fp8_max), torch.ones_like(tensor_amax))
    amax = xb.abs().amax(dim=-1, keepdim=True)
    bs = round_element(amax / fp4_max / ts, "fp8_e4m3")
    return bs, ts, torch.where(bs > 0, bs * ts, torch.ones_like(bs))


def round_to(x: torch.Tensor, fmt: str, cfg: dict, scale: torch.Tensor | None = None) -> torch.Tensor:
    """x rounded onto `fmt` (pinnbench fake_quant). `scale` (broadcastable against x) for the scaled formats; `cfg`
    holds mx_block and nvfp4_block."""
    kind, elem = FORMATS[fmt]
    if kind == "cast":
        return x.to(CAST_DTYPES[fmt]).to(x.dtype)
    if kind == "scaled":
        if scale is None:
            raise ValueError(f"{fmt} needs a scale")
        z = x / scale
        return (round_int(z, fmt) if elem is None else round_element(z, elem)) * scale
    if kind == "block":
        xb, pad = _blocks(x, cfg["mx_block"])
        s = _mx_scale(xb, elem)
        return _unblock(round_element(xb / s, elem) * s, pad)
    xb, pad = _blocks(x, cfg["nvfp4_block"])
    s = _nvfp4_scales(xb)[2]
    return _unblock(round_element(xb / s, "fp4_e2m1") * s, pad)


# ---- element codes ------------------------------------------------------------------------------------------------

def code_bits(elem: str) -> int:
    s = ELEMENTS[elem]
    return 1 + s["E"] + s["M"]


_TABLES: dict[str, torch.Tensor] = {}


def code_table(elem: str) -> torch.Tensor:
    """float32 value of every code (sign bit on top): codes past the grid (E4M3 S.1111.111, E5M2 exponent 31) are NaN
    (E5M2 mantissa 0: inf)."""
    if elem not in _TABLES:
        g = element_grid(elem)
        mag_bits = code_bits(elem) - 1
        mag = torch.full((2 ** mag_bits,), float("nan"))
        mag[:g.numel()] = g
        if elem == "fp8_e5m2":
            mag[31 << 2] = float("inf")
        _TABLES[elem] = torch.cat([mag, -mag])
    return _TABLES[elem]


def nan_code(elem: str) -> int | None:
    return {"fp8_e4m3": 0x7F, "fp8_e5m2": 0x7F}.get(elem)


def encode_elements(q: torch.Tensor, elem: str) -> torch.Tensor:
    """uint8 codes of values already on the element grid (sign bit = signbit, so -0.0 keeps its sign). A NaN takes the
    format's NaN code; a format without one refuses it."""
    g = element_grid(elem).to(q.device)
    nan = torch.isnan(q)
    a = torch.where(nan, torch.zeros_like(q), q.abs())
    idx = torch.searchsorted(g, a.contiguous())
    if not torch.equal(g[idx.clamp(max=g.numel() - 1)], a):
        raise ValueError(f"{elem}: a value is not on the grid (encode only rounded values)")
    codes = idx | (torch.signbit(q).to(torch.int64) << (code_bits(elem) - 1))
    if nan.any():
        nc = nan_code(elem)
        if nc is None:
            raise ValueError(f"{elem} has no NaN code: {int(nan.sum())} NaN values cannot be stored")
        codes = torch.where(nan, torch.full_like(codes, nc), codes)
    return codes.to(torch.uint8)


# ---- bit packing (4 bits: 2 per byte; 6 bits: 4 in 3 bytes; 8 bits: 1 per byte) -----------------------------------

def pack_bits(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """Flat uint8 codes (each < 2^bits) into bytes; zero codes pad the end."""
    c = codes.reshape(-1).to(torch.uint8)
    if bits == 8:
        return c.clone()
    if bits == 4:
        c = torch.nn.functional.pad(c, (0, (-c.numel()) % 2)).reshape(-1, 2)
        return c[:, 0] | (c[:, 1] << 4)
    if bits == 6:
        c = torch.nn.functional.pad(c, (0, (-c.numel()) % 4)).reshape(-1, 4)
        b0 = c[:, 0] | (c[:, 1] << 6)
        b1 = (c[:, 1] >> 2) | (c[:, 2] << 4)
        b2 = (c[:, 2] >> 4) | (c[:, 3] << 2)
        return torch.stack([b0, b1, b2], dim=1).reshape(-1)
    raise ValueError(f"no packing for {bits}-bit codes")


def unpack_bits(data: torch.Tensor, bits: int, n: int) -> torch.Tensor:
    """The first n codes (uint8) of pack_bits' bytes."""
    if bits == 8:
        return data[:n]
    if bits == 4:
        return torch.stack([data & 0xF, data >> 4], dim=1).reshape(-1)[:n]
    if bits == 6:
        b = data.reshape(-1, 3)
        b0, b1, b2 = b[:, 0], b[:, 1], b[:, 2]
        c = torch.stack([b0 & 0x3F, (b0 >> 6) | ((b1 & 0xF) << 2), (b1 >> 4) | ((b2 & 0x3) << 4), b2 >> 2], dim=1)
        return c.reshape(-1)[:n]
    raise ValueError(f"no unpacking for {bits}-bit codes")


# ---- pack / decode ------------------------------------------------------------------------------------------------

EXACT_INT_STEPS = 4      # integer codes tried on each side of round(x / scale)


def _int_step(c: torch.Tensor, direction: float) -> torch.Tensor:
    """The next float32 integer from c in `direction` (+1 / -1): c +- 1 where float32 is finer than the integers,
    the next float32 where it is coarser (above 2^24)."""
    nxt = torch.nextafter(c, torch.full_like(c, direction * math.inf))
    return torch.maximum(nxt, c + 1) if direction > 0 else torch.minimum(nxt, c - 1)


def _exact_ints(x: torch.Tensor, scale: torch.Tensor, fmt: str, start: torch.Tensor | None = None) -> torch.Tensor:
    """For x already on the grid (rounded weights): the float32 integer z with z * scale == x, the nearest to
    `start` (default round(x / scale)). round(x / scale) missed by one float32 step for 347 of 2,000,000 random INT32
    weights on the CPU, and by more for one Kovasznay weight on the RTX machine (2026-10-30, value not recorded), so
    up to EXACT_INT_STEPS integers on each side are tried, nearest first. Only a code that gives x back exactly is
    taken; where none does, `start` stays and the export's check stops."""
    m = INT_MAX[fmt]
    z = round_int(x / scale, fmt) if start is None else start
    out, found = z.clone(), z * scale == x
    up, dn = z, z
    for _ in range(EXACT_INT_STEPS):
        up, dn = _int_step(up, 1.0).clamp(-m, m), _int_step(dn, -1.0).clamp(-m, m)
        for c in (up, dn):
            hit = ~found & (c * scale == x)
            out, found = torch.where(hit, c, out), found | hit
    return out


def _int_codes(z: torch.Tensor, fmt: str) -> torch.Tensor:
    """Integer codes of z already rounded (float32 integers within +-max). INT32: 2^31 is stored as 2^31 - 1."""
    if torch.isnan(z).any():
        raise ValueError(f"{fmt} has no NaN code: {int(torch.isnan(z).sum())} NaN values cannot be stored")
    if fmt == "INT32":
        z64 = z.to(torch.float64).clamp(max=2.0 ** 31 - 1)
        return z64.to(torch.int64).to(torch.int32)
    ints = z.to(torch.int32)
    if fmt == "INT4":
        return (ints & 0xF).to(torch.uint8)                       # 4-bit two's complement
    return ints.to(INT_DTYPES[fmt])


def pack(x: torch.Tensor, fmt: str, cfg: dict, scale: torch.Tensor | None = None,
         like: torch.Tensor | None = None, rounded: bool = False) -> dict:
    """x (float32) rounded onto `fmt` and stored at the format's real width. Returns codes and scales; nothing else
    of x is kept. Scaled formats take `scale` (broadcastable against x, stored as given); block formats compute their
    block scales along the last dimension, as round_to does, from `like` when given (rounded weights: the FP32
    weights their rounding took the scales from), else from x. rounded=True: x is already on the grid (the desktop's
    rounded weights) and the integer formats take the codes that give x back exactly (`_exact_ints`). Runs on x's
    device."""
    kind, elem = FORMATS[fmt]
    x = x.to(torch.float32)
    if like is not None and (kind not in ("block", "nvfp4") or like.shape != x.shape):
        raise ValueError(f"`like` is for the block formats, with x's shape ({fmt}, {tuple(x.shape)})")
    out: dict = {"format": fmt, "shape": list(x.shape)}
    if kind == "cast":
        out["codes"] = x.to(CAST_DTYPES[fmt])
        return out
    if kind == "scaled":
        if scale is None:
            raise ValueError(f"{fmt} needs a scale")
        scale = scale.to(torch.float32)
        z = x / scale
        out["scale"] = scale.clone()
        if elem is None:
            codes = _int_codes(_exact_ints(x, scale, fmt) if rounded else round_int(z, fmt), fmt)
            out["codes"] = pack_bits(codes, 4) if fmt == "INT4" else codes.reshape(-1).clone()
        else:
            out["codes"] = pack_bits(encode_elements(round_element(z, elem), elem), code_bits(elem))
        return out
    block = cfg["mx_block"] if kind == "block" else cfg["nvfp4_block"]
    xb, pad = _blocks(x, block)
    lb = xb if like is None else _blocks(like.to(torch.float32), block)[0]
    if kind == "block":
        s = _mx_scale(lb, elem)
        e = torch.frexp(s).exponent - 1                                     # s = 2^e exactly: the float's own exponent
        out["block_exp"] = (e + 127).to(torch.uint8).reshape(-1)          # E8M0: 2^(code - 127)
    else:
        elem = "fp4_e2m1"
        bs, ts, s = _nvfp4_scales(lb)
        out["block_scale"] = encode_elements(bs.reshape(-1), "fp8_e4m3")
        out["tensor_scale"] = ts.reshape(())
    if not torch.isfinite(s).all():
        raise ValueError(f"{fmt}: a block scale is not finite (NaN or inf in the input)")
    q = _unblock(round_element(xb / s, elem), pad)
    out["block"], out["codes"] = block, pack_bits(encode_elements(q, elem), code_bits(elem))
    return out


def _element_values(p: dict, elem: str, n: int) -> torch.Tensor:
    return code_table(elem)[unpack_bits(p["codes"], code_bits(elem), n).to(torch.int64)]


def int_codes(p: dict) -> torch.Tensor:
    """The integers of a packed integer format, in the tensor's shape: int8 for INT4 (sign-extended) and INT8, int16 for
    INT16, int32 for INT32."""
    fmt, shape = p["format"], p["shape"]
    if FORMATS[fmt] != ("scaled", None):
        raise ValueError(f"{fmt} is not an integer format")
    if fmt == "INT4":
        c = unpack_bits(p["codes"], 4, math.prod(shape)).to(torch.int8)
        return (c - ((c & 0x8) << 1)).reshape(shape)                     # sign-extend 4 bits
    return p["codes"].reshape(shape)


def decode(p: dict) -> torch.Tensor:
    """The float32 tensor that `pack` stored."""
    fmt, shape = p["format"], p["shape"]
    kind, elem = FORMATS[fmt]
    n = math.prod(shape)
    if kind == "cast":
        return p["codes"].to(torch.float32)
    if kind == "scaled":
        z = int_codes(p).to(torch.float32) if elem is None else _element_values(p, elem, n).reshape(shape)
        return z * p["scale"]
    if kind == "block":
        s = torch.exp2(p["block_exp"].to(torch.float32) - 127)
    else:
        elem = "fp4_e2m1"
        bs = code_table("fp8_e4m3")[p["block_scale"].to(torch.int64)]
        s = torch.where(bs > 0, bs * p["tensor_scale"], torch.ones_like(bs))
    z = _element_values(p, elem, n).reshape(shape)
    zb, pad = _blocks(z, p["block"])
    return _unblock(zb * s.reshape(*zb.shape[:-1], 1), pad)


def round_stats(x: torch.Tensor, fmt: str, cfg: dict, scale: torch.Tensor | None = None) -> tuple[int, int, int]:
    """(values, saturated, underflow) of rounding x onto fmt: saturated = beyond the format's range (they clamp; FP16 /
    BF16: beyond the dtype's max, they become inf), underflow = nonzero values that round to zero. Scaled formats:
    beyond max x scale by more than 1e-6 of it (as exp14's clipped_fraction: a row's largest weight sits exactly at
    the max and must not count from round-off)."""
    kind, elem = FORMATS[fmt]
    if kind == "cast":
        sat = x.abs() > torch.finfo(CAST_DTYPES[fmt]).max
    elif kind == "scaled":
        sat = x.abs() > scale * format_max(fmt) * (1 + 1e-6)
    else:
        xb, pad = _blocks(x, cfg["mx_block"] if kind == "block" else cfg["nvfp4_block"])
        s = _mx_scale(xb, elem) if kind == "block" else _nvfp4_scales(xb)[2]
        sat = _unblock((xb / s).abs() > ELEMENTS[elem]["max"], pad)
    under = (x != 0) & (round_to(x, fmt, cfg, scale) == 0)
    return x.numel(), int(sat.sum()), int(under.sum())


def value_bits(fmt: str) -> int:
    """Bits of one element (FP32: 32); block and tensor scales not counted."""
    if fmt == "FP32":
        return 32
    kind, elem = FORMATS[fmt]
    if kind == "cast":
        return 16
    if elem is None:
        return {"INT4": 4, "INT8": 8, "INT16": 16, "INT32": 32}[fmt]
    return code_bits(elem)


def bytes_per_point(fmt: str, width: int, cfg: dict) -> float:
    """Bytes one point's `width` values take held in `fmt`: the elements plus the block scale bytes (MX: one E8M0
    byte, NVFP4: one E4M3 byte per block); a per-tensor scale is shared by all points and not counted."""
    kind = "fp32" if fmt == "FP32" else FORMATS[fmt][0]
    out = width * value_bits(fmt) / 8
    if kind in ("block", "nvfp4"):
        out += math.ceil(width / (cfg["mx_block"] if kind == "block" else cfg["nvfp4_block"]))
    return out


def stored_bytes(p: dict) -> int:
    """Bytes the packed tensor takes: codes plus every scale."""
    return sum(v.numel() * v.element_size() for k, v in p.items()
               if k in ("codes", "scale", "block_exp", "block_scale", "tensor_scale"))
