"""HK 1:1000 map sheet grid (e.g. ``11-NE-18B``) in HK1980 Grid coordinates.

1:20000 sheets 1-16 form a 4x4 grid of 15000 m x 12000 m whose NW corner is
(800000 + 15000*col, 848000 - 12000*row). Sheet 17 is an outlier at
(860000, 848000). Each is split into NW/NE/SW/SE quadrants (7500 x 6000 m),
each quadrant into 25 blocks (5x5, row-major from NW, 1500 x 1200 m) and each
block into A/B/C/D (NW/NE/SW/SE) 1:1000 sheets of 750 x 600 m.
"""
from __future__ import annotations

import re

SHEET_W, SHEET_H = 750.0, 600.0
_RE = re.compile(r"^(\d{1,2})-(NW|NE|SW|SE)-(\d{1,2})([A-D])$")
_QUAD = {"NW": (0, 0), "NE": (1, 0), "SW": (0, 1), "SE": (1, 1)}
_LETTER = {"A": (0, 0), "B": (1, 0), "C": (0, 1), "D": (1, 1)}


def _sheet20k_nw(num: int) -> tuple[float, float]:
    if num == 17:
        return 860000.0, 848000.0
    if not 1 <= num <= 16:
        raise ValueError(f"unknown 1:20000 sheet {num}")
    col, row = (num - 1) % 4, (num - 1) // 4
    return 800000.0 + 15000 * col, 848000.0 - 12000 * row


def is_sheet_name(name: str) -> bool:
    return bool(_RE.match(name))


def sheet_bounds(name: str) -> tuple[float, float, float, float]:
    """Return (xmin, ymin, xmax, ymax) in HK1980 for a 1:1000 sheet name."""
    m = _RE.match(name)
    if not m:
        raise ValueError(f"not a 1:1000 sheet name: {name}")
    num, quad, block, letter = int(m[1]), m[2], int(m[3]), m[4]
    if not 1 <= block <= 25:
        raise ValueError(f"bad block number in {name}")
    x, y = _sheet20k_nw(num)
    qc, qr = _QUAD[quad]
    x += qc * 7500
    y -= qr * 6000
    br, bc = divmod(block - 1, 5)
    x += bc * 1500
    y -= br * 1200
    lc, lr = _LETTER[letter]
    x += lc * SHEET_W
    y -= lr * SHEET_H
    return x, y - SHEET_H, x + SHEET_W, y


def sheet_at(e: float, n: float) -> str | None:
    """1:1000 sheet name containing HK1980 point (e, n)."""
    candidates = [17] + list(range(1, 17))
    for num in candidates:
        x0, y0 = _sheet20k_nw(num)
        if not (x0 <= e < x0 + 15000 and y0 - 12000 < n <= y0):
            continue
        dx, dy = e - x0, y0 - n
        qc, qr = int(dx // 7500), int(dy // 6000)
        quad = {(0, 0): "NW", (1, 0): "NE", (0, 1): "SW", (1, 1): "SE"}[(qc, qr)]
        dx -= qc * 7500
        dy -= qr * 6000
        bc, br = int(dx // 1500), int(dy // 1200)
        dx -= bc * 1500
        dy -= br * 1200
        lc, lr = int(dx // SHEET_W), int(dy // SHEET_H)
        letter = {(0, 0): "A", (1, 0): "B", (0, 1): "C", (1, 1): "D"}[(lc, lr)]
        return f"{num}-{quad}-{br * 5 + bc + 1}{letter}"
    return None


def sheets_in_bbox(xmin: float, ymin: float, xmax: float, ymax: float) -> list[str]:
    """All 1:1000 sheet names intersecting the bbox (HK1980)."""
    out = []
    eps = 1e-6
    x = xmin
    xs = []
    while x < xmax + SHEET_W:
        xs.append(min(x, xmax - eps))
        x += SHEET_W
    y = ymin
    ys = []
    while y < ymax + SHEET_H:
        ys.append(min(y, ymax - eps))
        y += SHEET_H
    for yy in ys:
        for xx in xs:
            s = sheet_at(xx, yy + eps)
            if s and s not in out:
                out.append(s)
    return sorted(out)
