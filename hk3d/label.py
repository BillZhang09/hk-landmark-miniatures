"""Engraved text on the underside of the model (name, coordinates, scale)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import manifold3d as m3d
import numpy as np

# Fonts with CJK coverage first (place names are often Chinese); matplotlib's
# bundled DejaVu Sans is the last resort and covers Latin only.
FONT_CANDIDATES = [
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
    "C:/Windows/Fonts/msjhbd.ttc",
    "C:/Windows/Fonts/msjh.ttc",
]


@lru_cache(maxsize=1)
def _font():
    from matplotlib.font_manager import FontProperties

    for f in FONT_CANDIDATES:
        if Path(f).exists():
            return FontProperties(fname=f)
    return FontProperties(family="DejaVu Sans", weight="bold")


def _has_cjk(s: str) -> bool:
    return any(ord(ch) > 0x2E80 for ch in s)


def text_section(lines: list[str], line_height: float) -> tuple[m3d.CrossSection, float, float]:
    """2D cross-section of centred multi-line text. Returns (section, width, height)."""
    from matplotlib.textpath import TextPath

    prop = _font()
    polys_all = []
    y = 0.0
    widths = []
    for i, line in enumerate(lines):
        size = line_height * (1.0 if i == 0 else 0.72)
        tp = TextPath((0, 0), line, size=size, prop=prop)
        polys = [p for p in tp.to_polygons(closed_only=True) if len(p) >= 3]
        if not polys:
            continue
        allp = np.concatenate(polys)
        x0, x1 = allp[:, 0].min(), allp[:, 0].max()
        widths.append(x1 - x0)
        y -= size * 1.25
        for p in polys:
            q = p.copy()
            q[:, 0] -= (x0 + x1) / 2
            q[:, 1] += y
            polys_all.append(q[:-1] if np.allclose(q[0], q[-1]) else q)
    if not polys_all:
        return m3d.CrossSection(), 0.0, 0.0
    cs = m3d.CrossSection([p.astype(np.float64) for p in polys_all], m3d.FillRule.EvenOdd)
    b = cs.bounds()  # (xmin, ymin, xmax, ymax)
    cs = cs.translate((-(b[0] + b[2]) / 2, -(b[1] + b[3]) / 2))
    return cs, b[2] - b[0], b[3] - b[1]


def engraving(lines: list[str], diameter: float, depth: float) -> m3d.Manifold | None:
    """Solid to subtract from the model so the text reads correctly from below."""
    lines = [ln for ln in lines if ln.strip()]
    if not lines:
        return None
    lh = diameter * 0.075
    cs, w, h = text_section(lines, lh)
    if w <= 0:
        return None
    fit = min(1.0, diameter * 0.62 / w, diameter * 0.45 / h)
    cs = cs.scale((fit, fit))
    # mirror in x: the text is read looking up at the underside
    cs = cs.mirror((1, 0))
    return m3d.Manifold.extrude(cs, depth + 0.5).translate((0, 0, -0.5))
