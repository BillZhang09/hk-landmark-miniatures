"""Split a model bigger than the printer into pieces that are printed one by
one and put back together.

Seams across the model are paths, not straight lines: each one is the
cheapest way across the model within the band the printer's bed allows, where
crossing a cell costs the height of what stands on it. So seams run down
streets and round buildings, and they cut through a building only where no
street gets past it.

Each piece then fits on the bed:

* in plan, x seams wander inside a band chosen so that no piece is wider
  than the bed (the same for y). A seam moves at most one cell per cell it
  advances (45 degrees), so an x seam and a y seam cross only once and every
  piece is one region;
* in height, a piece taller than the bed is cut level, at the heights in its
  allowed range where the section stays the same just above and below
  (not next to a roof or a setback, which would leave a thin wafer) and has
  room for a pin.

Pieces are aligned by pins: short pieces of filament (or dowels) pushed into
holes drilled across the seams. Holes in the base run horizontally across the
seam, with a 45 degree peak so they print without support; holes in a level
cut run vertically. Every piece carries its name (A1, B2, B2-2 for the
second level...) engraved underneath, and an assembly sheet shows where each
goes.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import manifold3d as m3d
import numpy as np
from scipy import ndimage

from .printable import _raster

BIG = 1.0e4


@dataclass
class Piece:
    name: str
    tile: tuple[int, int]        # (column, row), row 0 in the north
    level: int                   # 1 = standing on the base
    man: m3d.Manifold            # in model coordinates
    z0: float = 0.0
    z1: float = 0.0


@dataclass
class Plan:
    nx: int
    ny: int
    xpaths: list                 # per x seam: column index of the seam for every row
    ypaths: list                 # per y seam: row index of the seam for every column
    col: np.ndarray = field(repr=False)    # tile column of every cell
    row: np.ndarray = field(repr=False)    # tile row of every cell


def counts(extent: tuple[float, float], bed: tuple[float, float], margin: float) -> tuple[int, int]:
    """Pieces needed across x and y for a model ``extent`` wide on a bed."""
    return tuple(max(1, math.ceil(e / max(b - 2 * margin, 1.0) - 1e-9)) for e, b in zip(extent, bed))


def tile_name(i: int, j: int, level: int = 1) -> str:
    col = ""
    k = i
    while True:
        col = chr(ord("A") + k % 26) + col
        k = k // 26 - 1
        if k < 0:
            break
    return f"{col}{j + 1}" + (f"-{level}" if level > 1 else "")


# --------------------------------------------------------------- seams
def _seam(C: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Cheapest top-to-bottom path through cost rows ``C`` within columns
    lo..hi, moving at most one column per row. Returns a column per row."""
    W = C[:, lo:hi + 1]
    R, n = W.shape
    d = math.sqrt(2.0)
    E = W[0].copy()
    back = np.zeros((R, n), np.int8)
    inf = np.full(1, np.inf)
    for r in range(1, R):
        w = W[r]
        cand = np.stack([np.concatenate([inf, E[:-1]]) + d * w, E + w, np.concatenate([E[1:], inf]) + d * w])
        k = np.argmin(cand, axis=0)
        E = cand[k, np.arange(n)]
        back[r] = k.astype(np.int8) - 1
    path = np.empty(R, np.int64)
    c = int(np.argmin(E))
    for r in range(R - 1, -1, -1):
        path[r] = c
        c += int(back[r, c])
    return path + lo


def plan_seams(height: np.ndarray, inside: np.ndarray, xs: np.ndarray, ys: np.ndarray,
               bounds: tuple[float, float, float, float], bed: tuple[float, float], margin: float,
               keep_off: float = 1.5) -> Plan:
    """Seams for a model whose top surface is ``height`` (mm, cells centred on
    ``xs`` ascending and ``ys`` descending) and whose outline spans
    ``bounds`` = (x0, y0, x1, y1)."""
    cell = float(xs[1] - xs[0])
    x0, y0, x1, y1 = bounds
    nx, ny = counts((x1 - x0, y1 - y0), bed, margin)
    # what a seam cuts through: the height of anything within keep_off of
    # it, so seams stay a little away from walls; a small constant keeps
    # them straight where nothing is in the way
    k = 2 * int(round(keep_off / cell)) + 1
    C = np.where(inside, ndimage.maximum_filter(np.where(inside, height, 0.0), size=k), 0.0) + 0.01
    R = len(ys)

    def band(lo_edge, extent, n, b, coords):
        s = extent / n
        w = max(0.0, (b - 2 * margin - s) / 2)
        out = []
        for i in range(1, n):
            c = lo_edge + i * s
            a, z = c - w, c + w
            ia, iz = int(np.searchsorted(coords, a)), int(np.searchsorted(coords, z, side="right")) - 1
            ia, iz = max(0, ia), min(len(coords) - 1, iz)
            if iz < ia:
                ia = iz = int(np.argmin(np.abs(coords - c)))
            out.append((ia, iz))
        return out

    xpaths = [_seam(C, a, z) for a, z in band(x0, x1 - x0, nx, bed[0], xs)]
    # rows from the north: y seam j sits below row j
    ypaths = [_seam(C.T, a, z) for a, z in band(-y1, y1 - y0, ny, bed[1], -ys)]
    rr, cc = np.mgrid[0:R, 0:len(xs)]
    col = np.zeros((R, len(xs)), np.int32)
    for p in xpaths:
        col += cc > p[:, None]
    row = np.zeros((R, len(xs)), np.int32)
    for p in ypaths:
        row += rr > p[None, :]
    return Plan(nx, ny, xpaths, ypaths, col, row)


def regions(plan: Plan, xs: np.ndarray, ys: np.ndarray) -> dict[tuple[int, int], m3d.CrossSection]:
    """Outline of every piece in plan (CrossSections in model mm)."""
    cell = float(xs[1] - xs[0])
    top, bot = ys[0] + 10 * cell, ys[-1] - 10 * cell
    west, east = xs[0] - 10 * cell, xs[-1] + 10 * cell
    everything = m3d.CrossSection.square((2 * BIG, 2 * BIG)).translate((-BIG, -BIG))

    def cs(pts):
        return m3d.CrossSection([np.asarray(pts, np.float64)], m3d.FillRule.EvenOdd).simplify(0.25 * cell)

    west_of = [m3d.CrossSection()]
    for p in plan.xpaths:
        line = np.stack([xs[p], ys], 1)
        west_of.append(cs(np.concatenate([[[line[0, 0], top]], line, [[line[-1, 0], bot], [-BIG, bot],
                                                                       [-BIG, top]]])))
    west_of.append(everything)
    north_of = [m3d.CrossSection()]
    for p in plan.ypaths:
        line = np.stack([xs, ys[p]], 1)
        north_of.append(cs(np.concatenate([[[west, line[0, 1]]], line, [[east, line[-1, 1]], [east, BIG],
                                                                        [west, BIG]]])))
    north_of.append(everything)
    out = {}
    for i in range(plan.nx):
        colr = west_of[i + 1] - west_of[i]
        for j in range(plan.ny):
            out[(i, j)] = colr ^ (north_of[j + 1] - north_of[j])
    return out


# ---------------------------------------------------------------- pins
def teardrop_hole(d: float, length: float) -> m3d.Manifold:
    """Horizontal hole of diameter d along y, centred on the origin, with a
    45 degree peak on top so it prints without support."""
    r = d / 2
    t = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    pts = np.concatenate([np.stack([r * np.cos(t), r * np.sin(t)], 1), [[0.0, r * math.sqrt(2)]]])
    sec = m3d.CrossSection.hull_points(pts.astype(np.float64))
    # section in x-z, extruded along y
    return m3d.Manifold.extrude(sec, length).translate((0, 0, -length / 2)).rotate((90, 0, 0))


def min_base(d: float, wall: float = 0.8) -> float:
    """Base thick enough for a teardrop hole of diameter d through its middle."""
    return d / 2 + d / 2 * math.sqrt(2) + 2 * wall


def base_pins(plan: Plan, xs: np.ndarray, ys: np.ndarray, inside: np.ndarray, *, d: float,
              depth: float, spacing: float = 50.0, edge: float = 4.0) -> list[tuple[float, float, float, float]]:
    """Pins across the seams in the base: (x, y, ux, uy), (ux, uy) being the
    pin's direction (across the seam). Pins keep clear of the outline and of
    the seams crossing this one; a stretch of seam between two crossings
    gets one pin when short, otherwise one every ``spacing`` mm (at least two)."""
    cell = float(xs[1] - xs[0])
    clear = depth + d / 2 + edge
    dist = ndimage.distance_transform_edt(np.pad(inside, 1))[1:-1, 1:-1] * cell
    pins = []

    def along(rows, cols, piece, crossing):
        P = np.stack([xs[cols], ys[rows]], 1)
        ok = dist[rows, cols] >= clear
        for q in crossing:
            ok &= q >= clear / cell
        n, k = len(P), 0
        while k < n:
            if not ok[k]:
                k += 1
                continue
            a = k
            while k < n and ok[k] and piece[k] == piece[a]:
                k += 1
            seg = P[a:k]
            if len(seg) < 2:
                continue
            s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(seg, axis=0).T))])
            L = s[-1]
            if L < 2 * d:
                continue
            m = 1 if L < 30 else max(2, int(round(L / spacing)))
            for q in (np.arange(m) + 0.5) * L / m:
                i = int(np.clip(np.searchsorted(s, q), 1, len(seg) - 1))
                p = seg[i - 1] + (seg[i] - seg[i - 1]) * (q - s[i - 1]) / max(s[i] - s[i - 1], 1e-9)
                t = seg[min(len(seg) - 1, i + 4)] - seg[max(0, i - 5)]
                t = t / max(float(np.hypot(*t)), 1e-9)
                pins.append((float(p[0]), float(p[1]), float(t[1]), float(-t[0])))

    R, C = inside.shape
    for p in plan.xpaths:
        rows = np.arange(R)
        along(rows, p, plan.row[rows, p], [np.abs(rows - q[p]) for q in plan.ypaths])
    for p in plan.ypaths:
        cols = np.arange(C)
        along(p, cols, plan.col[p, cols], [np.abs(cols - q[p]) for q in plan.xpaths])
    return pins


def pin_holes(pins, *, z: float, d: float, depth: float) -> m3d.Manifold | None:
    if not pins:
        return None
    hole = teardrop_hole(d, 2 * depth)
    out = []
    for x, y, nx_, ny_ in pins:
        # the hole runs along y: turn y onto the pin's direction
        ang = math.degrees(math.atan2(-nx_, ny_))
        out.append(hole.rotate((0, 0, ang)).translate((x, y, z)))
    return m3d.Manifold.batch_boolean(out, m3d.OpType.Add)


# ------------------------------------------------------------- levels
def _section_stats(cs: m3d.CrossSection) -> tuple[float, list[tuple[float, float]]]:
    comps = []
    for c in cs.decompose():
        a = c.area()
        per = sum(float(np.hypot(*np.diff(np.vstack([p, p[:1]]), axis=0).T).sum()) for p in c.to_polygons())
        comps.append((a, 2 * a / max(per, 1e-9)))
    return cs.area(), comps


def level_cuts(T: m3d.Manifold, top: float, zmax: float, *, floor: float, need_r: float,
               step: float = 1.0) -> list[float]:
    """Heights to cut a piece ``top`` mm tall into pieces at most zmax tall."""
    n = math.ceil(top / zmax - 1e-9)
    if n <= 1:
        return []
    windows = [(max(top - (n - k) * zmax, floor), k * zmax) for k in range(1, n)]
    cand = sorted({round(h, 3) for a, b in windows for h in np.arange(a, b + 1e-6, step)})
    need = sorted({round(h + o, 3) for h in cand for o in (-2.0, 0.0, 5.0)})
    area = {}
    comps = {}
    for h in need:
        a, c = _section_stats(T.slice(h))
        area[h], comps[h] = a, c
    cost = {}
    amax = max(max(area[h] for h in cand), 1.0)
    for h in cand:
        a = max(area[h], 1.0)
        # a big section is a strong joint: never the tip of a spire
        c0 = 2.0 * (1.0 - area[h] / amax)
        c = c0 + 3.0 * (abs(area[round(h - 2.0, 3)] - area[h]) + abs(area[round(h + 5.0, 3)] - area[h])) / a
        for ca, rho in comps[h]:
            if rho < 1.0:
                c += 5.0
            elif rho < need_r:
                c += 1.0
        cost[h] = c
    # cheapest choice of one height per window with gaps of at most zmax
    best = {h: (cost[h], [h]) for h in cand if windows[0][0] - 1e-6 <= h <= windows[0][1] + 1e-6}
    for a, b in windows[1:]:
        nb = {}
        for h in cand:
            if not (a - 1e-6 <= h <= b + 1e-6):
                continue
            prev = [(v[0], v[1]) for g, v in best.items() if g < h and h - g <= zmax + 1e-6]
            if prev:
                c, path = min(prev, key=lambda v: v[0])
                nb[h] = (c + cost[h], path + [h])
        best = nb
    ok = [v for h, v in best.items() if top - h <= zmax + 1e-6]
    if not ok:  # numerical corner: fall back to equal parts
        return [top * k / n for k in range(1, n)]
    return min(ok, key=lambda v: v[0])[1]


def _mask(T: m3d.Manifold, h: float, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    return _raster(T.slice(h).to_polygons(), gx, gy)


def level_pins(T: m3d.Manifold, h: float, *, d: float, depth: float, wall: float = 1.2,
               res: float = 0.25) -> tuple[list[tuple[float, float]], np.ndarray, tuple]:
    """Vertical pins across the level cut at h: up to two per separate part
    of the section, where the hole has a wall of at least ``wall`` mm all
    along its depth. Returns the pins and the section mask (for labels)."""
    b = T.bounding_box()
    gx = np.arange(b[0] + res / 2, b[3], res)
    gy = np.arange(b[4] - res / 2, b[1], -res)
    if len(gx) < 3 or len(gy) < 3:
        return [], np.zeros((1, 1), bool), (gx, gy)
    M = _mask(T, h, gx, gy)
    both = M & _mask(T, h - depth + 0.2, gx, gy) & _mask(T, h + depth - 0.2, gx, gy)
    dist = ndimage.distance_transform_edt(np.pad(both, 1))[1:-1, 1:-1] * res
    need = d / 2 + wall
    lab, n = ndimage.label(M)
    pins = []
    for k in range(1, n + 1):
        dk = np.where(lab == k, dist, 0.0)
        if dk.max() < need:
            continue
        r0, c0 = np.unravel_index(int(np.argmax(dk)), dk.shape)
        pins.append((float(gx[c0]), float(gy[r0])))
        rr, cc = np.nonzero(dk >= need)
        sep = np.hypot((rr - r0) * res, (cc - c0) * res)
        if sep.size and sep.max() >= 4 * d + 2 * wall:
            q = int(np.argmax(sep))
            pins.append((float(gx[cc[q]]), float(gy[rr[q]])))
    return pins, M, (gx, gy)


def _label_spot(M: np.ndarray, grid: tuple, avoid: list, r_avoid: float) -> tuple[float, float, float]:
    gx, gy = grid
    if M.size <= 1:
        return 0.0, 0.0, 0.0
    res = float(gx[1] - gx[0])
    free = M.copy()
    if avoid:
        X, Y = np.meshgrid(gx, gy)
        for x, y in avoid:
            free &= np.hypot(X - x, Y - y) > r_avoid
    dist = ndimage.distance_transform_edt(np.pad(free, 1))[1:-1, 1:-1] * res
    r0, c0 = np.unravel_index(int(np.argmax(dist)), dist.shape)
    return float(gx[c0]), float(gy[r0]), float(dist[r0, c0])


def _text(lines, x, y, z, radius, depth, max_size=160.0):
    from .label import engraving

    if radius <= 0:
        return None
    # engraving() fits the text in a 0.62 x 0.45 box of the size it is
    # given; that box fits in a circle of 0.383 x the size
    size = min(radius / 0.383, max_size)
    t = engraving(lines, size, depth)
    return None if t is None else t.translate((x, y, z))


# --------------------------------------------------------------- split
@dataclass
class TileOptions:
    bed: tuple[float, float, float] = (256.0, 256.0, 256.0)
    margin: float = 5.0          # kept clear round each piece on the bed
    pin_mm: float = 1.75         # pin diameter (1.75 mm filament); 0 = no pins
    pin_clearance: float = 0.3   # hole = pin + this
    pin_depth: float = 6.0       # hole depth on each side of the seam
    engrave_depth: float = 0.6


def split(model: m3d.Manifold, height: np.ndarray, inside: np.ndarray, xs: np.ndarray, ys: np.ndarray,
          bounds: tuple, base_mm: float, opts: TileOptions, text_lines: list[str] | None = None,
          progress=None) -> tuple[list[Piece], dict, Plan, list]:
    """Cut ``model`` (model mm, base on z=0) into pieces that fit the bed."""
    def report(f, msg):
        if progress:
            progress(f, msg)

    cell = float(xs[1] - xs[0])
    bed = opts.bed
    plan = plan_seams(height, inside, xs, ys, bounds, bed[:2], opts.margin)
    shapes = regions(plan, xs, ys)
    d = opts.pin_mm + opts.pin_clearance if opts.pin_mm > 0 else 0.0
    zc = (base_mm + d / 2 - d / 2 * math.sqrt(2)) / 2   # equal walls under and over the hole
    pins = []
    cut = []
    if d and plan.nx * plan.ny > 1:
        pins = base_pins(plan, xs, ys, inside, d=d, depth=opts.pin_depth)
        h = pin_holes(pins, z=zc, d=d, depth=opts.pin_depth)
        if h is not None:
            cut.append(h)
    # name underneath each piece, clear of the pins along its edges
    if text_lines is not None:
        clear = opts.pin_depth + d + 4.0
        for (i, j), _ in shapes.items():
            m = inside & (plan.col == i) & (plan.row == j)
            if not m.any():
                continue
            dist = ndimage.distance_transform_edt(np.pad(m, 1))[1:-1, 1:-1] * cell
            r0, c0 = np.unravel_index(int(np.argmax(dist)), dist.shape)
            rad = float(dist[r0, c0]) - (clear if plan.nx * plan.ny > 1 else 2.0)
            t = _text([tile_name(i, j)] + text_lines, float(xs[c0]), float(ys[r0]), 0.0, rad, opts.engrave_depth)
            if t is not None:
                cut.append(t)
    if cut:
        model = model - m3d.Manifold.batch_boolean(cut, m3d.OpType.Add)

    zmax_bed = bed[2] - 1.0
    need_r = (d / 2 + 1.2) if d else 0.0
    pieces: list[Piece] = []
    level_pin_count = 0
    floating: list = []
    t0 = time.time()
    n_tiles = len(shapes)
    top_all = model.bounding_box()[5]
    for t_i, ((i, j), sec) in enumerate(sorted(shapes.items(), key=lambda kv: (kv[0][1], kv[0][0]))):
        report(t_i / n_tiles, f"Cutting piece {tile_name(i, j)}")
        if sec.is_empty():
            continue
        T = model ^ m3d.Manifold.extrude(sec, top_all + 10).translate((0, 0, -5))
        if T.is_empty() or T.volume() < 1.0:
            continue
        top = T.bounding_box()[5]
        hs = level_cuts(T, top, zmax_bed, floor=base_mm + 1.0, need_r=need_r) if top > zmax_bed else []
        labels = []
        if hs:
            holes = []
            for k, h in enumerate(hs):
                lp, M, grid = level_pins(T, h, d=d, depth=opts.pin_depth) if d else ([], *(_section_mask(T, h)))
                level_pin_count += len(lp)
                for x, y in lp:
                    holes.append(m3d.Manifold.cylinder(2 * opts.pin_depth, d / 2, circular_segments=24)
                                 .translate((x, y, h - opts.pin_depth)))
                x, y, rad = _label_spot(M, grid, lp, d / 2 + 1.5 if d else 0.0)
                labels.append(_text([tile_name(i, j, k + 2)], x, y, h, min(rad - 0.5, 20.0), 0.4)
                              if rad >= 2.5 else None)
            if holes:
                T = T - m3d.Manifold.batch_boolean(holes, m3d.OpType.Add)
        edges = [-5.0] + hs + [top + 5.0]
        for k in range(len(edges) - 1):
            lo, hi = edges[k], edges[k + 1]
            box = m3d.Manifold.cube((2 * BIG, 2 * BIG, hi - lo)).translate((-BIG, -BIG, lo))
            P = T ^ box
            if k and labels[k - 1] is not None:
                P = P - labels[k - 1]
            if P.is_empty() or P.volume() < 1e-3:
                continue
            P, fl = _clean(P)
            b = P.bounding_box()
            pieces.append(Piece(tile_name(i, j, k + 1), (i, j), k + 1, P, float(b[2]), float(b[5])))
            floating += [(pieces[-1], c) for c in fl]

    moved = len(floating)
    left = _reattach(pieces, floating, bed)
    sizes = []
    for p in pieces:
        b = p.man.bounding_box()
        sizes.append((b[3] - b[0], b[4] - b[1], b[5] - b[2]))
    stats = {
        "grid": f"{plan.nx}x{plan.ny}",
        "pieces": len(pieces),
        "levels": max((p.level for p in pieces), default=1),
        "pins": len(pins) + level_pin_count,
        "pin_mm": opts.pin_mm, "pin_length_mm": round(2 * opts.pin_depth - 1.0, 1) if d else 0,
        "largest_mm": [round(max(s[k] for s in sizes), 1) for k in range(3)] if sizes else [0, 0, 0],
        "fits": all(s[0] <= bed[0] and s[1] <= bed[1] and s[2] <= bed[2] for s in sizes),
        "reattached": moved - left,
        "floating": left,
        "seconds": round(time.time() - t0, 1),
    }
    return pieces, stats, plan, pins


def _clean(P: m3d.Manifold, crumb: float = 1.0) -> tuple[m3d.Manifold, list]:
    """The piece with its air pockets filled and without the parts that float
    above its bottom (returned apart: each was held up through a
    neighbouring piece); specks under ``crumb`` mm3 are dropped."""
    from .solid import standing

    P, loose, _ = standing(P, crumb)
    return P, loose


def _reattach(pieces: list, floating: list, bed) -> int:
    """Give each floating part to the piece next to it that holds it up (the
    one with the most material within 0.1 mm of it), if the two together
    still fit the bed. Returns how many parts are left floating."""
    left = 0
    for owner, c in floating:
        b = c.bounding_box()
        near = m3d.Manifold.cube((b[3] - b[0] + 0.2, b[4] - b[1] + 0.2, b[5] - b[2] + 0.2)).translate(
            (b[0] - 0.1, b[1] - 0.1, b[2] - 0.1))
        best, best_v = None, 0.0
        for q in pieces:
            if q is owner:
                continue
            qb = q.man.bounding_box()
            if qb[0] > b[3] + 0.1 or qb[3] < b[0] - 0.1 or qb[1] > b[4] + 0.1 or qb[4] < b[1] - 0.1 \
                    or qb[2] > b[5] + 0.1 or qb[5] < b[2] - 0.1:
                continue
            v = (q.man ^ near).volume()
            if v > best_v:
                best, best_v = q, v
        if best is not None:
            m = best.man + c
            mb = m.bounding_box()
            if mb[3] - mb[0] <= bed[0] and mb[4] - mb[1] <= bed[1] and mb[5] - mb[2] <= bed[2]:
                best.man, best.z0, best.z1 = m, float(mb[2]), float(mb[5])
                continue
        owner.man = owner.man + c
        left += 1
    return left


def _section_mask(T: m3d.Manifold, h: float, res: float = 0.25):
    b = T.bounding_box()
    gx = np.arange(b[0] + res / 2, b[3], res)
    gy = np.arange(b[4] - res / 2, b[1], -res)
    if len(gx) < 3 or len(gy) < 3:
        return np.zeros((1, 1), bool), (gx, gy)
    return _mask(T, h, gx, gy), (gx, gy)


def on_bed(p: Piece) -> m3d.Manifold:
    """The piece moved to stand on z=0, centred on the origin."""
    b = p.man.bounding_box()
    return p.man.translate((-(b[0] + b[3]) / 2, -(b[1] + b[4]) / 2, -b[2]))


# ------------------------------------------------------------ assembly
PIECE_COLORS = ["#c9b79c", "#9fb4c7", "#b9c99f", "#d1a6a6", "#b3a6d1", "#a6d1c7"]


def piece_color(p: Piece) -> str:
    i, j = p.tile
    return PIECE_COLORS[(i + 2 * j + 3 * (p.level - 1)) % len(PIECE_COLORS)]


def assembly_sheet(path, rgba: np.ndarray, plan: Plan, pins: list, pieces: list[Piece], xs, ys,
                   title: str = "") -> None:
    """Top view with the seams, the name of each piece and the pins."""
    from PIL import Image, ImageDraw, ImageFont

    from .label import FONT_CANDIDATES

    img = Image.fromarray(rgba, "RGBA")
    R, C = rgba.shape[:2]
    S = 1600 / max(R, C)
    img = img.resize((int(C * S), int(R * S)), Image.LANCZOS)
    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    bg.alpha_composite(img)
    dr = ImageDraw.Draw(bg)
    font = small = None
    for f in FONT_CANDIDATES:
        try:
            font = ImageFont.truetype(f, 40)
            small = ImageFont.truetype(f, 22)
            break
        except OSError:
            continue
    if font is None:
        font = small = ImageFont.load_default()
    for p in plan.xpaths:
        dr.line([((c + 0.5) * S, (r + 0.5) * S) for r, c in enumerate(p)], fill=(200, 30, 30, 255), width=4)
    for p in plan.ypaths:
        dr.line([((c + 0.5) * S, (r + 0.5) * S) for c, r in enumerate(p)], fill=(200, 30, 30, 255), width=4)
    cell = float(xs[1] - xs[0])

    def px(x, y):
        return ((x - xs[0]) / cell + 0.5) * S, ((ys[0] - y) / cell + 0.5) * S

    for x, y, *_ in pins:
        u, v = px(x, y)
        dr.ellipse([u - 6, v - 6, u + 6, v + 6], fill=(30, 30, 30, 255))
    levels: dict = {}
    for p in pieces:
        levels[p.tile] = max(levels.get(p.tile, 1), p.level)
    for (i, j), n in levels.items():
        m = (plan.col == i) & (plan.row == j) & (rgba[..., 3] > 0)
        if not m.any():
            continue
        dist = ndimage.distance_transform_edt(m)
        r0, c0 = np.unravel_index(int(np.argmax(dist)), dist.shape)
        u, v = (c0 + 0.5) * S, (r0 + 0.5) * S
        name = tile_name(i, j)
        dr.text((u, v), name, fill=(20, 20, 20, 255), font=font, anchor="mm", stroke_width=3,
                stroke_fill=(255, 255, 255, 255))
        if n > 1:
            dr.text((u, v + 38), f"{n} 層 levels", fill=(20, 20, 20, 255), font=small, anchor="mm",
                    stroke_width=2, stroke_fill=(255, 255, 255, 255))
    dr.text((16, 12), (title + "  " if title else "") + "N ↑", fill=(20, 20, 20, 255), font=small)
    bg.convert("RGB").save(path)


def assembly_text(pieces: list[Piece], stats: dict, opts: TileOptions, title: str = "") -> str:
    lines = [f"{title} — 分塊打印 Tiled print".strip(" —"), "",
             f"打印範圍 Bed: {opts.bed[0]:g} x {opts.bed[1]:g} x {opts.bed[2]:g} mm",
             f"分塊 Grid: {stats['grid']}，共 {stats['pieces']} 件 pieces，最多 {stats['levels']} 層 levels", ""]
    if stats["pins"]:
        lines += [f"定位銷 Pins: {stats['pins']} 支，直徑 {opts.pin_mm:g} mm、長 {stats['pin_length_mm']:g} mm"
                  f"（例如把 {opts.pin_mm:g} mm 耗材剪成 {stats['pin_length_mm']:g} mm 一段）。",
                  f"Cut {stats['pins']} pins of {opts.pin_mm:g} mm filament or dowel, "
                  f"{stats['pin_length_mm']:g} mm long.", ""]
    lines += ["每件的名稱刻在底部；A、B、C… 由西至東，1、2、3… 由北至南；-2、-3 是疊在上面的第二、三層。",
              "Each piece has its name engraved underneath: columns A, B, C... west to east, rows 1, 2, 3... "
              "north to south; -2, -3 are the levels stacked on top.", "",
              "件 Piece    闊 W    深 D    高 H (mm)    位於 at z (mm)"]
    for p in sorted(pieces, key=lambda p: (p.tile[1], p.tile[0], p.level)):
        b = p.man.bounding_box()
        lines.append(f"{p.name:<10} {b[3] - b[0]:6.1f} {b[4] - b[1]:6.1f} {b[5] - b[2]:6.1f}    "
                     f"{p.z0:6.1f} – {p.z1:6.1f}")
    return "\n".join(lines) + "\n"
