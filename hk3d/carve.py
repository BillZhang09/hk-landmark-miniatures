"""Precise landmark solids by carving a 2.5D block from several directions.

The heightmap pipeline samples every surface from straight above.  For a tall
landmark that has two visible costs: facades that are not aligned with the
sampling grid become fine stair-steps, and anything that can only be seen from
the side (the gap under a mast truss, a recessed podium) is filled in.

LandsD building meshes are rarely closed, so they cannot be used as solids
directly.  Instead we build a *depth hull*:

    L = Top  ∩  View(d1)  ∩  View(d2)  ∩ ...

* ``Top`` is the building's own heightmap, grown by one cell so its
  stair-steps lie outside the true facade.
* ``View(d)`` is everything behind the first surface a ray travelling in
  direction ``d`` hits.  The directions are the building's dominant facade
  normals (both ways), so each facade becomes an exact plane and building
  corners - where two views meet - come out sharp.

Every operand is a heightfield solid (manifold by construction) and the
booleans are done by manifold3d, so the result is always watertight.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

import manifold3d as m3d

from .solid import grid_solid


def facade_axes(V: np.ndarray, F: np.ndarray, max_axes: int = 4, min_frac: float = 0.06,
                min_sep_deg: float = 8.0) -> list[float]:
    """Dominant facade orientations (radians, modulo pi), largest first."""
    T = V[F]
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    area = np.linalg.norm(n, axis=1) / 2
    with np.errstate(invalid="ignore", divide="ignore"):
        n = n / np.linalg.norm(n, axis=1, keepdims=True)
    vert = np.isfinite(n).all(1) & (np.abs(n[:, 2]) < 0.1) & (area > 0)
    if not vert.any():
        return []
    ang = np.degrees(np.arctan2(n[vert, 1], n[vert, 0])) % 180.0
    hist, _ = np.histogram(ang, bins=360, range=(0, 180), weights=area[vert])
    # circular smoothing over +-1 degree
    hist = ndimage.uniform_filter1d(hist, 5, mode="wrap")
    out: list[float] = []
    total = hist.sum()
    floor = min_frac * total / 5
    if hist.max() < floor:
        # faceted: walls in many directions, none dominant.  Rather than no
        # side views at all, take the strongest directions there are
        floor = 0.5 * hist.max()
    for b in np.argsort(-hist):
        if hist[b] < floor or len(out) >= max_axes:
            break
        a = (b + 0.5) * 0.5
        if all(min(abs(a - o), 180 - abs(a - o)) >= min_sep_deg for o in out):
            out.append(a)
    # refine each axis to the area-weighted mean of nearby faces
    refined = []
    for a in out:
        d = (ang - a + 90) % 180 - 90
        near = np.abs(d) < 2.0
        w = area[vert][near]
        refined.append(np.radians(a + (np.average(d[near], weights=w) if w.sum() > 0 else 0.0)))
    return refined


def _scene(V: np.ndarray, F: np.ndarray):
    import open3d as o3d
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.core.Tensor(V.astype(np.float32)), o3d.core.Tensor(F.astype(np.uint32)))
    return sc


def _cast(scene, origins: np.ndarray, d: np.ndarray, full: bool = False):
    import open3d as o3d
    rays = np.concatenate([origins, np.broadcast_to(d, origins.shape)], -1).astype(np.float32)
    r = scene.cast_rays(o3d.core.Tensor(rays))
    if not full:
        return r["t_hit"].numpy()
    return {k: r[k].numpy() for k in ("t_hit", "primitive_ids", "primitive_uvs", "primitive_normals")}


class TextureSampler:
    """Colour of the building's diffuse texture at ray hits."""

    def __init__(self, uv: np.ndarray, tex: np.ndarray, files: list[str]):
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        self.uv, self.tex = uv, tex
        self.images = []
        for f in files:
            try:
                self.images.append(np.asarray(Image.open(f).convert("RGB")))
            except Exception:  # unreadable texture: treat as untextured
                self.images.append(None)

    def sample(self, prim: np.ndarray, bary: np.ndarray, hit: np.ndarray) -> np.ndarray:
        """RGB in 0..1 (NaN where there is no hit or no texture)."""
        out = np.full(prim.shape + (3,), np.nan, np.float32)
        pid = np.where(hit, prim, 0).astype(np.int64)
        w = np.stack([1 - bary[..., 0] - bary[..., 1], bary[..., 0], bary[..., 1]], -1)
        uv = (self.uv[pid] * w[..., None]).sum(-2)
        t = np.where(hit, self.tex[pid], -1)
        for i, im in enumerate(self.images):
            m = t == i
            if im is None or not m.any():
                continue
            h, wd = im.shape[:2]
            x = ((uv[m, 0] % 1.0) * wd).astype(np.int64).clip(0, wd - 1)
            y = ((1.0 - uv[m, 1] % 1.0) * h).astype(np.int64).clip(0, h - 1)
            out[m] = im[y, x] / 255.0
        return out


def flat_texture(rgb: np.ndarray, ok: np.ndarray, cell: float, size: float = 1.0,
                 tol: float = 1.0 / 255) -> np.ndarray:
    """Cells whose texture does not vary at all over ``size`` metres (every
    channel's standard deviation under ``tol``), and the half window around
    them where the fill blends into the photo."""
    w = max(3, int(round(size / cell)) | 1)
    m = ok.astype(np.float64)
    den = ndimage.uniform_filter(m, w)
    x = np.where(ok[..., None], np.nan_to_num(rgb), 0.0).astype(np.float64)
    sd = np.zeros(ok.shape)
    for c in range(3):
        mu = ndimage.uniform_filter(x[..., c], w) / np.maximum(den, 1e-9)
        var = ndimage.uniform_filter(x[..., c] ** 2, w) / np.maximum(den, 1e-9) - mu * mu
        sd = np.maximum(sd, np.sqrt(np.maximum(var, 0.0)))
    flat = ok & (den > 0.5) & (sd < tol)
    if flat.any():
        flat = ndimage.binary_dilation(flat, iterations=w // 2) & ok
    return flat


def _line_se(length: int, ang: float) -> np.ndarray:
    n = length // 2
    t = np.arange(-n, n + 1)
    se = np.zeros((2 * n + 1, 2 * n + 1), bool)
    se[np.round(-t * np.sin(ang)).astype(int) + n, np.round(t * np.cos(ang)).astype(int) + n] = True
    return se


def _long_bands(p: np.ndarray, cell: float, min_width: float, min_length: float) -> np.ndarray:
    """Keep the parts of a mask that are bands at least ``min_width`` wide
    running straight for ``min_length`` in some direction, in pieces whose
    overall extent is at least twice that."""
    k = max(2, int(round(min_width / cell)))
    p = ndimage.binary_opening(p, np.ones((k, k), bool))
    n = max(3, int(round(min_length / cell)) | 1)
    out = np.zeros_like(p)
    for a in np.radians(np.arange(0.0, 180.0, 15.0)):  # a band k wide tolerates +-atan(k/n)
        out |= ndimage.binary_opening(p, _line_se(n, a))
    lab, nlab = ndimage.label(out, np.ones((3, 3), bool))
    if nlab:
        objs = ndimage.find_objects(lab)
        ext = np.array([np.hypot(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start) for sl in objs]) * cell
        out = np.isin(lab, 1 + np.nonzero(ext >= 2 * min_length)[0])
    return out


def facade_pattern(rgb: np.ndarray, ok: np.ndarray, cell: float, contrast: float = 0.03,
                   max_sat: float = 0.15, min_width: float = 0.75, min_length: float = 4.0,
                   background: float = 8.0, colour_tol: float = 0.1, grid_edge: float = 0.03,
                   smooth: float = 0.0) -> np.ndarray:
    """Structural lines painted on a facade texture: bracing, frames, spandrels.

    Seeds are cells lighter than the facade around them (``background``
    metres) and nearly grey (cladding, not a coloured reflection) that form
    bands at least ``min_width`` wide - so thin window mullions drop out;
    the texture is first averaged over ``smooth`` metres, so a band that is
    itself finely striped still reads as one -
    running straight for at least ``min_length`` - so blotchy reflections drop
    out.  Glass that mirrors something bright can hide a band from that test,
    so the cladding colour is then learnt from the seeds, and cells of that
    colour - or grey cells of a similar tone that the window grid does not
    cross - are added where they form such bands connected to a seed.
    """
    okf = ok.astype(np.float32)
    raw = np.nan_to_num(rgb)
    if smooth > 0:
        # average out what is finer than the print can show (masked blur)
        sg = smooth / cell
        den = np.maximum(ndimage.gaussian_filter(okf, sg), 1e-6)
        rgb = np.stack([ndimage.gaussian_filter(raw[..., i] * okf, sg) / den for i in range(3)], -1)
        rgb = np.where(ok[..., None], rgb, np.nan).astype(np.float32)
    L = np.nan_to_num(rgb.mean(-1))
    sat = np.nan_to_num(rgb.max(-1) - rgb.min(-1), nan=1.0)
    w = max(3, int(round(background / cell)))
    bg = ndimage.uniform_filter(np.where(ok, L, 0), w) / np.maximum(ndimage.uniform_filter(okf, w), 1e-6)
    strong = ok & (L > bg + contrast) & (sat < max_sat)
    seeds = _long_bands(strong, cell, min_width, min_length)
    if seeds.sum() < 50:
        return seeds
    ref = np.median(rgb[seeds], axis=0)
    diff = np.abs(np.nan_to_num(rgb, nan=9.0) - ref).max(-1)
    alike = ok & (diff < colour_tol)
    # cladding is also plain where glazing carries the window grid: grey cells
    # of roughly the seeds' tone with no texture edge nearby
    L0 = raw.mean(-1)
    g = np.hypot(ndimage.sobel(L0, 0), ndimage.sobel(L0, 1)) / 8.0
    plain = ok & ~ndimage.binary_dilation(g > grid_edge) & (sat < max_sat) & (diff < 2.5 * colour_tol)
    grown = _long_bands(strong | alike | plain, cell, min_width, min_length)
    lab, nlab = ndimage.label(grown, np.ones((3, 3), bool))
    if not nlab:
        return seeds
    has_seed = ndimage.maximum(seeds, lab, np.arange(1, nlab + 1)) > 0
    return np.isin(lab, 1 + np.nonzero(has_seed)[0]) | seeds


def _disk(r: int) -> np.ndarray:
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return xx * xx + yy * yy <= r * r


def _widen(mask: np.ndarray, width_cells: float) -> np.ndarray:
    """Grow the parts of a mask that are thinner than ``width_cells`` (they
    are at least 3/4 of it) so every band can be printed."""
    if not mask.any():
        return mask
    core = ndimage.binary_opening(mask, _disk(max(1, int(np.ceil(width_cells / 2)))))
    thin = mask & ~core
    if not thin.any():
        return mask
    return mask | ndimage.binary_dilation(thin, _disk(max(1, int(round(width_cells / 4)))))


def _straighten(p: np.ndarray, aspect: float = 4.0, cover: float = 0.5) -> np.ndarray:
    """Upright and level bands as clean rectangles.

    A pier or spandrel read from a photo has ragged sides (windows, tiles,
    shading eat into it).  Each piece at least ``aspect`` times taller than
    wide (or wider than tall) becomes the columns (rows) it covers over at
    least ``cover`` of its length.  Only pieces that fill most of their
    bounding box are bands: bracing joined to a frame, or a grid, is
    elongated too, but mostly empty, and is left as it is.
    """
    lab, n = ndimage.label(p, np.ones((3, 3), bool))
    if not n:
        return p
    out = p.copy()
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if max(h, w) < aspect * min(h, w):
            continue
        m = lab[sl] == i
        if m.mean() < 0.75:
            continue
        cov = m.mean(0) if h > w else m.mean(1)
        keep = cov >= cover
        if not keep.any():
            continue
        out[sl] &= ~m
        if h > w:
            out[sl] |= keep[None, :]
        else:
            out[sl] |= keep[:, None]
    return out


def _drop_grid_strips(p: np.ndarray, cell: float, pitch: float, min_length: float) -> np.ndarray:
    """Remove horizontal strips that belong to a line grid too fine to print.

    When the building's horizontal lines repeat every ``pitch`` metres and
    that is too fine to draw, none of them is drawn; but the structure rules
    would still keep some, wherever the texture shows them most clearly.  A
    strip (under two pitches tall, running across the facade) goes when
    another such strip lies 0.7 to 4.3 pitches above or below it (up to three
    lines may be missing between): that is the grid.  A lone horizontal
    piece, or a short one - a tie between braces, the crossing of two braces
    - stays.
    """
    n = max(3, int(round(min_length / cell)) | 1)
    runs = ndimage.binary_opening(p, np.ones((1, n), bool))
    if not runs.any():
        return p
    lab, k = ndimage.label(runs, np.ones((3, 3), bool))
    P = pitch / cell
    rows, keep = [], []
    wide = min(4 * min_length / cell, 0.5 * p.shape[1])
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        rows.append(0.5 * (sl[0].start + sl[0].stop))
        # a wall two lines tall is no grid line (bands of a grid may touch:
        # up to about one pitch); nor is a short piece - grid lines run
        # across the facade, the strips at a brace crossing do not
        keep.append(h >= 2 * P or w < wide)
    rows = np.array(rows)
    gone = np.zeros(k + 1, bool)
    for i in range(k):
        if keep[i]:
            continue
        d = np.abs(rows - rows[i])
        d[i] = np.inf
        gone[i + 1] = ((d >= 0.7 * P) & (d <= 4.3 * P) & ~np.array(keep)).any()
    if not gone.any():
        return p
    # take the strips' ragged ends with them, and what is then left too short
    # to count as structure (the length rule of regularise_pattern)
    cut = ndimage.binary_dilation(gone[lab], iterations=2)
    out = p & ~cut
    lab2, k2 = ndimage.label(out, np.ones((3, 3), bool))
    if k2:
        near = ndimage.binary_dilation(cut, iterations=2)
        touched = np.unique(lab2[near & out])
        for sl_i, sl in enumerate(ndimage.find_objects(lab2), 1):
            if sl_i in touched and np.hypot(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start) * cell < 4 * min_length:
                out[sl] &= lab2[sl] != sl_i
    return out


def _drop_reflections(p: np.ndarray, L: np.ndarray, grad: np.ndarray, gref: float, ok: np.ndarray,
                      cell: float, min_area: float = 4.0, ragged: float = 1.2, busy: float = 1.0,
                      contrast: float = 0.3) -> np.ndarray:
    """Remove pieces of the structure mask that are reflections after all.

    On dark glass the window grid has little contrast, so a reflection can
    look plain and survive :func:`regularise_pattern`.  Structure, whatever
    its tone, has a straight outline; a reflection's outline is ragged.  So a
    piece goes when its outline is ragged (``ragged`` times longer than the
    same outline smoothed over 0.4 m) and either its texture is busy inside
    (mean gradient over ``busy`` times ``gref``, the facade's median) or it
    hardly stands out from the facade around it (less than ``contrast``
    standard deviations).  Measured on 24 buildings: braces, piers and walls
    have outlines within 1.05-1.12 of smooth; reflections 1.2-1.6.
    """
    lab, n = ndimage.label(p, np.ones((3, 3), bool))
    if not n:
        return p
    out = p.copy()
    pad, near, far = int(round(3.0 / cell)) + 2, max(1, int(round(0.5 / cell))), int(round(3.0 / cell))
    sm = max(1.0, 0.4 / cell)
    H, W = p.shape
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        if (lab[sl] == i).sum() * cell * cell < min_area:
            continue
        s2 = (slice(max(0, sl[0].start - pad), min(H, sl[0].stop + pad)),
              slice(max(0, sl[1].start - pad), min(W, sl[1].stop + pad)))
        M = lab[s2] == i
        inn = ndimage.binary_erosion(M, iterations=2)
        ring = ndimage.binary_dilation(M, iterations=far) & ~ndimage.binary_dilation(M, iterations=near) \
            & ok[s2] & ~p[s2]
        if inn.sum() < 10 or ring.sum() < 20:
            continue
        bnd = M & ~ndimage.binary_erosion(M)
        Ms = ndimage.gaussian_filter(M.astype(np.float32), sm) > 0.5
        rag = bnd.sum() / max((Ms & ~ndimage.binary_erosion(Ms)).sum(), 1)
        if rag <= ragged:
            continue
        Lr = L[s2]
        con = (Lr[inn].mean() - Lr[ring].mean()) / max(float(Lr[ring].std()), 1e-3)
        gin = grad[s2][inn].mean() / max(gref, 1e-4)
        if gin > busy or con < contrast:
            out[s2] &= ~M
    return out


def _panel_room(lab_s: np.ndarray, si: int, ssl: tuple, t: np.ndarray, margin: int,
                step: float) -> np.ndarray:
    """Where on surface ``si`` (window ``ssl`` of the view) panels may be
    recessed.

    The outline stays solid ``margin`` cells deep where it is the silhouette
    or an outside corner, so the building keeps its shape.  Where the surface
    meets a wall standing at least ``step`` in front of it - an inside corner:
    a recessed balcony, a step back - nothing is outlined by that edge, and
    the recess may come within two cells of the wall.  Residential wings are
    mostly such strips between protruding walls.
    """
    H, W = lab_s.shape
    k = margin + 1
    es = (slice(max(0, ssl[0].start - k), min(H, ssl[0].stop + k)),
          slice(max(0, ssl[1].start - k), min(W, ssl[1].stop + k)))
    S = lab_s[es] == si
    T = t[es]
    idx = ndimage.distance_transform_edt(~S, return_distances=False, return_indices=True)
    Tn = T[idx[0], idx[1]]
    front = ~S & np.isfinite(T) & (T < Tn - step)
    solid = np.pad(S | front, 1)
    room = (ndimage.distance_transform_edt(solid)[1:-1, 1:-1] > margin) \
        & (ndimage.distance_transform_edt(np.pad(S, 1))[1:-1, 1:-1] > 2)
    r0, c0 = ssl[0].start - es[0].start, ssl[1].start - es[1].start
    return room[r0:r0 + (ssl[0].stop - ssl[0].start), c0:c0 + (ssl[1].stop - ssl[1].start)] & (lab_s[ssl] == si)


_COMPONENTS: list | None = None     # set to a list to collect per-piece figures (calibration)


def regularise_pattern(p: np.ndarray, surf: np.ndarray, busy: np.ndarray, cell: float,
                       width: float, min_length: float, floor_pitch: float | None = None) -> np.ndarray:
    """Keep the part of the raw pattern on one facade that is trustworthy
    structure: bracing, frames, piers.

    Texture photos are full of reflections, so the raw pattern is patchy.
    These rules separate structure from reflections on any building:

    * structure is opaque.  Inside a brace, pier or wall the texture is
      plain; a bright patch that the window grid still runs through
      (``busy`` marks texture edges) is glass mirroring something.  Fat
      patches (wider than six ribs) and the slender rest are judged
      separately, so a reflection touching a brace does not survive with it;
    * structure is long: what is left must hang together over four times
      ``min_length``, or right across the facade;
    * where the facade gets floor lines (``floor_pitch`` metres, from
      :func:`floor_bands`), horizontal strips thinner than a storey are
      spandrels - the floor lines already show them, evenly.
    """
    p = p & surf
    if not p.any():
        return p
    r = max(2, int(round(max(6.0, 6 * width) / cell / 2)))
    core = ndimage.distance_transform_edt(np.pad(p, 1)) >= r
    fat = (ndimage.distance_transform_edt(~core) <= r + 1)[1:-1, 1:-1] & p if core.any() else np.zeros_like(p)
    keep = np.zeros_like(p)
    for part in (fat, p & ~fat):
        lab, n = ndimage.label(part, np.ones((3, 3), bool))
        if not n:
            continue
        idx = np.arange(1, n + 1)
        inside = ndimage.binary_erosion(part, iterations=2)
        area = np.asarray(ndimage.sum(part, lab, idx))
        ins = np.asarray(ndimage.sum(inside, lab, idx))
        edges = np.asarray(ndimage.sum(busy & inside, lab, idx))
        glassy = edges > 0.32 * np.maximum(ins, 1)
        if _COMPONENTS is not None:
            _COMPONENTS.extend(zip(area * cell * cell, ins / np.maximum(area, 1), edges / np.maximum(ins, 1)))
        keep |= np.isin(lab, idx[~glassy])
    p = keep
    n = max(3, int(round(min_length / cell)) | 1)
    if floor_pitch:
        runs = ndimage.binary_opening(p, np.ones((1, n), bool))
        storey = ndimage.binary_opening(runs, np.ones((max(3, int(round(floor_pitch / cell))), 1), bool))
        p = p & ~(runs & ~storey)
    lab, n = ndimage.label(p, np.ones((3, 3), bool))
    if not n:
        return p
    hs, ws = np.nonzero(surf.any(1)), np.nonzero(surf.any(0))
    sh, sw = hs[0][-1] - hs[0][0] + 1, ws[0][-1] - ws[0][0] + 1
    ok = []
    for sl in ndimage.find_objects(lab):
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        ok.append(np.hypot(h, w) * cell >= 4 * min_length or h >= 0.8 * sh or w >= 0.8 * sw)
    return np.isin(lab, np.arange(1, n + 1)[np.array(ok)])


def floor_bands(sig: np.ndarray, cnt: np.ndarray, cell: float, width: float,
                max_pitch: float = 8.0, min_corr: float = 0.2,
                min_pitch: float = 2.5) -> tuple[np.ndarray | None, dict]:
    """Rows of the building that carry a floor line (spandrel, slab edge).

    ``sig`` is, per height row, how strongly the texture changes vertically
    in step along whole facades (summed over every facade of the building),
    ``cnt`` the number of cells behind it.  Floor lines are at the same
    height on every side, so adding the sides up lets them rise out of the
    reflections, which differ from side to side.  The lines must repeat
    (autocorrelation of at least ``min_corr`` at some pitch up to
    ``max_pitch`` metres, and again at twice that pitch) and the pitch must leave room for a rib and a gap
    of ``width`` each - otherwise the building gets no floor lines at all
    rather than a random selection.  A pitch below ``min_pitch`` metres is
    no storey but a window grid (curtain-wall modules): those lines are
    left off too, so they never drown the structure lines.  Returns (row
    mask or None, stats).
    """
    from scipy.signal import find_peaks

    valid = cnt >= max(16, 8.0 / cell)
    if valid.sum() * cell < 3 * max_pitch:
        return None, {}
    g = np.where(valid, sig / np.maximum(cnt, 1), 0.0)
    g = np.where(valid, g, np.median(g[valid]))
    h = g - ndimage.median_filter(g, size=int(2 * max_pitch / cell) | 1, mode="nearest")
    th = max(2, int(round(width / cell)))
    S = ndimage.uniform_filter1d(h, th, mode="nearest")
    S = np.where(valid, S - S[valid].mean(), 0.0)
    # the pitch is rarely a whole number of cells: correlate at quarter cells,
    # or a fine window grid would only show at twice its pitch
    up = 4
    Su = np.interp(np.arange(len(S) * up) / up, np.arange(len(S)), S)
    den = float((Su * Su).sum())
    if den <= 0:
        return None, {}
    lags = np.arange(max(4, int(0.75 / cell)) * up, int(max_pitch / cell) * up + 1)
    r = np.array([float((Su[:-l] * Su[l:]).sum()) for l in lags]) / den
    top = np.nonzero((r[1:-1] >= r[:-2]) & (r[1:-1] >= r[2:]))[0] + 1
    if not len(top) or r[top].max() < min_corr:
        return None, {"floor_corr": round(float(r.max()), 2)}
    first = top[r[top] >= 0.5 * r[top].max()][0]      # the shortest strong one: not a multiple
    Pu = int(lags[first])
    P = Pu / up
    info = {"floor_pitch_m": round(P * cell, 2), "floor_corr": round(float(r[first]), 2)}
    # a real pitch shows again at twice the distance; a chance peak does not
    twice = max(float((Su[:-l] * Su[l:]).sum()) for l in (2 * Pu - 1, 2 * Pu, 2 * Pu + 1)) / den
    if twice < 0.5 * r[first]:
        return None, info
    if P < 2.25 * th or P * cell < min_pitch:        # unprintable, or a window grid
        # unless the storey has a second, lighter line half way up (a
        # transom, a sill): then the lines first repeat at half a storey, but
        # the storey itself correlates clearly better.  A uniform grid
        # correlates the same at every multiple and stays a grid.
        def corr(lag: int) -> float:
            return max(float((Su[:-k] * Su[k:]).sum()) for k in range(lag - up, lag + up + 1)
                       if 0 < k < len(Su)) / den
        for k in (2, 3):
            rk = corr(k * Pu)
            if rk >= 1.15 * r[first] and k * P >= 2.25 * th and min_pitch <= k * P * cell <= max_pitch \
                    and corr(2 * k * Pu) >= 0.5 * rk:
                info.update(floor_grid_m=info["floor_pitch_m"], floor_pitch_m=round(k * P * cell, 2),
                            floor_corr=round(rk, 2))
                Pu, P = k * Pu, k * P
                break
        else:
            info["floor_lines"] = 0
            return None, info
    pk, prop = find_peaks(S, distance=max(1, int(0.7 * P)), height=0.0)
    if len(pk) < 4:
        return None, info
    hp = prop["peak_heights"]
    pk = pk[hp >= 0.35 * np.median(np.sort(hp)[len(hp) // 2:])]
    rows = np.zeros(len(sig), bool)
    for c in pk:
        a = max(0, int(c) - th // 2)
        rows[a:a + th] = True
    info["floor_lines"] = int(len(pk))
    return rows, info


def mullion_columns(L: np.ndarray, ok: np.ndarray, cell: float, width: float, seen: float = 1.0,
                    max_pitch: float = 8.0, min_corr: float = 0.25, min_contrast: float = 0.12,
                    min_lines: int = 4, pitch: float | None = None) -> tuple[np.ndarray | None, dict]:
    """Columns of one facade surface that carry a mullion (or a pier, a
    panel joint) worth drawing, as a mask over the columns of ``ok``.

    ``L`` is the texture luminance.  A mullion runs the whole height, so
    the horizontal texture change summed down each column (signed: a line
    adds up, a reflection cancels out) rises at it.  As with the floor
    lines, the columns must repeat at a regular pitch - and again at twice
    it - or the surface gets none.  Then three checks:

    * printable: the pitch must leave room for a rib and a recess, judged
      at the ``seen`` scale (2.25 times it).  A finer curtain-wall grid is
      left off, unless every second to fourth mullion is clearly heavier
      (it correlates 15% better, and looks 1.3 times as contrasty as the
      ones between) - then those, the major mullions, are drawn;
    * visible: at the print's scale (the texture blurred over a quarter of
      ``seen``) the lines must stand out from the bays between them by
      ``min_contrast`` of the surface's own spread of tone, in the top,
      middle and bottom third alike.  Hairline panel joints across ribbon
      windows are regular too, but drawn as ribs they would turn a
      horizontal building vertical;
    * there are at least ``min_lines`` of them.

    With ``pitch`` (metres: what another surface of the building showed)
    only pitches within 8% of it are tried, and half the correlation and
    contrast are enough - a shaded face shows the same mullions fainter.

    Returns (column mask ``width`` wide per line, or None; stats).
    """
    L = np.nan_to_num(L).astype(np.float32)
    fm = ok.astype(np.float32)

    def blur(s):
        return ndimage.gaussian_filter(L * fm, s) / np.maximum(ndimage.gaussian_filter(fm, s), 1e-6)

    Ls = blur(1.0)
    same = np.zeros_like(ok)
    same[:, 1:-1] = ok[:, 1:-1] & ok[:, 2:] & ok[:, :-2]
    dx = np.zeros_like(Ls)
    dx[:, 1:-1] = Ls[:, 2:] - Ls[:, :-2]

    def signal(rows=None):
        m = same if rows is None else same & rows[:, None]
        return np.abs((dx * m).sum(0)), m.sum(0).astype(np.float64)

    A, N = signal()
    info: dict = {}
    if not N.any():
        return None, info
    valid = N >= 0.5 * N.max()
    span = valid.sum() * cell
    th = max(2, int(round(width / cell)))
    least = 2.25 * seen / cell                       # finest printable pitch, in cells
    if span < (min_lines - 1) * least * cell or valid.sum() < 3:
        return None, info
    g = np.where(valid, A / np.maximum(N, 1), 0.0)
    g = np.where(valid, g, np.median(g[valid]))
    h = g - ndimage.median_filter(g, size=int(2 * max_pitch / cell) | 1, mode="nearest")
    h = np.where(valid, h - h[valid].mean(), 0.0)
    # the fine pitch from the unsmoothed signal: smoothing over a rib would
    # hide a grid finer than two ribs and let a multiple pass for it
    up = 4
    hu = np.interp(np.arange(len(h) * up) / up, np.arange(len(h)), h)
    den = float((hu * hu).sum())
    if den <= 0:
        return None, info

    def corr(lag: int) -> float:
        return max((float((hu[:-k] * hu[k:]).sum()) for k in range(lag - up, lag + up + 1)
                    if 0 < k < len(hu)), default=0.0) / den

    top_lag = int(min(max_pitch, span / 3) / cell) * up
    if pitch:
        lags = np.arange(int(0.92 * pitch / cell * up), min(top_lag, int(1.08 * pitch / cell * up)) + 1)
        min_corr, min_contrast = 0.5 * min_corr, 0.5 * min_contrast
    else:
        lags = np.arange(int(0.75 / cell) * up, top_lag + 1)
    if len(lags) < 3:
        return None, info
    r = np.array([float((hu[:-l] * hu[l:]).sum()) for l in lags]) / den
    top = np.nonzero((r[1:-1] >= r[:-2]) & (r[1:-1] >= r[2:]))[0] + 1
    if not len(top) or r[top].max() < min_corr:
        return None, info
    if pitch:
        first = top[np.argmax(r[top])]
    else:
        first = top[r[top] >= 0.5 * r[top].max()][0]   # the shortest strong one
    Pu = int(lags[first])
    r0 = float(r[first])
    info = {"mullion_pitch_m": round(Pu / up * cell, 2), "mullion_corr": round(r0, 2)}
    if r0 < min_corr or corr(2 * Pu) < 0.5 * r0:
        return None, info
    k = 1
    if Pu / up < least:
        for k in (2, 3, 4):
            rk = corr(k * Pu)
            if rk >= 1.15 * r0 and k * Pu / up >= least and k * Pu / up * cell <= max_pitch \
                    and corr(2 * k * Pu) >= 0.5 * rk:
                info.update(mullion_grid_m=info["mullion_pitch_m"], mullion_pitch_m=round(k * Pu / up * cell, 2),
                            mullion_corr=round(rk, 2))
                break
        else:
            return None, info
    P = k * Pu / up
    # lines on a regular lattice, each moved to the strongest column near it
    Sm = ndimage.uniform_filter1d(h, th, mode="nearest")
    x = np.nonzero(valid)[0]
    nb = max(4, int(round(P)))
    b = np.minimum((np.mod(x, P) / P * nb).astype(int), nb - 1)
    fold = np.bincount(b, Sm[x], nb) / np.maximum(np.bincount(b, minlength=nb), 1)
    phi = (np.argmax(fold) + 0.5) * P / nb
    rad = max(1, int(round(0.2 * P)))
    cols = []
    for c0 in np.arange(phi, len(A), P):
        c0 = int(round(c0))
        lo, hi = max(0, c0 - rad), min(len(A), c0 + rad + 1)
        if valid[lo:hi].any():
            cols.append(lo + int(np.argmax(np.where(valid[lo:hi], Sm[lo:hi], -np.inf))))
    if len(cols) < min_lines:
        return None, info
    # visible at the print's scale, over the whole height
    Lb = blur(max(1.0, 0.25 * seen / cell))
    spread = float(np.std(Lb[ok]))
    if spread <= 0:
        return None, info

    def contrast(cs, rows=None):
        m = ok if rows is None else ok & rows[:, None]
        n = m.sum(0)
        cm = (Lb * m).sum(0) / np.maximum(n, 1)
        v = n >= 0.5 * max(n.max(), 1)
        out = []
        for c in cs:
            c = int(round(c))
            a, z = max(0, int(c - P / 2)), min(len(cm), int(c + P / 2) + 1)
            bay = np.zeros(len(cm), bool)
            bay[a:z] = True
            bay[max(0, c - th):c + th + 1] = False
            bay &= v
            seg = v[max(0, c - th // 2):c + th // 2 + 1]
            if bay.any() and seg.any():
                ref = np.median(cm[bay])
                out.append(float(np.abs(cm[max(0, c - th // 2):c + th // 2 + 1][seg] - ref).max()))
        return float(np.median(out)) if out else 0.0

    rows_on = np.nonzero(ok.any(1))[0]
    thirds = []
    for part in np.array_split(rows_on, 3):
        m = np.zeros(ok.shape[0], bool)
        m[part] = True
        thirds.append(contrast(cols, m) / spread)
    seenc = contrast(cols)
    info.update(mullion_contrast=round(seenc / spread, 2), mullion_thirds=[round(t, 2) for t in thirds])
    if seenc < (0.005 if pitch else 0.01) or min(thirds) < min_contrast:
        return None, info
    if k > 1:
        minor = [c + j * P / k for c in cols for j in range(1, k) if c + j * P / k < len(A)]
        if seenc < 1.3 * contrast(minor):
            return None, info
    mask = np.zeros(len(A), bool)
    for c in cols:
        mask[max(0, c - th // 2):c - th // 2 + th] = True
    info["mullion_lines"] = len(cols)
    return mask, info


def _minmod(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The smaller of two slopes when they agree in sign, else none."""
    return np.where(a * b > 0, np.where(np.abs(a) < np.abs(b), a, b), 0).astype(a.dtype)


def _view_solid(scene, V: np.ndarray, ang: float, cell: float, R: float, z0: float, z1: float,
                sub: int = 3, slack: float = 0.25, ground: np.ndarray | None = None,
                min_open: float = 0.0, info: dict | None = None) -> m3d.Manifold:
    """Everything behind the first surface hit by rays travelling along ``ang``.

    One ray per grid point is exact on any continuous surface, but lets a thin
    or slanted part (a fin, a truss member) slip between rays here and there,
    and since the views are intersected, every such miss bites a piece out of
    it.  So ``sub`` x ``sub`` rays are cast per cell (``sub`` odd: the middle
    one is the grid point's own), and next to the point sample stands a
    conservative depth: the nearest of what the rays in the point's own and
    the eight neighbouring cells say about it.  Their plain depths would call
    every slanted face too near, so each ray speaks for the plane it lies on
    (its depth carried to the grid point along the one-sided local slope, but
    no nearer than the plain nearest depth).  Where the two agree within
    ``slack`` cells - flat, sloped, creased and curved faces - the point
    sample stands and the surface keeps its true position.  Where the
    conservative depth is nearer by twice that - an edge, a part the middle
    ray missed - it wins: the near surface is continued over its neighbours,
    so a part seen edge-on comes out up to a cell too fat, never bitten.  In
    between the two are blended, so nothing flickers.
    """
    if sub % 2 == 0:
        raise ValueError("sub must be odd")
    d = np.array([np.cos(ang), np.sin(ang), 0.0])
    u = np.array([np.sin(ang), -np.cos(ang), 0.0])          # keeps the frame right-handed
    zhat = np.array([0.0, 0.0, 1.0])
    pu = V[:, 0] * u[0] + V[:, 1] * u[1]
    us = np.arange(pu.min() - 2 * cell, pu.max() + 2 * cell, cell)
    zs = np.arange(z1 + cell, z0 - cell, -cell)              # descending, as grid_solid expects
    off = (np.arange(sub) + 0.5) / sub - 0.5                 # ray offsets inside one cell
    uf = (us[:, None] + off[None, :] * cell).ravel()
    zf = (zs[:, None] - off[None, :] * cell).ravel()
    miss = np.float32(1e6)                                   # stands in for "no hit" in the arithmetic
    tf = np.empty((len(zf), len(uf)), np.float32)
    step = max(1, 4_000_000 // max(len(uf), 1))              # rows of rays per batch
    for i in range(0, len(zf), step):
        U, Zg = np.meshgrid(uf, zf[i:i + step])
        ti = _cast(scene, U[..., None] * u + Zg[..., None] * zhat - d * R, d)
        tf[i:i + step] = np.where(np.isfinite(ti), ti, miss)
    # slope per ray spacing along the rows and columns of rays; one-sided at
    # an edge, so a ray never borrows the slope of the jump next to it
    pad = np.pad(tf, 1, mode="edge")
    gz = _minmod(tf - pad[:-2, 1:-1], pad[2:, 1:-1] - tf)
    gu = _minmod(tf - pad[1:-1, :-2], pad[1:-1, 2:] - tf)
    nz, nu = len(zs), len(us)
    wide = sub * 3                                           # rays across a grid point's 3 x 3 cells
    ext = [np.pad(a, sub, mode="edge") for a in (tf, gz, gu)]
    plain = np.full((nz, nu), miss, np.float32)
    plane = np.full((nz, nu), miss, np.float32)
    for a in range(wide):
        da = a - (wide - 1) / 2                              # ray's row offset from the grid point
        for b in range(wide):
            db = b - (wide - 1) / 2
            t_, gz_, gu_ = (x[a:a + sub * nz:sub, b:b + sub * nu:sub] for x in ext)
            np.minimum(plain, t_, out=plain)
            np.minimum(plane, t_ - gz_ * np.float32(da) - gu_ * np.float32(db), out=plane)
    safe = np.maximum(plain, plane)
    own = tf[sub // 2::sub, sub // 2::sub]
    gap = own - safe                                         # never negative: own is one of the rays
    w = np.clip(gap / (slack * cell) - 1.0, 0.0, 1.0)
    w = w * w * (3.0 - 2.0 * w)
    t = np.where(w >= 1.0, safe, own - w * gap)
    t = np.where(t < miss / 2, t, np.nan)
    # extend each column's lowest hit down to the floor so the building
    # always reaches the ground it stands on (rows run top -> bottom) -
    # except through real openings at its foot (see _ground_openings)
    hit = np.isfinite(t)
    if hit.any():
        last = np.where(hit.any(0), hit.shape[0] - 1 - np.argmax(hit[::-1], axis=0), -1)
        rows = np.arange(hit.shape[0])[:, None]
        below = (rows > last[None, :]) & (last[None, :] >= 0)
        if ground is not None and min_open > 0:
            gu = _ground_line(ground, u, us, cell)
            op = _ground_openings(hit, below, gu, zs, cell, min_open)
            below &= ~op
            if info is not None:
                info["open_m2"] = info.get("open_m2", 0.0) + float(op.sum()) * cell * cell
        t = np.where(below, t[np.maximum(last, 0), np.arange(t.shape[1])][None, :], t)
    eps = 0.01 * cell
    W = np.where(np.isfinite(t), np.maximum(2 * R - t, eps), eps).astype(np.float32)
    # frame (u, z, w) -> world: p = u*u + z*zhat + d*(R - w)
    sol = grid_solid(us.astype(np.float32), zs.astype(np.float32), W)
    M = np.column_stack([u, [0.0, 0.0, 1.0], -d, d * R])
    return sol.transform(M)


def _ground_line(ground: np.ndarray, u: np.ndarray, us: np.ndarray, cell: float) -> np.ndarray:
    """Highest ground under the building along each ray of a view.

    ``ground`` is (n, 3): x, y, ground elevation of the cells of the
    building's footprint.  A ray at height z clears the ground all the way
    across the building only above this line.  NaN where no footprint cell
    falls in a column."""
    gu = np.full(len(us), -np.inf)
    i = np.floor((ground[:, 0] * u[0] + ground[:, 1] * u[1] - us[0]) / cell + 0.5).astype(int)
    ok = (i >= 0) & (i < len(us)) & np.isfinite(ground[:, 2])
    np.maximum.at(gu, i[ok], ground[ok, 2])
    gu = ndimage.maximum_filter1d(gu, 3, mode="nearest")      # a column the footprint just misses
    return np.where(np.isfinite(gu), gu, np.nan)


def _ground_openings(hit: np.ndarray, below: np.ndarray, gu: np.ndarray, zs: np.ndarray, cell: float,
                     min_open: float, tol: float = 0.5) -> np.ndarray:
    """Cells under a building that a view really sees through: an arcade, a
    plaza under a raised floor, the space under an overhang or a bridge.

    Below a column's lowest hit the view normally fills down to the floor,
    because a LandsD building often stops short of the ground: a tower
    modelled from the roof of the podium it stands on, or a block a little
    above the terrain.  A gap above the ground line (plus ``tol``) is kept
    open only if

    * it can be printed: at least ``min_open`` metres wide, and that high in
      nine of ten of its columns - which also leaves the thin wedge under a
      building that stops short on a slope filled, and
    * the building is not simply floating in this view: either the gap is
      closed in left and right by the building (an arcade, the space between
      the legs of a bridge), or the building stands on the ground in at
      least a tenth of the view (it reaches within 1 m of the ground line),
      so the gap is a real overhang beside it.
    """
    above = zs[:, None] > (gu[None, :] + tol)               # NaN ground: never
    cand = below & above
    if not cand.any():
        return cand
    cols = hit.any(0)
    sky = ~cols                                              # columns beside the building
    last = hit.shape[0] - 1 - np.argmax(hit[::-1], axis=0)
    on = cols & np.isfinite(gu)
    grounded = bool(on.any()) and float(np.mean(zs[last[on]] - gu[on] < 1.0)) >= 0.1
    lab, n = ndimage.label(cand)
    keep = np.zeros(n + 1, bool)
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        m = lab[sl] == i
        heights = m.sum(0) * cell
        if (sl[1].stop - sl[1].start) * cell < min_open or np.percentile(heights, 10) < min_open:
            continue
        if grounded:
            keep[i] = True
            continue
        c0, c1 = sl[1].start, sl[1].stop
        if c0 == 0 or c1 == hit.shape[1] or sky[c0 - 1] or sky[c1]:
            continue
        # every row of it closed in left and right by the building or by ground
        r = np.arange(sl[0].start, sl[0].stop)
        rows_on = m.any(1)
        left = np.argmax(m, axis=1) + c0 - 1
        right = m.shape[1] - 1 - np.argmax(m[:, ::-1], axis=1) + c0 + 1
        side = np.ones(len(r), bool)
        for col in (left, right):
            ok = (col >= 0) & (col < hit.shape[1])
            cc = np.clip(col, 0, hit.shape[1] - 1)
            side &= ok & ~sky[cc] & (hit[r, cc] | ~above[r, cc] | ~below[r, cc])
        if side[rows_on].all():
            keep[i] = True
    return keep[lab] & cand


def _runs_to_rects(mask: np.ndarray, x0: float, y0: float, cell: float) -> list:
    """Row runs of a boolean image as rectangles (x right, y down from the
    top-left cell centre ``x0, y0``)."""
    pad = np.zeros((mask.shape[0], mask.shape[1] + 2), np.int8)
    pad[:, 1:-1] = mask
    dm = np.diff(pad, axis=1)
    r0, c0 = np.nonzero(dm == 1)
    _r1, c1 = np.nonzero(dm == -1)
    xa, xb = x0 + (c0 - 0.5) * cell, x0 + (c1 - 0.5) * cell
    ya, yb = y0 - (r0 + 0.5) * cell, y0 - (r0 - 0.5) * cell
    return [np.array([[xa[i], ya[i]], [xb[i], ya[i]], [xb[i], yb[i]], [xa[i], yb[i]]]) for i in range(len(r0))]


def _fill_cross_section(cs, u0: float, z0: float, cell: float, shape: tuple) -> np.ndarray:
    """Cells whose centre ``(u0 + c*cell, z0 - r*cell)`` lies inside the
    cross-section: an exact scanline fill of its outline (holes included),
    so the debug image shows the recess that is really cut."""
    rings = [np.asarray(r, np.float64) for r in cs.to_polygons() if len(r) >= 3]
    out = np.zeros(shape, bool)
    if not rings:
        return out
    a = np.concatenate(rings)
    b = np.concatenate([np.roll(r, -1, axis=0) for r in rings])
    flat = a[:, 1] == b[:, 1]
    a, b = a[~flat], b[~flat]
    zlo, zhi = np.minimum(a[:, 1], b[:, 1]), np.maximum(a[:, 1], b[:, 1])
    rows = np.arange(shape[0])
    zs = z0 - rows * cell
    for r in rows[(zs >= zlo.min()) & (zs < zhi.max())]:
        z = zs[r]
        hit = (zlo <= z) & (z < zhi)
        if not hit.any():
            continue
        t = (z - a[hit, 1]) / (b[hit, 1] - a[hit, 1])
        x = np.sort(a[hit, 0] + t * (b[hit, 0] - a[hit, 0]))
        for x0, x1 in zip(x[0::2], x[1::2]):                 # even-odd: inside between pairs
            c0 = max(int(np.ceil((x0 - u0) / cell)), 0)
            c1 = min(int(np.ceil((x1 - u0) / cell)), shape[1])
            if c1 > c0:
                out[r, c0:c1] = True
    return out


def relief_cutters(V: np.ndarray, F: np.ndarray, axes: list[float], sampler: TextureSampler,
                   depth: float, width: float, cell: float = 0.125, min_area: float = 6.0,
                   debug: str | None = None, height: float | None = None,
                   chamfer: float | None = None) -> tuple[list, dict]:
    """Prisms that recess the glazing between the structure and floor lines
    painted on the building's texture.

    For every facade direction the texture is sampled where side rays hit.
    :func:`facade_pattern` and :func:`regularise_pattern` find the structure
    (bracing, frames, piers) per facade; :func:`floor_bands` finds the floor
    lines of the whole building, which are then drawn on every facade whose
    own texture agrees with them.  What lies between - the panels - is
    vectorised: each connected panel on one surface becomes a polygon with
    straightened edges, extruded ``depth`` metres into the facade along the
    surface.  Subtracting the prisms leaves structure and floor lines
    standing as crisp ribs at least ``width`` wide; floor lines, which the
    printer builds from layers rather than lines, are ``height`` tall
    (default: ``width``).  Nothing here is specific to one building.  Returns (list of manifolds in the local metre frame,
    stats).
    """
    scene = _scene(V, F)
    R = float(np.hypot(*np.abs(V[:, :2]).max(0))) + 10.0
    z0, z1 = float(V[:, 2].min()), float(V[:, 2].max())
    zhat = np.array([0.0, 0.0, 1.0])
    zs = np.arange(z1 + cell, z0 - cell, -cell)
    H = len(zs)
    # what counts as structure is judged at a fixed real-world scale (the one
    # validated on six towers); a finer printer draws thinner ribs, but must
    # not make reflections and window mullions pass for structure
    seen = max(width, 1.0)
    margin = max(2, int(np.ceil(seen / cell)))         # keep clear of outlines: corners stay solid
    min_len = max(4.0, 4 * seen)
    # a panel is worth recessing once it is two ribs in area (between the
    # floor lines of a narrow residential wing, say); narrower than a rib it
    # is dropped below anyway
    min_panel = min(min_area, 2 * seen * seen)
    out: list = []
    stats = {"relief_panels": 0, "relief_area_m2": 0.0}
    views = []
    sig, cnt = np.zeros(H), np.zeros(H)
    view = 0
    for a in axes:
        for ang in (a, a + np.pi):
            view += 1
            d = np.array([np.cos(ang), np.sin(ang), 0.0])
            u = np.array([np.sin(ang), -np.cos(ang), 0.0])
            pu = V[:, 0] * u[0] + V[:, 1] * u[1]
            us = np.arange(pu.min() - 2 * cell, pu.max() + 2 * cell, cell)
            U, Zg = np.meshgrid(us, zs)
            r = _cast(scene, U[..., None] * u + Zg[..., None] * zhat - d * R, d, full=True)
            t = r["t_hit"]
            hit = np.isfinite(t)
            n = r["primitive_normals"]
            # each face is textured from the axis it faces most directly
            align = [np.abs(n[..., 0] * np.cos(b) + n[..., 1] * np.sin(b)) for b in axes]
            mine = np.abs(n[..., 0] * d[0] + n[..., 1] * d[1])
            # near-vertical facades only: sloped roofs are often half hidden
            # from the side (a recess would stop at the shadow of whatever
            # hides them) and their texture is an aerial photo, not cladding
            face = hit & (mine > 0.9) & (mine >= np.max(align, axis=0) - 1e-6)
            if not face.any():
                continue
            rgb = sampler.sample(r["primitive_ids"], r["primitive_uvs"], face)
            face &= np.isfinite(rgb).all(-1)
            # where LandsD had no photo, the texture is one flat colour (grey
            # 128, also 99, 177, ...): no photo is uniform over a metre, and
            # flat, light and plain it would pass for cladding
            face &= ~flat_texture(rgb, face, cell)
            # judge the texture at the scale the print can show: bands under
            # 3/4 of a rib are noise, finer stripes are averaged
            pat = facade_pattern(rgb, face, cell, min_width=0.75 * seen, smooth=0.25 * seen,
                                 min_length=min_len, background=max(8.0, 8 * seen))
            # one surface at a time: pieces of facade between creases and
            # steps in depth (second differences of the depth map)
            tz = np.where(hit, t, 0.0).astype(np.float64)
            edge = ~hit
            d2 = np.abs(tz[:, :-2] - 2 * tz[:, 1:-1] + tz[:, 2:]) > 0.25 * cell
            edge[:, 1:-1] |= d2 | ~hit[:, :-2] | ~hit[:, 2:]
            d2 = np.abs(tz[:-2] - 2 * tz[1:-1] + tz[2:]) > 0.25 * cell
            edge[1:-1] |= d2 | ~hit[:-2] | ~hit[2:]
            lab_s, ns = ndimage.label(face & ~edge)
            size = np.bincount(lab_s.ravel(), minlength=ns + 1)
            big = size * cell * cell >= min_area
            big[0] = False
            lab_s = (np.cumsum(big) * big)[lab_s]      # drop small surfaces, renumber the rest
            ns = int(big.sum())
            if not ns:
                continue
            # texture edges: where the window grid shows, and - for the floor
            # lines - how much each row of each surface changes vertically in
            # step (signed sum: a line adds up, a reflection cancels out)
            fm = face.astype(np.float32)
            L = np.nan_to_num(rgb).mean(-1)
            grad = np.hypot(ndimage.sobel(L, 0), ndimage.sobel(L, 1)) / 8.0
            busy = grad > 0.03
            Ls = ndimage.gaussian_filter(L * fm, 1.0) / np.maximum(ndimage.gaussian_filter(fm, 1.0), 1e-6)
            dz = np.zeros_like(Ls)
            dz[1:-1] = Ls[2:] - Ls[:-2]
            same = np.zeros(lab_s.shape, bool)
            same[1:-1] = (lab_s[1:-1] > 0) & (lab_s[2:] == lab_s[1:-1]) & (lab_s[:-2] == lab_s[1:-1])
            key = (lab_s * H + np.arange(H)[:, None])[same]
            A = np.abs(np.bincount(key, weights=dz[same], minlength=(ns + 1) * H).reshape(ns + 1, H))
            N = np.bincount(key, minlength=(ns + 1) * H).reshape(ns + 1, H)
            sig += A[1:].sum(0)
            cnt += N[1:].sum(0)
            views.append((view, d, u, us, t, mine, rgb, face, pat, lab_s, busy, A, N, L, grad))
    height = width if height is None else height
    rows, info = floor_bands(sig, cnt, cell, height)
    # lines repeating too finely to print (a window grid, or storeys too low
    # for this scale) are left off; then the structure rules must not draw
    # some of them either, as horizontal strips here and there
    grid = info.get("floor_pitch_m") if rows is None and info.get("floor_lines") == 0 else None
    stats.update(info)
    th = max(2, int(round(height / cell)))
    clear = None if rows is None else ~ndimage.binary_dilation(rows, iterations=max(1, th // 2))
    floors_on = mullions_on = 0
    # floor lines go on each surface whose own texture has them too; there,
    # mullions complete them to a window grid (never alone: vertical lines
    # without the floors would turn a building of ribbon windows into fins).
    # A mullion pitch found on one surface is then looked for, more
    # leniently, on the building's others - a shaded face shows it fainter
    fl, mull, area = {}, {}, {}
    for vi, (view, d, u, us, t, mine, rgb, face, pat, lab_s, busy, A, N, L, grad) in enumerate(views):
        if rows is None:
            break
        for si, ssl in enumerate(ndimage.find_objects(lab_s), 1):
            rs, cl = rows[ssl[0]], clear[ssl[0]]
            wide = N[si, ssl[0]] >= max(8, 2.0 / cell)
            e = A[si, ssl[0]] / np.maximum(N[si, ssl[0]], 1)
            if (rs & wide).sum() < 3 * th or (cl & wide).sum() < 3 * th \
                    or e[rs & wide].mean() < 1.3 * e[cl & wide].mean():
                continue
            fl[vi, si] = rs
            area[vi, si] = (int((lab_s[ssl] == si).sum()), int((lab_s[ssl] == si).any(0).sum()) * cell)
            mull[vi, si] = mullion_columns(L[ssl], lab_s[ssl] == si, cell, width, seen)
    found = sorted({i["mullion_pitch_m"] for m, i in mull.values() if m is not None})
    for key, (m, i) in list(mull.items()):
        if m is None and found:
            vi, si = key
            lab_s, L = views[vi][9], views[vi][13]
            ssl = ndimage.find_objects(lab_s)[si - 1]
            for p in found:
                m2, i2 = mullion_columns(L[ssl], lab_s[ssl] == si, cell, width, seen, pitch=p)
                if m2 is not None:
                    mull[key] = (m2, i2)
                    break
    # on a building, not a face: found on less than half the facade that has
    # floor lines and is wide enough for them, they would be a random subset
    # of a symmetric tower
    if found:
        fits = {k: a for k, (a, w) in area.items() if w >= 3 * min(found)}
        share = sum(fits.get(k, 0) for k, (m, i) in mull.items() if m is not None) / max(sum(fits.values()), 1)
        stats["mullion_share"] = round(share, 2)
        if share < 0.5:
            mull = {}
    for vi, (view, d, u, us, t, mine, rgb, face, pat, lab_s, busy, A, N, L, grad) in enumerate(views):
        gref = float(np.median(grad[face]))
        reg = np.zeros_like(pat)                      # what is finally kept proud, for the debug image
        pan = np.zeros_like(pat)                      # and what is really recessed
        stru = np.zeros_like(pat)                     # structure alone, without the floor lines
        cand = np.zeros_like(pat)                     # ribs a surface would get, cut or not
        for si, ssl in enumerate(ndimage.find_objects(lab_s), 1):
            surf = lab_s[ssl] == si
            su, sz, st = us[ssl[1]], zs[ssl[0]], t[ssl]
            inner = _panel_room(lab_s, si, ssl, t, margin, step=max(0.25, 2 * cell))
            rs = fl.get((vi, si))
            ps = regularise_pattern(pat[ssl], surf, busy[ssl], cell, seen, min_len,
                                    None if rs is None else info["floor_pitch_m"])
            if rs is None and grid:
                ps = _drop_grid_strips(ps, cell, grid, min_len)
            ps = _widen(_straighten(ps), width / cell) & surf
            ps = _drop_reflections(ps, L[ssl], grad[ssl], gref, surf, cell)
            if debug:
                stru[ssl] |= ps
            mc = None
            if rs is not None:
                ps |= surf & rs[:, None]
                mc = mull.get((vi, si), (None, None))[0]
                if mc is not None:
                    ps |= surf & mc[None, :]
            # what this surface would get; the debug image and the counts
            # take it only once the cut is really made (below)
            if debug:
                cand[ssl] |= ps & surf
            # relief only where the surface really carries structure
            frac = float(ps[surf].mean())
            if not 0.02 <= frac <= 0.7:
                continue
            # smooth the outline of the ribs before it is vectorised
            ribs = ndimage.gaussian_filter(ps.astype(np.float32), 1.5) > 0.5
            lab, nlab = ndimage.label(inner & ~ribs)
            polys = []
            for i, sl in enumerate(ndimage.find_objects(lab), 1):
                m = lab[sl] == i
                if m.sum() * cell * cell < min_panel:
                    continue
                polys += _runs_to_rects(m, su[sl[1].start], sz[sl[0].start], cell)
            if not polys:
                continue
            cs = m3d.CrossSection(polys, m3d.FillRule.Positive).simplify(2.0 * cell)
            # drop recesses narrower than one rib, keep corners sharp
            q = 0.5 * min(width, height)                     # a recess narrower than a line cannot print
            cs = cs.offset(-q, m3d.JoinType.Miter, 4.0).offset(q, m3d.JoinType.Miter, 4.0)
            if cs.is_empty():
                continue
            # counts and the debug image from the outline that is cut, not a raster copy of it
            npanel = len(cs.decompose())
            area = float(cs.area())
            rr, cc = np.nonzero(surf)
            along = depth / float(np.median(mine[ssl][rr, cc]))  # ray length for ``depth`` across the face
            front = 0.3                                           # start in front of the facade: a clean cut
            prism = cs.extrude(along + front)
            if chamfer:
                # lower the cut as it goes in (``chamfer`` m down per m of
                # depth): the rib over each recess then overhangs at 45
                # degrees on the print instead of flat, so it prints
                # without sagging; the outline on the facade is unchanged
                k = chamfer * depth / along
                prism = prism.transform(np.array([[1.0, 0, 0, 0], [0, 1.0, k, -k * along], [0, 0, 1.0, 0]]))
            # is it a plane  t = c0*u + c1*z + c2 ?
            if len(rr) > 20000:
                sel = np.random.default_rng(0).choice(len(rr), 20000, replace=False)
                rr, cc = rr[sel], cc[sel]
            tu, tzz, tv = su[cc], sz[rr], st[rr, cc].astype(np.float64)
            coef, *_ = np.linalg.lstsq(np.column_stack([tu, tzz, np.ones(len(rr))]), tv, rcond=None)
            res = np.abs(tu * coef[0] + tzz * coef[1] + coef[2] - tv)
            if np.percentile(res, 95) <= max(0.05, 0.25 * cell):
                # (u, z, q) -> world: u*u^ + z*z^ + d*(plane(u, z) + along - q - R)
                M = np.column_stack([u + d * coef[0], zhat + d * coef[1], -d, d * (coef[2] + along - R)])
                prism = prism.transform(M)
            else:
                # curved: bend the prism to the depth map of this surface
                near = ndimage.distance_transform_edt(~surf, return_distances=False, return_indices=True)
                T = ndimage.gaussian_filter(st[near[0], near[1]].astype(np.float64), 1.0)

                def bend(P, T=T, su=su, sz=sz, along=along):
                    col = (P[:, 0] - su[0]) / cell
                    row = (sz[0] - P[:, 1]) / cell
                    tt = ndimage.map_coordinates(T, [row, col], order=1, mode="nearest")
                    w = tt + along - P[:, 2] - R
                    return P[:, 0:1] * u + P[:, 1:2] * zhat + w[:, None] * d

                prism = prism.refine_to_length(max(1.0, 4 * cell)).warp_batch(bend)
                stats["relief_curved"] = stats.get("relief_curved", 0) + 1
            if prism.status() != m3d.Error.NoError or prism.is_empty():
                continue
            out.append(prism)
            stats["relief_panels"] += npanel
            stats["relief_area_m2"] += area
            reg[ssl] |= ps & surf
            if debug:
                pan[ssl] |= _fill_cross_section(cs, su[0], sz[0], cell, surf.shape)
            if rs is not None:
                floors_on += 1
                mullions_on += mc is not None
        if debug:
            _debug_view(debug, view, rgb, face, pat, reg, pan, stru, cand)
    if rows is not None:
        stats["floor_surfaces"] = floors_on
        stats["mullion_surfaces"] = mullions_on
        if found and mullions_on:
            stats["mullion_pitch_m"] = found
    stats["relief_area_m2"] = round(float(stats["relief_area_m2"]), 1)
    return out, stats


def _debug_view(folder: str, view: int, rgb, face, pat, reg, pan=None, stru=None, cand=None) -> None:
    """Texture as the rays see it, and what became of it: grey facade, red
    raw pattern rejected, amber ribs found on a surface that got no cut
    (too little or too much structure, or no recess wide enough), blue the
    panels really recessed and black the ribs really left standing.  Blue
    and black are only drawn once the cut exists, so the image shows the
    geometry, not the intention."""
    from pathlib import Path

    from PIL import Image
    Path(folder).mkdir(parents=True, exist_ok=True)
    img = np.nan_to_num(rgb, nan=1.0)
    Image.fromarray((img * 255).astype(np.uint8)).save(f"{folder}/view{view}_rgb.png")
    m = np.where(face[..., None], [215, 215, 215], [255, 255, 255])
    m = np.where((pat & ~reg)[..., None], [235, 150, 150], m)
    if cand is not None:
        m = np.where((cand & ~reg)[..., None], [240, 190, 90], m)
    if pan is not None:
        m = np.where(pan[..., None], [170, 205, 240], m)
    m = np.where(reg[..., None], [30, 30, 30], m)
    Image.fromarray(m.astype(np.uint8)).save(f"{folder}/view{view}_pat.png")
    if stru is not None:
        Image.fromarray((stru & reg).astype(np.uint8) * 255).save(f"{folder}/view{view}_struct.png")


def carve(V: np.ndarray, F: np.ndarray, top: np.ndarray, xs: np.ndarray, ys: np.ndarray,
          cell: float, z_floor: float, axes: list[float] | None = None,
          side_cell: float | None = None, ground: np.ndarray | None = None,
          min_open: float = 0.0) -> tuple[m3d.Manifold, dict]:
    """Depth hull of one building.

    V, F   -- the building's triangles in local metres (x, y relative to the
              model centre, z in mPD)
    top    -- its heightmap (mPD) on the model grid ``xs`` (ascending) x ``ys``
              (descending), NaN where the building is absent
    z_floor -- the solid is extended down to this elevation
    ground -- optional ground elevation (mPD) on the same grid; with
              ``min_open`` > 0, openings at the building's foot at least that
              many metres high and wide are kept open instead of filled
    Returns the solid in the same local metre frame and some stats.
    """
    fp = np.isfinite(top)
    if not fp.any():
        return m3d.Manifold(), {"axes": 0}
    rows, cols = np.nonzero(ndimage.binary_dilation(fp, iterations=3))
    r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    T = top[r0:r1, c0:c1]
    grow = ndimage.binary_dilation(np.isfinite(T), iterations=1)
    Tg = ndimage.grey_dilation(np.where(np.isfinite(T), T, -1e9), size=(3, 3))
    Ttop = np.where(grow, Tg - z_floor, 0.01 * cell).astype(np.float32)
    top_solid = grid_solid(xs[c0:c1].astype(np.float32), ys[r0:r1].astype(np.float32),
                           np.maximum(Ttop, 0.01 * cell)).translate((0.0, 0.0, z_floor))

    if axes is None:
        axes = facade_axes(V, F)
    scene = _scene(V, F)
    R = float(np.hypot(*np.abs(V[:, :2]).max(0))) + 10.0
    z1 = float(np.nanmax(T)) + 1.0
    L = top_solid
    views = []
    stats: dict = {"axes": len(axes), "axes_deg": [round(float(np.degrees(a)), 1) for a in axes]}
    gpts = None
    if ground is not None and min_open > 0:
        Xg, Yg = np.meshgrid(xs, ys)
        on = fp & np.isfinite(ground)
        gpts = np.column_stack([Xg[on], Yg[on], ground[on]]) if on.any() else None
    for a in axes:
        for ang in (a, a + np.pi):
            views.append(_view_solid(scene, V, ang, side_cell or cell, R, z_floor, z1,
                                     ground=gpts, min_open=min_open, info=stats))
    for v in views:
        L = L ^ v
    stats["views"] = len(views)
    if "open_m2" in stats:
        stats["open_m2"] = round(stats["open_m2"], 1)    # opening area summed over the side views
    return L, stats


def decimate(man: m3d.Manifold, agg: float = 3.0) -> m3d.Manifold:
    """Quadric decimation that keeps the solid watertight.

    manifold3d's own ``simplify`` hardly merges large coplanar areas, so the
    flat facades of a carved tower would stay as millions of grid triangles.
    fast-simplification (quadric error metric) collapses them quickly.  A
    low aggressiveness only takes near-zero-error collapses: on the Bank of
    China tower 4.0M -> 0.56M triangles with 99% of the surface within 4 mm
    (real scale) and the worst point 0.18 m off.  The result is rejected if
    it is not manifold or its volume moves by more than 0.1%.
    """
    try:
        import fast_simplification as fs
    except ImportError:  # optional dependency
        return man.simplify(0.02)
    mesh = man.to_mesh()
    V = np.asarray(mesh.vert_properties)[:, :3].astype(np.float64)
    F = np.asarray(mesh.tri_verts).astype(np.int32)
    V2, F2 = fs.simplify(V, F, target_reduction=0.99, agg=agg)
    if len(F2) == 0:
        return man
    cand = m3d.Manifold(m3d.Mesh(vert_properties=V2.astype(np.float32), tri_verts=F2.astype(np.uint32)))
    if cand.status() != m3d.Error.NoError or abs(cand.volume() - man.volume()) > 1e-3 * man.volume():
        return man
    return cand
