"""Make carved (truly 3D) parts printable without supports.

An FDM printer lays each layer on the one below.  A layer may stick out past
the one under it by about one layer height (a 45 degree overhang), or span a
gap between two supports up to some length (a bridge); anything else - the
underside of a cantilever, a ceiling too wide to bridge - droops or falls.

:func:`support_fill` slices a part the way a slicer does (at the printer's
layer height and the model's cell size), walks the layers from the top down
and, wherever a layer would hang over air, adds material under it that
narrows by the overhang allowance on every layer: a 45 degree bracket under a
cantilever, a sloped haunch under a wide ceiling.  Cells that a slicer can
bridge - on a straight run of air, closed by material at both ends, no longer
than ``max_bridge_mm`` - are left open, so arcades, passages and the space
under a footbridge stay open.  The added material is returned as one
watertight solid to be unioned with the part.

Only parts that can overhang need this: the 2.5D height field never does.
"""
from __future__ import annotations

import manifold3d as m3d
import numpy as np
from scipy import ndimage

from . import solid


def _row_bridges(m: np.ndarray, a: np.ndarray, max_cells: float) -> np.ndarray:
    n = m.shape[1]
    idx = np.broadcast_to(np.arange(n), m.shape)
    b = np.maximum.accumulate(np.where(~m, idx, -1), axis=1)                    # last non-M at or before
    f = np.minimum.accumulate(np.where(~m, idx, n)[:, ::-1], axis=1)[:, ::-1]   # first non-M at or after
    rows = np.arange(m.shape[0])[:, None]
    ok = m & (b >= 0) & (f < n) & (f - b - 1 <= max_cells)
    return ok & a[rows, np.clip(b, 0, n - 1)] & a[rows, np.clip(f, 0, n - 1)]


_CROSS = ndimage.generate_binary_structure(2, 1)


def bridgeable(M: np.ndarray, anchor: np.ndarray, max_cells: float, directions: int = 4) -> np.ndarray:
    """Cells of ``M`` (air under something) a slicer can bridge: they lie on a
    straight run of ``M`` cells whose two ends are ``anchor`` cells (not open
    air, not the edge of the array) and that is at most ``max_cells`` long.
    Runs are tried in ``directions`` evenly spaced directions (4: along the
    rows, the columns and both diagonals).

    A slicer lays each connected bridge area in one direction.  So each
    connected patch of ``M`` gets the one direction that bridges most of
    it, and only the cells bridged that way count.  Taking every cell's own
    best direction passed patches no single direction spans: the deck under
    the IFC2 podium, 119 mm2 in one layer, of which a slicer bridged
    two-thirds with 10-40 mm lines over air."""
    out = np.zeros_like(M)
    if not M.any():
        return out
    # a bridge's two anchors are within max_cells + 1 of every cell on it, so
    # a window that much bigger than the overhang gives the same answer
    rr, cc = np.nonzero(M)
    pad = int(max_cells) + 2
    r0, r1 = max(rr.min() - pad, 0), min(rr.max() + pad + 1, M.shape[0])
    c0, c1 = max(cc.min() - pad, 0), min(cc.max() + pad + 1, M.shape[1])
    m, a = M[r0:r1, c0:c1], anchor[r0:r1, c0:c1]
    per = []
    for k in range(directions):
        ang = 180.0 * k / directions
        if ang == 0.0:
            ok = _row_bridges(m, a, max_cells)
        elif ang == 90.0:
            ok = _row_bridges(m.T, a.T, max_cells).T
        else:
            mr = ndimage.rotate(m.astype(np.uint8), ang, order=0, reshape=True) > 0
            ar = ndimage.rotate(a.astype(np.uint8), ang, order=0, reshape=True) > 0
            okr = _row_bridges(mr, ar, max_cells)                 # rotation keeps the cell size
            back = ndimage.rotate(okr.astype(np.uint8), -ang, order=0, reshape=True) > 0
            h0 = (back.shape[0] - m.shape[0]) // 2
            w0 = (back.shape[1] - m.shape[1]) // 2
            ok = back[h0:h0 + m.shape[0], w0:w0 + m.shape[1]]
        per.append(ok & m)
    lab, n = ndimage.label(m, np.ones((3, 3), bool))
    idx = np.arange(1, n + 1)
    cover = np.stack([ndimage.sum(ok, lab, idx) for ok in per])       # (directions, patches)
    best = np.concatenate([[0], cover.argmax(0)])                      # per label, 0 = background
    pick = best[lab]
    sub = np.zeros_like(m)
    for k, ok in enumerate(per):
        sub |= ok & (pick == k) & (lab > 0)
    out[r0:r1, c0:c1] = sub
    return out


def _lean(over: np.ndarray, solid_k: np.ndarray, reach: float, strut: np.ndarray | None = None,
          carried: np.ndarray | None = None, k: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """What layer k carries under the overhanging patches ``over`` of the
    layer above; also which of it is a strut and which was put there for a
    loose fragment (see below), for the next layer down.

    A patch hanging from the part leans on the material ``solid_k`` of layer
    k: it is carried except for a strip ``reach`` cells wide along its far
    side, so the support narrows at 45 degrees towards that material - a
    bracket under a cantilever, a pointed arch under a wide span.

    A loose fragment touches no material of layer k: leaning would shrink it
    to nothing in the air, since it is joined to the part nowhere above. It
    is carried whole and a copy moved ``reach`` cells towards the nearest
    material; from then on only that copy (``strut``) goes on down, moved
    again on every layer - a 45 degree strut - until it stands on material.
    Each layer of a strut covers both its own place and the one above it, so
    consecutive layers overlap (a strut one cell wide stepping one cell per
    layer would only touch along an edge). What the layer above put down
    for a fragment besides its strut (``carried``) is held by the strut and
    needs nothing more.

    Distances are taken in a window round the patches, widened until it
    holds the nearest material of every patch cell, so the answer is the
    same as over the whole layer."""
    rr, cc = np.nonzero(over)
    H, W = over.shape
    m = 16
    while True:
        r0, r1 = max(rr.min() - m, 0), min(rr.max() + m + 1, H)
        c0, c1 = max(cc.min() - m, 0), min(cc.max() + m + 1, W)
        whole = r0 == 0 and c0 == 0 and r1 == H and c1 == W
        s = solid_k[r0:r1, c0:c1]
        o = over[r0:r1, c0:c1]
        if s.any():
            dist, (ir, ic) = ndimage.distance_transform_edt(~s, return_indices=True)
            if whole or dist[o].max() <= m:
                break
        elif whole:
            return over.copy(), over.copy(), over.copy()
        m *= 2
    sw = np.zeros_like(o) if strut is None else strut[r0:r1, c0:c1] & o
    cw = np.zeros_like(o) if carried is None else carried[r0:r1, c0:c1]
    free = o & ~cw & ~sw                     # overhang of the part itself
    lab, n = ndimage.label(free)
    ids = np.arange(1, n + 1)
    far = np.concatenate([[0.0], np.asarray(ndimage.maximum(dist, lab, ids))])
    near = np.concatenate([[0.0], np.asarray(ndimage.minimum(dist, lab, ids))])
    loose = near > 1.5
    loose[0] = False
    add = free & ~loose[lab] & (dist <= far[lab] - reach)
    # the cores that go on down: fresh fragments whole, struts as they came
    core = (free & loose[lab]) | sw
    st = np.zeros_like(o)
    elbow = np.zeros_like(o)
    step = int(np.floor((k + 1) * reach + 1e-9) - np.floor(k * reach + 1e-9))
    clab, cn = ndimage.label(core, np.ones((3, 3), bool))
    for i in range(1, cn + 1):
        pr, pc = np.nonzero(clab == i)
        j = int(np.argmin(dist[pr, pc]))
        d = np.array([ir[pr[j], pc[j]] - pr[j], ic[pr[j], pc[j]] - pc[j]], float)
        dr = dc = 0
        if step and np.hypot(*d) > 0:
            u = d / np.hypot(*d) * step
            dr, dc = int(round(u[0])), int(round(u[1]))
            if dr == 0 and dc == 0:
                dr, dc = (int(np.sign(d[0])), 0) if abs(d[0]) >= abs(d[1]) else (0, int(np.sign(d[1])))
        st[np.clip(pr + dr, 0, o.shape[0] - 1), np.clip(pc + dc, 0, o.shape[1] - 1)] = True
        if dr and dc:                        # a diagonal step: join the two by a face, not a corner
            elbow[np.clip(pr + dr, 0, o.shape[0] - 1), pc] = True
    # where a strut steps onto material it ends, overlapping that material
    # (one cell) so the two fuse; the rest goes on
    held = core | st | elbow
    add |= held
    st &= ~s
    out = [np.zeros_like(over) for _ in range(3)]
    for full, part in zip(out, (add, st, held)):
        full[r0:r1, c0:c1] = part
    return out[0], out[1], out[2]


def _rects(mask: np.ndarray, x_lo: float, y_hi: float, cell: float) -> list:
    """Row runs of a mask (rows going down from ``y_hi``, columns right from
    ``x_lo``, cells of ``cell``) as rectangles, counter-clockwise."""
    pad = np.zeros((mask.shape[0], mask.shape[1] + 2), np.int8)
    pad[:, 1:-1] = mask
    dm = np.diff(pad, axis=1)
    r0, c0 = np.nonzero(dm == 1)
    _, c1 = np.nonzero(dm == -1)
    xa, xb = x_lo + c0 * cell, x_lo + c1 * cell
    yt, yb = y_hi - r0 * cell, y_hi - (r0 + 1) * cell
    return [np.array([[xa[i], yb[i]], [xb[i], yb[i]], [xb[i], yt[i]], [xa[i], yt[i]]]) for i in range(len(r0))]


def _disk(r: int) -> np.ndarray:
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return xx * xx + yy * yy <= r * r + 0.5


def _tiles(V: np.ndarray, F: np.ndarray, z_min: float, pad: float, coarse: float = 1.0,
           extra: np.ndarray | None = None) -> list:
    """XY boxes (x0, y0, x1, y1, z) around the down-facing faces of a mesh,
    padded by ``pad`` - the only places a part can overhang - with the
    highest point of those faces. The lowest faces of every separate solid
    count as down-facing too: a loose fragment with a pointed or sloping
    bottom has no face looking down, but its first layer is printed on air.
    ``extra``: more places, (n, 2, 3) boxes (lowest, highest corner) found
    by comparing layers (see ``_scan``) - a tip hanging from a wall above
    has neither."""
    P = V[F]
    n = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    a = np.linalg.norm(n, axis=1)
    nz = n[:, 2] / np.maximum(a, 1e-12)
    down = (nz < -0.6) & (a > 1e-9)
    down |= _island_bottoms(V, F, P)
    down &= P[:, :, 2].max(1) > z_min
    P = P[down]
    if extra is not None and len(extra):
        P = np.concatenate([P, np.stack([extra[:, 0], extra[:, 1], extra[:, 1]], 1)])
    if not len(P):
        return []
    down = np.ones(len(P), bool)
    lo, hi = P[down, :, :2].min(1), P[down, :, :2].max(1)
    g0 = lo.min(0) - pad - coarse
    shape = np.ceil((hi.max(0) + pad + coarse - g0) / coarse).astype(int) + 1
    occ = np.zeros((shape[1], shape[0]), bool)
    i0 = np.floor((lo - g0) / coarse).astype(int)
    i1 = np.floor((hi - g0) / coarse).astype(int)
    for (c0, r0), (c1, r1) in zip(i0, i1):
        occ[r0:r1 + 1, c0:c1 + 1] = True
    k = int(np.ceil(pad / coarse))
    occ = ndimage.binary_dilation(occ, iterations=k)
    lab, _ = ndimage.label(occ)
    cen = ((P[down, :, :2].mean(1) - g0) / coarse).astype(int)
    which = lab[cen[:, 1], cen[:, 0]]
    ztop = P[down, :, 2].max(1)
    out = []
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        z = float(ztop[which == i].max()) if (which == i).any() else z_min
        out.append((g0[0] + sl[1].start * coarse, g0[1] + sl[0].start * coarse,
                    g0[0] + sl[1].stop * coarse, g0[1] + sl[0].stop * coarse, z))
    return out


def _scan(man: m3d.Manifold, base_mm: float, cell: float, layer: float) -> np.ndarray:
    """Boxes (n, 2, 3) round every place where a layer reaches more than a
    cell past the one below it (edge neighbours only) or starts on nothing:
    the model sliced layer by layer and compared, whatever its faces look
    like."""
    lo, hi = np.asarray(man.bounding_box()[:3]), np.asarray(man.bounding_box()[3:])
    jit = np.float32(0.4371 * cell)
    xs = (np.arange(lo[0] - cell, hi[0] + cell, cell) + jit).astype(np.float32)
    ys = (np.arange(hi[1] + cell, lo[1] - cell, -cell) + jit).astype(np.float32)
    cross = ndimage.generate_binary_structure(2, 1)
    boxes = []
    prev = None
    k = 0
    while True:
        z = base_mm + (k + 0.5) * layer
        if z >= hi[2]:
            break
        cur = _raster(man.slice(z).to_polygons(), xs, ys)
        if prev is not None:
            flag = cur & ~ndimage.binary_dilation(prev, cross)
            if flag.any():
                lab, _ = ndimage.label(flag, np.ones((3, 3), bool))
                for sl in ndimage.find_objects(lab):
                    boxes.append([[xs[sl[1].start] - cell, ys[sl[0].stop - 1] - cell, z - layer],
                                  [xs[sl[1].stop - 1] + cell, ys[sl[0].start] + cell, z]])
        prev = cur
        k += 1
    return np.asarray(boxes, np.float64).reshape(-1, 2, 3)


def _island_bottoms(V: np.ndarray, F: np.ndarray, P: np.ndarray, tol: float = 1e-3) -> np.ndarray:
    """Faces touching the lowest point of the separate solid they belong to
    (air pockets left out)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]).astype(np.int64)
    g = coo_matrix((np.ones(len(e), np.int8), (e[:, 0], e[:, 1])), shape=(len(V), len(V)))
    n, lab = connected_components(g, directed=False)
    shell = lab[F[:, 0]]
    vol = np.bincount(shell, np.einsum("ij,ij->i", P[:, 0], np.cross(P[:, 1], P[:, 2])) / 6.0, minlength=n)
    zmin = np.full(n, np.inf)
    np.minimum.at(zmin, lab, V[:, 2])
    return (vol[shell] > 0) & (P[:, :, 2].min(1) <= zmin[shell] + tol)


def _raster(polys: list, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Cells (centres ``xs`` ascending x ``ys`` descending, evenly spaced)
    inside a slice's polygons, by even-odd scanlines."""
    out = np.zeros((len(ys), len(xs)), bool)
    if not polys:
        return out
    A = np.concatenate([np.asarray(p, np.float64) for p in polys])
    B = np.concatenate([np.roll(np.asarray(p, np.float64), -1, axis=0) for p in polys])
    cell = float(xs[1] - xs[0])
    y0, x0 = float(ys[0]), float(xs[0])
    lo, hi = np.minimum(A[:, 1], B[:, 1]), np.maximum(A[:, 1], B[:, 1])
    # rows whose centre y satisfies lo <= y < hi
    r_lo = np.floor((y0 - hi) / cell).astype(np.int64) + 1
    r_hi = np.floor((y0 - lo) / cell).astype(np.int64)
    r_lo, r_hi = np.maximum(r_lo, 0), np.minimum(r_hi, len(ys) - 1)
    n = np.maximum(r_hi - r_lo + 1, 0)
    if not n.sum():
        return out
    e = np.repeat(np.arange(len(A)), n)
    r = r_lo[e] + (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n))
    y = y0 - r * cell
    f = (y - A[e, 1]) / (B[e, 1] - A[e, 1])
    x = A[e, 0] + f * (B[e, 0] - A[e, 0])
    o = np.lexsort((x, r))
    r, x = r[o], x[o]
    ra, xa, xb = r[0::2], x[0::2], x[1::2]
    ca = np.clip(np.ceil((xa - x0) / cell).astype(np.int64), 0, len(xs))
    cb = np.clip(np.ceil((xb - x0) / cell).astype(np.int64), 0, len(xs))
    D = np.zeros((len(ys), len(xs) + 1), np.int32)
    np.add.at(D, (ra, ca), 1)
    np.add.at(D, (ra, cb), -1)
    return np.cumsum(D[:, :-1], axis=1) > 0


def _occupancy(man: m3d.Manifold, xs: np.ndarray, ys: np.ndarray, z0: float, layer: float, K: int) -> np.ndarray:
    """(K, ny, nx) occupancy at layer centres z0 + (k + 0.5) * layer: the
    part sliced the way a slicer does, exactly."""
    box = m3d.Manifold.cube((float(xs[-1] - xs[0]) + 4 * float(xs[1] - xs[0]),
                             float(ys[0] - ys[-1]) + 4 * float(xs[1] - xs[0]), (K + 2) * layer)).translate(
        (float(xs[0]) - 2 * float(xs[1] - xs[0]), float(ys[-1]) - 2 * float(xs[1] - xs[0]), z0 - layer))
    part = man ^ box
    out = np.zeros((K, len(ys), len(xs)), bool)
    for k in range(K):
        out[k] = _raster(part.slice(z0 + (k + 0.5) * layer).to_polygons(), xs, ys)
    return out


def relief_guard(man: m3d.Manifold, cutters: list, base_mm: float, layer: float,
                 min_mm: float, share: float = 0.25) -> tuple[list, dict]:
    """The relief panels (``cutters``) that may be cut into ``man``.

    A facade relief recesses panels a fixed depth into the surface.  Where
    the model is thin in plan - a fin, a skirt, a wall between two facades
    both carrying relief - recesses from the two sides meet or leave a web
    too thin to print, and the cut splits the part.  Each panel is looked at in the layers it spans (as
    a slicer sees the model, at most five of them): where more than
    ``share`` of what it would take lies in parts of ``man`` narrower than
    ``min_mm`` (twice the relief depth and a printed line), the panel is
    left out whole - trimming panels to the thin parts left ragged edges
    and slivers (ICC's loose fragments went from 19 to 519).  A panel that
    only clips a thin part at its edge is still cut.

    After that, every layer of the cut model is compared with the one
    below: a panel whose cut leaves a piece starting on nothing (a neck
    between the sample layers cut through) is left out too.

    Returns (the panels to cut, stats with the panels and volume left out)."""
    stats = {"relief_panels_kept": 0, "relief_kept_mm3": 0.0}
    h = 0.5 * min_mm
    J = m3d.JoinType.Round
    slices: dict = {}
    out = []
    for c in cutters:
        bb = c.bounding_box()
        k0 = int(np.ceil((bb[2] - base_mm) / layer - 0.5))
        k1 = int(np.floor((bb[5] - base_mm) / layer - 0.5))
        ks = sorted(set(np.linspace(k0, k1, 5).round().astype(int))) if k1 >= k0 else []
        took = thin_took = 0.0
        for k in ks:
            z = float(base_mm + (k + 0.5) * layer)
            if k not in slices:
                slices[k] = man.slice(z)
            r = c.slice(z) ^ slices[k]
            if r.is_empty():
                continue
            b = r.bounds()
            win = m3d.CrossSection.square((b[2] - b[0] + 4 * min_mm, b[3] - b[1] + 4 * min_mm)).translate(
                (b[0] - 2 * min_mm, b[1] - 2 * min_mm))
            a = slices[k] ^ win
            thin = a - a.offset(-h, J, circular_segments=12).offset(h + 1e-4, J, circular_segments=12)
            took += r.area()
            thin_took += (r ^ thin).area()
        if took > 0 and thin_took > share * took:
            stats["relief_panels_kept"] += 1
            stats["relief_kept_mm3"] += (man ^ c).volume()
        else:
            out.append(c)
    # the sample layers can miss a neck between them: cut, then look at
    # every layer for parts the cut left standing on nothing.  The panels
    # that did it are not cut there - a box round the loose part, a
    # millimetre beyond it and from a millimetre below it, is taken out of
    # them - and the cut is looked at again (at most three rounds).  A panel
    # still to blame after that is not cut at all.  Leaving out the whole
    # panel at once took the relief off 100 mm of an ICC tower face (476
    # mm3) for one small part
    for rnd in range(4):
        if not out:
            break
        bad = _panels_cutting_loose(man, out, base_mm, layer)
        if not bad:
            break
        for i in sorted(bad, reverse=True):
            before = (man ^ out[i]).volume()
            if rnd < 3:
                hole = m3d.Manifold.batch_boolean(bad[i], m3d.OpType.Add)
                out[i] = out[i] - hole
                stats["relief_panels_trimmed"] = stats.get("relief_panels_trimmed", 0) + 1
                stats["relief_kept_mm3"] += before - (man ^ out[i]).volume()
                if out[i].is_empty():
                    del out[i]
            else:
                stats["relief_panels_kept"] += 1
                stats["relief_cut_loose"] = stats.get("relief_cut_loose", 0) + 1
                stats["relief_kept_mm3"] += before
                del out[i]
    stats["relief_kept_mm3"] = round(stats["relief_kept_mm3"], 2)
    return out, stats


def _panels_cutting_loose(man: m3d.Manifold, panels: list, base_mm: float, layer: float,
                          min_area: float = 0.0025, margin: float = 1.0) -> dict:
    """The panels whose cut leaves part of ``man`` starting on nothing (a
    piece of a layer that rests on nothing in the layer below where before
    the cut it did), each with the boxes round those parts it is not to
    cut: ``margin`` beyond the part on every side and from ``margin`` below
    it to a layer above.  The panels to blame are those cutting into the
    layer below near the part."""
    cut = m3d.Manifold.batch_boolean(panels, m3d.OpType.Add)
    post = man - cut
    zb = (man ^ cut).bounding_box()
    if (man ^ cut).is_empty():
        return {}
    k0 = max(1, int(np.floor((zb[2] - base_mm) / layer)))
    k1 = int(np.ceil((man.bounding_box()[5] - base_mm) / layer))
    boxes = [p.bounding_box() for p in panels]
    bad: dict = {}
    J = m3d.JoinType.Miter
    prev_post = post.slice(float(base_mm + (k0 - 0.5) * layer))
    prev_pre = man.slice(float(base_mm + (k0 - 0.5) * layer))
    for k in range(k0, k1):
        z = float(base_mm + (k + 0.5) * layer)
        cur_post, cur_pre = post.slice(z), man.slice(z)
        if not cur_post.is_empty():
            below_post = prev_post.offset(1e-3, J)
            below_pre = prev_pre.offset(1e-3, J)
            for c in cur_post.decompose():
                if c.area() < min_area or not (c ^ below_post).is_empty():
                    continue
                # stood on something before the cut?
                if (c ^ below_pre).is_empty():
                    continue
                b = c.bounds()
                m = margin
                win = m3d.CrossSection.square((b[2] - b[0] + 2 * m, b[3] - b[1] + 2 * m)).translate((b[0] - m, b[1] - m))
                hole = m3d.Manifold.cube((b[2] - b[0] + 2 * m, b[3] - b[1] + 2 * m, m + 1.5 * layer)).translate(
                    (b[0] - m, b[1] - m, z - m))
                for i, pb in enumerate(boxes):
                    if pb[2] <= z and pb[5] >= z - 2 * layer and pb[0] <= b[2] + m and pb[3] >= b[0] - m \
                            and pb[1] <= b[3] + m and pb[4] >= b[1] - m:
                        if not (panels[i].slice(z - layer) ^ win).is_empty() or not (panels[i].slice(z) ^ win).is_empty():
                            bad.setdefault(i, []).append(hole)
        prev_post, prev_pre = cur_post, cur_pre
    return bad


def widen_thin(man: m3d.Manifold, base_mm: float, cell: float, layer: float,
               width_mm: float) -> tuple[m3d.Manifold | None, dict]:
    """Material that widens the parts of ``man`` too thin to print.

    A carved part can be thinner than a printed line in plan even where it
    looks solid from the side: a mast truss seen edge-on, the fins of a
    crown.  A slicer then lays a dotted trail of tiny loops, or nothing, and
    what stands on it starts in mid-air.  Each layer is sliced (as a slicer
    does) and opened with a disk ``width_mm`` across; what the opening takes
    away is thinner than that.  Such a part is widened by the disk's radius
    and a cell on every side (so it is at least ``width_mm`` across whatever
    its direction on the grid) when it reaches more than three radii from
    anything wider - a member, a bar, a fin - or is a free piece at least
    ``width_mm`` long.
    The corner of a wider part, which the opening also clips, stays as it is
    unless it is sharper than about 25 degrees, so facades keep their edges.
    Specks smaller than ``width_mm`` are left alone (a slicer drops them),
    unless they carry something: a column stands on the layer below and
    holds up, through the specks above it, a wider part (a deck on two thin
    posts would otherwise be printed on air).  A tip that carries nothing -
    the top of a crown fin, a post on a roof - is left for the slicer to
    drop: widening it turned IFC2's crown fins into stacks of blocks.

    Returns (solid or None, stats with the added volume)."""
    stats = {"widened_mm3": 0.0}
    r = max(1, int((width_mm / cell - 1) // 2))
    if width_mm <= 0 or man.is_empty():
        return None, stats
    lo, hi = np.asarray(man.bounding_box()[:3]), np.asarray(man.bounding_box()[3:])
    jit = np.float32(0.0137 * cell)
    xs = (np.arange(lo[0] - 2 * r * cell, hi[0] + 2 * r * cell, cell) + jit).astype(np.float32)
    ys = (np.arange(hi[1] + 2 * r * cell, lo[1] - 2 * r * cell, -cell) + jit).astype(np.float32)
    disk = _disk(r)
    long_ = width_mm / cell
    x_lo, y_hi = float(xs[0]) - 0.5 * cell, float(ys[0]) + 0.5 * cell
    out = []
    shape = (len(ys), len(xs))
    # every layer, packed (a slicer's view of the model)
    packed = []
    while base_mm + (len(packed) + 0.5) * layer < hi[2]:
        z = base_mm + (len(packed) + 0.5) * layer
        packed.append(np.packbits(_raster(man.slice(z).to_polygons(), xs, ys), axis=None))
    n = len(packed)

    def layer_at(k):
        if 0 <= k < n:
            return np.unpackbits(packed[k], count=shape[0] * shape[1]).reshape(shape).astype(bool)
        return np.zeros(shape, bool)

    def specks(c):
        """Mask of the free pieces of c shorter than the target width."""
        lab, m = ndimage.label(c, np.ones((3, 3), bool))
        out_ = np.zeros_like(c)
        for i, sl in enumerate(ndimage.find_objects(lab), start=1):
            if max(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start) < long_:
                out_[sl] |= lab[sl] == i
        return out_

    # from the top down: specks that carry something (a wider part above,
    # or specks above that do), standing on the layer below
    carrying = [None] * n
    holds = np.zeros(shape, bool)        # what the layer above needs held up
    cur = layer_at(n - 1)
    for k in range(n - 1, -1, -1):
        below = layer_at(k - 1)
        sp = specks(cur) if cur.any() else np.zeros(shape, bool)
        mem = np.zeros(shape, bool)
        if sp.any() and holds.any():
            lab, _ = ndimage.label(sp, np.ones((3, 3), bool))
            on = np.unique(lab[sp & below])
            keep_ids = np.intersect1d(np.unique(lab[sp & holds]), on)
            if len(keep_ids):
                mem = np.isin(lab, keep_ids) & sp
        if mem.any():
            carrying[k] = np.packbits(mem, axis=None)
        holds = (cur & ~sp) | mem
        cur = below

    for k in range(n):
        z = base_mm + (k + 0.5) * layer
        cur = layer_at(k)
        if not cur.any():
            continue
        rr, cc = np.nonzero(cur)
        r0, r1 = max(rr.min() - 3 * r, 0), min(rr.max() + 3 * r + 1, cur.shape[0])
        c0, c1 = max(cc.min() - 3 * r, 0), min(cc.max() + 3 * r + 1, cur.shape[1])
        c = cur[r0:r1, c0:c1]
        mem = (np.unpackbits(carrying[k], count=shape[0] * shape[1]).reshape(shape).astype(bool)[r0:r1, c0:c1]
               if carrying[k] is not None else np.zeros_like(c))
        core = ndimage.binary_opening(c, disk)
        thin = c & ~core
        if not thin.any():
            continue
        far = ndimage.distance_transform_edt(~core) > 3 * r if core.any() else np.ones_like(c)
        seed = thin & far
        # free pieces shorter than the target width are specks: leave them,
        # unless they carry something (found above)
        lab, _ = ndimage.label(c, np.ones((3, 3), bool))
        for i, sl in enumerate(ndimage.find_objects(lab), start=1):
            if max(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start) < long_:
                piece = lab[sl] == i
                if not (piece & mem[sl]).any():
                    seed[sl] &= ~piece
        if not seed.any():
            continue
        grow = ndimage.binary_propagation(seed, mask=thin)
        add = ndimage.binary_dilation(grow, _disk(r + 1)) & ~c   # a cell more: widths measured across the grid
        if not add.any():
            continue
        stats["widened_mm3"] += float(add.sum()) * cell * cell * layer
        full = np.zeros_like(cur)
        full[r0:r1, c0:c1] = add
        # overlap the layers next to it a little so the slabs fuse
        cs = m3d.CrossSection(_rects(full, x_lo, y_hi, cell), m3d.FillRule.Positive).simplify(0.01 * cell)
        if not cs.is_empty():
            out.append(cs.extrude(1.02 * layer).translate((0.0, 0.0, z - 0.51 * layer)))
    stats["widened_mm3"] = round(stats["widened_mm3"], 2)
    if not out:
        return None, stats
    return m3d.Manifold.batch_boolean(out, m3d.OpType.Add), stats


def _shift_open(a: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """out[q] = a[q + (dy, dx)], True where that falls off the grid."""
    out = np.ones_like(a)
    H, W = a.shape
    if abs(dy) >= H or abs(dx) >= W:
        return out
    out[max(-dy, 0):H - max(dy, 0), max(-dx, 0):W - max(dx, 0)] = \
        a[max(dy, 0):H - max(-dy, 0), max(dx, 0):W - max(-dx, 0)]
    return out


def _sight(clear: np.ndarray, seen: np.ndarray, step: tuple, toward_end: bool) -> None:
    """Mark in ``seen`` the voxels of ``clear`` from which a straight line
    of sight runs out of the grid through ``clear`` only.  Axis 0 is the
    line's main axis (the viewer at its far end if ``toward_end``); each
    step along it towards the viewer moves the line by ``step`` (fractions
    of a voxel) across the other two axes - a digital line, rounded from
    the start."""
    n = clear.shape[0]
    order = range(n - 1, -1, -1) if toward_end else range(n)
    prev = prev_o = None
    for t, i in enumerate(order):
        # t steps away from the viewer
        o = (int(round(-t * step[0])), int(round(-t * step[1])))
        v = clear[i].copy() if prev is None else clear[i] & _shift_open(prev, prev_o[0] - o[0], prev_o[1] - o[1])
        seen[i] |= v
        prev, prev_o = v, o


def fill_hollows(man: m3d.Manifold, base_mm: float, cell: float, layer: float,
                 min_open_mm: float, azimuths: int = 32,
                 elevations: tuple = (0.0, 8.0, 15.0, 22.0, 30.0, 45.0, 60.0, 75.0)) -> tuple[m3d.Manifold | None, dict]:
    """Material that makes solid everything inside a model that cannot be
    seen from outside.

    Where carved parts meet - a tower on its podium, a skirt over a roof, a
    building on the base - they leave cracks and hollows between them, and
    a model only needs what a viewer sees.  The model is sliced into voxels
    (``cell`` across, one ``layer`` high).  The viewer looks from the sky
    and from the sides (``azimuths`` directions at each of ``elevations``
    degrees, and straight down), never from below.  Lines of sight run
    through openings at least ``min_open_mm`` across - so a crack seen head
    on shows only its mouth - and air within ``min_open_mm`` of what they
    reach (a facade recess) stays open; all other air is filled.  Fill
    hanging over open air more steeply than 45 degrees, and fins of fill
    thinner than a voxel or two standing out of the model, are left out.
    Returns (solid or None, stats)."""
    stats = {"hollow_mm3": 0.0}
    if man.is_empty() or min_open_mm <= 0:
        return None, stats
    lo, hi = np.asarray(man.bounding_box()[:3]), np.asarray(man.bounding_box()[3:])
    r1 = max(1, int(min_open_mm / (2 * cell) + 1e-6))            # an opening of min_open_mm stays open
    r2 = max(1, int(round(min_open_mm / cell)))                  # depth seen into a crack
    # distances are physical: voxels are ``layer`` tall and ``cell`` wide,
    # so a step of one cell reaches cell / layer voxels up or down (with
    # layers as tall as cells, the 6-neighbour cross)
    step = _l1_ball(1.0, layer / cell)
    rz = int(np.ceil(r1 * cell / layer - 1e-6))                  # r1 cells, in layers
    m = (r1 + 2) * cell
    jit = np.float32(0.0137 * cell)
    xs = (np.arange(lo[0] - m, hi[0] + m, cell) + jit).astype(np.float32)
    ys = (np.arange(hi[1] + m, lo[1] - m, -cell) + jit).astype(np.float32)
    K = int(np.ceil((hi[2] - base_mm) / layer)) + 1
    if K < 2:
        return None, stats
    # rz solid layers below (the base), rz + 1 empty ones on top (the sky)
    air = np.ones((rz + K + rz + 1, len(ys), len(xs)), bool)
    air[:rz] = False
    for k in range(K):
        air[rz + k] = ~_raster(man.slice(base_mm + (k + 0.5) * layer).to_polygons(), xs, ys)
    clear = ndimage.binary_erosion(air, _l1_ball(r1, layer / cell), border_value=1)
    # one copy of clear and seen per main axis (z, y, x), contiguous along it
    lay = {0: (0, 1, 2), 1: (1, 0, 2), 2: (2, 0, 1)}
    views = {}
    vox = np.array([layer, cell, cell])
    dirs = [np.array([1.0, 0.0, 0.0])]
    for e in elevations:
        for j in range(azimuths):
            th = 2 * np.pi * j / azimuths
            dirs.append(np.array([np.sin(np.radians(e)), np.cos(np.radians(e)) * np.sin(th),
                                  np.cos(np.radians(e)) * np.cos(th)]))
    for d in dirs:
        u = d / vox
        p = int(np.argmax(np.abs(u)))
        if p not in views:
            c = np.ascontiguousarray(clear.transpose(lay[p]))
            views[p] = (c, np.zeros_like(c))
        q = [i for i in lay[p] if i != p]
        _sight(*views[p], (u[q[0]] / abs(u[p]), u[q[1]] / abs(u[p])), u[p] > 0)
    del clear
    seen = np.zeros_like(air)
    for p, (c, sv) in views.items():
        seen |= sv.transpose(np.argsort(lay[p]))
    del views
    keep = ndimage.binary_dilation(seen, step, iterations=r1 + r2, mask=air)
    del seen
    fill = air & ~keep
    fill[:rz] = False
    fill[rz + K:] = False
    del keep
    # no fins standing out of the model, nothing hanging over open air
    body = ~air | fill
    fill &= ndimage.binary_opening(body, step, border_value=1)
    del body
    # edge neighbours only: a cell resting on a corner below starts on air;
    # one cell out per cell of height (45 degrees) - with layers lower than
    # a cell, only every cell / layer layers
    sq = ndimage.generate_binary_structure(2, 1)
    every = max(1, int(round(cell / layer)))
    for k in range(rz, rz + K):
        below = ~air[k - 1] | fill[k - 1]
        fill[k] &= ndimage.binary_dilation(below, sq) if (k - rz) % every == 0 else below
    # weld it on: reach a voxel into the model wherever no open air is near,
    # or the cell steps of the slabs leave hairline gaps at the true surface
    cross = ndimage.generate_binary_structure(3, 1)
    open_air = air & ~fill
    weld = ndimage.binary_dilation(fill, cross) & ~air & ~ndimage.binary_dilation(open_air, cross)
    del open_air
    stats["hollow_mm3"] = round(float(fill.sum()) * cell * cell * layer, 2)
    fill = (fill | weld)[rz:rz + K]
    del weld, air
    if not fill.any():
        return None, stats
    x_lo, y_hi = float(xs[0]) - 0.5 * cell, float(ys[0]) + 0.5 * cell
    slabs = []
    for k in np.nonzero(fill.any((1, 2)))[0]:
        # reach a little into the layers next to it, so the slabs fuse
        cs = m3d.CrossSection(_rects(fill[k], x_lo, y_hi, cell), m3d.FillRule.Positive).simplify(0.01 * cell)
        if not cs.is_empty():
            slabs.append(cs.extrude(1.02 * layer).translate((0.0, 0.0, base_mm + (k - 0.01) * layer)))
    if not slabs:
        return None, stats
    return m3d.Manifold.batch_boolean(slabs, m3d.OpType.Add), stats


def _l1_ball(r: float, aspect: float) -> np.ndarray:
    """Voxels (z, y, x) within physical L1 distance ``r`` cells of the
    centre, a voxel being ``aspect`` cells tall: with aspect 1 and r = n,
    the 6-neighbour cross dilated n - 1 times."""
    nz = int(np.floor(r / aspect + 1e-9))
    n = int(np.floor(r + 1e-9))
    z, y, x = np.mgrid[-nz:nz + 1, -n:n + 1, -n:n + 1]
    return np.abs(z) * aspect + np.abs(y) + np.abs(x) <= r + 1e-9


def _at_least_a_line(add: np.ndarray, solid_: np.ndarray, w: int, shadow: np.ndarray) -> np.ndarray:
    """``add`` (the support of one layer) grown wherever it, together with
    the ``solid_`` beside it, is narrower than ``w`` cells, so every cell
    of it lies in a ``w`` x ``w`` block of material.

    Free-standing pieces (touching no ``solid_`` of the same layer: a strut,
    a column under a loose fragment) grow freely: a strut one cell wide under
    the tip of a cone hanging from a wall was 0.18 mm across for 11 layers.
    Support against the side of the part grows only within the ``shadow`` of
    what it holds up (the layer above).  Growing it outwards too put cells
    out over air that needed support in turn: at 0.08 mm layers, where a
    45 degree bracket steps out less than a cell per layer, the support of
    IFC2 grew from 0.9 to 5.7 cm3.  Not growing it at all lost most of what
    the growth gained at 0.2 mm (IFC2 unsupported 14.2 -> 17.2 mm3)."""
    if w <= 1 or not add.any():
        return add
    u = add | solid_
    thin = add & ~ndimage.binary_opening(u, np.ones((w, w), bool))
    if not thin.any():
        return add
    lab, n = ndimage.label(add, np.ones((3, 3), bool))
    touching = np.zeros(n + 1, bool)
    touching[np.unique(lab[ndimage.binary_dilation(solid_, _CROSS) & add])] = True
    touching[0] = False
    grow = lambda t: ndimage.binary_dilation(t, np.ones((2 * w - 1, 2 * w - 1), bool)) & ~solid_
    out = add.copy()
    free, fused = thin & ~touching[lab], thin & touching[lab]
    if free.any():
        out |= grow(free)
    if fused.any():
        out |= grow(fused) & shadow
    return out


def support_fill(man: m3d.Manifold, rest: tuple | None, base_mm: float, cell: float, layer: float,
                 max_bridge_mm: float = 10.0, line_mm: float = 0.4, angle_deg: float = 45.0,
                 jitter: float = 0.0137, outline: m3d.CrossSection | None = None,
                 scan: bool = False) -> tuple[m3d.Manifold | None, dict]:
    """Material to add under ``man`` (print mm, z up, the base top at
    ``base_mm``) so that it prints without supports.

    rest -- optional (xs, ys, Z): the 2.5D surface the part stands on (grid
            x ascending, y descending, Z in mm), counted as support.
    outline -- optional model outline (plan view).  Support is built from
            grid cells; where it reaches the outline the cut would leave the
            cell steps as a comb of notches along the rim, so within two
            cells of the outline each slab is grown by a cell and a half
            and clipped to the outline itself: its edge there is the rim.
    scan -- also look wherever slicing the model layer by layer finds a
            layer reaching past the one below or starting on nothing (see
            ``_scan``), not only round down-facing faces.
    Returns (solid or None, stats)."""
    stats = {"support_mm3": 0.0, "bridged_mm2": 0.0}
    V, F = solid.to_arrays(man)
    if not len(F):
        return None, stats
    reach = layer / np.tan(np.radians(angle_deg)) / cell            # overhang allowance, cells per layer
    wide = max(1, int(np.ceil(line_mm / cell - 1e-6)))              # cells across a printed line
    pad = max_bridge_mm + 4 * cell
    tiles = _tiles(V.astype(np.float64), F, base_mm + layer, pad,
                   extra=_scan(man, base_mm, cell, layer) if scan else None)
    if not tiles:
        return None, stats
    if rest is not None:
        rxs, rys, RZ = rest
    out = []
    rim = None
    if outline is not None and not outline.is_empty():
        rim = outline - outline.offset(-2 * cell, m3d.JoinType.Miter, 4.0)
    jit = np.float32(jitter * cell)                                  # keep cell centres off mesh vertices
    for x0, y0, x1, y1, ztop in tiles:
        xs = (np.arange(x0, x1, cell) + 0.5 * cell + jit).astype(np.float32)
        ys = (np.arange(y1, y0, -cell) - 0.5 * cell + jit).astype(np.float32)
        # nothing above the highest down-facing face can need support
        K = int(np.ceil((ztop - base_mm) / layer)) + 2
        if K < 2 or len(xs) < 3 or len(ys) < 3:
            continue
        O = _occupancy(man, xs, ys, base_mm, layer, K)
        zc = base_mm + (np.arange(K) + 0.5) * layer
        if rest is not None:
            cols = np.interp(xs, rxs, np.arange(len(rxs)))
            rows = np.interp(-ys, -rys, np.arange(len(rys)))
            R, C = np.meshgrid(rows, cols, indexing="ij")
            Zr = ndimage.map_coordinates(RZ, [R, C], order=1, mode="nearest")
            O |= Zr[None] >= zc[:, None, None]
        Fill = np.zeros_like(O)
        req = O[K - 1]
        strut = np.zeros_like(O[0])
        carried = np.zeros_like(O[0])
        for k in range(K - 2, -1, -1):
            over = req & ~O[k]                       # layer k+1 over air
            if not over.any():
                req = O[k]
                strut[:] = False
                carried[:] = False
                continue
            br = bridgeable(over, O[k], max_bridge_mm / cell, directions=12)
            if br.any():
                # a slicer lays a bridge as one patch: the edge of a span
                # running across the grid, which steps out of its own row
                # by a cell, goes with the lines beside it.  No further: a
                # strip along a curved rim (ICC) is spanned by a chord, and
                # what lies outside the chord hangs from one side
                br |= over & ndimage.binary_dilation(br, np.ones((3, 3), bool))
            over &= ~br
            stats["bridged_mm2"] += float(br.sum()) * cell * cell
            if not over.any():
                req = O[k]
                strut[:] = False
                carried[:] = False
                continue
            # layer k carries each overhanging patch except a strip one
            # allowance wide along its far side, measured from the material
            # of layer k: so every layer reaches at most that far past the
            # one below it - a 45 degree bracket under a cantilever, a
            # pointed arch under a span too wide to bridge - and every patch
            # is followed down until it stands on something
            if O[k].any():
                add, strut, carried = _lean(over, O[k], reach, strut, carried, k)
            else:                                    # nothing beside it to lean on: straight down
                add = over
                strut = over.copy()
                carried = over.copy()
            # what is added must itself print: nowhere narrower than a
            # line (a strut one cell wide under the tip of a cone hanging
            # from a wall was 0.18 mm across for 11 layers)
            add = _at_least_a_line(add, O[k], wide, req)    # req: layer k+1
            Fill[k] = add
            req = O[k] | add
        if not Fill.any():
            continue
        stats["support_mm3"] += float(Fill.sum()) * cell * cell * layer
        # one slab per layer, exactly as the printer will lay it down; each
        # also takes the cells of the layers next to it that are solid in
        # the part, so a run reaches one layer into the part above and the
        # support below and fuses with them
        E = Fill.copy()
        E[1:] |= Fill[:-1] & O[1:]
        E[:-1] |= Fill[1:] & O[:-1]
        x_lo, y_hi = float(xs[0]) - 0.5 * cell, float(ys[0]) + 0.5 * cell
        for k in np.nonzero(E.any((1, 2)))[0]:
            rects = _rects(E[k], x_lo, y_hi, cell)
            cs = m3d.CrossSection(rects, m3d.FillRule.Positive).simplify(0.01 * cell)
            if cs.is_empty():
                continue
            if rim is not None:
                edge = cs.offset(1.5 * cell, m3d.JoinType.Miter, 4.0) ^ rim
                if not edge.is_empty():
                    cs = (cs + edge) ^ outline
            z0 = base_mm + k * layer - 0.01 * layer
            out.append(cs.extrude(1.02 * layer).translate((0.0, 0.0, z0)))
    stats["support_mm3"] = round(stats["support_mm3"], 2)
    stats["bridged_mm2"] = round(stats["bridged_mm2"], 2)
    if not out:
        return None, stats
    return m3d.Manifold.batch_boolean(out, m3d.OpType.Add), stats
