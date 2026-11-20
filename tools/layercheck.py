"""Would this model print as sliced?  An independent layer-by-layer check.

    layercheck.py model.stl [--base MM] [--layers 0.08,0.12,0.2] [--first-layer MM]
                            [--line 0.4] [--bridge 10] [--min-area 0.0025] [--json out.json]

Slices the model the way a slicer does (planes at the middle of each layer,
from the bed up) and compares each layer with the one below, using exact
cross-sections - none of the generator's own rasters, candidate windows or
support rules, so its blind spots are not shared:

  births      parts of a layer that touch nothing in the layer below: they
              would be printed on air (the tip of a cone hanging from a wall,
              a speck under a canopy);
  overhangs   material reaching further past the layer below than half a
              line width (a perimeter wider than that out over air sags).
              Each connected patch is tested as a slicer bridges it - one
              direction for the whole patch, every line of it resting on the
              layer below on both sides for at least a line width, within the
              bridge length.  Patches that pass are bridges (with their
              longest span), the lines of them that do not hold and the
              patches that fail are unsupported.  Areas are summed over the
              layers; times the layer height (``mm3``) they compare across
              layer heights;
  thin        members narrower than a line width that stay so for at least
              three layers (a slicer drops or fattens them): whole pieces,
              and ribs, fins and necks joined to something thicker.

Layers are sliced at their middles as a slicer lays them: the first layer
(``--first-layer``, default the layer height), then one layer height each.

The base height (``--base``) is read from the build's result.json (next to
the model, or <name>_result.json) if not given; without either it stops
rather than guess.  Reports each layer height separately: layer phases
differ, so a pass at 0.2 mm says nothing about 0.08 mm.
"""
import argparse
import json
import sys
from pathlib import Path

import manifold3d as m3d
import numpy as np
import trimesh
from PIL import Image, ImageDraw

RES = 0.05            # raster for the bridge test, mm


def load(path: str) -> m3d.Manifold:
    m = trimesh.load(path, process=True)
    m.merge_vertices()
    man = m3d.Manifold(m3d.Mesh(np.asarray(m.vertices, np.float32), np.asarray(m.faces, np.uint32)))
    if man.status() != m3d.Error.NoError:
        sys.exit(f"{path}: not a closed manifold ({man.status()})")
    return man


def base_height(path: Path, given: float | None) -> float:
    if given is not None:
        return given
    for p in (path.with_name("result.json"), path.with_name(path.stem + "_result.json")):
        if p.exists():
            return float(json.loads(p.read_text())["params"]["base_mm"])
    sys.exit(f"{path.name}: no result.json beside it - give the base height with --base")


def raster(cs: m3d.CrossSection, x0: float, y1: float, w: int, h: int) -> np.ndarray:
    """Cells (RES, rows down from y1, columns right from x0) inside cs, even-odd."""
    acc = np.zeros((h, w), bool)
    for poly in cs.to_polygons():
        P = np.asarray(poly)
        if len(P) < 3:
            continue
        im = Image.new("1", (w, h), 0)
        ImageDraw.Draw(im).polygon([((x - x0) / RES - 0.5, (y1 - y) / RES - 0.5) for x, y in P], fill=1)
        acc ^= np.asarray(im, bool)
    return acc


def bridge_test(patch: m3d.CrossSection, cur: m3d.CrossSection, prev: m3d.CrossSection,
                bridge: float, anchor: float = 0.4, step_deg: float = 3.0, max_samples: int = 2500,
                ok_share: float = 0.98):
    """Best single direction for bridging ``patch``: (passes, angle, longest span, share ok).

    A line of the bridge holds where, on both sides, it runs on over the
    layer below for ``anchor`` mm: an end resting on a sliver of the layer
    below is not anchored."""
    b = patch.bounds()
    x0, y0, x1, y1 = b[0] - bridge, b[1] - bridge, b[2] + bridge, b[3] + bridge
    w, h = int(np.ceil((x1 - x0) / RES)) + 1, int(np.ceil((y1 - y0) / RES)) + 1
    win = m3d.CrossSection.square((x1 - x0, y1 - y0)).translate((x0, y0))
    C = raster(cur ^ win, x0, y1, w, h)
    A = raster(prev ^ win, x0, y1, w, h)
    S = raster(patch, x0, y1, w, h)
    rr, cc = np.nonzero(S)
    if not len(rr):
        return True, 0.0, 0.0, 1.0
    if len(rr) > max_samples:
        pick = np.random.default_rng(0).choice(len(rr), max_samples, replace=False)
        rr, cc = rr[pick], cc[pick]
    steps = np.arange(1, int((bridge + anchor) / RES) + 1)
    la = max(1, int(round(anchor / RES)))
    best = (False, 0.0, np.inf, 0.0)
    for deg in np.arange(0.0, 180.0, step_deg):
        t = np.radians(deg)
        half = []
        for sgn in (1, -1):
            r = np.rint(rr[:, None] - sgn * np.sin(t) * steps[None]).astype(int)
            c = np.rint(cc[:, None] + sgn * np.cos(t) * steps[None]).astype(int)
            inside = (r >= 0) & (r < h) & (c >= 0) & (c < w)
            r, c = np.clip(r, 0, h - 1), np.clip(c, 0, w - 1)
            anc = A[r, c] & inside
            mat = C[r, c] & inside
            # the anchor: ``la`` steps in a row where the line (this
            # layer) lies on the layer below
            held = anc & mat
            run = np.ones_like(anc)
            for j in range(la):
                run[:, :len(steps) - j] &= held[:, j:]
            run[:, len(steps) - la + 1:] = False
            hit = run.any(1)
            first = np.where(hit, run.argmax(1), len(steps))
            # every step before the anchor is material of this layer (or
            # the layer below, where an anchor too short to hold lies)
            broken = ~(mat | anc) & (np.arange(len(steps))[None] < first[:, None])
            ok = hit & ~broken.any(1)
            half.append((ok, (first + 1) * RES))
        span = half[0][1] + half[1][1]
        ok = half[0][0] & half[1][0] & (span <= bridge)
        share = ok.mean()
        longest = float(span[ok].max()) if ok.any() else np.inf
        if share > best[3] or (share == best[3] and longest < best[2]):
            best = (share >= ok_share, float(deg), longest, float(share))
    return best


def max_step(patch: m3d.CrossSection, prev: m3d.CrossSection, lo: float, hi: float = 4.0) -> float:
    """How far ``patch`` reaches past ``prev``: the least offset of prev that covers it."""
    if (patch - prev.offset(hi, m3d.JoinType.Round, circular_segments=24)).area() > 1e-6:
        return float("inf")
    for _ in range(10):
        mid = 0.5 * (lo + hi)
        if (patch - prev.offset(mid, m3d.JoinType.Round, circular_segments=24)).area() > 1e-6:
            lo = mid
        else:
            hi = mid
    return round(hi, 3)


def min_width(cs: m3d.CrossSection, hi: float) -> float:
    """Width of the widest strip that fits: bisection on inward offsets."""
    lo = 0.0
    for _ in range(12):
        mid = 0.5 * (lo + hi)
        if cs.offset(-mid / 2, m3d.JoinType.Round, circular_segments=16).is_empty():
            hi = mid
        else:
            lo = mid
    return round(hi, 3)


def thin_parts(cs: m3d.CrossSection, line: float, min_area: float) -> list:
    """Parts of a layer narrower than a line: whole pieces, and ribs, fins
    and necks joined to something thicker (opening the piece with a disk a
    line across takes them off; corners it rounds are not counted - they
    are shorter than two lines)."""
    J = m3d.JoinType.Round
    r = 0.475 * line
    out = []
    for c in cs.decompose():
        if c.area() < min_area:
            continue
        inner = c.offset(-r, J, circular_segments=16)
        if inner.is_empty():
            out.append(c)
            continue
        for t in (c - inner.offset(r + 1e-4, J, circular_segments=16)).decompose():
            b = t.bounds()
            if t.area() >= min_area and np.hypot(b[2] - b[0], b[3] - b[1]) >= 2 * line:
                out.append(t)
    return out


def check(man: m3d.Manifold, base: float, layer: float, line: float, bridge: float, min_area: float,
          first: float | None = None) -> dict:
    top = man.bounding_box()[5]
    allow = 0.5 * line
    first = first or layer
    # the middle of each layer as a slicer lays them: the first layer, then
    # ``layer`` each (the generator samples the same planes)
    zs = np.concatenate([[0.5 * first], np.arange(first + 0.5 * layer, top, layer)])
    births, overhangs, bridges = [], [], []
    thin_runs, open_runs = [], []          # thin members: [z0, z1, cs, widths]
    prev = None
    for z in zs:
        cur = man.slice(float(z))
        if prev is not None and not cur.is_empty():
            # births: pieces touching nothing below
            touched = prev.offset(1e-3, m3d.JoinType.Miter)
            born = []
            for comp in cur.decompose():
                if (comp ^ touched).area() < 1e-9 and comp.area() >= min_area:
                    b = comp.bounds()
                    born.append(comp)
                    births.append({"z": round(float(z), 3), "area_mm2": round(comp.area(), 4),
                                   "x": [round(b[0], 2), round(b[2], 2)], "y": [round(b[1], 2), round(b[3], 2)]})
            # overhangs beyond half a line width
            beyond = cur - prev.offset(allow, m3d.JoinType.Round, circular_segments=32)
            for bc in born:
                beyond = beyond - bc
            for patch in beyond.decompose():
                if patch.area() < min_area:
                    continue
                ok, deg, span, share = bridge_test(patch, cur, prev, bridge)
                b = patch.bounds()
                near = prev ^ m3d.CrossSection.square((b[2] - b[0] + 8, b[3] - b[1] + 8)).translate((b[0] - 4, b[1] - 4))
                step = max_step(patch, near, allow)
                rec = {"z": round(float(z), 3), "area_mm2": round(patch.area(), 4), "step_mm": step,
                       "x": [round(b[0], 2), round(b[2], 2)], "y": [round(b[1], 2), round(b[3], 2)],
                       "angle_deg": deg, "span_mm": None if not np.isfinite(span) else round(span, 2),
                       "bridged_share": round(share, 3)}
                (bridges if ok else overhangs).append(rec)
                if ok and share < 1.0:
                    # the lines of a passing patch that do not hold still hang
                    overhangs.append({**rec, "area_mm2": round((1.0 - share) * patch.area(), 4),
                                      "rest_of_bridge": True})
        # thin members, followed from layer to layer
        if not cur.is_empty():
            pieces = thin_parts(cur, line, min_area)
            nxt, used = [], set()
            for c in pieces:
                grown = c.offset(0.05, m3d.JoinType.Miter)
                j = next((j for j, r in enumerate(open_runs) if j not in used and not (r[2] ^ grown).is_empty()), None)
                w = min_width(c, line)
                if j is None:
                    nxt.append([float(z), float(z), c, [w]])
                else:
                    used.add(j)
                    r = open_runs[j]
                    nxt.append([r[0], float(z), c, r[3] + [w]])
            thin_runs += [r for j, r in enumerate(open_runs) if j not in used]
            open_runs = nxt
        else:
            thin_runs += open_runs
            open_runs = []
        prev = cur
    thin_runs += open_runs
    thin = []
    for r in thin_runs:
        if len(r[3]) >= 3:
            b = r[2].bounds()
            thin.append({"z": [round(r[0], 3), round(r[1], 3)], "layers": len(r[3]),
                         "min_width_mm": min(r[3]), "x": [round(b[0], 2), round(b[2], 2)],
                         "y": [round(b[1], 2), round(b[3], 2)]})
    total = lambda L: round(sum(e["area_mm2"] for e in L), 3)
    # areas summed over layers depend on how many layers there are; times
    # the layer height they are volumes, comparable across layer heights
    vol = lambda L: round(sum(e["area_mm2"] for e in L) * layer, 3)
    # a perimeter reaching out up to three-quarters of a line still prints
    # (sagging a little); past that it hangs in the air
    severe = [e for e in overhangs if e["step_mm"] > 0.75 * line]
    return {"layer_mm": layer, "first_layer_mm": first, "layers": len(zs), "allowance_mm": allow, "bridge_mm": bridge,
            "births": {"count": len(births), "area_mm2": total(births),
                       "largest": sorted(births, key=lambda e: -e["area_mm2"])[:15]},
            "severe": {"count": len(severe), "area_mm2": total(severe), "mm3": vol(severe),
                       "step_over_mm": round(0.75 * line, 3),
                       "largest": sorted(severe, key=lambda e: -e["area_mm2"])[:20]},
            "unsupported": {"count": len(overhangs), "area_mm2": total(overhangs), "mm3": vol(overhangs),
                            "largest": sorted(overhangs, key=lambda e: -e["area_mm2"])[:15], "all": overhangs},
            "bridges": {"count": len(bridges), "area_mm2": total(bridges),
                        "longest_span_mm": max((e["span_mm"] for e in bridges if e["span_mm"]), default=0.0),
                        "longest": sorted(bridges, key=lambda e: -(e["span_mm"] or 0))[:10]},
            "thin": {"count": len(thin), "members": sorted(thin, key=lambda e: e["min_width_mm"])[:20]}}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model")
    ap.add_argument("--base", type=float)
    ap.add_argument("--layers", default="0.08,0.12,0.2")
    ap.add_argument("--line", type=float, default=0.4)
    ap.add_argument("--bridge", type=float, default=10.0)
    ap.add_argument("--min-area", type=float, default=0.0025)
    ap.add_argument("--first-layer", type=float, help="the slicer's first layer height (default: the layer height)")
    ap.add_argument("--json")
    a = ap.parse_args()
    path = Path(a.model)
    base = base_height(path, a.base)
    man = load(a.model)
    out = {"model": path.name, "base_mm": base, "line_mm": a.line, "results": []}
    for layer in (float(s) for s in a.layers.split(",")):
        r = check(man, base, layer, a.line, a.bridge, a.min_area, a.first_layer)
        out["results"].append(r)
        print(f"{path.name} @ {layer} mm: births {r['births']['count']} ({r['births']['area_mm2']} mm2), "
              f"unsupported {r['unsupported']['count']} ({r['unsupported']['area_mm2']} mm2 over layers = {r['unsupported']['mm3']} mm3; "
              f"reaching over {r['severe']['step_over_mm']} mm: {r['severe']['count']}, {r['severe']['area_mm2']} mm2), "
              f"bridges {r['bridges']['count']} (longest {r['bridges']['longest_span_mm']} mm), "
              f"thin members {r['thin']['count']}")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
