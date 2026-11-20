"""Which step of the build made a defect?  Follows layercheck's findings back
through the stages of the build.

    stagetrace.py stages/ check.json [--model out.stl] [--top 10] [--layer 0.2]
                  [--line 0.4] [--bridge 10] [--json trace.json]

``stages/`` is the folder a build writes with HK3D_DEBUG_STAGES set: the
model as it stood after each step, in print millimetres -

  0_ground         the ground and the other buildings (as assembled)
  1_source         the carved landmark, scaled, before anything else
  2_trimmed        cut to the outline and decimated, spires thickened
  3_widened        thin load-bearing members widened
  4_supported      support material and bridges added
  5_relief         the facade relief cut
  6_assembled      ground + landmark
  7_filled         sealed hollows filled
  8_final_support  the whole-model support pass
  9_standing       fragments that stand on nothing removed

``check.json`` is layercheck's output for the finished model.  The largest
unsupported patches and births of the chosen layer height are looked at in
every stage: at the same layer, in the same place (the patch's box), is
there material, and is it held by the layer below (beyond half a line, and
not bridgeable in one direction, it hangs)?  The landmark stages 1-5 stand
on 0_ground, so they are checked together with it.

The first stage where a defect hangs is where it came from:
1_source - in the source (or the carving of it); 2_trimmed - cropping took
its support away; 3_widened, 5_relief - made by that step; 6_assembled -
the landmark and the ground do not meet; 7-9 - made or left by the
assembly passes.  A defect that hangs from 1_source and is never fixed is
one the support passes did not reach.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import manifold3d as m3d
import numpy as np

_spec = importlib.util.spec_from_file_location("layercheck", Path(__file__).with_name("layercheck.py"))
lc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lc)

MARK = {"absent": ".", "held": "o", "hangs": "X", "born": "B"}   # in the printed table
LANDMARK = ("1_source", "2_trimmed", "3_widened", "4_supported", "5_relief")


def box(x0, y0, x1, y1, z0, z1) -> m3d.Manifold:
    return m3d.Manifold.cube((x1 - x0, y1 - y0, z1 - z0)).translate((x0, y0, z0))


def square(x0, y0, x1, y1) -> m3d.CrossSection:
    return m3d.CrossSection.square((x1 - x0, y1 - y0)).translate((x0, y0))


def look(man: m3d.Manifold, d: dict, layer: float, line: float, bridge: float, margin: float = 0.5) -> dict:
    """The state of the defect ``d`` (a layercheck record) in ``man``."""
    z = d["z"]
    (x0, x1), (y0, y1) = d["x"], d["y"]
    at = square(x0 - margin, y0 - margin, x1 + margin, y1 + margin)
    # slice a crop wide enough for the bridge test's anchors
    w = bridge + 1.0
    crop = man ^ box(x0 - w, y0 - w, x1 + w, y1 + w, z - 1.5 * layer, z + 0.5 * layer)
    cur, prev = crop.slice(float(z)), crop.slice(float(z - layer))
    # material in the defect's own box (the margin would take in the wall
    # an overhang stands out from)
    here = cur ^ square(x0, y0, x1, y1)
    if here.area() < 1e-6:
        return {"state": "absent", "area_mm2": 0.0}
    # pieces touching nothing below
    touched = prev.offset(1e-3, m3d.JoinType.Miter)
    born = [c for c in cur.decompose() if (c ^ touched).area() < 1e-9 and not (c ^ at).is_empty()]
    if born:
        return {"state": "born", "area_mm2": round(sum((c ^ at).area() for c in born), 4)}
    allow = 0.5 * line
    beyond = cur - prev.offset(allow, m3d.JoinType.Round, circular_segments=32)
    hang = 0.0
    for patch in beyond.decompose():
        part = (patch ^ at).area()
        if part < 1e-6:
            continue
        ok, _deg, _span, share = lc.bridge_test(patch, cur, prev, bridge)
        hang += part * (1.0 - share) if ok else part
    if hang < 0.0025:
        return {"state": "held", "area_mm2": round(here.area(), 4)}
    return {"state": "hangs", "area_mm2": round(hang, 4)}


def origin(states: list) -> str | None:
    """The first stage from which the defect hangs (or is born) in every
    stage after it, up to the end."""
    first = None
    for name, s in states:
        bad = s["state"] in ("hangs", "born")
        if bad and first is None:
            first = name
        elif not bad:
            first = None
    return first


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stages")
    ap.add_argument("check")
    ap.add_argument("--model", help="the finished model, traced as the last stage")
    ap.add_argument("--layer", type=float, default=0.2)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--line", type=float, default=0.4)
    ap.add_argument("--bridge", type=float, default=10.0)
    ap.add_argument("--json")
    a = ap.parse_args()
    folder = Path(a.stages)
    rep = json.loads(Path(a.check).read_text())
    res = next((r for r in rep["results"] if abs(r["layer_mm"] - a.layer) < 1e-6), None)
    if res is None:
        sys.exit(f"{a.check}: no result for {a.layer} mm layers")
    defects = [{"kind": "unsupported", **d} for d in res["unsupported"]["largest"][:a.top]]
    defects += [{"kind": "birth", **d} for d in res["births"]["largest"][:max(1, a.top // 2)]]
    paths = sorted(folder.glob("[0-9]_*.stl"))
    if not paths:
        sys.exit(f"{folder}: no stage files (build with HK3D_DEBUG_STAGES={folder})")
    stages = {p.stem: lc.load(str(p)) for p in paths}
    ground = stages.get("0_ground")
    order = [(n, (m + ground) if n in LANDMARK and ground is not None else m) for n, m in stages.items()]
    if a.model:
        order.append(("final", lc.load(a.model)))
    print("per stage: . no material there  o held  X hangs  B printed on air")
    out = []
    for d in defects:
        states = [(n, look(m, d, a.layer, a.line, a.bridge)) for n, m in order]
        where = "ground" if ground is not None and look(ground, d, a.layer, a.line, a.bridge)["state"] != "absent" else "landmark"
        rec = {**{k: d[k] for k in ("kind", "z", "x", "y", "area_mm2")}, "in": where,
               "from": origin(states), "stages": dict(states)}
        out.append(rec)
        line = "  ".join(f"{n.split('_', 1)[-1] if n != 'final' else n}:{MARK[s['state']]}" for n, s in states)
        print(f"{d['kind']:11s} z {d['z']:6.2f}  x {d['x']}  y {d['y']}  {d['area_mm2']:.3f} mm2  "
              f"[{where}]  from {rec['from']}\n    {line}")
    if a.json:
        Path(a.json).write_text(json.dumps({"stages": [n for n, _ in order], "defects": out}, indent=1))


if __name__ == "__main__":
    main()
