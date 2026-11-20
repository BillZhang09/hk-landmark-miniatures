"""Per-building figures from relief_probe.py debug images.

usage: relief_metrics.py PROBE_DIR [CELL]
kept_m2      structure (or floor line) area kept proud, all views
flat_frac    share of it standing on texture with no variation at all (LandsD's
             fill colour where there is no photo): should be 0
blob_m2      kept area in pieces that are neither thin bands (<= 6 m wide) nor
             whole-surface strips: big irregular areas, typically reflections
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

d = Path(sys.argv[1])
cell = float(sys.argv[2]) if len(sys.argv) > 2 else 0.125
res = json.loads((d / "results.json").read_text()) if (d / "results.json").exists() else {}
print(f"{'building':12} {'panels':>6} {'cut m2':>8} {'floors':>6} {'kept m2':>8} {'flat%':>6} {'blob m2':>8}")
out = {}
for b in sorted(p for p in d.iterdir() if p.is_dir()):
    kept = flat = blob = 0.0
    for v in b.glob("view*_pat.png"):
        pat = np.asarray(Image.open(v)).astype(int)
        rgb = np.asarray(Image.open(str(v).replace("_pat", "_rgb"))).astype(float)
        k = (pat == 30).all(-1)
        if not k.any():
            continue
        L = rgb.mean(-1)
        sd = np.sqrt(np.maximum(ndimage.uniform_filter(L * L, 5) - ndimage.uniform_filter(L, 5) ** 2, 0))
        fl = (np.ptp(rgb, -1) <= 2) & (sd < 0.5) & (L < 250)
        kept += k.sum()
        flat += (k & fl).sum()
        r = max(2, int(round(3.0 / cell)))      # wider than 6 m
        core = ndimage.distance_transform_edt(k) >= r
        lab, n = ndimage.label(k, np.ones((3, 3), bool))
        if n:
            fat = np.unique(lab[core])
            blob += np.isin(lab, fat[fat > 0]).sum() if core.any() else 0
    st = res.get(b.name, {})
    row = {"panels": st.get("relief_panels"), "cut_m2": st.get("relief_area_m2"), "floors": st.get("floor_lines"),
           "kept_m2": round(kept * cell * cell), "flat_pct": round(100 * flat / max(kept, 1), 1),
           "blob_m2": round(blob * cell * cell)}
    out[b.name] = row
    print(f"{b.name:12} {str(row['panels']):>6} {str(row['cut_m2']):>8} {str(row['floors']):>6} "
          f"{row['kept_m2']:>8} {row['flat_pct']:>6} {row['blob_m2']:>8}")
(d / "metrics.json").write_text(json.dumps(out, indent=1))
