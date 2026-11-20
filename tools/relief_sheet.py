"""Contact sheet of one building's relief debug images (from relief_probe.py).

usage: relief_sheet.py PROBE_DIR/BUILDING OUT.png [HEIGHT]
Top row: texture as the side rays see it, one column per facade view.
Bottom row: grey = facade, red = pattern found but rejected, amber = ribs
found on a surface that got no cut, black = kept
proud (structure, floor lines), blue = recessed panels.
"""
import sys
from pathlib import Path
from PIL import Image
d = Path(sys.argv[1]); out = sys.argv[2]; H = int(sys.argv[3]) if len(sys.argv) > 3 else 700
cols = []
for v in sorted(d.glob("view*_rgb.png"), key=lambda p: int(p.stem[4:-4])):
    a = Image.open(v); b = Image.open(str(v).replace("_rgb", "_pat"))
    # crop to content (non-white)
    import numpy as np
    m = (np.asarray(b) != 255).any(-1)
    if not m.any(): continue
    ys, xs = np.nonzero(m); box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    a, b = a.crop(box), b.crop(box)
    s = H / a.height
    a = a.resize((max(1, int(a.width * s)), H)); b = b.resize(a.size, Image.NEAREST)
    pair = Image.new("RGB", (a.width, 2 * H + 6), "white"); pair.paste(a, (0, 0)); pair.paste(b, (0, H + 6))
    cols.append(pair)
W = sum(c.width for c in cols) + 8 * len(cols)
sh = Image.new("RGB", (W, 2 * H + 6), "white"); x = 0
for c in cols: sh.paste(c, (x, 0)); x += c.width + 8
sh.save(out)
print(out, sh.size)
