"""Would this model print without supports?  printcheck.py <model.stl> [layer mm] [bridge mm] [cell mm] [--base MM]

Slices the model at the layer height and reports how much material a
support-free print would still be missing: the volume printable.support_fill
would add (0 = every layer stands on the one below, within 45 degrees or as a
bridge no longer than the given span).

This re-runs the generator's own support rule with other ray offsets, so it
shares that rule's blind spots; tools/layercheck.py is the independent
layer-by-layer check.  The base height is read from the build's result.json
(``result.json`` or ``<name>_result.json`` beside the model) unless given
with --base; without either it stops rather than guess."""
import json, sys, time
from pathlib import Path

import manifold3d as m3d
import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hk3d import printable  # noqa: E402

args = sys.argv[1:]
zb = None
if "--base" in args:
    i = args.index("--base")
    zb = float(args[i + 1])
    del args[i:i + 2]
path = args[0]
layer = float(args[1]) if len(args) > 1 else 0.2
bridge = float(args[2]) if len(args) > 2 else 10.0
cell = float(args[3]) if len(args) > 3 else 0.1
if zb is None:
    for res in (Path(path).with_name("result.json"), Path(path).with_name(Path(path).stem + "_result.json")):
        if res.exists():
            zb = float(json.loads(res.read_text())["params"]["base_mm"])
            break
    else:
        sys.exit(f"{Path(path).name}: no result.json beside it - give the base height with --base")
m = trimesh.load(path, process=True)
m.merge_vertices()
man = m3d.Manifold(m3d.Mesh(np.asarray(m.vertices, np.float32), np.asarray(m.faces, np.uint32)))
if man.status() != m3d.Error.NoError:
    sys.exit(f"not a closed manifold: {man.status()}")
t = time.time()
# rays offset differently from the ones the build used, so they do not
# run through the vertices of the material it added
_, st = printable.support_fill(man, None, zb, cell, layer, bridge, jitter=0.4371)
print(f"{Path(path).name}: unsupported {st['support_mm3']:.2f} mm3, bridged {st['bridged_mm2']:.1f} mm2 "
      f"(base {zb} mm, layer {layer} mm, bridge {bridge} mm, {time.time() - t:.0f} s)")
