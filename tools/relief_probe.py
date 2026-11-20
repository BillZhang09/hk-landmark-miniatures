"""Run the facade-relief texture step on single buildings and save debug images.

usage: relief_probe.py APP DATA OUT name=sheet/BUILDING/ID ...
Parameters match the landmark preset (0.4 mm/m, 0.4 mm line, 0.2 mm layer, 0.3 mm relief).

Besides the debug images it writes ``results.json`` (the numbers per
building, each with its full ``sheet/BUILDING/ID`` path) and
``manifest.json``: the code version (commit and a hash of the source that ran), the parameters and a SHA-256 of every
FBX and texture read, so a run can be repeated and compared later.
"""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

APP, DATA, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
sys.path.insert(0, str(APP))
from hk3d import carve, fbx  # noqa: E402

S = 0.4                       # mm per m
width, height, depth = 0.4 / S, 2 * 0.2 / S, 0.3 / S


def sha256(path) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()
    except OSError:
        return None


def rel(path) -> str:
    try:
        return str(Path(path).resolve().relative_to(DATA.resolve()))
    except ValueError:
        return str(path)


def git(*args) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(APP), *args], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


from hk3d import pipeline  # noqa: E402

OUT.mkdir(parents=True, exist_ok=True)
manifest = {
    "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "commit": git("rev-parse", "HEAD"),
    "uncommitted_changes": bool(git("status", "--porcelain", "--", "hk3d", "tools")),
    # the code that actually ran, committed or not: one hash over every source file
    "source_sha256": hashlib.sha256(b"".join(
        str(f.relative_to(APP.resolve())).encode() + b"\0" + f.read_bytes()
        for f in sorted([*(APP.resolve() / "hk3d").rglob("*.py"), Path(__file__).resolve()]))).hexdigest(),
    "pipeline_version": pipeline.PIPELINE_VERSION,
    "params": {"mm_per_m": S, "width_m": width, "height_m": height, "depth_m": depth, "chamfer": 1.0},
    "buildings": {},
}
res = {}
for arg in sys.argv[4:]:
    name, p = arg.split("=", 1)
    fp = DATA / p / (p.split("/")[-1] + ".fbx")
    t0 = time.time()
    entry = manifest["buildings"][name] = {"building": p, "fbx": rel(fp), "fbx_sha256": sha256(fp)}
    tx = fbx.load_textured(str(fp))
    if tx is None:
        res[name] = {"building": p, "note": "no UVs / load_textured None"}
        print(name, res[name], flush=True)
        continue
    V = tx["V"].astype(np.float64).copy()
    c = V[:, :2].mean(0)
    V[:, :2] -= c
    F = tx["F"]
    entry["textures"] = {rel(f): sha256(f) for f in tx["files"]}
    axes = carve.facade_axes(V, F)
    sampler = carve.TextureSampler(tx["uv"], tx["tex"], tx["files"])
    tex_frac = float((tx["tex"] >= 0).mean())
    sizes = [sampler.images[i].shape[:2] if sampler.images[i] is not None else None for i in range(len(tx["files"]))]
    try:
        cut, st = carve.relief_cutters(V, F, axes, sampler, depth, width, height=height,
                                       chamfer=1.0, debug=str(OUT / name))
    except Exception as exc:
        cut, st = [], {"error": repr(exc)}
    st.update(building=p, axes_deg=[round(float(np.degrees(a)), 1) for a in axes], ntex=len(tx["files"]),
              tex_frac=round(tex_frac, 3), tex_sizes=[list(s) if s else None for s in sizes][:6],
              zrange=[round(float(V[:, 2].min()), 1), round(float(V[:, 2].max()), 1)],
              cutters=len(cut), seconds=round(time.time() - t0, 1))
    res[name] = st
    print(name, json.dumps(st, ensure_ascii=False), flush=True)
(OUT / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
(OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
