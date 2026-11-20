"""Command line: ``hk3d serve | index | build | sheets | download``."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _common(p):
    p.add_argument("--data", default=os.environ.get("HK3D_DATA", "."),
                   help="folder containing LandsD sheet folders (e.g. 11-NE-18B)")
    p.add_argument("--cache", default=os.environ.get("HK3D_CACHE", "cache"), help="cache / output folder")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="hk3d", description="LandsD 3D Digital Map -> 3D-printable miniatures")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="run the web app")
    _common(s)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)

    s = sub.add_parser("index", help="scan local data and print sheet summary")
    _common(s)

    s = sub.add_parser("build", help="build a model without the web UI")
    _common(s)
    loc = s.add_mutually_exclusive_group(required=True)
    loc.add_argument("--latlon", nargs=2, type=float, metavar=("LAT", "LON"))
    loc.add_argument("--hk80", nargs=2, type=float, metavar=("E", "N"))
    s.add_argument("--radius", type=float, default=300, help="real-world radius in metres")
    s.add_argument("--diameter", type=float, default=100, help="printed diameter in mm")
    s.add_argument("--base", type=float, default=3, help="base thickness in mm")
    s.add_argument("--zscale", type=float, default=1, help="vertical exaggeration")
    s.add_argument("--detail", type=float, default=0.1, help="heightmap cell size on the print (mm)")
    s.add_argument("--min-feature", type=float, default=None,
                   help="remove ground noise thinner than this (mm); default 0.75 x nozzle")
    s.add_argument("--nozzle", type=float, default=0.4,
                   help="printer nozzle (mm); sets --spire, --min-feature, --line-width and --layer-height defaults (default 0.4)")
    s.add_argument("--shape", choices=["circle", "hexagon", "square"], default="circle")
    s.add_argument("--layers", default=None, help="comma list: terrain,ground,infrastructure,water,vegetation,building")
    s.add_argument("--multicolor", action="store_true", help="split into coloured parts (3MF + per-part STL)")
    s.add_argument("--name", default="", help="label for files and the engraving")
    s.add_argument("--no-engrave", action="store_true", help="do not engrave name/coordinates underneath")
    s.add_argument("--openings", type=float, default=1.0,
                   help="keep see-through openings at a building's foot (arcades, plazas under raised floors) "
                        "and the space under bridges at least this high and wide (mm); 0 = fill them (default 1.0)")
    s.add_argument("--supports", action="store_true",
                   help="you will print with supports: do not add 45-degree material under overhangs "
                        "(default: the model prints without supports)")
    s.add_argument("--bridge", type=float, default=10.0,
                   help="longest span your printer bridges (mm): ceilings and decks up to this span stay open "
                        "without supports (default 10)")
    s.add_argument("--bed", default="256x256x256",
                   help="printer build volume XxYxZ (mm): a bigger model is cut into pieces that fit, "
                        "with pin holes and names to put them together (default 256x256x256)")
    s.add_argument("--no-tile", action="store_true", help="never cut the model into pieces")
    s.add_argument("--pin", type=float, default=1.75,
                   help="diameter of the pins aligning the pieces (mm; default 1.75 = filament); 0 = no pin holes")
    s.add_argument("--line-width", type=float, default=None,
                   help="printer line width (mm): facade ribs are at least this wide; default = nozzle")
    s.add_argument("--layer-height", type=float, default=None,
                   help="printer layer height (mm): floor lines are at least two layers tall; default nozzle / 2")
    s.add_argument("--first-layer", type=float, default=None,
                   help="the slicer's first layer height (mm); default = layer height. Sets where the layers fall")
    s.add_argument("--spire", type=float, default=None,
                   help="thicken masts/spires on buildings to at least this width (mm); 0 = off; "
                        "default 1.5 x nozzle")
    s.add_argument("--precise", action="store_true",
                   help="carve the centre building from its facade directions (flat facades, see-through gaps)")
    s.add_argument("--precise-all", action=argparse.BooleanOptionalAction, default=None,
                   help="with --precise: carve every building in the model, not only the centre one "
                        "(default: on with --landmark, else off; --no-precise-all for the centre only)")
    s.add_argument("--relief", type=float, default=None,
                   help="with --precise: recess facade panels between the bracing/frames seen in the "
                        "building texture by this much (mm); needs textured FBX; default 0.3 with --landmark, else 0")
    s.add_argument("--trim-edge", action="store_true",
                   help="drop buildings cut by the outline, keeping the one at the centre")
    s.add_argument("--road-recess", type=float, default=1.0,
                   help="sink roads this many printed layers below the pavement beside them so the street "
                        "network shows on a one-colour print; 0 = true heights (default 1)")
    s.add_argument("--ground-clean", type=float, default=2.5,
                   help="remove cars/bushes/noise smaller than this (m) from the ground; 0 = off")
    s.add_argument("--holder", action="store_true", help="also output a display ring the model drops into")
    s.add_argument("--landmark", action="store_true",
                   help="preset for a single-landmark piece: R 100 m, 80 mm, 0.05 mm detail, 5 mm base, holder, trimmed edge, every building carved precisely with facade relief, trees")
    s.add_argument("--out", default=None, help="output folder (default: cache/jobs/<hash>)")

    s = sub.add_parser("sheets", help="list 1:1000 sheets needed for an area")
    s.add_argument("--latlon", nargs=2, type=float, required=True, metavar=("LAT", "LON"))
    s.add_argument("--radius", type=float, default=300)

    s = sub.add_parser("landmarks", help="list the tallest local buildings near a point")
    _common(s)
    s.add_argument("--latlon", nargs=2, type=float, required=True, metavar=("LAT", "LON"))
    s.add_argument("--radius", type=float, default=600)
    s.add_argument("--limit", type=int, default=10)

    s = sub.add_parser("download", help="download sheets from the LandsD Open3Dhk API")
    _common(s)
    s.add_argument("sheets", nargs="+")
    s.add_argument("--format", choices=["FBX0", "FBX"], default="FBX0",
                   help="FBX0 = untextured (~50 MB/sheet, enough for printing)")

    a = ap.parse_args(argv)

    if a.cmd == "serve":
        import uvicorn

        from .server import create_app
        print(f"Data: {Path(a.data).resolve()}\nOpen http://{a.host}:{a.port}")
        uvicorn.run(create_app(a.data, a.cache), host=a.host, port=a.port, log_level="warning")
        return

    if a.cmd == "sheets":
        from .coords import wgs84_to_hk80
        from .sheets import sheets_in_bbox
        e, n = wgs84_to_hk80(*a.latlon)
        print(f"HK1980 E {e:.1f} N {n:.1f}")
        print("\n".join(sheets_in_bbox(e - a.radius, n - a.radius, e + a.radius, n + a.radius)))
        return

    if a.cmd == "download":
        from .download import download_sheet
        for sh in a.sheets:
            def prog(done, total, sh=sh):
                print(f"\r{sh}: {done / 1e6:.0f}{f'/{total / 1e6:.0f}' if total else ''} MB", end="", flush=True)
            path = download_sheet(sh, Path(a.data), a.format, progress=prog)
            print(f"\n -> {path}")
        return

    from .catalog import load_or_scan

    def scan_prog(done, total, _p):
        if done % 50 == 0 or done == total:
            print(f"\rIndexing {done}/{total}", end="", file=sys.stderr, flush=True)

    cat = load_or_scan(a.data, a.cache, scan_prog)
    if a.cmd == "index":
        print(f"\n{len(cat.entries)} FBX files, {len(cat.errors)} errors")
        for name, s in sorted(cat.sheets().items()):
            print(f"{name:12s} files={s['files']:4d} tris={s['ntri']:>10,d} {s['groups']}")
        return

    if a.cmd == "landmarks":
        from .coords import hk80_to_wgs84, wgs84_to_hk80
        e, n = wgs84_to_hk80(*a.latlon)
        print(f"\n{'top mPD':>8} {'height':>7} {'size m':>11} {'dist':>6}  {'HK1980 E/N':<20} lat, lon")
        for b in cat.landmarks(e, n, a.radius, a.limit):
            la, lo = hk80_to_wgs84(b["e"], b["n"])
            print(f"{b['top_m']:8.1f} {b['height_m']:7.1f} {b['width_m']:5.0f}x{b['depth_m']:<5.0f} {b['dist_m']:6.0f}  "
                  f"{b['e']:.1f} {b['n']:.1f}   {la:.6f}, {lo:.6f}")
        return

    from .coords import wgs84_to_hk80
    from .pipeline import ModelParams, build, input_signature
    e, n = a.hk80 if a.hk80 else wgs84_to_hk80(*a.latlon)
    if a.spire is None:
        a.spire = round(1.5 * a.nozzle, 2)
    if a.min_feature is None:
        a.min_feature = round(0.75 * a.nozzle, 2)
    if a.line_width is None:
        a.line_width = a.nozzle
    if a.layer_height is None:
        a.layer_height = round(a.nozzle / 2, 3)
    if a.landmark:
        defaults = {"radius": 300, "diameter": 100, "detail": 0.1, "base": 3}
        preset = {"radius": 100, "diameter": 80, "detail": 0.05, "base": 5}
        for k, v in preset.items():  # only override values the user left at default
            if getattr(a, k) == defaults[k]:
                setattr(a, k, v)
        a.holder = True
        a.trim_edge = True
        a.precise = True
        if a.precise_all is None:
            a.precise_all = True
        if a.relief is None:
            a.relief = 0.3
        if not a.layers:
            a.layers = "terrain,ground,infrastructure,bridge,water,vegetation,building"
    kw = dict(e=e, n=n, radius_m=a.radius, diameter_mm=a.diameter, base_mm=a.base, z_scale=a.zscale,
              detail_mm=a.detail, min_feature_mm=a.min_feature, shape=a.shape, multicolor=a.multicolor, name=a.name,
              engrave=not a.no_engrave, ground_clean_m=a.ground_clean, holder=a.holder,
              spire_mm=a.spire, openings_mm=a.openings, line_mm=a.line_width, layer_mm=a.layer_height, first_layer_mm=a.first_layer,
              support_free=not a.supports, bridge_mm=a.bridge, trim_edge=a.trim_edge, precise_landmark=a.precise,
              precise_all=bool(a.precise_all) and a.precise, facade_relief_mm=a.relief or 0.0,
              road_recess_layers=a.road_recess,
              tile=not a.no_tile, bed_mm=[float(v) for v in a.bed.lower().split("x")], pin_mm=a.pin)
    if a.layers:
        kw["groups"] = [g.strip() for g in a.layers.split(",") if g.strip()]
    params = ModelParams(**kw)
    out = Path(a.out) if a.out else Path(a.cache) / "jobs" / params.key(input_signature(cat, params))

    def prog(f, msg):
        print(f"\r[{f * 100:5.1f}%] {msg:60.60s}", end="", file=sys.stderr, flush=True)

    res = build(params, cat, out, Path(a.cache), prog)
    print(file=sys.stderr)
    print(json.dumps({"out": str(out.resolve()), **res["stats"], "files": res["files"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
