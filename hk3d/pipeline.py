"""End-to-end: location + print parameters -> printable model files."""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Callable

import manifold3d as m3d
import numpy as np
from scipy import ndimage

from . import printable, solid, tiles
from .catalog import Catalog, file_version
from .coords import hk80_to_wgs84
from .layers import BASE_COLOR, GROUP_ORDER, GROUPS, PART_MODE, SURFACE_GROUPS
from .raster import make_grid, raster_groups

MIN_CELL_M = 0.1      # finer than this is below the source data resolution
MAX_CELLS = 3000      # per side; keeps memory/time bounded
TILE_MARGIN = 5.0     # mm kept clear round a piece on the bed
PIN_CLEARANCE = 0.3   # pin hole = pin diameter + this


# Bump whenever the same parameters would give a different model, so results
# cached by the web app under the old key are not served again.
PIPELINE_VERSION = 24


@dataclass
class ModelParams:
    e: float                      # HK1980 easting of the centre
    n: float                      # HK1980 northing of the centre
    radius_m: float = 300.0       # real-world radius of the area
    diameter_mm: float = 100.0    # printed diameter (or side for square)
    base_mm: float = 3.0          # solid base under the lowest point
    z_scale: float = 1.0          # vertical exaggeration
    detail_mm: float = 0.1        # heightmap cell size on the print
    groups: list = field(default_factory=lambda: [g for g in GROUP_ORDER if GROUPS[g]["default"]])
    min_feature_mm: float = 0.3   # remove spikes thinner than this on ground/trees (0 = off)
    ground_clean_m: float = 2.5   # remove cars/bushes/noise smaller than this from the ground (0 = off)
    spire_mm: float = 0.6         # thicken masts/spires on buildings to at least this width (0 = off)
    line_mm: float = 0.4          # printer line width: the narrowest rib left standing on a facade
    layer_mm: float = 0.2         # printer layer height: horizontal ribs are at least two layers tall
    first_layer_mm: float | None = None   # the slicer's first layer (default: layer_mm); sets where layers fall
    trim_edge: bool = False       # drop tall buildings cut by the outline (except the one at the centre)
    precise_landmark: bool = False  # carve the centre building from several directions (flat facades)
    precise_all: bool = False       # with precise_landmark: carve every building, not only the centre one
    openings_mm: float = 1.0        # keep see-through openings at a building's foot (arcades, plazas under
                                    # raised floors) and under bridges at least this high and wide (0 = fill)
    support_free: bool = True       # add 45 degree material under overhangs a printer cannot print in the air
    bridge_mm: float = 10.0         # support_free: longest span the printer bridges (open under a ceiling)
    facade_relief_mm: float = 0.0   # precise landmark: recess facade panels between the bracing/frames
                                    # seen in its texture by this much (0 = off)
    road_recess_layers: float = 1.0  # sink roads this many printed layers below the pavement beside them,
                                     # so the street network reads on a one-colour print (0 = true heights)
    shape: str = "circle"         # circle | hexagon | square
    multicolor: bool = False      # split into separately-coloured bodies
    skin_mm: float = 0.6          # thickness of road/water colour layers
    floor_m: float | None = None  # elevation mapped to the base top (default: lowest point)
    simplify_mm: float = 0.02     # mesh simplification tolerance
    name: str = ""                # optional label for files / engraving
    engrave: bool = True          # engrave name, coordinates and scale underneath
    engrave_depth_mm: float = 0.6
    holder: bool = False          # also output a display ring the model drops into
    holder_wall_mm: float = 5.0
    holder_clearance_mm: float = 0.3
    tile: bool = True             # cut a model bigger than the printer into pieces that fit (tiles.py)
    bed_mm: list = field(default_factory=lambda: [256.0, 256.0, 256.0])  # printer build volume x, y, z
    pin_mm: float = 1.75          # tile: diameter of the pins aligning the pieces (0 = no pin holes)

    def validate(self) -> None:
        if not 10 <= self.radius_m <= 3000:
            raise ValueError("radius_m must be within 10-3000 m")
        if not 10 <= self.diameter_mm <= 2000:
            raise ValueError("diameter_mm must be within 10-2000 mm")
        if not 0.5 <= self.base_mm <= 50:
            raise ValueError("base_mm must be within 0.5-50 mm")
        if not 0.1 <= self.z_scale <= 10:
            raise ValueError("z_scale must be within 0.1-10")
        if not 0.02 <= self.detail_mm <= 1:
            raise ValueError("detail_mm must be within 0.02-1 mm")
        if self.shape not in ("circle", "hexagon", "square"):
            raise ValueError("shape must be circle, hexagon or square")
        if not 0 <= self.road_recess_layers <= 10:
            raise ValueError("road_recess_layers must be within 0-10")
        bad = [g for g in self.groups if g not in GROUPS]
        if bad:
            raise ValueError(f"unknown groups {bad}")
        if not self.groups:
            raise ValueError("select at least one layer")
        if not 0 <= self.spire_mm <= 3:
            raise ValueError("spire_mm must be within 0-3 mm")
        if not 0 <= self.facade_relief_mm <= 1:
            raise ValueError("facade_relief_mm must be within 0-1 mm")
        if not 0 <= self.openings_mm <= 20:
            raise ValueError("openings_mm must be within 0-20 mm")
        if not 0.1 <= self.line_mm <= 1.5:
            raise ValueError("line_mm must be within 0.1-1.5 mm")
        if not 0.02 <= self.layer_mm <= 0.8:
            raise ValueError("layer_mm must be within 0.02-0.8 mm")
        if self.first_layer_mm is not None and not 0.02 <= self.first_layer_mm <= 1.0:
            raise ValueError("first_layer_mm must be within 0.02-1 mm")
        if not 0 <= self.bridge_mm <= 50:
            raise ValueError("bridge_mm must be within 0-50 mm")
        if len(self.bed_mm) != 3 or not all(50 <= b <= 2000 for b in self.bed_mm):
            raise ValueError("bed_mm must be three sizes within 50-2000 mm")
        if not 0 <= self.pin_mm <= 5:
            raise ValueError("pin_mm must be within 0-5 mm")

    def tiles_across(self) -> tuple[int, int]:
        """Pieces across x and y the outline needs on the printer's bed."""
        if not self.tile:
            return 1, 1
        b = solid.shape_prism(self.shape, self.diameter_mm, 1.0).bounding_box()
        return tiles.counts((b[3] - b[0], b[4] - b[1]), tuple(self.bed_mm[:2]), TILE_MARGIN)

    @property
    def mm_per_m(self) -> float:
        return self.diameter_mm / (2 * self.radius_m)

    def rib_width_m(self) -> float:
        """Real width of the narrowest vertical or slanted facade rib: one line."""
        return self.line_mm / self.mm_per_m

    def print_base(self) -> float:
        """The layer boundary a slicer puts nearest the top of the base:
        the printability passes slice the model where the printer's layers
        fall (first layer, then ``layer_mm`` each), not from the base up."""
        first = self.first_layer_mm or self.layer_mm
        return round(first + round((self.base_mm - first) / self.layer_mm) * self.layer_mm, 6)

    def rib_height_m(self) -> float:
        """Real height of the narrowest horizontal facade rib: two layers."""
        return 2 * self.layer_mm / (self.mm_per_m * self.z_scale)

    def cell_m(self) -> float:
        cell = max(self.detail_mm / self.mm_per_m, MIN_CELL_M)
        return max(cell, 2 * self.radius_m / (MAX_CELLS - 6))

    def key(self, inputs: str = "") -> str:
        """Cache key of the model: the parameters, the pipeline version and
        ``inputs`` (see :func:`input_signature`), so a model is built again
        when the data it is made of changes."""
        d = asdict(self)
        d["e"], d["n"] = round(self.e, 2), round(self.n, 2)
        d["groups"] = sorted(self.groups)
        d["pipeline"] = PIPELINE_VERSION
        if inputs:
            d["inputs"] = inputs
        return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]


Progress = Callable[[float, str], None]

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".tga"}


def input_signature(catalog: Catalog, params: ModelParams) -> str:
    """Fingerprint of the data a model of ``params`` is built from: every
    catalogued file in the area (path, size, modification time), the sheets
    that cover it (they decide what is sea), and - when the facade relief
    reads building textures - the texture images beside those files.  A
    download, a replaced file or a rescan changes it."""
    r = params.radius_m * 1.05 + 10.0
    x0, y0, x1, y1 = params.e - r, params.n - r, params.e + r, params.n + r
    textures = params.precise_landmark and params.facade_relief_mm > 0
    items = []
    for e in catalog.query(x0, y0, x1, y1):
        items.append(["f", e.path, e.version])
        if textures and e.group == "building":
            try:
                for f in (catalog.root / e.path).parent.iterdir():
                    if f.suffix.lower() in _IMAGE_SUFFIXES:
                        st = f.stat()
                        items.append(["t", e.path, f.name, file_version(st.st_size, st.st_mtime)])
            except OSError:
                items.append(["t", e.path, "unreadable"])
    for name, sh in catalog.sheets().items():
        b = sh.get("bounds")
        if b is not None and b[0] <= x1 and b[2] >= x0 and b[1] <= y1 and b[3] >= y0:
            items.append(["s", name])
    return hashlib.sha1(json.dumps(sorted(items)).encode()).hexdigest()[:16]

CLASS_IDS = {"base": 0, "infrastructure": 1, "water": 2, "vegetation": 3, "building": 4}
CLASS_OF_GROUP = {"terrain": 0, "ground": 0, "infrastructure": 1, "bridge": 1, "water": 2, "vegetation": 3,
                  "building": 4}
CLASS_COLORS = {0: BASE_COLOR, 1: GROUPS["infrastructure"]["color"], 2: GROUPS["water"]["color"],
                3: GROUPS["vegetation"]["color"], 4: GROUPS["building"]["color"]}
CLASS_NAMES = {v: k for k, v in CLASS_IDS.items()}
HOLDER_COLOR = "#5b3f2e"


def _shape_mask(shape: str, X: np.ndarray, Y: np.ndarray, r: float) -> np.ndarray:
    if shape == "circle":
        return X ** 2 + Y ** 2 <= r ** 2
    if shape == "square":
        return (np.abs(X) <= r) & (np.abs(Y) <= r)
    # hexagon with flat top/bottom edges (vertices at 0deg, 60deg, ...)
    ax, ay = np.abs(X), np.abs(Y)
    return (ay <= r * np.sqrt(3) / 2) & (np.sqrt(3) * ax + ay <= np.sqrt(3) * r)


def _disk(k: int) -> np.ndarray:
    r = (k - 1) / 2
    yy, xx = np.mgrid[:k, :k]
    return (xx - r) ** 2 + (yy - r) ** 2 <= r ** 2 + 0.5


def build(params: ModelParams, catalog: Catalog, out_dir: Path, cache_dir: Path | None = None,
          progress: Progress | None = None) -> dict:
    params.validate()
    base_asked = params.base_mm
    tiled_xy = params.tiles_across() != (1, 1)
    if tiled_xy and params.pin_mm > 0:
        # the pins between pieces run through the base
        params = replace(params, base_mm=max(params.base_mm, round(tiles.min_base(params.pin_mm + PIN_CLEARANCE), 1)))
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)

    def report(f, msg):
        if progress:
            progress(min(max(f, 0.0), 1.0), msg)

    s = params.mm_per_m
    cell = params.cell_m()
    grid = make_grid(params.e, params.n, params.radius_m, cell)
    cov = catalog.coverage_fraction(params.e - params.radius_m, params.n - params.radius_m,
                                    params.e + params.radius_m, params.n + params.radius_m)
    if cov == 0:
        raise ValueError("No local 3D map data covers this location. Download the map sheets first.")

    # ---------------------------------------------------------- 1. rasterise
    report(0.0, f"Rasterising {grid.n}x{grid.n} cells at {cell:.2f} m")
    order = [g for g in GROUP_ORDER if g in params.groups]
    owners: dict = {}
    H = raster_groups(catalog, grid, order, cache_dir,
                      progress=lambda f, m: report(0.02 + 0.5 * f, m),
                      owners=owners if (params.trim_edge or params.precise_landmark) else None)

    xs, ys = grid.local_coords()
    X, Y = np.meshgrid(xs, ys)
    inside = _shape_mask(params.shape, X, Y, params.radius_m)
    trimmed = 0
    cleared = np.zeros(inside.shape, bool)   # where something cut by the outline was taken away
    if params.trim_edge and "building" in owners:
        before = np.isfinite(H["building"])
        H["building"], trimmed = _trim_edge(H["building"], *owners["building"], inside, X, Y,
                                            heights=_entry_heights(catalog, owners["building"][1]))
        cleared |= before & ~np.isfinite(H["building"])

    # buildings carved from several directions instead of 2.5D: the one at
    # the centre, or (precise_all) every building of some size
    land_L, land_top, land_stats, land_cut = None, None, {}, []
    if params.precise_landmark and "building" in owners:
        owner, paths = owners["building"]
        groups = [_centre_ids(owner, X, Y)]
        if params.precise_all:
            seen = set(groups[0])
            vals, counts = np.unique(owner[np.isfinite(H["building"]) & inside & (owner >= 0)], return_counts=True)
            order_ = np.argsort(-counts)
            groups += [[int(vals[i])] for i in order_ if vals[i] not in seen and counts[i] * cell * cell >= 60.0]
        groups = [g for g in groups if g]
        # files cut away at the rim: no cell left with a height but some owned
        gone = set(np.unique(owner[(owner >= 0) & ~np.isfinite(H["building"])]).tolist()) - \
            set(np.unique(owner[(owner >= 0) & np.isfinite(H["building"])]).tolist())
        below = _parts_below(groups, owner, paths, catalog, grid, skip=gone)
        Ls, tops, t_carve = [], [], time.time()
        for gi, ids in enumerate(groups):
            report(0.52, f"Carving building facades {gi + 1}/{len(groups)}")
            try:
                # the panels are recessed by facade_relief_mm; the ribs left
                # standing are one printed line wide and two layers tall
                L, top_i, st, cut = _carve_landmark(
                    catalog, grid, H, owner, paths, X, Y, cell, ids=ids,
                    relief_m=params.facade_relief_mm / s,
                    relief_width_m=params.rib_width_m(), relief_height_m=params.rib_height_m(),
                    min_open_m=params.openings_mm / s,
                    chamfer=1.0 / params.z_scale if params.support_free else None)
            except Exception as exc:  # keep this building 2.5D rather than fail the build
                L, top_i, st, cut = None, None, {"error": str(exc)}, []
            if gi == 0:
                land_stats = dict(st)
            if L is None:
                if gi:
                    land_stats["failed"] = land_stats.get("failed", 0) + 1
                continue
            Ls.append(L)
            tops.append(top_i)
            land_cut += cut
            if gi:
                for k in ("relief_panels", "relief_area_m2", "relief_curved", "open_m2"):
                    if k in st:
                        land_stats[k] = round(land_stats.get(k, 0) + st[k], 1)
        # the parts of the podiums (or whatever a group stands on) that the
        # group hides from above: carved from their own meshes over the group's
        # footprint as well, as plain solids - their facades already had their
        # relief with their own group, and mixing their texture into the
        # group above would confuse its floor lines
        stacked = 0
        for gi, ids, cap in below:
            extra = np.isin(owner, groups[gi])
            try:
                L, top_i, st, _ = _carve_landmark(catalog, grid, H, owner, paths, X, Y, cell, ids=ids,
                                                  min_open_m=params.openings_mm / s, extra=extra, cap=cap)
            except Exception:  # the group above keeps standing on what it had
                L = None
            if L is not None:
                Ls.append(L)
                tops.append(top_i)
                stacked += 1
        if Ls:
            land_L = Ls[0] if len(Ls) == 1 else m3d.Manifold.batch_boolean(Ls, m3d.OpType.Add)
            with np.errstate(all="ignore"):
                land_top = np.fmax.reduce(np.stack(tops), axis=0)
            H["building"] = np.where(np.isfinite(land_top), np.nan, H["building"])
            land_stats.update(buildings=len(Ls), seconds=round(time.time() - t_carve, 1))
            if stacked:
                land_stats["stacked_groups"] = stacked
    if params.trim_edge and "ground" in H and "terrain" in H:
        before = np.isfinite(H["ground"])
        H["ground"] = _trim_edge_ground(H["ground"], H["terrain"], inside)
        cleared |= before & ~np.isfinite(H["ground"])

    report(0.53, "Cleaning ground surface")
    H = _clean_ground(H, order, cell, params.ground_clean_m, cleared=cleared)
    stack = np.stack([H[g] for g in order])
    with np.errstate(all="ignore"):
        top = np.fmax.reduce(stack, axis=0)
    covered = np.isfinite(top)
    if land_top is not None:
        # buildings carved as solids were taken out of the heightmap above,
        # but they are geometry all the same
        covered = covered | np.isfinite(land_top)
    coverage = float(covered[inside].mean())
    if coverage == 0:
        raise ValueError("No 3D geometry found inside the selected area.")
    # no geometry inside a sheet we have is sea; outside one it is unknown
    have = _sheet_mask(catalog, grid)
    with np.errstate(all="ignore"):
        ground = np.fmax.reduce(np.stack([H[g] for g in order if g in SURFACE_GROUPS])) \
            if any(g in SURFACE_GROUPS for g in order) else np.full_like(top, np.nan)
    top_f, unknown = _fill_missing(top, ground, have, cell)
    valid = inside & ~unknown
    floor = params.floor_m if params.floor_m is not None else float(np.nanmin(top_f[valid]))
    ceiling = float(np.nanmax(top_f[valid]))
    if land_top is not None and np.isfinite(land_top[inside]).any():
        # precisely carved buildings are taken out of ``top``; their tops still count
        ceiling = max(ceiling, float(np.nanmax(land_top[inside])))

    # semantic class of the visible top surface (later groups win ties)
    cls = np.zeros(top.shape, dtype=np.uint8)
    tol = max(0.15, cell * 0.5)
    for g in order:
        h = H[g]
        cls[np.isfinite(h) & (h >= top_f - tol)] = CLASS_OF_GROUP[g]

    def to_mm(h_m):
        return (params.base_mm + np.clip(h_m - floor, 0, None) * s * params.z_scale).astype(np.float32)

    Z = to_mm(top_f)
    report(0.55, "Filtering unprintable features")
    cell_mm = cell * s
    keep = ndimage.binary_dilation(cls == CLASS_OF_GROUP["building"], iterations=1)
    if "bridge" in order:
        keep |= np.isfinite(H["bridge"]) & (H["bridge"] >= top_f - tol)
    Z = _filter_unprintable(Z, cls == CLASS_OF_GROUP["building"], keep, params.min_feature_mm,
                            params.spire_mm, cell_mm)
    Z = np.maximum(Z, params.base_mm)
    road_stats = {}
    if "infrastructure" in order and params.road_recess_layers > 0:
        Z, cls, road_stats = _recess_roads(Z, cls, H, top_f, order, inside & ~unknown, tol, cell_mm,
                                           params.road_recess_layers * params.layer_mm, params.line_mm)
    zmax = float(Z[inside].max())
    # bridge decks standing free above the road keep the space under them
    Zlow, cls_low, deck_tb, deck_stats = Z, cls, None, {}
    if "bridge" in order and params.openings_mm > 0:
        report(0.56, "Opening the space under bridges")
        # carved buildings are not in the 2.5D surface, but footbridges end in them
        stand = Z if land_top is None else np.fmax(Z, to_mm(np.where(np.isfinite(land_top), land_top, floor)))
        Zlow, cls_low, deck_tb, deck_stats = _bridge_decks(catalog, grid, H, order, top_f, Z, cls, to_mm,
                                                           params, cell_mm, inside, tol, stand=stand)
    land_mm = None
    if land_L is not None:
        report(0.57, "Carved buildings: printability")
        land_mm, sup = _landmark_to_mm(land_L, land_top, to_mm, floor, s, params, xs, ys, cell_mm, land_cut,
                                       rest=Zlow)
        land_stats.update(sup)
        zmax = max(zmax, land_mm.bounding_box()[5])
    if os.environ.get("HK3D_DEBUG"):
        np.savez_compressed(out_dir / "debug_surface.npz", Z=Z, cls=cls, cell_mm=cell_mm)

    xs_mm, ys_mm = (xs * s).astype(np.float32), (ys * s).astype(np.float32)
    prism = solid.shape_prism(params.shape, params.diameter_mm, zmax + 5)

    diag = solid.quad_diagonals(Z)
    deck = None
    if deck_tb is not None:
        deck = (solid.grid_solid(xs_mm, ys_mm, deck_tb[0], diag) - solid.grid_solid(xs_mm, ys_mm, deck_tb[1], diag))
        deck = deck ^ solid.shape_prism(params.shape, params.diameter_mm, zmax + 5)
        if deck.is_empty():
            deck = None

    _solids: dict[int, object] = {}

    def hsolid(Zs):
        # the base surface B is reused by every colour part: build it once
        k = id(Zs)
        if k not in _solids:
            _solids[k] = solid.grid_solid(xs_mm, ys_mm, Zs, diag)
        return _solids[k]

    def body(Ztop, Zbot=None):
        m = solid.grid_solid(xs_mm, ys_mm, Ztop, diag)
        if Zbot is not None:
            # subtract before cutting the outline: the two height fields share
            # one triangulation, so coincident areas cancel exactly here
            m = m - hsolid(Zbot)
        m = m ^ prism
        return m.simplify(params.simplify_mm) if params.simplify_mm > 0 else m

    if land_mm is not None:
        land_mm = land_mm ^ prism

    # ----------------------------------------------------- 2. build solids
    parts: list[tuple[str, str, np.ndarray, np.ndarray]] = []
    report(0.6, "Building watertight solid")
    lat, lon = hk80_to_wgs84(params.e, params.n)
    text_cut = None
    if params.engrave and params.base_mm >= params.engrave_depth_mm + 1.0:
        from .label import engraving
        text_cut = engraving([
            params.name,
            f"{lat:.5f}°N  {lon:.5f}°E",
            f"R {params.radius_m:g} m  ·  1:{round(1000 / s):,}",
            "Map © Lands Department, HKSAR",
        ], params.diameter_mm, params.engrave_depth_mm)

    def engrave(m):
        return m - text_cut if text_cut is not None else m

    rest = body(Zlow)
    if deck is not None:
        rest = rest + deck
    plain = rest + land_mm if land_mm is not None else rest
    _stage("0_ground", rest)
    _stage("6_assembled", plain)
    hollow_stats = {}
    if land_mm is not None and params.openings_mm > 0:
        # carved parts meet with cracks and hollows between them; whatever a
        # viewer cannot see from outside is printed solid
        add, hollow_stats = printable.fill_hollows(plain, params.print_base(), max(cell_mm, 0.5 * params.line_mm),
                                                   params.layer_mm, params.openings_mm)
        if add is not None:
            land_mm = land_mm + (add ^ prism)
            plain = rest + land_mm
            if os.environ.get("HK3D_DEBUG_SUPPORT"):
                solid.write_stl(Path(os.environ["HK3D_DEBUG_SUPPORT"]) / "hollow.stl", *solid.to_arrays(add))
        _stage("7_filled", plain)
    if land_mm is not None and params.support_free:
        # the facade relief, cut after the landmark's support pass, deepens
        # the step under a ledge on a facade that already leans out (ICC's
        # skirt: 0.56-0.77 mm in one layer), and the parts meet the ground
        # and each other only now: one more pass over the whole model, which
        # also compares every layer with the one below so nothing is missed
        # for lacking a down-facing face
        t = time.time()
        add, fst = printable.support_fill(
            plain, None, params.print_base(), max(cell_mm, 0.5 * params.line_mm), params.layer_mm, params.bridge_mm,
            params.line_mm, outline=solid.shape_prism(params.shape, params.diameter_mm, 2.0).slice(0.0), scan=True)
        hollow_stats = {**hollow_stats, "final_support_mm3": fst["support_mm3"],
                        "final_support_seconds": round(time.time() - t, 1)}
        if add is not None:
            land_mm = land_mm + (add ^ prism)
            plain = rest + land_mm
        _stage("8_final_support", plain)
    # everything stands on the base; a loose fragment (a sliver the facade
    # relief cut off a thin fin) would be printed in the air: leave it out
    plain, loose, pockets = solid.standing(plain)
    loose_stats = {"loose_parts": len(loose), "loose_mm3": round(sum(p.volume() for p in loose), 2),
                   "pockets_filled": pockets, **hollow_stats}
    if loose and land_mm is not None:
        land_mm = land_mm - m3d.Manifold.batch_boolean(loose, m3d.OpType.Add)
    _stage("9_standing", plain)
    full = engrave(plain)
    Vf, Ff = solid.to_arrays(full)
    if params.multicolor:
        # under a free-standing deck the parts follow the road below it
        Z_top, cls_top = Z, cls
        Z, cls = Zlow, cls_low
        ground_groups = [g for g in order if g in SURFACE_GROUPS]
        if ground_groups:
            with np.errstate(all="ignore"):
                gm = np.fmax.reduce(np.stack([H[g] for g in ground_groups]), axis=0)
            G = np.minimum(to_mm(np.where(np.isfinite(gm), gm, np.where(unknown, top_f, 0.0))), Z)
        else:
            G = np.full_like(Z, params.base_mm)
        # label map: 0 = base, otherwise the part that owns the cell's top surface
        k = int(round(params.min_feature_mm / cell_mm))
        min_cells = max(4, k * k)
        label = np.zeros(Z.shape, dtype=np.uint8)
        present = []
        present_cls = {CLASS_OF_GROUP[x] for x in order}
        for g in PART_MODE:
            if CLASS_OF_GROUP[g] in present_cls:
                m = _clean_mask((cls == CLASS_OF_GROUP[g]) & (label == 0), label == 0, min_cells)
                if (m & inside).any():
                    present.append(g)
                    label[m] = CLASS_OF_GROUP[g]
        B = Z.copy()
        for g in present:
            m = label == CLASS_OF_GROUP[g]
            under = G if PART_MODE[g] == "column" else Z
            B[m] = np.minimum(under[m], Z[m] - params.skin_mm)
        B = np.clip(B, params.base_mm * 0.5, Z).astype(np.float32)
        n_parts = len(present) + 1
        report(0.65, "Splitting colour parts: base")
        V, F = solid.to_arrays(engrave((hsolid(B) ^ prism).simplify(params.simplify_mm)))
        parts.append(("base", BASE_COLOR, V, F))
        for i, g in enumerate(present):
            report(0.65 + 0.2 * (i + 1) / n_parts, f"Splitting colour parts: {g}")
            T = np.where(label == CLASS_OF_GROUP[g], Z, B).astype(np.float32)
            m = body(T, B)
            if m.is_empty() or m.volume() < 1e-3:
                continue
            V, F = solid.to_arrays(m)
            parts.append((g, GROUPS[g]["color"], V, F))
        if deck is not None:
            V, F = solid.to_arrays(deck)
            parts.append(("bridge", GROUPS["bridge"]["color"], V, F))
        if land_mm is not None:
            lm = land_mm - rest
            if not lm.is_empty():
                V, F = solid.to_arrays(lm)
                parts.append(("landmark", GROUPS["building"]["color"], V, F))
    else:
        parts.append(("model", BASE_COLOR, Vf, Ff))

    if params.multicolor:
        Z, cls = Z_top, cls_top

    # what the model looks like from above: carved buildings are not in Z
    Zview, cls_view = Z, cls
    if land_top is not None:
        lt = to_mm(np.where(np.isfinite(land_top), land_top, floor))
        above = np.isfinite(land_top) & (lt > Z)
        Zview, cls_view = np.where(above, lt, Z), np.where(above, CLASS_IDS["building"], cls).astype(np.uint8)

    # ----------------------------------------- 2b. pieces for a small printer
    pieces, tile_stats = [], {}
    if params.tile and (tiled_xy or zmax > params.bed_mm[2] - 1.0):
        report(0.86, "Cutting into pieces that fit the printer")
        opts = tiles.TileOptions(bed=tuple(params.bed_mm), margin=TILE_MARGIN, pin_mm=params.pin_mm,
                                 pin_clearance=PIN_CLEARANCE, engrave_depth=params.engrave_depth_mm)
        b = prism.bounding_box()
        lines = []
        if params.engrave:
            lines = [params.name, f"R {params.radius_m:g} m  ·  1:{round(1000 / s):,}",
                     "Map © Lands Department, HKSAR"]
        if params.base_mm < params.engrave_depth_mm + 1.0:
            lines = None
        pieces, tile_stats, tplan, tpins = tiles.split(
            plain, Zview, inside, xs_mm, ys_mm, (b[0], b[1], b[3], b[4]), params.base_mm, opts, lines,
            progress=lambda f, m: report(0.86 + 0.06 * f, m))
        if base_asked != params.base_mm:
            tile_stats["base_raised_mm"] = params.base_mm

    # --------------------------------------------------------- 3. export
    report(0.92, "Writing files")
    stem = _slug(params.name) or f"hk3d_{int(round(params.e))}_{int(round(params.n))}_r{int(params.radius_m)}"
    files = {}
    if pieces:
        title = params.name or stem
        zf = {f"{stem}_{p.name}.stl": solid.stl_bytes(*solid.to_arrays(tiles.on_bed(p)), p.name) for p in pieces}
        tiles.assembly_sheet(out_dir / "assembly.png", _topview_rgba(Zview, cls_view, inside, cell_mm), tplan, tpins,
                             pieces, xs_mm, ys_mm, title)
        zf["assembly.png"] = (out_dir / "assembly.png").read_bytes()
        zf["assembly.txt"] = tiles.assembly_text(pieces, tile_stats, opts, title).encode("utf-8")
        solid.write_zip(out_dir / f"{stem}_pieces.zip", zf)
        files["pieces_zip"] = f"{stem}_pieces.zip"
        files["assembly"] = "assembly.png"
    solid.write_stl(out_dir / f"{stem}.stl", Vf, Ff, stem)
    files["stl"] = f"{stem}.stl"
    solid.write_3mf(out_dir / f"{stem}.3mf", parts, title=params.name or stem)
    files["3mf"] = f"{stem}.3mf"
    if params.multicolor:
        solid.write_zip(out_dir / f"{stem}_parts.zip",
                        {f"{stem}_{n}.stl": solid.stl_bytes(V, F, n) for n, _c, V, F in parts})
        files["parts_zip"] = f"{stem}_parts.zip"
        preview_parts = parts
    else:
        if land_mm is not None:
            # colour the 2.5D part by class and show the carved landmark as building
            Vr, Fr = solid.to_arrays(engrave(rest))
            Vl, Fl = solid.to_arrays(land_mm)
            preview_parts = list(_classify_faces(Vr, Fr, cls, Z, xs_mm, ys_mm, params.base_mm)) + [
                ("building", GROUPS["building"]["color"], Vl, Fl)]
        else:
            preview_parts = _classify_faces(Vf, Ff, cls, Z, xs_mm, ys_mm, params.base_mm)
    if pieces:
        # show where the model comes apart
        preview_parts = [(p.name, tiles.piece_color(p), *solid.to_arrays(p.man)) for p in pieces]
    if params.holder and not tiled_xy:
        ring_floor = 2.0                      # mm of ring under the model (not the elevation floor)
        ring = solid.holder_ring(params.shape, params.diameter_mm, params.base_mm, floor=ring_floor,
                                 wall=params.holder_wall_mm, clearance=params.holder_clearance_mm)
        Vh, Fh = solid.to_arrays(ring)
        solid.write_stl(out_dir / f"{stem}_holder.stl", Vh, Fh, f"{stem}_holder")
        files["holder"] = f"{stem}_holder.stl"
        # preview only: show the model sitting in its ring
        Vp = Vh.copy()
        Vp[:, 2] -= ring_floor
        preview_parts = list(preview_parts) + [("holder", HOLDER_COLOR, Vp, Fh)]
    report(0.96, "Writing preview")
    solid.write_glb(out_dir / "preview.glb", preview_parts)
    files["glb"] = "preview.glb"
    _write_topview(out_dir / "topview.png", Zview, cls_view, inside, cell_mm)
    files["png"] = "topview.png"

    result = {
        "params": asdict(params),
        "lat": lat, "lon": lon,
        "files": files,
        "stats": {
            "grid_cells": grid.n, "cell_m": round(cell, 3), "cell_mm": round(cell_mm, 4),
            "scale": f"1:{round(1000 / s):,}",
            "elev_min_m": round(floor, 2), "elev_max_m": round(ceiling, 2),
            "height_mm": round(zmax, 2),
            "triangles": int(len(Ff)),
            "parts": [{"name": n, "triangles": int(len(F))} for n, _c, _V, F in parts],
            # share of the area with map data (sea included); the rest was
            # filled from the surrounding ground
            "data_coverage": round(float(valid.sum() / max(inside.sum(), 1)), 4),
            "edge_buildings_removed": trimmed,
            "landmark": land_stats,
            "bridges": deck_stats,
            "roads": road_stats,
            "tiles": tile_stats,
            "cleanup": loose_stats,
            "volume_cm3": round(full.volume() / 1000, 2),
            "seconds": round(time.time() - t0, 1),
        },
    }
    result["status"] = build_status(params, result["stats"])
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    report(1.0, "Done")
    return result


def _stage(name: str, man) -> None:
    """With HK3D_DEBUG_STAGES set to a folder, keep the model as it stands
    after each step (print mm) there as <name>.stl, so a defect in the
    output can be traced to the step that made it (tools/stagetrace.py)."""
    d = os.environ.get("HK3D_DEBUG_STAGES")
    if d and man is not None and not man.is_empty():
        Path(d).mkdir(parents=True, exist_ok=True)
        solid.write_stl(Path(d) / f"{name}.stl", *solid.to_arrays(man))


def build_status(params: ModelParams, stats: dict) -> dict:
    """What a build is, without dressing a failure up as a result.

    ``generated``: the files were written and nothing was left out or
    worked around.  ``needs_review``: something was - a landmark that could
    not be carved (printed as a plain block instead), facade relief that
    failed, missing map data, fragments removed.  A build never claims to
    have passed a slicer or a test print: that is checked outside it
    (tools/layercheck.py, the slicer's preview, a test print)."""
    issues = []
    lm = stats.get("landmark") or {}
    if params.precise_landmark:
        if lm.get("error"):
            issues.append({"code": "landmark_failed", "detail": lm["error"]})
        elif lm.get("note"):
            issues.append({"code": "landmark_missing", "detail": lm["note"]})
        if lm.get("failed"):
            issues.append({"code": "buildings_failed", "detail": f"{lm['failed']} more carved buildings failed"})
        if lm.get("relief_error"):
            issues.append({"code": "relief_failed", "detail": lm["relief_error"]})
        if params.facade_relief_mm > 0 and lm and not lm.get("error") and lm.get("textured") is False:
            issues.append({"code": "relief_no_texture", "detail": "no facade texture: facades printed flat"})
    if stats.get("data_coverage", 1.0) < 0.99:
        issues.append({"code": "missing_data", "detail": f"map data covers {stats['data_coverage']:.1%} of the area"})
    cl = stats.get("cleanup") or {}
    if cl.get("loose_mm3", 0.0) > 0.5:
        issues.append({"code": "fragments_removed", "detail": f"{cl['loose_parts']} loose fragments ({cl['loose_mm3']} mm3) removed"})
    if (stats.get("tiles") or {}).get("floating"):
        issues.append({"code": "floating_pieces", "detail": f"{stats['tiles']['floating']} floating parts need support"})
    return {"level": "needs_review" if issues else "generated", "issues": issues,
            "slicer_verified": False, "test_printed": False}


def _entry_heights(catalog: Catalog, paths: list) -> np.ndarray:
    """Height (m, top minus base, base clamped at sea level) of each file in ``paths``."""
    bb = {e.path: e.bbox for e in catalog.entries}
    return np.array([bb[p][5] - max(bb[p][2], 0.0) if p in bb else np.inf for p in paths])


def _parts_below(groups: list, owner: np.ndarray, paths: list, catalog: Catalog, grid,
                 skip: set | None = None, min_share: float = 0.25, tol_m: float = 1.0,
                 ring_m: float = 3.0) -> list:
    """The files each group of buildings to carve stands on.

    A tower is often a file of its own that starts on the roof of its podium
    (another file).  Each group is carved only where its own roof is the top
    surface, so the part of the podium under the tower belonged to no group:
    the tower started at the podium roof, the podium kept only the strips
    showing around it, and the tower printed on thin piers.  A file stands
    under a group when it ends no higher than the group's lowest point
    (within ``tol_m``) and at least ``min_share`` of the group's footprint is
    inside the file's plan box, or that much of the box is under the
    footprint.  A podium can also rise above the tower's foot elsewhere (a
    mall round a station, its roof higher away from the tower): such a file
    stands under the group too when it starts below the group's lowest
    point and is what shows from above on at least ``min_share`` of a ring
    ``ring_m`` wide round the footprint - it surrounds the tower.  Under the
    footprint it is taken up to the group's lowest point (``cap``), no
    higher.  Files in ``skip`` (removed at the rim) are left out.
    Returns [(group index, [file ids], cap mPD), ...] for the groups that
    have any."""
    bb = {e.path: e.bbox for e in catalog.entries}
    box = [bb.get(p) for p in paths]
    xs, ys = grid.local_coords()
    cx, cy = grid.cx, grid.cy
    skip = skip or set()
    out = []
    for gi, g in enumerate(groups):
        base = min((box[i][2] for i in g if box[i] is not None), default=None)
        foot = np.isin(owner, g)
        n_foot = int(foot.sum())
        if base is None or not n_foot:
            continue
        ring = None
        below = []
        for j, b in enumerate(box):
            if b is None or j in g or j in skip:
                continue
            if b[5] > base + tol_m:
                if b[2] >= base - tol_m:
                    continue
                if ring is None:
                    k = max(1, int(round(ring_m / grid.cell)))
                    ring = owner[ndimage.binary_dilation(foot, iterations=k) & ~foot]
                if not len(ring) or (ring == j).mean() < min_share:
                    continue
            ci = np.nonzero((xs >= b[0] - cx) & (xs <= b[3] - cx))[0]
            ri = np.nonzero((ys >= b[1] - cy) & (ys <= b[4] - cy))[0]
            if not len(ci) or not len(ri):
                continue
            win = foot[ri.min():ri.max() + 1, ci.min():ci.max() + 1]
            if win.sum() >= min_share * n_foot or win.mean() >= min_share:
                below.append(j)
        if below:
            out.append((gi, below, float(base)))
    return out


def _centre_ids(owner: np.ndarray, X: np.ndarray, Y: np.ndarray, keep_r: float = 5.0) -> list[int]:
    return sorted(set(np.unique(owner[(np.hypot(X, Y) <= keep_r) & (owner >= 0)]).tolist()))


def _trim_edge(Hb: np.ndarray, owner: np.ndarray, paths: list, inside: np.ndarray,
               X: np.ndarray, Y: np.ndarray, keep_r: float = 5.0, heights: np.ndarray | None = None,
               keep_frac: float = 0.3) -> tuple[np.ndarray, int]:
    """Remove tall buildings that the outline cuts through, so the rim shows
    no sliced towers.  Buildings within ``keep_r`` metres of the centre (the
    landmark itself) are always kept.  With ``heights`` (per file), cut
    buildings lower than ``keep_frac`` of the landmark stay, so the rim keeps
    its street-level detail.  Returns (heights, number removed)."""
    cut = set(np.unique(owner[~inside & (owner >= 0)]).tolist())
    centre = set(_centre_ids(owner, X, Y, keep_r))
    if heights is not None and centre:
        limit = keep_frac * max(heights[i] for i in centre)
        cut = {i for i in cut if heights[i] > limit}
    drop = np.array(sorted(cut - centre), dtype=np.int32)
    if not len(drop):
        return Hb, 0
    Hb = Hb.copy()
    Hb[np.isin(owner, drop)] = np.nan
    return Hb, len(drop)


def _carve_landmark(catalog: Catalog, grid, H: dict, owner: np.ndarray, paths: list,
                    X: np.ndarray, Y: np.ndarray, cell: float, relief_m: float = 0.0,
                    relief_width_m: float = 1.0, relief_height_m: float | None = None,
                    ids: list | None = None, min_open_m: float = 0.0, chamfer: float | None = None,
                    extra: np.ndarray | None = None, cap: float | None = None):
    """Depth-hull solid (local metres, mPD) of the building files ``ids``
    (default: whatever stands at the centre), plus the prisms that cut its
    facade relief.  ``extra`` marks cells hidden from above by something
    else where these files may stand too (their top there is taken as the
    highest point of the files, at most ``cap``; the side views give the
    true outline).
    Returns (solid, its heightmap with NaN elsewhere, stats, cutters) or
    (None, None, stats, [])."""
    from . import carve, fbx
    from .raster import _MESH_CACHE

    if ids is None:
        ids = _centre_ids(owner, X, Y)
    if not ids:
        return None, None, {"note": "no building at the centre"}, []
    mask = np.isin(owner, ids)
    top = np.where(mask, H["building"], np.nan)
    if extra is not None:
        bb = {e.path: e.bbox for e in catalog.entries}
        zt = max((bb[paths[i]][5] for i in ids if paths[i] in bb), default=np.nan)
        if cap is not None:
            zt = min(zt, cap)
        top = np.where(extra & ~mask & np.isfinite(zt), np.float32(zt), top)
    Vs, Fs, UVs, Ts, files, off = [], [], [], [], [], 0
    for i in ids:
        path = str(catalog.root / paths[i])
        tx = None
        if relief_m > 0:
            try:
                tx = fbx.load_textured(path)
            except Exception:  # odd FBX: carve without relief
                tx = None
        parts = [(tx["V"], tx["F"])] if tx else _MESH_CACHE.get(path)
        for V, F in parts:
            Vl = V.astype(np.float64).copy()
            Vl[:, 0] -= grid.cx
            Vl[:, 1] -= grid.cy
            Vs.append(Vl)
            Fs.append(F + off)
            off += len(V)
            if tx:
                UVs.append(tx["uv"])
                Ts.append(np.where(tx["tex"] >= 0, tx["tex"] + len(files), -1))
            else:
                UVs.append(np.zeros((len(F), 3, 2)))
                Ts.append(np.full(len(F), -1))
        if tx:
            files += tx["files"]
    V, F = np.concatenate(Vs), np.concatenate(Fs)
    sampler = None
    if relief_m > 0 and files:
        sampler = carve.TextureSampler(np.concatenate(UVs), np.concatenate(Ts), files)
    ground = [H[g] for g in ("terrain", "ground", "infrastructure") if g in H]
    lows = [float(np.nanmin(g)) for g in ground if np.isfinite(g).any()]
    if not lows:
        # no ground layer chosen: the surface around it is filled as sea
        # (0 mPD, see _fill_missing), so the building must reach down to it
        lows = [0.0]
    z_floor = float(min(lows + [V[:, 2].min()])) - 2.0
    t = time.time()
    xs, ys = grid.local_coords()
    # facades are planes, so the side views can be coarser than the top one
    axes = carve.facade_axes(V, F)
    with np.errstate(all="ignore"):
        gsurf = np.fmax.reduce(np.stack(ground)) if ground else None
    L, st = carve.carve(V, F, top, xs, ys, cell, z_floor, axes=axes, side_cell=max(cell, 0.25),
                        ground=gsurf, min_open=min_open_m)
    if L.is_empty():  # forces the (lazy) booleans to run
        return None, None, {"note": "carving produced no solid"}, []
    cutters: list = []
    if relief_m > 0:
        st["textured"] = bool(sampler)
        if sampler is not None:
            try:
                cutters, rst = carve.relief_cutters(V, F, axes, sampler, relief_m, relief_width_m,
                                                    height=relief_height_m, chamfer=chamfer,
                                                    debug=os.environ.get("HK3D_DEBUG_RELIEF"))
                st.update(rst)
            except Exception as exc:  # keep the flat facades rather than fail
                st["relief_error"] = str(exc)
    st.update(files=[paths[i] for i in ids], seconds=round(time.time() - t, 1))
    return L, top, st, cutters


def _landmark_to_mm(L, land_top: np.ndarray, to_mm, floor: float, s: float, params, xs, ys, cell_mm: float,
                    cutters: list | None = None, rest: np.ndarray | None = None):
    """Scale the carved landmark into print millimetres, add thickened spires,
    trim it to the base, decimate it, make it printable without supports
    (on the 2.5D surface ``rest`` it stands on) and cut the facade relief
    (after the decimation, so the ribs keep their exact straight edges).
    Returns (solid, stats)."""
    from . import carve, printable

    zs = s * params.z_scale
    Lm = L.scale((s, s, zs)).translate((0.0, 0.0, params.base_mm - floor * zs))
    _stage("1_source", Lm)
    # masts/spires: the same 2.5D thickening as everywhere else, unioned in
    Zl = to_mm(np.where(np.isfinite(land_top), land_top, floor))
    Zt = _thicken_spires(Zl, np.isfinite(land_top), params.spire_mm, cell_mm)
    grown = Zt > Zl + 1e-4
    if grown.any():
        region = ndimage.binary_dilation(grown, iterations=2)
        r, c = np.nonzero(ndimage.binary_dilation(region, iterations=2))
        r0, r1, c0, c1 = r.min(), r.max() + 1, c.min(), c.max() + 1
        Zs = np.where(region, Zt, params.base_mm * 0.5)[r0:r1, c0:c1].astype(np.float32)
        Lm = Lm + solid.grid_solid((xs[c0:c1] * s).astype(np.float32), (ys[r0:r1] * s).astype(np.float32), Zs)
    # the outline prism must reach above the top of the building, however
    # tall it is and whatever the vertical exaggeration
    big = max(10 * params.diameter_mm, Lm.bounding_box()[5] + 5.0)
    keep = m3d_box(big, params.base_mm * 0.5, Lm.bounding_box()[5] + 1)
    # cut to the outline before the support pass, so a part whose support
    # stood outside it is held up again
    Lm = carve.decimate(Lm ^ keep ^ solid.shape_prism(params.shape, params.diameter_mm, big))
    _stage("2_trimmed", Lm)
    st: dict = {}
    wide: dict = {}
    if params.spire_mm > 0:
        # carved members thinner than a printable width in plan (a mast
        # truss seen edge-on, crown fins, parapets) widened layer by layer,
        # before the support pass so it holds up what they now carry
        add, wide = printable.widen_thin(Lm, params.print_base(), max(cell_mm, 0.25 * params.line_mm),
                                         params.layer_mm, params.spire_mm)
        if add is not None:
            Lm = carve.decimate(Lm + (add ^ keep ^ solid.shape_prism(params.shape, params.diameter_mm, big)))
        _stage("3_widened", Lm)
    if params.support_free:
        t = time.time()
        sup, st = printable.support_fill(
            Lm, None if rest is None else ((xs * s).astype(np.float32), (ys * s).astype(np.float32), rest),
            # supports need no finer slicing than half a printed line
            params.print_base(), max(cell_mm, 0.5 * params.line_mm), params.layer_mm, params.bridge_mm, params.line_mm,
            outline=solid.shape_prism(params.shape, params.diameter_mm, 2.0).slice(0.0))
        if os.environ.get("HK3D_DEBUG_SUPPORT"):
            d = Path(os.environ["HK3D_DEBUG_SUPPORT"])
            d.mkdir(parents=True, exist_ok=True)
            solid.write_stl(d / "land.stl", *solid.to_arrays(Lm))
            if rest is not None:
                np.savez_compressed(d / "rest.npz", xs=xs * s, ys=ys * s, Z=rest)
            if sup is not None:
                solid.write_stl(d / "support.stl", *solid.to_arrays(sup))
        if sup is not None:
            Lm = Lm + carve.decimate(sup)
        st = {"support_mm3": st["support_mm3"], "bridged_mm2": st["bridged_mm2"],
              "support_seconds": round(time.time() - t, 1)}
        _stage("4_supported", Lm)
    if cutters:
        import manifold3d as m3d
        place = lambda c: c.scale((s, s, zs)).translate((0.0, 0.0, params.base_mm - floor * zs))
        # where a part is too thin for recesses (from both sides) and a
        # printable web between them, its panels are not cut
        panels, kept = printable.relief_guard(Lm, [place(c) for c in cutters], params.print_base(),
                                              params.layer_mm, 2 * params.facade_relief_mm + params.line_mm)
        st = {**st, **kept}
        if panels:
            Lm = Lm - m3d.Manifold.batch_boolean(panels, m3d.OpType.Add)
        _stage("5_relief", Lm)
    if os.environ.get("HK3D_DEBUG_SUPPORT"):
        solid.write_stl(Path(os.environ["HK3D_DEBUG_SUPPORT"]) / "landmark.stl", *solid.to_arrays(Lm))
    return Lm, {**wide, **st}


def _bridge_decks(catalog: Catalog | None, grid, H: dict, order: list, top_f: np.ndarray, Z: np.ndarray,
                  cls: np.ndarray, to_mm, params, cell_mm: float, inside: np.ndarray, tol: float,
                  U: np.ndarray | None = None, stand: np.ndarray | None = None):
    """Keep the space under bridge decks that stand free above the road.

    The 2.5D surface fills everything under a deck.  Where the bridge layer
    is the top surface, the first bridge surface above whatever lies beneath
    (road, ground, a lower building) is its underside.  The space between is
    kept open where it is at least ``openings_mm`` high and wide, under a
    deck at least three layers thick; with ``support_free``, only where a
    printer can bridge it: on a straight run no longer than ``bridge_mm``
    between two things that stand up to the deck (piers, abutments, the
    buildings a footbridge connects), tried in 12 directions.

    Returns (surface with the road under open decks, classes of that
    surface, (deck top, deck bottom) height fields or None, stats)."""
    from . import printable
    from .raster import raster_up

    none = (Z, cls, None, {})
    bt = H["bridge"]
    on = np.isfinite(bt) & (bt >= top_f - tol) & inside
    if not on.any():
        return none
    others = [g for g in order if g != "bridge"]
    with np.errstate(all="ignore"):
        under = np.fmax.reduce(np.stack([H[g] for g in others])) if others else np.full_like(bt, np.nan)
        ground = np.fmax.reduce(np.stack([H[g] for g in others if g in SURFACE_GROUPS])) \
            if any(g in SURFACE_GROUPS for g in others) else under
    start = np.where(on & np.isfinite(under) & (under < bt - 0.5), under + 0.3, np.nan).astype(np.float32)
    if U is None:                            # (tests pass the underside in)
        U = raster_up(catalog, grid, "bridge", start)
    U = np.where(np.isfinite(start), U, np.nan)
    ok = on & np.isfinite(U)
    if not ok.any():
        return none
    Gmm = to_mm(np.where(ok, under, 0.0))
    Ub = np.minimum(to_mm(np.where(ok, U, 0.0)), Z - 3 * params.layer_mm)   # a deck three layers thick
    C = ok & (Ub - Gmm >= params.openings_mm)
    k = max(1, int(round(params.openings_mm / cell_mm)))
    C = ndimage.binary_opening(C, structure=_disk(k))
    stats = {"bridge_clear_m2": round(float(C.sum()) * (cell_mm / params.mm_per_m) ** 2, 1)}
    if params.support_free and C.any():
        g_mm = to_mm(np.where(np.isfinite(ground), ground, np.where(np.isfinite(under), under, top_f)))
        anchor = ~C & ((Z if stand is None else stand) - g_mm >= params.openings_mm)
        # a pier beyond the outline is cut away with it, leaving the deck's
        # end hanging at the rim: only what stays a line width inside counts
        anchor &= ndimage.binary_erosion(inside, structure=_disk(max(1, int(round(params.line_mm / cell_mm)))))
        C &= printable.bridgeable(C, anchor, params.bridge_mm / cell_mm, directions=12)
        C = ndimage.binary_opening(C, structure=_disk(k))
    stats["bridge_open_m2"] = round(float(C.sum()) * (cell_mm / params.mm_per_m) ** 2, 1)
    if not C.any():
        return Z, cls, None, stats
    Zlow = np.where(C, Gmm, Z).astype(np.float32)
    # the classes of what is under the deck
    cls_low = cls.copy()
    cls_low[C] = CLASS_IDS["base"]
    for g in others:
        h = H[g]
        cls_low[C & np.isfinite(h) & (h >= under - tol)] = CLASS_OF_GROUP[g]
    # deck slab between its underside and its top; elsewhere both surfaces
    # take the nearest deck cell's top, so they cancel exactly and the
    # deck's edge tapers within one cell, its underside sloping up
    near = ndimage.distance_transform_edt(~C, return_distances=False, return_indices=True)
    Tn = Z[near[0], near[1]]
    top = np.where(C, Z, Tn).astype(np.float32)
    bot = np.where(C, Ub, Tn).astype(np.float32)
    return Zlow, cls_low, (top, bot), stats


def m3d_box(size: float, z0: float, z1: float):
    import manifold3d as m3d
    return m3d.Manifold.cube((size, size, z1 - z0), center=True).translate((0, 0, (z0 + z1) / 2))


def _recess_roads(Z: np.ndarray, cls: np.ndarray, H: dict, top_f: np.ndarray, order: list,
                  area: np.ndarray, tol: float, cell_mm: float, depth_mm: float,
                  line_mm: float) -> tuple[np.ndarray, np.ndarray, dict]:
    """Sink the visible road surface ``depth_mm`` below everything around it.

    At map scale a kerb is a fraction of a printed layer, so roads and
    pavements print as one flat surface.  Lowering the roads by whole
    layers draws the street network on a one-colour print, the way the
    facade relief draws floor lines.  The road outline is tidied first, in
    print units: the edge is smoothed, lanes narrower than a printed line
    and specks smaller than three lines square are dropped, and small holes
    (a car, a patch of other surface) are filled - only with ground, never
    with a building, bridge, water or tree.  Bridge decks stay at their
    height.  The colour classes follow the same outline, so a multicolour
    road part and the recess coincide.  Returns (Z, cls, stats).
    """
    road = np.isfinite(H["infrastructure"]) & (H["infrastructure"] >= top_f - tol) & area
    if "bridge" in H:
        road &= ~(np.isfinite(H["bridge"]) & (H["bridge"] >= top_f - tol))
    ground = cls == CLASS_OF_GROUP["ground"]
    free = (road | ground) & area                   # cells the outline may move over
    if not road.any():
        return Z, cls, {}
    line = max(line_mm / cell_mm, 1.0)              # one printed line, in cells
    m = ndimage.gaussian_filter(road.astype(np.float32), max(line / 4, 0.7)) > 0.5
    m = ndimage.binary_opening(m, _disk(max(2, int(round(line)))))   # a printed line across
    min_cells = int(np.ceil((3 * line) ** 2))
    lab, n = ndimage.label(m)
    if n:
        m &= (np.bincount(lab.ravel(), minlength=n + 1) >= min_cells)[lab]
    hole_lab, nh = ndimage.label(~m)
    if nh:
        small = np.bincount(hole_lab.ravel(), minlength=nh + 1) < min_cells
        small[0] = False
        m |= small[hole_lab] & free
    m &= free
    out = np.where(m, Z - depth_mm, Z).astype(np.float32)
    cls = cls.copy()
    cls[m & ground] = CLASS_OF_GROUP["infrastructure"]
    cls[road & ~m] = CLASS_OF_GROUP["ground"]
    return out, cls, {"road_recess_mm": round(depth_mm, 3), "road_cells": int(m.sum()),
                      "road_dropped_cells": int((road & ~m).sum()), "road_filled_cells": int((m & ~road).sum())}


def _trim_edge_ground(Hg: np.ndarray, Ht: np.ndarray, inside: np.ndarray, min_h: float = 15.0) -> np.ndarray:
    """Some podiums and decks are modelled in the ground photo-mesh instead of
    BUILDING.  Remove raised parts of it (more than ``min_h`` m above the DTM)
    that the outline cuts through; the DTM fills the gap later."""
    with np.errstate(invalid="ignore"):
        raised = np.isfinite(Hg) & np.isfinite(Ht) & (Hg - Ht > min_h)
    if not raised.any():
        return Hg
    lab, n = ndimage.label(raised, structure=np.ones((3, 3)))
    cut = np.unique(lab[~inside & raised])
    cut = cut[cut > 0]
    if not len(cut):
        return Hg
    Hg = Hg.copy()
    Hg[np.isin(lab, cut)] = np.nan
    return Hg


def _filter_unprintable(Z: np.ndarray, building: np.ndarray, keep: np.ndarray, min_feature_mm: float,
                        spire_mm: float, cell_mm: float) -> np.ndarray:
    """Remove ground and tree noise thinner than ``min_feature_mm`` and make
    masts printable.

    On buildings and bridges (``keep``) the filter only removes isolated
    one-cell spikes; ground and trees are filtered fully.  Masts are found on
    the surface before filtering and widened after it: a mast one cell thin
    would otherwise be taken for such a spike and be gone before it could be
    widened.
    """
    raw = Z
    k = int(round(min_feature_mm / cell_mm))
    if k >= 2:
        Zo = ndimage.grey_opening(Z, footprint=_disk(k))
        Zb = ndimage.grey_opening(Z, size=(2, 2))
        Z = np.where(keep, np.maximum(Zb, np.minimum(Z, Zo)), Zo).astype(np.float32)
    return _thicken_spires(Z, building, spire_mm, cell_mm, source=raw)


def _sheet_mask(catalog: Catalog, grid) -> np.ndarray:
    """Cells that lie on a map sheet available locally."""
    xs, ys = grid.local_coords()
    ex, ny = grid.cx + xs, grid.cy + ys
    have = np.zeros((len(ys), len(xs)), bool)
    for s in catalog.sheets().values():
        x0, y0, x1, y1 = s["bounds"]
        have |= ((ny >= y0) & (ny <= y1))[:, None] & ((ex >= x0) & (ex <= x1))[None, :]
    return have


def _fill_missing(top: np.ndarray, ground: np.ndarray, have: np.ndarray, cell: float,
                  sea: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """The surface to print where there is no geometry, and where that is unknown.

    LandsD covers all land of a sheet, so an empty cell on a sheet we have is
    sea (0 mPD).  An empty cell on a sheet we do not have is unknown: it gets
    the ground around it carried in (nearest ground, then smoothed over about
    10 m) - never sea level, which would cut a cliff into a hillside and drag
    the base down with it - and is reported, not used to place the base.
    Buildings are not carried in.  Returns (surface, unknown mask).
    """
    hole = ~np.isfinite(top)
    unknown = hole & ~have
    out = np.where(hole, sea, top)
    if not unknown.any():
        return out, unknown
    src = np.isfinite(ground) & ~hole
    if not src.any():
        src, ground = ~hole, top
    if not src.any():
        return out, unknown
    _, (ri, ci) = ndimage.distance_transform_edt(~src, return_indices=True)
    near = ground[ri, ci].astype(np.float64)
    sig = max(10.0 / cell, 1.0)
    w = ndimage.gaussian_filter(unknown.astype(np.float64), sig)
    sm = ndimage.gaussian_filter(np.where(unknown, near, 0.0), sig) / np.maximum(w, 1e-9)
    # blend from the nearest ground at the boundary to the smooth fill inside
    d = ndimage.distance_transform_edt(unknown) * cell
    a = np.clip(d / 10.0, 0.0, 1.0)
    fill = (1 - a) * near + a * np.where(w > 1e-6, sm, near)
    out = np.where(unknown, fill, out).astype(np.float32)
    return out, unknown


def _thicken_spires(Z: np.ndarray, building: np.ndarray, width_mm: float, cell_mm: float,
                    source: np.ndarray | None = None) -> np.ndarray:
    """Widen masts and spires that are too thin to print, keeping their height.

    A cell belongs to a spire when it stands more than 0.5 mm above every cell
    on a ring 0.75-1.0 * max(``width_mm``, 0.6 mm) away.  Building corners, walls and
    sloped roof ridges always have neighbours of similar height somewhere on
    that ring, so only free-standing thin features qualify.  Each spire cell
    spreads its height over a disk ``width_mm`` across, so a feature of
    width d ends up about d + width_mm wide, and at least ``width_mm``.
    Growth stays on the building footprint so facades are not padded.
    Spires are found on ``source`` (default ``Z``) and added to ``Z``.
    """
    src = Z if source is None else source
    k = int(round(width_mm / cell_mm))
    if width_mm <= 0 or k < 3 or not building.any():
        return Z
    # the detection ring must clear the spire itself, so it never shrinks
    # below 0.45-0.6 mm even when the target width is small
    r_out = max(k, round(0.6 / cell_mm)) / 2
    r_in = 0.75 * r_out
    yy, xx = np.mgrid[-int(r_out):int(r_out) + 1, -int(r_out):int(r_out) + 1]
    rr = np.hypot(xx, yy)
    ring = (rr >= r_in) & (rr <= r_out + 0.25)
    around = ndimage.maximum_filter(src, footprint=ring)
    spire = building & (src - around > 0.5)
    if not spire.any():
        return Z
    grown = ndimage.grey_dilation(np.where(spire, src, 0).astype(np.float32), footprint=_disk(k))
    zone = ndimage.binary_dilation(building, iterations=1)
    return np.where(zone, np.maximum(Z, grown), Z).astype(np.float32)


def _spread(P: np.ndarray, cell: float, max_m: float = 64.0, coarse_m: float = 1.0) -> np.ndarray:
    """Smoothly extend the known (finite) values of ``P`` over the gaps: a
    pyramid of normalised Gaussian averages, the finest scale with enough
    known cells nearby winning, blended so no seams show.  NaN where nothing
    is known within about ``max_m``."""
    f = max(1, int(round(coarse_m / cell)))
    h, w = P.shape
    ph, pw = -h % f, -w % f
    ok = np.pad(np.isfinite(P), ((0, ph), (0, pw)))
    v = np.pad(np.where(np.isfinite(P), P, 0.0), ((0, ph), (0, pw))).astype(np.float64)
    shp = (ok.shape[0] // f, f, ok.shape[1] // f, f)
    k = ok.reshape(shp).sum(axis=(1, 3)).astype(np.float64)
    num = v.reshape(shp).sum(axis=(1, 3))
    c = f * cell
    out = np.full(k.shape, np.nan)
    sig = max_m
    while sig >= 2 * c - 1e-9:
        s_ = sig / c
        n_ = ndimage.gaussian_filter(num, s_, mode="nearest")
        d_ = ndimage.gaussian_filter(k, s_, mode="nearest")
        with np.errstate(all="ignore"):
            est = np.where(d_ > 1e-9, n_ / np.maximum(d_, 1e-9), 0.0)
        # weight: how well this scale is supported (d_ is the known-cell density)
        wgt = np.clip(d_ / (0.2 * f * f), 0.0, 1.0)
        out = np.where(np.isfinite(out), wgt * est + (1 - wgt) * out, np.where(wgt > 1e-3, est, np.nan))
        sig /= 2
    have = np.isfinite(out)
    if not have.any():
        return np.full(P.shape, np.nan, np.float32)
    if not have.all():
        _, (ri, ci) = ndimage.distance_transform_edt(~have, return_indices=True)
        out = out[ri, ci]
    full = ndimage.zoom(out, f, order=1, mode="nearest", grid_mode=True)[:h, :w]
    have = np.repeat(np.repeat(have, f, axis=0), f, axis=1)[:h, :w]
    return np.where(have, full, np.nan).astype(np.float32)


def _clean_ground(H: dict, order: list, cell: float, clean_m: float,
                  cleared: np.ndarray | None = None) -> dict:
    """Tidy the ground before it becomes a print surface.

    1. The photo-mesh layers (ground, roads, water) are the measured surface
       wherever they exist.  The bare-earth DTM is a coarse TIN whose
       triangles span tens of metres, so it only fills where the photo-mesh
       has none (e.g. under removed trees), smoothed over about 5 m so its
       facets do not show on hillsides.  It never lifts the photo-mesh: that
       would raise roads beside a cliff, and water beside a quay.
    2. Narrow gaps in the photo-mesh (typically a ring around each building)
       are filled from the nearest real ground instead of the lower DTM,
       which otherwise leaves a moat around buildings.
    3. Cars, bushes, lamp posts and photogrammetry noise are removed with an
       open-then-close filter (structuring size ``clean_m``).
    4. Water keeps its own level wherever it is the top of the photo-mesh:
       a pond, a fountain or the sea is flat and known, so neither the DTM
       nor the filter may move it.
    5. Where a building or podium cut by the outline was removed
       (``cleared``), nothing was ever measured: the photo-mesh stops at the
       building and the DTM is interpolated across the footprint, often
       metres above the street around it.  These cells are filled by
       smoothly extending the photo-mesh around them instead.
    Objects on the ground (buildings, bridges, trees) are not touched.
    """
    H = dict(H)
    surf = [g for g in order if g in SURFACE_GROUPS]
    if not surf:
        return H
    photo = [g for g in surf if g != "terrain"]
    with np.errstate(all="ignore"):
        P = np.fmax.reduce(np.stack([H[g] for g in photo])) if photo else np.full_like(H[surf[0]], np.nan)
    T = np.full_like(P, np.nan)
    if "terrain" in H:
        t = H["terrain"]
        ok = np.isfinite(t)
        if ok.any():
            sigma = max(5.0 / cell, 0.5)  # DTM triangles are tens of metres wide
            num = ndimage.gaussian_filter(np.where(ok, t, 0.0).astype(np.float64), sigma)
            den = ndimage.gaussian_filter(ok.astype(np.float64), sigma)
            T = np.where(ok, num / np.maximum(den, 1e-6), np.nan).astype(np.float32)
    water = None
    if "water" in photo:
        w = H["water"]
        with np.errstate(all="ignore"):
            others = [H[g] for g in photo if g != "water"]
            rest = np.fmax.reduce(np.stack(others)) if others else np.full_like(w, np.nan)
            water = np.isfinite(w) & ~(rest > w + 0.05)        # water is the top surface here
    valid = np.isfinite(P)
    if valid.any() and not valid.all():
        dist, (ri, ci) = ndimage.distance_transform_edt(~valid, return_indices=True)
        gap = (~valid) & (dist * cell <= 3.0)
        P = np.where(gap, P[ri, ci], P)
        valid = valid | gap
    if cleared is not None and (cleared & ~valid).any() and valid.any():
        # extend the land around it: never the sea, which would pull a
        # quayside site down to the water
        src = valid & ~water if water is not None else valid
        sp = _spread(np.where(src, P, np.nan), cell)
        use = cleared & ~valid & np.isfinite(sp)
        T = np.where(use, sp, T)
    S = np.where(valid, P, T)
    has = np.isfinite(S)
    if not has.any():
        return H
    if clean_m > 0:
        k = int(round(clean_m / cell))
        if k >= 2:
            fill = float(np.nanmin(S))
            S0 = np.where(has, S, fill)
            S0 = ndimage.grey_closing(ndimage.grey_opening(S0, size=(k, k)), size=(k, k))
            S = np.where(has, S0, np.nan)
    if water is not None:
        S = np.where(water, H["water"], S)
    # write the cleaned surface back: each surface group keeps its footprint
    # (so colours/classes stay right) but takes the cleaned height
    for g in photo:
        H[g] = np.where(np.isfinite(H[g]), S, np.nan).astype(np.float32)
    if "terrain" in H:
        # the terrain layer carries the full cleaned surface (gap fills included)
        H["terrain"] = S.astype(np.float32)
    else:
        H[photo[0]] = np.where(valid, S, H[photo[0]]).astype(np.float32)
    return H


def _clean_mask(m: np.ndarray, free: np.ndarray, min_cells: int) -> np.ndarray:
    """Make a colour-part mask printable: drop specks and resolve diagonal-only
    contacts (which would make two bodies share a single edge)."""
    lab, n = ndimage.label(m)
    if n:
        sizes = np.bincount(lab.ravel())
        m = (sizes >= min_cells)[lab] & m
    # close 1-cell gaps: two regions one cell apart would meet along a
    # zero-thickness edge once interpolated
    m = m | (ndimage.binary_closing(m, structure=np.ones((3, 3), bool)) & free)
    for _ in range(8):
        a, b, c, d = m[:-1, :-1], m[:-1, 1:], m[1:, :-1], m[1:, 1:]
        p1 = a & d & ~b & ~c
        p2 = b & c & ~a & ~d
        if not (p1.any() or p2.any()):
            break
        fb, fc_ = free[:-1, 1:], free[1:, :-1]
        fa, fd = free[:-1, :-1], free[1:, 1:]
        m = m.copy()
        # p1 (NW+SE): fill NE if free, else SW if free, else drop SE
        m[:-1, 1:] |= p1 & fb
        m[1:, :-1] |= p1 & ~fb & fc_
        m[1:, 1:] &= ~(p1 & ~fb & ~fc_)
        # p2 (NE+SW): fill NW if free, else SE if free, else drop SW
        m[:-1, :-1] |= p2 & fa
        m[1:, 1:] |= p2 & ~fa & fd
        m[1:, :-1] &= ~(p2 & ~fa & ~fd)
    return m


def _slug(name: str) -> str:
    keep = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name.strip())
    return keep.strip("_")[:60]


def _classify_faces(V, F, cls, Z, xs_mm, ys_mm, base_mm):
    """Split a single body into per-class submeshes for a coloured preview."""
    tri = V[F]
    cen = tri.mean(1)
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nz = nrm[:, 2] / np.maximum(np.linalg.norm(nrm, axis=1), 1e-12)
    cell = float(xs_mm[1] - xs_mm[0])
    col = np.clip(((cen[:, 0] - xs_mm[0]) / cell + 0.5).astype(int), 0, len(xs_mm) - 1)
    row = np.clip(((ys_mm[0] - cen[:, 1]) / cell + 0.5).astype(int), 0, len(ys_mm) - 1)
    # walls take the class of the highest neighbouring cell
    zi = ndimage.maximum_filter(Z, size=3)
    pick = np.zeros(Z.shape, dtype=np.int64)
    best = np.full(Z.shape, -np.inf)
    ny, nx = Z.shape
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            sh = np.roll(np.roll(Z, dy, 0), dx, 1)
            better = sh > best
            best = np.where(better, sh, best)
            pick = np.where(better, (dy + 1) * 3 + (dx + 1), pick)
    rr, cc = np.mgrid[:ny, :nx]
    dyy, dxx = pick // 3 - 1, pick % 3 - 1
    wall_cls = cls[(rr - dyy) % ny, (cc - dxx) % nx]
    del zi
    fc = np.where(nz > 0.5, cls[row, col], wall_cls[row, col])
    fc = np.where((nz < -0.5) | (cen[:, 2] < base_mm - 1e-3), 0, fc)
    out = []
    for c in np.unique(fc):
        out.append((CLASS_NAMES[int(c)], CLASS_COLORS[int(c)], V, F[fc == c]))
    return out


def _write_topview(path, Z, cls, inside, cell_mm):
    from PIL import Image

    im = Image.fromarray(_topview_rgba(Z, cls, inside, cell_mm), "RGBA")
    if im.width > 1024:
        im = im.resize((1024, 1024), Image.LANCZOS)
    im.save(path)


def _topview_rgba(Z, cls, inside, cell_mm):
    gy, gx = np.gradient(Z.astype(np.float64), cell_mm)
    az, alt = np.radians(315), np.radians(45)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    shade = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    shade = np.clip(0.35 + 0.75 * shade, 0, 1)
    pal = np.array([[int(CLASS_COLORS[i][j:j + 2], 16) for j in (1, 3, 5)] for i in range(5)], dtype=np.float64)
    rgb = pal[cls] * shade[..., None]
    a = np.where(inside, 255, 0)
    return np.dstack([np.clip(rgb, 0, 255), a]).astype(np.uint8)
