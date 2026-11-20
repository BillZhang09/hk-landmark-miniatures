import os
from pathlib import Path

import manifold3d as m3d
import numpy as np
import pytest

from hk3d import sheets, solid
from hk3d.coords import hk80_to_wgs84, wgs84_to_hk80
from hk3d.fbx import _triangulate

DATA = Path(os.environ.get("HK3D_DATA", Path(__file__).resolve().parents[2]))
HAS_DATA = (DATA / "11-NE-18B").is_dir()


def test_sheet_bounds_known():
    assert sheets.sheet_bounds("11-NE-18B") == (841250.0, 819800.0, 842000.0, 820400.0)


@pytest.mark.parametrize("name", ["1-NW-1A", "11-NE-18B", "16-SE-25D", "6-SW-13C", "17-NE-3B"])
def test_sheet_roundtrip(name):
    x0, y0, x1, y1 = sheets.sheet_bounds(name)
    assert sheets.sheet_at((x0 + x1) / 2, (y0 + y1) / 2) == name


def test_sheets_in_bbox():
    got = sheets.sheets_in_bbox(841000, 819700, 841400, 820500)
    assert got == ["11-NE-13C", "11-NE-13D", "11-NE-18A", "11-NE-18B", "11-NE-18C", "11-NE-18D"]


def test_coords_roundtrip():
    e, n = wgs84_to_hk80(22.3105, 114.2245)
    lat, lon = hk80_to_wgs84(e, n)
    assert abs(lat - 22.3105) < 1e-7 and abs(lon - 114.2245) < 1e-7
    assert 830000 < e < 850000 and 810000 < n < 830000


def test_triangulate_mixed_polygons():
    # a quad (0,1,2,3) and a triangle (4,5,6), FBX encodes the last index as ~i
    pvi = np.array([0, 1, 2, ~3, 4, 5, ~6])
    tris = _triangulate(pvi)
    assert tris.tolist() == [[0, 1, 2], [0, 2, 3], [4, 5, 6]]


def test_grid_solid_is_closed_and_cuts_cleanly():
    import trimesh

    n = 60
    xs = np.linspace(-30, 30, n, dtype=np.float32)
    ys = xs[::-1].copy()
    X, Y = np.meshgrid(xs, ys)
    Z = (3 + 5 * np.exp(-(X ** 2 + Y ** 2) / 200) + 10 * ((np.abs(X) < 5) & (np.abs(Y) < 5))).astype(np.float32)
    m = solid.grid_solid(xs, ys, Z) ^ solid.shape_prism("circle", 50, 40)
    V, F = solid.to_arrays(m.simplify(0.01))
    tm = trimesh.Trimesh(V, F)
    assert tm.is_watertight and tm.is_winding_consistent and tm.volume > 0
    assert np.allclose(tm.bounds[:, :2], [[-25, -25], [25, 25]], atol=0.05)


@pytest.mark.skipif(not HAS_DATA, reason="LandsD sample data not available")
@pytest.mark.parametrize("shape", ["circle", "hexagon", "square"])
def test_build_end_to_end(tmp_path, shape):
    import trimesh

    from hk3d.catalog import load_or_scan
    from hk3d.pipeline import ModelParams, build

    cat = load_or_scan(DATA, tmp_path / "cache")
    p = ModelParams(e=841625, n=820100, radius_m=120, diameter_mm=60, detail_mm=0.2, shape=shape,
                    multicolor=(shape == "circle"))
    res = build(p, cat, tmp_path / "out", tmp_path / "cache")
    tm = trimesh.load(tmp_path / "out" / res["files"]["stl"])
    assert tm.is_watertight
    assert res["stats"]["data_coverage"] == 1.0
    assert abs(tm.bounds[1][0] - 30) < 0.1
    if p.multicolor:
        assert len(res["stats"]["parts"]) >= 2


def test_separate_touching_makes_two_cubes_manifold():
    import manifold3d as m3d
    import trimesh

    a = m3d.Manifold.cube((1, 1, 1))
    b = m3d.Manifold.cube((1, 1, 1)).translate((1, 1, 0))  # shares one vertical edge with a
    V, F = solid.to_arrays(a + b)
    tm = trimesh.Trimesh(V, F)  # merges vertices by position, like a slicer
    assert tm.is_watertight and abs(tm.volume - 2) < 1e-2


def test_engraving_section_fits_and_is_mirrored():
    from hk3d.label import engraving

    e = engraving(["樂華邨 Lok Wah", "22.31990°N 114.22159°E"], 100, 0.6)
    assert e is not None and e.status().name == "NoError"
    lo, hi = np.array(e.bounding_box()[:3]), np.array(e.bounding_box()[3:])
    assert hi[0] - lo[0] <= 62.5 and abs(hi[2] - 0.6) < 1e-6


def test_3mf_assembly_has_components_and_bambu_config(tmp_path):
    import zipfile

    import manifold3d as m3d

    parts = []
    for i, name in enumerate(["base", "building"]):
        V, F = solid.to_arrays(m3d.Manifold.cube((1, 1, 1)).translate((i * 2, 0, 0)))
        parts.append((name, "#ffffff", V, F))
    solid.write_3mf(tmp_path / "t.3mf", parts, title="demo")
    with zipfile.ZipFile(tmp_path / "t.3mf") as z:
        model = z.read("3D/3dmodel.model").decode()
        cfg = z.read("Metadata/model_settings.config").decode()
    assert model.count("<component ") == 2 and 'name="demo"' in model
    assert 'value="building"' in cfg and 'key="extruder" value="2"' in cfg



def test_3mf_coordinates_read_back_as_the_stl_holds_them(tmp_path):
    """Vertices a few microns apart (as separate_touching leaves them) stay
    apart in the 3MF: rounded to 4 decimals they merged into zero-area
    triangles (BoC v22: 15,824 of them)."""
    import re
    import zipfile

    V = np.array([[10.123401, 0, 0], [10.123409, 0, 0], [10.123405, 1, 0], [100.5, 50.25, 171.36]])
    F = np.array([[0, 1, 2], [0, 2, 3]])
    solid.write_3mf(tmp_path / "t.3mf", [("m", "#ffffff", V, F)])
    with zipfile.ZipFile(tmp_path / "t.3mf") as z:
        s = z.read("3D/3dmodel.model").decode()
    got = np.array(re.findall(r'<vertex x="([^"]+)" y="([^"]+)" z="([^"]+)"', s), float).astype(np.float32)
    assert np.array_equal(got, V.astype(np.float32))


def test_3mf_names_with_quotes_stay_valid_xml(tmp_path):
    import xml.etree.ElementTree as ET
    import zipfile

    import manifold3d as m3d

    names = ['HK "test" <tower>', "Jardine & Co's", "中銀大廈"]
    parts = []
    for i, name in enumerate(names):
        V, F = solid.to_arrays(m3d.Manifold.cube((1, 1, 1)).translate((i * 2, 0, 0)))
        parts.append((name, "#ffffff", V, F))
    title = '中環 "Central" & more'
    solid.write_3mf(tmp_path / "t.3mf", parts, title=title)
    with zipfile.ZipFile(tmp_path / "t.3mf") as z:
        model = ET.fromstring(z.read("3D/3dmodel.model"))
        cfg = ET.fromstring(z.read("Metadata/model_settings.config"))
    got = {e.get("name") for e in model.iter() if e.tag.endswith("object")}
    assert set(names) | {title} <= got                    # names round-trip unchanged
    assert {e.get("value") for e in cfg.iter("metadata") if e.get("key") == "name"} == set(names) | {title}

@pytest.mark.parametrize("shape", ["circle", "hexagon", "square"])
def test_holder_ring_fits_model(shape):
    import trimesh

    ring = solid.holder_ring(shape, 80, 5, wall=5, clearance=0.3, floor=2)
    V, F = solid.to_arrays(ring)
    tm = trimesh.Trimesh(V, F)
    assert tm.is_watertight and tm.volume > 0
    assert abs(tm.bounds[1][2] - 7) < 1e-4            # floor + pocket depth
    # the model (size 80) must fit through the pocket mouth at z = 6.9 with clearance
    sec = ring.slice(6.0)
    model = solid._frustum(shape, 80, 80, 1).slice(0.5)
    assert (sec ^ model).area() < 1e-6


def test_ground_cleaning_keeps_masts_and_removes_cars():
    from hk3d.pipeline import _clean_ground

    n, cell = 200, 0.2
    ground = np.full((n, n), 10.0, np.float32)
    ground[50:60, 50:70] += 1.5                       # a 2 m x 4 m "car"
    building = np.full((n, n), np.nan, np.float32)
    building[100:140, 100:140] = 60.0
    building[118:122, 118:122] = 90.0                 # a 0.8 m mast on the roof
    H = {"terrain": np.full((n, n), 9.0, np.float32), "ground": ground, "building": building}
    out = _clean_ground(H, ["terrain", "ground", "building"], cell, 2.5)
    assert out["ground"][55, 60] == pytest.approx(10.0)
    assert np.nanmax(out["building"]) == 90.0


def test_ground_cleaning_keeps_water_level_and_measured_ground():
    """A coarse DTM must not lift water beside a quay or a road beside a
    slope, whether or not the noise filter is on."""
    from hk3d.pipeline import _clean_ground

    n, cell = 200, 0.25
    xx = np.tile(np.arange(n), (n, 1)) * cell
    quay = xx < 25.0                                   # 10 m platform, sea from x = 25 m
    ground = np.where(quay, 10.0, np.nan).astype(np.float32)
    water = np.where(quay, np.nan, 0.0).astype(np.float32)
    road = np.full((n, n), np.nan, np.float32)
    road[:40, :60] = 10.0                              # a road on the platform...
    dtm = np.clip(10 - (xx - 15) / 2, 0, 10).astype(np.float32)   # ...under a TIN that slopes 20 m wide
    dtm[:40, :60] = 14.0                               # and lies 4 m above it
    order = ["terrain", "ground", "infrastructure", "water"]
    for clean in (0.0, 2.5):
        H = {"terrain": dtm.copy(), "ground": ground.copy(), "infrastructure": road.copy(), "water": water.copy()}
        out = _clean_ground(H, order, cell, clean)
        assert np.nanmax(np.abs(out["water"][~quay])) == 0.0   # the sea stays at 0 mPD up to the quay
        assert np.nanmax(out["ground"]) == pytest.approx(10.0)
        assert np.nanmax(out["infrastructure"]) == pytest.approx(10.0)
        assert np.nanmax(np.abs(out["terrain"] - np.where(quay, 10.0, 0.0))) < 1e-5


def test_ground_under_a_trimmed_tower_continues_the_street():
    """Where a tower cut by the outline is removed, the photo-mesh has no
    ground under it and the DTM is a guess (here a mound 9 m above the
    street).  The footprint must take the street level around it, not show
    the guess as a ramp; real DTM-only ground elsewhere is left alone."""
    from hk3d.pipeline import _clean_ground

    n, cell = 400, 0.125
    yy, xx = np.mgrid[0:n, 0:n] * cell
    tower = (xx > 20) & (yy > 10) & (yy < 40)          # 30 m x 30 m, running off the edge
    tree = (xx > 2) & (xx < 8) & (yy > 2) & (yy < 8)    # ground hidden by a tree, not cleared
    street = 6.0 + 0.02 * yy                            # gently sloping street
    ground = np.where(tower | tree, np.nan, street).astype(np.float32)
    dtm = (street + 9.0 * np.exp(-(((xx - 40) / 12) ** 2 + ((yy - 25) / 10) ** 2))).astype(np.float32)
    dtm[tree] = 7.5
    order = ["terrain", "ground", "building"]
    H = {"terrain": dtm, "ground": ground, "building": np.full((n, n), np.nan, np.float32)}
    old = _clean_ground(dict(H), order, cell, 2.5)["terrain"]
    assert np.nanmax(old[tower] - street[tower]) > 5     # the ramp this guards against
    out = _clean_ground(dict(H), order, cell, 2.5, cleared=tower)["terrain"]
    assert np.nanmax(np.abs(out[tower] - street[tower])) < 0.3
    assert np.abs(np.diff(out, axis=1)).max() < 0.1      # no step where the fill meets the street (0.04 mm at 1:2,500)
    assert out[40, 40] == pytest.approx(old[40, 40])    # the tree's ground is not touched


def test_missing_sheets_continue_the_ground_and_do_not_set_the_base():
    """Cells without any geometry are sea on a sheet we have, but unknown on
    one we do not: those must not become a 0 mPD pit that drags the base down."""
    from hk3d.pipeline import _fill_missing

    n, cell = 200, 0.5
    yy, xx = np.mgrid[:n, :n]
    land = (100.0 + 0.05 * xx).astype(np.float32)      # a hilltop at 100-105 mPD
    top = np.where(xx < 120, land, np.nan).astype(np.float32)
    top[50:60, 50:60] = 140.0                          # a building, not carried into the gap
    ground = np.where(xx < 120, land, np.nan).astype(np.float32)
    have = xx < 150                                     # the sheet east of x = 150 is missing
    out, unknown = _fill_missing(top, ground, have, cell)
    assert (unknown == (xx >= 150)).all()
    assert (out[(xx >= 120) & (xx < 150)] == 0.0).all()   # on our sheet but empty: sea
    assert out[unknown].min() > 99.0 and out[unknown].max() < 106.0
    assert out[~(xx >= 120)].min() >= 100.0             # floor from real data stays on the hill
    # nothing missing: unchanged
    out, unknown = _fill_missing(land, land, np.ones((n, n), bool), cell)
    assert not unknown.any() and (out == land).all()


def test_one_cell_masts_survive_the_noise_filter():
    from hk3d.pipeline import _filter_unprintable

    cell = 0.05                                         # mm per cell
    Z = np.full((80, 80), 3.0, np.float32)
    Z[20:60, 20:60] = 40.0
    Z[40, 40] = 70.0                                    # a mast one cell thin
    Z[5, 5] = 4.0                                       # a one-cell speck on the ground
    building = Z > 10
    out = _filter_unprintable(Z, building, building, 0.3, 0.6, cell)
    assert out.max() == 70.0
    mast = out >= 70.0
    assert min(mast.any(0).sum(), mast.any(1).sum()) * cell >= 0.6 - 1e-6
    assert out[5, 5] == 3.0


def test_facade_ribs_follow_line_width_and_layer_height():
    """Ribs are one printed line wide and floor lines two layers tall, set by
    the printer rather than tied to the spire width; a fine nozzle keeps the
    floor lines that a coarse one has to drop."""
    from hk3d import carve
    from hk3d.pipeline import ModelParams

    coarse = ModelParams(e=0, n=0, radius_m=100, diameter_mm=80)                    # 1:2500
    fine = ModelParams(e=0, n=0, radius_m=100, diameter_mm=80, line_mm=0.2, layer_mm=0.1, spire_mm=0.3)
    assert coarse.rib_width_m() == pytest.approx(1.0) and coarse.rib_height_m() == pytest.approx(1.0)
    assert fine.rib_width_m() == pytest.approx(0.5) and fine.rib_height_m() == pytest.approx(0.5)
    tall = ModelParams(e=0, n=0, radius_m=100, diameter_mm=80, z_scale=2.0)
    assert tall.rib_height_m() == pytest.approx(0.5) and tall.rib_width_m() == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    cell, n = 0.125, 1600
    cnt = np.full(n, 400.0)
    # 3 m storeys at 1:5000: 0.6 mm apart on the print
    city = dict(e=0, n=0, radius_m=250, diameter_mm=100)
    sig = 2.0 + 0.5 * rng.random(n)
    sig[::24] += 6.0
    rows, info = carve.floor_bands(sig * cnt, cnt, cell, ModelParams(**city).rib_height_m())
    assert rows is None and info["floor_lines"] == 0   # 0.4 mm lines need a 0.9 mm pitch
    rows, info = carve.floor_bands(sig * cnt, cnt, cell, ModelParams(**city, line_mm=0.2, layer_mm=0.1).rib_height_m())
    assert abs(info["floor_pitch_m"] - 3.0) < 0.1 and info["floor_lines"] > 60
    # a 1.5 m window grid is no storey, however fine the nozzle
    grid = 2.0 + 0.5 * rng.random(n)
    grid[::12] += 6.0
    rows, info = carve.floor_bands(grid * cnt, cnt, cell, fine.rib_height_m())
    assert rows is None and abs(info["floor_pitch_m"] - 1.5) < 0.1


def test_spire_thickening_widens_masts_but_not_corners_or_ridges():
    from hk3d.pipeline import _thicken_spires

    n, cell = 200, 0.05                                # mm per cell
    Z = np.full((n, n), 5.0, np.float32)
    building = np.zeros((n, n), bool)
    building[40:160, 40:160] = True
    Z[40:160, 40:160] = 60.0                          # tower with sharp corners
    yy, xx = np.mgrid[:n, :n]
    ridge = (np.abs(xx - 100) < 30) & (yy >= 40) & (yy < 70)
    Z[ridge] = np.maximum(Z[ridge], 70 - 0.5 * np.abs(xx - 100)[ridge])  # sloped ridge
    Z[118:121, 98:101] = 90.0                          # 0.15 mm mast, 30 mm tall
    out = _thicken_spires(Z, building, 0.6, cell)
    mast = out >= 90.0
    width = (mast.any(0).sum(), mast.any(1).sum())
    assert min(width) * cell >= 0.6 - 1e-6           # now printable
    assert out.max() == 90.0                           # height unchanged
    changed = out != Z
    assert not changed[40:45, 40:45].any()            # tower corner untouched
    assert not changed[40:70, 70:130].any()           # ridge untouched


def test_trim_edge_drops_cut_buildings_but_keeps_centre():
    from hk3d.pipeline import _trim_edge

    n = 101
    xs = np.linspace(-50, 50, n)
    X, Y = np.meshgrid(xs, xs)
    inside = np.hypot(X, Y) <= 40
    H = np.full((n, n), np.nan, np.float32)
    owner = np.full((n, n), -1, np.int32)
    H[40:60, 40:60], owner[40:60, 40:60] = 300, 0     # landmark at the centre
    H[45:55, 85:100], owner[45:55, 85:100] = 200, 1   # tower cut by the rim
    H[15:25, 45:55], owner[15:25, 45:55] = 30, 2      # fully inside
    out, removed = _trim_edge(H, owner, ["a", "b", "c"], inside, X, Y)
    assert removed == 1
    assert np.isnan(out[50, 90]) and out[50, 50] == 300 and out[20, 50] == 30


def test_fbx_download_upgrades_fbx0_sheet(tmp_path, monkeypatch):
    import io
    import zipfile

    from hk3d import download

    sheet = "11-SW-14A"
    old = tmp_path / sheet
    (old / "BUILDING").mkdir(parents=True)
    (old / "BUILDING" / "old.fbx").write_bytes(b"x")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(f"{sheet}/BUILDING/new.fbx", b"y")
        zf.writestr(f"{sheet}/GENERIC/g.fbx", b"z")
    calls = []

    class Resp(io.BytesIO):
        headers = {"Content-Length": str(len(buf.getvalue()))}
        def __enter__(self): return self
        def __exit__(self, *a): pass

    def fake_urlopen(req, timeout=0):
        calls.append(req.full_url)
        return Resp(buf.getvalue())

    monkeypatch.setattr(download.urllib.request, "urlopen", fake_urlopen)
    download.download_sheet(sheet, tmp_path, "FBX0")          # already present: no-op
    assert calls == [] and not download.is_textured(old)
    download.download_sheet(sheet, tmp_path, "FBX")           # upgrade
    assert len(calls) == 1 and download.is_textured(old)
    assert (old / "BUILDING" / "new.fbx").exists() and not (old / "BUILDING" / "old.fbx").exists()
    download.download_sheet(sheet, tmp_path, "FBX")           # already textured: no-op
    assert len(calls) == 1
    assert not list(tmp_path.glob(".dl-*"))                     # temp folders cleaned up


def test_trim_edge_ground_removes_cut_podium_only():
    from hk3d.pipeline import _trim_edge_ground

    n = 101
    xs = np.linspace(-50, 50, n)
    X, Y = np.meshgrid(xs, xs)
    inside = np.hypot(X, Y) <= 40
    T = np.full((n, n), 5.0, np.float32)
    G = np.full((n, n), 6.0, np.float32)
    G[40:60, 85:100] = 55.0                   # podium in the ground mesh, cut by the rim
    G[40:60, 40:60] = 40.0                    # deck fully inside
    G[10:20, 45:55] = 15.0                    # low raised area touching the rim (< 15 m)
    out = _trim_edge_ground(G, T, inside)
    assert np.isnan(out[50, 90]) and out[50, 50] == 40.0 and out[12, 50] == 15.0 and out[30, 30] == 6.0


def test_landmarks_and_snap_pick_the_right_building():
    from hk3d.catalog import Catalog, Entry

    def ent(name, group, bbox):
        return Entry(path=f"S/{name}.fbx", sheet="11-SW-14A", folder="BUILDING", group=group,
                     bbox=bbox, ntri=1, size=1, mtime=0)

    cat = Catalog(root=Path("."), entries=[
        ent("tower", "building", [990, 990, 5, 1010, 1010, 300]),
        ent("podium", "building", [900, 900, 5, 1100, 1100, 40]),     # box covers the tower too
        ent("low", "building", [1200, 1000, 5, 1220, 1020, 30]),       # under min_height
        ent("far", "building", [5000, 5000, 5, 5020, 5020, 400]),
        ent("ground", "ground", [0, 0, 0, 9000, 9000, 900]),
    ])
    names = [b["top_m"] for b in cat.landmarks(1000, 1000, radius=600)]
    assert names == [300.0]                                   # podium < 40 m tall, far is out of range
    hit = cat.building_at(1003, 997)
    assert hit["top_m"] == 300.0 and (hit["e"], hit["n"]) == (1000.0, 1000.0)
    assert cat.building_at(950, 950)["top_m"] == 40.0         # only the podium covers this point
    assert cat.building_at(3000, 3000) is None


def _open_box(w, d, h, ang):
    """Walls and roof of a rotated box (no floor), like a LandsD building mesh."""
    x, y = w / 2, d / 2
    V = np.array([[-x, -y, 0], [x, -y, 0], [x, y, 0], [-x, y, 0],
                  [-x, -y, h], [x, -y, h], [x, y, h], [-x, y, h]], float)
    c, s = np.cos(ang), np.sin(ang)
    V[:, :2] = V[:, :2] @ np.array([[c, s], [-s, c]])
    F = np.array([[0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6],
                  [3, 0, 4], [3, 4, 7], [4, 5, 6], [4, 6, 7]])
    return V, F


def test_carve_open_mesh_gives_watertight_flat_facades():
    import manifold3d as m3d

    from hk3d import carve
    ang = np.radians(30)
    V, F = _open_box(20, 10, 30, ang)
    cell = 0.5
    xs = np.arange(-20, 20, cell) + cell / 2
    ys = np.arange(20, -20, -cell) - cell / 2
    X, Y = np.meshgrid(xs, ys)
    lx = X * np.cos(ang) + Y * np.sin(ang)
    ly = -X * np.sin(ang) + Y * np.cos(ang)
    top = np.where((np.abs(lx) < 10) & (np.abs(ly) < 5), 30.0, np.nan)
    assert sorted(np.degrees(carve.facade_axes(V, F)) % 180) == pytest.approx([30, 120], abs=0.6)
    L, st = carve.carve(V, F, top, xs, ys, cell, z_floor=-1.0)
    assert st["views"] == 4
    assert L.status() == m3d.Error.NoError and L.genus() == 0
    # the true box from z=-1 to 30 is 20 x 10 x 31 = 6200 m3; a 0.5 m
    # heightmap alone would be off by its stair-steps around the outline
    assert L.volume() == pytest.approx(6200, rel=0.01)
    # facade points lie on the exact rotated planes
    P = np.asarray(L.to_mesh().vert_properties)[:, :3].astype(np.float64)
    side = P[(P[:, 2] > 1) & (P[:, 2] < 29)]
    c, s = np.cos(ang), np.sin(ang)
    lp = np.abs(np.column_stack([side[:, 0] * c + side[:, 1] * s, -side[:, 0] * s + side[:, 1] * c]))
    dist = np.minimum(np.abs(lp[:, 0] - 10), np.abs(lp[:, 1] - 5))
    assert np.percentile(dist, 95) < 0.05


def test_carve_keeps_thin_slanted_fins_whole():
    """Blades thinner than two side-view cells, aligned with no facade axis,
    leaning and standing in rows (crown fins, truss members) must not be
    bitten into pieces by rays that slip past them onto what is behind."""
    import manifold3d as m3d

    from hk3d import carve
    V, F = _open_box(20, 20, 30, 0.0)
    cell, side = 0.125, 0.25                              # as in a landmark build
    lean = np.array([[1, 0, 0.8, 0], [0, 1, 0, 0], [0, 0, 1, 0]], float)      # x shifts with height
    one = m3d.Manifold.cube((6.0, 0.3, 10.0), center=True).translate((0, 0, 5)).transform(lean).rotate((0, 0, 10))
    blades = [one.translate((0, y, 30)) for y in (-6, -4.5, -3, -1.5, 0, 1.5)]
    fins = m3d.Manifold.batch_boolean(blades, m3d.OpType.Add)
    Vs, Fs, off = [V], [F], len(V)
    for bl in blades:
        bm = bl.to_mesh()
        Vs.append(np.asarray(bm.vert_properties)[:, :3].astype(float))
        Fs.append(np.asarray(bm.tri_verts) + off)
        off += len(Vs[-1])
    V2, F2 = np.vstack(Vs), np.vstack(Fs)
    xs = np.arange(-15, 15, cell) + cell / 2
    ys = np.arange(15, -15, -cell) - cell / 2
    X, Y = np.meshgrid(xs, ys)
    scene = carve._scene(V2, F2)
    down = carve._cast(scene, np.stack([X, Y, np.full_like(X, 100.0)], -1), np.array([0.0, 0.0, -1.0]))
    top = np.where(np.isfinite(down), 100.0 - down, np.nan)
    L, _ = carve.carve(V2, F2, top, xs, ys, cell, z_floor=-1.0, axes=[0.0, np.pi / 2, np.pi / 4, 3 * np.pi / 4],
                     side_cell=side)
    assert L.status() == m3d.Error.NoError
    # point-sampled side views used to lose 6-12% here and leave 33 pieces
    assert (L ^ fins).volume() > 0.99 * fins.volume()            # all of every blade is there
    above = L ^ m3d.Manifold.cube((40, 40, 20), center=True).translate((0, 0, 40.5))
    assert above.volume() < 3 * fins.volume()                    # only about a cell fatter
    assert len(above.decompose()) <= len(blades)                 # no loose fragments


def test_carve_keeps_sloped_roofs_where_they_are():
    """Protecting thin parts must not fatten smooth ones: a steep roof plane
    is seen at a slant by every view and still has to come out in place."""
    import manifold3d as m3d

    from hk3d import carve
    V, F = _open_box(20, 20, 30, 0.0)
    V[[5, 6], 2] = 50.0                                   # roof rises 45 degrees towards +x
    cell, side = 0.125, 0.25
    xs = np.arange(-15, 15, cell) + cell / 2
    ys = np.arange(15, -15, -cell) - cell / 2
    X, Y = np.meshgrid(xs, ys)
    scene = carve._scene(V, F)
    down = carve._cast(scene, np.stack([X, Y, np.full_like(X, 100.0)], -1), np.array([0.0, 0.0, -1.0]))
    top = np.where(np.isfinite(down), 100.0 - down, np.nan)
    L, _ = carve.carve(V, F, top, xs, ys, cell, z_floor=-1.0, axes=[0.0, np.pi / 2], side_cell=side)
    assert L.status() == m3d.Error.NoError
    # 20 x 20 x 31 below the eaves plus the wedge; a roof one cell too high
    # all over would add 1%
    assert L.volume() == pytest.approx(20 * 20 * 31 + 20 * 20 * 20 / 2, rel=0.002)
    P = np.asarray(L.to_mesh().vert_properties)[:, :3].astype(np.float64)
    roof = P[(np.abs(P[:, 0]) < 9) & (np.abs(P[:, 1]) < 9) & (P[:, 2] > 29)]
    assert np.percentile(np.abs(roof[:, 2] - (40 + roof[:, 0])), 95) < 0.03


def _carve_mesh(man, cell=0.25, ground=0.0, min_open=2.0, axes=None):
    """Carve a closed manifold given as a LandsD-like triangle soup."""
    from hk3d import carve
    me = man.to_mesh()
    V = np.asarray(me.vert_properties)[:, :3].astype(float)
    F = np.asarray(me.tri_verts)
    xs = np.arange(-25, 25, cell) + cell / 2
    ys = np.arange(25, -25, -cell) - cell / 2
    X, Y = np.meshgrid(xs, ys)
    down = carve._cast(carve._scene(V, F), np.stack([X, Y, np.full_like(X, 100.0)], -1), np.array([0.0, 0.0, -1.0]))
    top = np.where(np.isfinite(down), 100.0 - down, np.nan)
    G = np.full_like(top, ground) if np.isscalar(ground) else ground
    L, _ = carve.carve(V, F, top, xs, ys, cell, z_floor=-1.0, axes=axes or [0.0, np.pi / 2],
                       side_cell=cell, ground=G, min_open=min_open)
    return L


def test_carve_keeps_the_space_under_a_bridge():
    """A deck on two legs: the space between the legs is a real opening and
    must stay open (the side views used to fill everything under a hit down
    to the floor, here 144 m3 of it)."""
    import manifold3d as m3d
    deck = m3d.Manifold.cube((30, 6, 1.5)).translate((-15, -3, 6))
    legs = [m3d.Manifold.cube((2, 6, 6)).translate((x, -3, 0)) for x in (-15, 13)]
    bridge = m3d.Manifold.batch_boolean([deck] + legs, m3d.OpType.Add)
    gap = m3d.Manifold.cube((24, 4, 5)).translate((-12, -2, 0.5))      # between the legs, under the deck
    L = _carve_mesh(bridge)
    assert L.status() == m3d.Error.NoError
    assert (L ^ gap).volume() < 0.02 * gap.volume()
    assert (L ^ deck).volume() > 0.98 * deck.volume()
    assert all((L ^ g).volume() > 0.98 * g.volume() for g in legs)
    # printing it is optional: below the size limit, or with it off, filled as before
    assert (_carve_mesh(bridge, min_open=0.0) ^ gap).volume() > 0.98 * gap.volume()
    assert (_carve_mesh(bridge, min_open=6.0) ^ gap).volume() > 0.98 * gap.volume()


def test_carve_keeps_the_space_under_an_overhang():
    """An upper block cantilevered 8 m past a grounded one: the space under
    it is open to the side, and the building stands on the ground, so it is
    an overhang to keep, not a building that stops short."""
    import manifold3d as m3d
    low = m3d.Manifold.cube((12, 10, 6)).translate((-12, -5, 0))
    high = m3d.Manifold.cube((20, 10, 8)).translate((-12, -5, 6))
    under = m3d.Manifold.cube((7, 9, 5)).translate((0.5, -4.5, 0.5))
    L = _carve_mesh(low + high)
    assert (L ^ under).volume() < 0.02 * under.volume()
    assert (L ^ (low + high)).volume() > 0.98 * (low + high).volume()


def test_carve_still_grounds_buildings_that_stop_short():
    """A building whose mesh ends above the ground (common in LandsD, on
    slopes and flat ground alike) leaves a gap all the way under it: that is
    no opening, and the building must still reach the ground."""
    import manifold3d as m3d
    tower = m3d.Manifold.cube((20, 20, 30)).translate((-10, -10, 2.5))  # 2.5 m above flat ground
    L = _carve_mesh(tower)
    assert L.slice(1.0).area() == pytest.approx(400, rel=0.02)
    # on a slope the gap is a wedge; still filled
    xs = np.arange(-25, 25, 0.25) + 0.125
    slope = np.tile(np.clip((xs + 10) * 0.12, 0, None), (len(xs), 1))  # 0 m to 2.4 m across it
    tower = m3d.Manifold.cube((20, 20, 30)).translate((-10, -10, 2.6))
    L = _carve_mesh(tower, ground=slope)
    assert L.slice(1.0).area() == pytest.approx(400, rel=0.02)
    # standing on the uphill end, the gap under the rest is a wedge: filled too
    downhill = np.tile(np.clip((10 - xs) * 0.3, 0, None), (len(xs), 1))  # 6 m at x = -10 to 0 m at x = 10
    tower = m3d.Manifold.cube((20, 20, 30)).translate((-10, -10, 6.0))
    L = _carve_mesh(tower, ground=downhill)
    assert L.slice(3.0).area() == pytest.approx(400, rel=0.02)


def test_separate_touching_catches_float32_collisions():
    # 100.000004999 and 100.000005001 fall in different 1e-5 rounding bins but
    # become the same float32 once written to STL
    a, b = 100 + 0.5e-5 - 1e-9, 100 + 0.5e-5 + 1e-9
    V = np.array([[a, 0, 0], [101, 0, 0], [a, 1, 0], [b, 0, 0], [99, 0, 0], [b, -1, 0]], float)
    F = np.array([[0, 1, 2], [3, 4, 5]])
    assert np.unique(V.astype(np.float32), axis=0).shape[0] == 5
    W = solid.separate_touching(V, F)
    assert np.unique(W.astype(np.float32), axis=0).shape[0] == 6
    assert np.abs(W - V).max() < 0.01


def _x_facade(h=240, w=160, band=5, seed=0, mullion=1):
    """Dark noisy glass with a light X brace, a light frame and thin mullions."""
    rng = np.random.default_rng(seed)
    img = 0.25 + 0.08 * rng.standard_normal((h, w, 3))
    for k in range(mullion):     # mullions, ``mullion`` pixels wide
        img[k::8, :] += 0.15
        img[:, k::8] += 0.15
    yy, xx = np.mgrid[:h, :w]
    brace = (np.abs(yy * w / h - xx) < band) | (np.abs(yy * w / h - (w - xx)) < band)
    brace |= (xx < band) | (xx >= w - band)
    # a photo of cladding is never perfectly flat (flat colour is LandsD's fill
    # where it has no photo, and is ignored)
    img[brace] = 0.75 + 0.02 * rng.standard_normal(int(brace.sum()))[:, None]
    return np.clip(img, 0, 1).astype(np.float32), brace


def test_facade_pattern_finds_bracing_not_mullions():
    from hk3d import carve
    rgb, brace = _x_facade()
    pat = carve.facade_pattern(rgb, np.ones(brace.shape, bool), cell=0.25)
    assert (pat & brace).sum() > 0.85 * brace.sum()     # the brace is found
    assert (pat & ~brace).sum() < 0.05 * brace.sum()    # mullions and noise are not


def test_relief_cutters_recess_panels_with_straight_edges(tmp_path):
    import manifold3d as m3d
    from PIL import Image

    from hk3d import carve
    # mullions two pixels wide: the rays sample every row of them, as they do
    # on real textures; their 0.9 m grid is too fine for 0.5 m ribs
    rgb, brace = _x_facade(h=256, w=256, band=12, mullion=2)
    Image.fromarray((rgb * 255).astype(np.uint8)).save(tmp_path / "t.png")
    V, F = _open_box(20, 20, 30, 0.0)
    # planar-map each wall along its own horizontal axis; the roof is untextured
    uv = np.zeros((len(F), 3, 2))
    for i, f in enumerate(F[:8]):
        P = V[f]
        n = np.cross(P[1] - P[0], P[2] - P[0])
        ax = 1 if abs(n[0]) > abs(n[1]) else 0
        uv[i, :, 0] = (P[:, ax] + 10) / 20
        uv[i, :, 1] = P[:, 2] / 30
    tex = np.r_[np.zeros(8, int), -np.ones(2, int)]
    sampler = carve.TextureSampler(uv, tex, [str(tmp_path / "t.png")])
    axes = carve.facade_axes(V, F)
    cutters, st = carve.relief_cutters(V, F, axes, sampler, depth=0.4, width=0.5)
    assert len(cutters) == 4 and st["relief_panels"] >= 16      # four triangles per wall
    assert st["floor_lines"] == 0                               # the grid is not drawn as floors
    box = m3d.Manifold.cube((20, 20, 31)).translate((-10, -10, -1))
    cut = m3d.Manifold.batch_boolean(cutters, m3d.OpType.Add)
    L = box - cut
    assert L.status() == m3d.Error.NoError and L.genus() == 0
    lost = box.volume() - L.volume()
    # four 20 x 30 m walls, most of each recessed 0.4 m (the X and frame stay)
    assert 0.5 * 4 * 600 * 0.4 < lost < 4 * 600 * 0.4
    # vector panels: a handful of straight edges, not one step per cell
    assert cut.num_tri() < 2000
    # nothing is cut deeper than the relief or outside the walls
    P = np.asarray(L.to_mesh().vert_properties)[:, :3]
    assert np.abs(P[:, :2]).max() <= 10 + 1e-6
    inner = P[(np.abs(P[:, :2]).max(1) < 10 - 1e-3) & (P[:, 2] > 0.5) & (P[:, 2] < 29.5)]
    assert np.abs(np.abs(inner[:, :2]).max(1) - 9.6).max() < 0.02


def test_flat_texture_marks_fill_not_photo():
    from hk3d import carve
    rng = np.random.default_rng(2)
    rgb = (0.5 + 0.03 * rng.standard_normal((80, 80, 3))).astype(np.float32)   # a quiet photo
    rgb[40:, :40] = 128 / 255                                                   # LandsD's fill
    flat = carve.flat_texture(rgb, np.ones((80, 80), bool), cell=0.125)
    assert flat[44:, :36].all()                     # the fill, and the seam around it
    assert not flat[:30].any() and not flat[:, 50:].any()


def test_facade_axes_of_a_faceted_tower():
    from hk3d import carve
    # a 72-sided prism: no direction holds enough wall for the usual rule,
    # which used to give no axes at all
    k = 72
    a = np.radians(np.arange(k) * 360 / k)
    r = 20.0
    ring = np.c_[r * np.cos(a), r * np.sin(a)]
    V = np.r_[np.c_[ring, np.zeros(k)], np.c_[ring, np.full(k, 50.0)]]
    F = np.array([f for i in range(k) for f in ((i, (i + 1) % k, k + i), ((i + 1) % k, k + (i + 1) % k, k + i))])
    axes = carve.facade_axes(V, F)
    assert 1 <= len(axes) <= 4


def test_straighten_squares_ragged_piers_only():
    from hk3d import carve
    p = np.zeros((100, 60), bool)
    p[5:95, 10:18] = True
    p[np.arange(5, 95, 3), 17] = False                # ragged edge
    p[np.arange(5, 95, 4), 18] = True
    yy, xx = np.mgrid[:100, :60]
    diag = (np.abs(yy * 0.5 - xx + 10) < 3) & (xx > 30)
    p |= diag
    q = carve._straighten(p)
    assert q[5:95, 10:17].all() and not q[5:95, 19].any()
    assert (q[:, 30:] == p[:, 30:]).all()             # a slanted brace is left alone


def test_panel_room_runs_to_walls_in_front_only():
    from hk3d import carve
    # a 6 m wide recessed strip (t = 11) between two walls 1.5 m in front (t = 9.5),
    # empty sky above it: the outline to the sky keeps its margin, the walls do not
    t = np.full((60, 80), np.inf)
    t[10:, 10:70] = 9.5
    t[10:, 16:64] = 11.0
    lab = np.zeros(t.shape, int)
    lab[10:, 16:64] = 1
    ssl = ndimage_find(lab)
    room = carve._panel_room(lab, 1, ssl, t, margin=8, step=0.25)
    rows = np.nonzero(room.any(1))[0]
    cols = np.nonzero(room.any(0))[0]
    assert rows[0] >= 8                       # 1 m below the top edge
    assert cols[0] <= 3 and cols[-1] >= 48 - 4    # within two cells of the side walls


def ndimage_find(lab):
    from scipy import ndimage
    return ndimage.find_objects(lab)[0]


def test_floor_bands_follow_real_floors_only():
    from hk3d import carve
    rng = np.random.default_rng(1)
    cell, n = 0.125, 1600                                # a 200 m tower
    cnt = np.full(n, 400.0)
    # a line every 4 m, with a taller plant floor half way up that shifts the rest
    lines = np.r_[np.arange(40, 800, 32), np.arange(860, n - 40, 32)]
    sig = 2.0 + 0.5 * rng.random(n)
    for c in lines:
        sig[c:c + 3] += 6.0
    rows, info = carve.floor_bands(sig * cnt, cnt, cell, width=1.0)
    assert abs(info["floor_pitch_m"] - 4.0) < 0.2 and abs(info["floor_lines"] - len(lines)) <= 2
    found = np.array([rows[c - 2:c + 5].any() for c in lines])
    assert found.mean() > 0.95                           # both halves, despite the shift
    assert rows.mean() < 0.3                             # ribs, not slabs
    # a window grid finer than the print can show: no lines rather than some of them
    fine = 2.0 + 0.5 * rng.random(n)
    fine[::10] += 6.0
    rows, info = carve.floor_bands(fine * cnt, cnt, cell, width=1.0)
    assert rows is None and info["floor_lines"] == 0
    # reflections only: nothing repeats
    rows, info = carve.floor_bands((2.0 + rng.random(n)) * cnt, cnt, cell, width=1.0)
    assert rows is None


def test_floor_bands_see_storeys_behind_transoms():
    from hk3d import carve
    rng = np.random.default_rng(4)
    cell, n = 0.125, 1600
    cnt = np.full(n, 400.0)
    # a 3.5 m storey: a strong sill line, and a weaker transom half way up
    sill = np.arange(40, n - 40, 28)
    sig = 2.0 + 0.5 * rng.random(n)
    for c in sill:
        sig[c:c + 3] += 6.0
        sig[c + 14:c + 17] += 3.0
    rows, info = carve.floor_bands(sig * cnt, cnt, cell, width=1.0)
    assert abs(info["floor_pitch_m"] - 3.5) < 0.2 and info.get("floor_grid_m", 0) < 2.0
    assert np.array([rows[c - 2:c + 5].any() for c in sill]).mean() > 0.9     # the sills,
    assert not np.array([rows[c + 15] for c in sill]).mean() > 0.1            # not the transoms
    # equally strong lines every 1.75 m: a uniform grid, still left off
    even = 2.0 + 0.5 * rng.random(n)
    for c in np.arange(40, n - 40, 14):
        even[c:c + 3] += 6.0
    rows, info = carve.floor_bands(even * cnt, cnt, cell, width=1.0)
    assert rows is None and info["floor_lines"] == 0


def _mullion_facade(rng, pitch_cells, contrast, H=800, W=320, every=None, major=0.0):
    """Glass (0.35, noisy, with ribbon-window bands) and dark vertical lines
    3 cells wide every ``pitch_cells``; every ``every``-th one ``major``
    darker still."""
    L = 0.35 + 0.03 * rng.standard_normal((H, W))
    L[(np.arange(H) % 28) < 10] += 0.25                       # spandrels: a storey of 3.5 m
    xs = np.arange(10, W - 4, pitch_cells)
    for j, x in enumerate(xs):
        x = int(round(x))
        L[:, x:x + 3] -= contrast + (major if every and j % every == 0 else 0.0)
    return L.astype(np.float32), xs


def test_mullion_columns_draw_visible_mullions():
    from hk3d import carve
    rng = np.random.default_rng(5)
    L, xs = _mullion_facade(rng, 28.8, 0.15)                    # 3.6 m bays, clear mullions
    m, info = carve.mullion_columns(L, np.ones(L.shape, bool), 0.125, 1.0)
    assert m is not None and abs(info["mullion_pitch_m"] - 3.6) < 0.15
    assert all(m[int(round(x)) - 2:int(round(x)) + 5].any() for x in xs[1:-1])
    assert m.mean() < 0.4                                       # ribs, not the whole surface


def test_mullion_columns_skip_grids_and_hairlines_draw_major_mullions():
    from hk3d import carve
    rng = np.random.default_rng(6)
    ok = np.ones((800, 320), bool)
    # a curtain-wall grid of 1.5 m: too fine to print, all alike
    L, _ = _mullion_facade(rng, 12, 0.15)
    assert carve.mullion_columns(L, ok, 0.125, 1.0)[0] is None
    # every fourth of them heavier: those (6 m) are drawn
    L, xs = _mullion_facade(rng, 12, 0.1, every=4, major=0.15)
    m, info = carve.mullion_columns(L, ok, 0.125, 1.0)
    assert m is not None and abs(info["mullion_pitch_m"] - 6.0) < 0.2
    assert all(m[int(x) - 2:int(x) + 5].any() for x in xs[4:-4:4])
    # hairline joints across ribbon windows: regular, but too faint to draw
    L, _ = _mullion_facade(rng, 24, 0.012)
    assert carve.mullion_columns(L, ok, 0.125, 1.0)[0] is None
    # stripes over the top third only (a mirrored neighbour): not drawn
    L, _ = _mullion_facade(rng, 28.8, 0.15)
    L[300:] = _mullion_facade(rng, 28.8, 0.0)[0][300:]
    assert carve.mullion_columns(L, ok, 0.125, 1.0)[0] is None


def test_drop_grid_strips_removes_grid_keeps_brace_crossings():
    from hk3d import carve
    cell = 0.125
    p = np.zeros((800, 320), bool)                        # 100 x 40 m facade
    for r in range(40, 400, 14):                          # grid lines every 1.75 m, across
        p[r:r + 6, 10:300] = True
    yy, xx = np.mgrid[:800, :320]
    brace = ((np.abs((yy - 400) - xx) < 6) | (np.abs((yy - 400) - (320 - xx)) < 6)) & (yy >= 400)
    p |= brace
    q = carve._drop_grid_strips(p, cell, 1.75, 4.0)
    assert not q[40:390].any()                            # the grid, ends and all
    assert (q & brace).sum() > 0.97 * brace.sum()         # the X, crossing included


def test_regularise_pattern_keeps_structure_drops_reflections():
    from hk3d import carve
    h, w, cell = 400, 240, 0.25                          # a 100 x 60 m facade
    yy, xx = np.mgrid[:h, :w]
    brace = (np.abs(yy * w / h - xx) < 5) | (np.abs(yy * w / h - (w - xx)) < 5)
    blob = (yy - 100) ** 2 + (xx - 60) ** 2 < 30 ** 2     # glass mirroring something bright
    wall = (yy > 300) & (xx > 150) & (xx < 215)           # a plain wall panel
    stray = (np.abs(yy - 200) < 3) & (xx > 20) & (xx < 60)  # a 10 m streak
    grid = (yy % 8 < 2) | (xx % 6 < 2)                    # the window grid shows through glass
    busy = grid & ~brace & ~wall & ~stray
    surf = np.ones((h, w), bool)
    out = carve.regularise_pattern(brace | blob | wall | stray, surf, busy, cell, width=1.0, min_length=4.0)
    only = lambda m: m & ~ndimage_dilate(brace)           # noqa: E731  - away from the brace
    assert (out & brace).sum() > 0.9 * brace.sum()       # all but the bit under the reflection
    assert (out & only(wall)).sum() > 0.95 * only(wall).sum()
    assert (out & only(blob)).sum() < 0.05 * only(blob).sum()
    assert not (out & only(stray)).any()
    # with floor lines drawn on the facade, spandrel strips are left to them
    span = (yy % 16 < 4) & (xx > 40) & (xx < 200)
    out = carve.regularise_pattern(brace | span, surf, busy & ~span, cell, 1.0, 4.0, floor_pitch=4.0)
    assert (out & only(span)).sum() < 0.05 * only(span).sum()
    assert (out & brace & ~ndimage_dilate(span)).sum() > 0.9 * (brace & ~ndimage_dilate(span)).sum()


def ndimage_dilate(m, it=6):
    from scipy import ndimage
    return ndimage.binary_dilation(m, iterations=it)


def test_relief_cutters_draw_floor_lines_on_every_side(tmp_path):
    import manifold3d as m3d
    from PIL import Image

    from hk3d import carve
    rng = np.random.default_rng(3)
    h, w = 480, 160                                       # 60 m of glass, 8 px per metre
    img = 0.25 + 0.05 * rng.standard_normal((h, w, 3))
    img[:, ::12] += 0.1                                   # mullions
    for c in range(16, h - 8, 32):                        # a light spandrel every 4 m
        img[c:c + 5] = 0.7
    for _ in range(12):                                   # bright reflections
        y, x = rng.integers(0, h - 60), rng.integers(0, w - 40)
        img[y:y + 60, x:x + 40] += 0.3 * rng.random()
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(tmp_path / "t.png")
    V, F = _open_box(20, 20, 60, 0.0)
    uv = np.zeros((len(F), 3, 2))
    for i, f in enumerate(F[:8]):
        P = V[f]
        nrm = np.cross(P[1] - P[0], P[2] - P[0])
        ax = 1 if abs(nrm[0]) > abs(nrm[1]) else 0
        uv[i, :, 0] = (P[:, ax] + 10) / 20
        uv[i, :, 1] = P[:, 2] / 60
    tex = np.r_[np.zeros(8, int), -np.ones(2, int)]
    sampler = carve.TextureSampler(uv, tex, [str(tmp_path / "t.png")])
    cutters, st = carve.relief_cutters(V, F, carve.facade_axes(V, F), sampler, depth=0.4, width=0.5)
    assert abs(st["floor_pitch_m"] - 4.0) < 0.2 and st["floor_surfaces"] == 4
    assert 12 <= st["floor_lines"] <= 16
    assert st["relief_panels"] >= 4 * 12                  # one panel per storey per side
    box = m3d.Manifold.cube((20, 20, 61)).translate((-10, -10, -1))
    L = box - m3d.Manifold.batch_boolean(cutters, m3d.OpType.Add)
    assert L.status() == m3d.Error.NoError
    lost = box.volume() - L.volume()
    assert 0.6 * 4 * 1200 * 0.4 < lost < 0.95 * 4 * 1200 * 0.4   # the glass, not the lines



def _floor_facade_sampler(tmp_path, w_m=20.0):
    from PIL import Image

    from hk3d import carve
    rng = np.random.default_rng(3)
    h, w = 480, 160                                       # 60 m of glass, 8 px per metre
    img = 0.25 + 0.05 * rng.standard_normal((h, w, 3))
    img[:, ::12] += 0.1
    for c in range(16, h - 8, 32):                        # a light spandrel every 4 m
        img[c:c + 5] = 0.7
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(tmp_path / "t.png")
    V, F = _open_box(w_m, w_m, 60, 0.0)
    uv = np.zeros((len(F), 3, 2))
    for i, f in enumerate(F[:8]):
        P = V[f]
        nrm = np.cross(P[1] - P[0], P[2] - P[0])
        ax = 1 if abs(nrm[0]) > abs(nrm[1]) else 0
        uv[i, :, 0] = (P[:, ax] + w_m / 2) / w_m
        uv[i, :, 1] = P[:, 2] / 60
    tex = np.r_[np.zeros(8, int), -np.ones(2, int)]
    return V, F, carve.TextureSampler(uv, tex, [str(tmp_path / "t.png")])


def _debug_colours(folder):
    from PIL import Image
    px = np.concatenate([np.asarray(Image.open(p)).reshape(-1, 3) for p in sorted(folder.glob("view*_pat.png"))])
    return {name: int((px == c).all(1).sum()) for name, c in
            (("blue", (170, 205, 240)), ("black", (30, 30, 30)), ("amber", (240, 190, 90)))}


@pytest.mark.parametrize("w_m", [3.25, 20.0])
def test_relief_debug_and_counts_follow_the_real_cut(tmp_path, w_m):
    """The reported area is the recess the model really loses (the audit's
    3.25 m tower read 15% low when it came from a raster copy), and the blue
    in the debug image is that same outline.  Landmark preset: 1 m lines,
    0.75 m deep, 45-degree chamfer."""
    import manifold3d as m3d

    from hk3d import carve
    V, F, sampler = _floor_facade_sampler(tmp_path, w_m)
    depth = 0.75
    cutters, st = carve.relief_cutters(V, F, carve.facade_axes(V, F), sampler, depth=depth, width=1.0,
                                       height=1.0, chamfer=1.0, debug=str(tmp_path / "ok"))
    px = _debug_colours(tmp_path / "ok")
    assert cutters and st["floor_surfaces"] == 4 and px["black"] > 0
    h = w_m / 2
    box = m3d.Manifold.cube((w_m, w_m, 61)).translate((-h, -h, -1))
    lost = box.volume() - (box - m3d.Manifold.batch_boolean(cutters, m3d.OpType.Add)).volume()
    assert abs(lost / depth - st["relief_area_m2"]) < 0.01 * st["relief_area_m2"]
    assert abs(px["blue"] * 0.125 ** 2 - st["relief_area_m2"]) < 0.03 * st["relief_area_m2"]


def test_relief_debug_shows_nothing_cut_when_every_recess_is_too_narrow(tmp_path, monkeypatch):
    from hk3d import carve
    V, F, sampler = _floor_facade_sampler(tmp_path)
    # every panel turns into a 5 cm sliver, which the printable-width filter removes
    monkeypatch.setattr(carve, "_runs_to_rects", lambda m, u0, z0, cell:
                        [[(u0, z0), (u0 + 0.05, z0), (u0 + 0.05, z0 + 0.05), (u0, z0 + 0.05)]])
    cutters, st = carve.relief_cutters(V, F, carve.facade_axes(V, F), sampler, depth=0.4, width=0.5,
                                       debug=str(tmp_path / "none"))
    px = _debug_colours(tmp_path / "none")
    assert not cutters
    assert st["relief_panels"] == 0 and st["relief_area_m2"] == 0 and st["floor_surfaces"] == 0
    assert px["blue"] == 0 and px["black"] == 0       # nothing shown as cut
    assert px["amber"] > 0                            # but the rejected candidates are visible


def test_cache_key_changes_with_the_pipeline(monkeypatch):
    from hk3d import pipeline
    p = pipeline.ModelParams(e=834691.0, n=815600.0)
    k = p.key()
    assert k == pipeline.ModelParams(e=834691.0, n=815600.0).key()
    monkeypatch.setattr(pipeline, "PIPELINE_VERSION", pipeline.PIPELINE_VERSION + 1)
    assert p.key() != k                                   # old cached models are not served again


# ------------------------------------------------------------------ printing without supports
def _box(x0, y0, z0, x1, y1, z1):
    import manifold3d as m3d
    return m3d.Manifold.cube((x1 - x0, y1 - y0, z1 - z0)).translate((x0, y0, z0))


def _support(man, bridge=10.0):
    from hk3d import printable
    return printable.support_fill(man, None, 3.0, 0.1, 0.2, bridge, 0.4)


def test_cantilevers_get_a_45_degree_bracket_that_needs_nothing_more():
    # a 2 mm pillar carrying a slab that sticks out 4 mm on each side
    T = _box(-1, -1, 2, 1, 1, 10) + _box(-5, -1, 10, 5, 1, 11)
    fill, st = _support(T)
    assert fill is not None
    # two 45 degree brackets: 2 x (4 x 4 / 2) x 2 mm wide = 32 mm3
    assert 25 < st["support_mm3"] < 40
    # the bracket stays as wide as the slab (no keel hanging under it)
    assert abs(fill.bounding_box()[1] + 1.0) < 0.3 and abs(fill.bounding_box()[4] - 1.0) < 0.3
    _, again = _support(T + fill)
    assert again["support_mm3"] < 0.5


def test_spans_a_printer_can_bridge_stay_open_wider_ones_become_pointed_arches():
    arch = lambda w: _box(-w / 2 - 2, -1, 2, -w / 2, 1, 10) + _box(w / 2, -1, 2, w / 2 + 2, 1, 10) \
        + _box(-w / 2 - 2, -1, 10, w / 2 + 2, 1, 11)
    fill, st = _support(arch(6))
    assert fill is None and st["bridged_mm2"] > 10
    fill, st = _support(arch(14))
    # two haunches meeting in a point: 14 x 7 / 2 x 2 mm = 98 mm3
    assert fill is not None and 80 < st["support_mm3"] < 110
    _, again = _support(arch(14) + fill)
    assert again["support_mm3"] < 0.5
    assert _support(arch(14), bridge=20)[0] is None
    # floating with nothing beside it: carried straight down
    fill, st = _support(_box(-3, -3, 8, 3, 3, 9))
    assert fill is not None and _support(_box(-3, -3, 8, 3, 3, 9) + fill)[1]["support_mm3"] < 0.5


def test_bridge_runs_are_found_in_any_direction():
    from hk3d.printable import bridgeable
    n = 80
    yy, xx = np.mgrid[:n, :n]
    u = (xx - 40 + yy - 40) / np.sqrt(2)
    v = (xx - 40 - (yy - 40)) / np.sqrt(2)
    deck = (np.abs(v) <= 4) & (np.abs(u) <= 15)                      # 30 cells long, diagonal
    piers = (np.abs(v) <= 4) & (np.abs(u) > 15) & (np.abs(u) <= 19)
    assert bridgeable(deck, piers, 40, directions=2)[deck].mean() == 0
    assert bridgeable(deck, piers, 40, directions=12)[deck].mean() > 0.7
    assert bridgeable(deck, piers, 20, directions=12)[deck].mean() == 0
    # open on one side: a cantilever, never a bridge
    assert bridgeable(deck, piers & (u < 0), 40, directions=12)[deck].mean() == 0


def test_footbridge_keeps_the_road_under_it_when_printable():
    from hk3d import pipeline
    cell = 0.25                                     # m; 1:2500 -> 0.1 mm cells
    n = 200
    ys, xs = np.mgrid[:n, :n]
    road = np.full((n, n), 5.0, np.float32)
    bridge = np.full((n, n), np.nan, np.float32)
    deck = (ys >= 80) & (ys < 120) & (xs >= 50) & (xs < 150)         # 10 m wide, 25 m long
    ends = (ys >= 80) & (ys < 120) & (((xs >= 40) & (xs < 50)) | ((xs >= 150) & (xs < 160)))
    bridge[deck | ends] = 13.0
    U = np.full((n, n), np.nan, np.float32)
    U[deck] = 11.5                                  # 6.5 m clear, 1.5 m deck
    U[ends] = 5.3                                   # abutments stand on the road
    H = {"infrastructure": road, "bridge": bridge}
    order = ["infrastructure", "bridge"]
    top = np.fmax(road, bridge)
    s = 0.4
    to_mm = lambda h: (5.0 + np.clip(h - 5.0, 0, None) * s).astype(np.float32)
    Z = to_mm(top)
    cls = np.ones((n, n), np.uint8)
    inside = np.ones((n, n), bool)

    def run(**kw):
        p = pipeline.ModelParams(e=0, n=0, radius_m=40, diameter_mm=32, layer_mm=0.2, **kw)
        return pipeline._bridge_decks(None, None, H, order, top, Z, cls, to_mm, p, cell * s, inside, 0.15, U=U)

    Zlow, _, tb, st = run(bridge_mm=12.0)          # a 10 mm span: bridged
    assert tb is not None and st["bridge_open_m2"] > 0.8 * deck.sum() * cell * cell
    assert np.allclose(Zlow[100, 60:140], Z[0, 0])  # the road shows under the deck
    assert np.all(tb[1][100, 60:140] >= Z[0, 0] + 1.0) and np.all(tb[0][100, 60:140] > tb[1][100, 60:140] + 0.5)
    _, _, tb, st = run(bridge_mm=5.0)              # too long to bridge: filled as before
    assert tb is None and st["bridge_open_m2"] == 0
    _, _, tb, st = run(bridge_mm=5.0, support_free=False)   # printing with supports: open anyway
    assert tb is not None
    _, _, tb, st = run(bridge_mm=12.0, openings_mm=3.0)     # 2.6 mm clear is less than asked for
    assert tb is None
    # the outline cuts the deck: the far abutment is cut away with it, so the
    # deck would end in the air at the rim; it is filled instead
    inside[:, 130:] = False
    _, _, tb, st = run(bridge_mm=12.0)
    assert tb is None and st["bridge_open_m2"] == 0


def _city(n=300, cell=1.0, tower=None, seed=1):
    from hk3d import solid as _solid

    xs = (-n / 2 + np.arange(n) + 0.5) * cell
    ys = xs[::-1].copy()
    X, Y = np.meshgrid(xs, ys)
    R = n * cell / 2 - 5
    inside = np.hypot(X, Y) <= R
    Z = np.full(X.shape, 5.0, np.float32)
    rng = np.random.default_rng(seed)
    for _ in range(40):
        cx, cy = rng.uniform(-R + 20, R - 20, 2)
        w, h = rng.uniform(8, 30, 2)
        Z[(abs(X - cx) < w / 2) & (abs(Y - cy) < h / 2)] = 5 + rng.uniform(10, 60)
    if tower:
        Z[(abs(X - 30) < 10) & (abs(Y + 20) < 10)] = tower
    m = _solid.grid_solid(xs.astype(np.float32), ys.astype(np.float32), Z) ^ _solid.shape_prism("circle", 2 * R, 500)
    return m, Z, inside, xs, ys, (-R, -R, R, R)


def test_pieces_fit_the_bed_and_seams_go_round_buildings():
    from hk3d import tiles

    m, Z, inside, xs, ys, b = _city()
    opts = tiles.TileOptions(bed=(180, 180, 180))
    pieces, st, plan, pins = tiles.split(m, Z, inside, xs, ys, b, 5.0, opts, ["T"])
    assert st["grid"] == "2x2" and st["fits"] and st["pieces"] == 4
    for p in pieces:
        bb = p.man.bounding_box()
        assert p.man.status() == m3d.Error.NoError
        assert bb[3] - bb[0] <= 180 - 2 * opts.margin + 1e-6 and bb[4] - bb[1] <= 180 - 2 * opts.margin + 1e-6
    # nothing lost but the pin holes and the names underneath
    total = sum(p.man.volume() for p in pieces)
    assert 0 < m.volume() - total < 0.01 * m.volume()
    # the seams cross far fewer buildings than straight cuts through the middle
    built = (Z > 6) & inside
    on_seam = sum(built[np.arange(len(ys)), p].sum() for p in plan.xpaths) + \
        sum(built[p, np.arange(len(xs))].sum() for p in plan.ypaths)
    straight = built[:, len(xs) // 2].sum() + built[len(ys) // 2, :].sum()
    assert on_seam < 0.5 * straight
    # pins sit on the seams, across them
    assert len(pins) >= 4
    for x, y, ux, uy in pins:
        assert abs(np.hypot(ux, uy) - 1) < 1e-6


def test_a_tower_taller_than_the_printer_is_cut_into_levels_with_pins():
    from hk3d import tiles

    m, Z, inside, xs, ys, b = _city(tower=400)
    opts = tiles.TileOptions(bed=(180, 180, 180))
    pieces, st, plan, pins = tiles.split(m, Z, inside, xs, ys, b, 5.0, opts, [])
    assert st["fits"] and st["levels"] == 3
    upper = [p for p in pieces if p.level > 1]
    assert [p.name for p in upper] == ["B2-2", "B2-3"]
    # two cuts through a 20 mm tower: each gets pin holes
    assert st["pins"] - len(pins) >= 2
    for p in upper:
        assert p.man.status() == m3d.Error.NoError and p.z1 - p.z0 <= 180


def test_pin_hole_runs_across_the_seam_and_peaks_upward():
    from hk3d import tiles

    h = tiles.pin_holes([(10.0, 20.0, 1.0, 0.0)], z=3.0, d=2.0, depth=6.0)
    bb = h.bounding_box()
    assert np.allclose([bb[0], bb[3]], [4.0, 16.0], atol=0.01)       # along x, 6 mm each side
    assert np.allclose([bb[1], bb[4]], [19.0, 21.0], atol=0.01)
    assert np.isclose(bb[2], 2.0, atol=0.01) and np.isclose(bb[5], 3.0 + np.sqrt(2), atol=0.01)


def test_tiles_needed_follow_the_bed():
    from hk3d import tiles
    from hk3d.pipeline import ModelParams

    assert tiles.counts((600, 600), (256, 256), 5) == (3, 3)
    assert tiles.counts((240, 100), (256, 256), 5) == (1, 1)
    assert ModelParams(e=0, n=0, diameter_mm=600).tiles_across() == (3, 3)
    assert ModelParams(e=0, n=0, diameter_mm=600, tile=False).tiles_across() == (1, 1)
    assert ModelParams(e=0, n=0, diameter_mm=80).tiles_across() == (1, 1)
    assert tiles.tile_name(0, 0) == "A1" and tiles.tile_name(27, 2, 2) == "AB3-2"


def test_an_overhang_cut_off_by_a_seam_goes_with_the_piece_holding_it_up():
    from hk3d import tiles

    n, cell = 200, 1.0
    xs = (-n / 2 + np.arange(n) + 0.5) * cell
    ys = xs[::-1].copy()
    X, Y = np.meshgrid(xs, ys)
    inside = (np.abs(X) <= 95) & (np.abs(Y) <= 95)
    base = m3d.Manifold.cube((190, 190, 5)).translate((-95, -95, 0))
    column = m3d.Manifold.cube((8, 150, 30)).translate((-20, -75, 5))
    deck = m3d.Manifold.cube((35, 150, 5)).translate((-20, -75, 30))      # reaches over the seam at x ~ 0
    model = base + column + deck
    Z = np.where(inside, 5.0, 0.0)
    Z[(X > -20) & (X < 15) & (np.abs(Y) < 75)] = 35.0
    pieces, st, plan, pins = tiles.split(model, Z, inside, xs, ys, (-95, -95, 95, 95), 5.0,
                                         tiles.TileOptions(bed=(120, 260, 100), pin_mm=0), None)
    assert st["grid"] == "2x1" and st["reattached"] >= 1 and st["floating"] == 0
    west = next(p for p in pieces if p.name == "A1")
    assert west.man.bounding_box()[3] > 14.9                             # the whole deck is on the west piece
    assert abs(sum(p.man.volume() for p in pieces) - model.volume()) < 1.0


def test_a_loose_fragment_with_a_pointed_bottom_is_held_up():
    from hk3d import printable

    # an upside-down pyramid standing on nothing: no face looks down by more than 45 degrees
    pyr = m3d.Manifold.cylinder(6.0, 0.01, 3.0, 4).translate((0, 0, 10))
    frag = pyr + m3d.Manifold.cube((20, 20, 1)).translate((-10, -10, 0))  # beside a base plate
    sup, st = printable.support_fill(frag, None, 1.0, 0.2, 0.2, 10.0)
    assert sup is not None and st["support_mm3"] > 0
    assert (frag + sup).decompose().__len__() == 1


def test_a_loose_fragment_beside_a_wall_is_carried_down_not_left_hanging():
    from hk3d import printable

    wall = m3d.Manifold.cube((2, 10, 20)).translate((-1, -5, 0))
    frag = m3d.Manifold.cube((1, 1, 1)).translate((5, 0, 12))           # 4 mm from the wall, in the air
    part = wall + frag + m3d.Manifold.cube((20, 20, 1)).translate((-10, -10, -1))   # on a base plate
    sup, st = printable.support_fill(part, None, 0.0, 0.2, 0.2, 10.0)
    assert sup is not None
    assert len((part + sup).decompose()) == 1                            # held up all the way down


def _synthetic_area(tmp_path, monkeypatch, w=6.0, h=20.0):
    """One open-box building of ``w`` x ``w`` x ``h`` m standing at (1000, 1000)
    on a sheet of its own, served to the pipeline as if it were LandsD data."""
    from hk3d import raster
    from hk3d.catalog import Catalog, Entry

    V, F = _open_box(w, w, h, 0.0)
    V[:, 0] += 1000.0
    V[:, 1] += 1000.0
    V[:, 2] += 5.0
    f = tmp_path / "data" / "11-SW-14A" / "BUILDING" / "B1" / "B1.fbx"
    f.parent.mkdir(parents=True)
    f.write_bytes(b"synthetic")
    monkeypatch.setattr(raster, "load_meshes", lambda path: [("b", V.copy(), F.copy())])
    monkeypatch.setattr(raster, "_MESH_CACHE", raster.MeshCache())
    import hk3d.pipeline as pl
    monkeypatch.setattr(pl, "_sheet_mask", lambda catalog, grid: np.ones((grid.n, grid.n), bool))
    st = f.stat()
    monkeypatch.setattr(Catalog, "coverage_fraction", lambda self, *a: 1.0)
    return Catalog(root=tmp_path / "data", entries=[Entry(
        path="11-SW-14A/BUILDING/B1/B1.fbx", sheet="11-SW-14A", folder="BUILDING", group="building",
        bbox=[V[:, 0].min(), V[:, 1].min(), 5.0, V[:, 0].max(), V[:, 1].max(), 5.0 + h],
        ntri=len(F), size=st.st_size, mtime=st.st_mtime)])


@pytest.mark.parametrize("precise", [False, True])
def test_building_only_area_builds_with_or_without_precise_carving(tmp_path, monkeypatch, precise):
    import trimesh

    from hk3d.pipeline import ModelParams, build

    cat = _synthetic_area(tmp_path, monkeypatch)
    p = ModelParams(e=1000, n=1000, radius_m=10, diameter_mm=20, detail_mm=0.2, groups=["building"],
                    precise_landmark=precise, engrave=False, tile=False, support_free=False)
    res = build(p, cat, tmp_path / "out", tmp_path / "cache")
    tm = trimesh.load(tmp_path / "out" / res["files"]["stl"])
    assert tm.is_watertight
    # the tower (5-25 mPD) on sea level: 25 m at 1 mm/m above the base
    assert abs(tm.bounds[1][2] - (p.base_mm + 25.0)) < 0.3
    assert abs(res["stats"]["elev_max_m"] - 25.0) < 0.3     # carved or not, the report agrees


def test_tall_landmark_is_not_clipped(tmp_path, monkeypatch):
    import trimesh

    from hk3d.pipeline import ModelParams, build

    cat = _synthetic_area(tmp_path, monkeypatch, w=6.0, h=500.0)
    p = ModelParams(e=1000, n=1000, radius_m=10, diameter_mm=10, base_mm=3, detail_mm=0.2,
                    groups=["building"], precise_landmark=True, engrave=False, tile=False,
                    support_free=False)
    p.validate()
    res = build(p, cat, tmp_path / "out", tmp_path / "cache")
    tm = trimesh.load(tmp_path / "out" / res["files"]["stl"])
    assert tm.is_watertight
    assert abs(tm.bounds[1][2] - (3.0 + 505.0 * 0.5)) < 0.5      # 5-505 mPD at 0.5 mm per m


@pytest.mark.parametrize("stack,wing", [(True, False), (False, False), (True, True)])
def test_tower_on_its_podium_keeps_the_podium_under_it(tmp_path, monkeypatch, stack, wing):
    """A tower file standing on a podium file hides most of the podium from
    above.  Carved one file at a time, the podium kept only the strip showing
    around the tower, and the tower stood on thin piers (V3, 11-NE-17B).
    ``wing``: the podium file also has a wing taller than the tower's foot
    beside it (as the mall round ICC's foot, 11-NW-24C): it still fills in
    under the tower, up to the tower's foot."""
    import trimesh

    import hk3d.pipeline as pl
    from hk3d import raster
    from hk3d.catalog import Catalog, Entry
    from hk3d.pipeline import ModelParams, build

    flat = (np.array([[-40, -40, 0], [40, -40, 0], [40, 40, 0], [-40, 40, 0]], float),
            np.array([[0, 1, 2], [0, 2, 3]]))
    # as in V3, the tower's lowest storey is a narrow core and the floors
    # above spread out: carved alone, the core is what continues downwards
    Vt, Ft = _open_box(8.0, 8.0, 18.0, 0.0)
    Vt[:, 2] += 2.0
    Vc, Fc = _open_box(2.0, 2.0, 2.0, 0.0)
    tower = (np.vstack([Vt, Vc]), np.vstack([Ft, Fc + len(Vt)]))
    pod = _open_box(10.0, 10.0, 10.0, 0.0)
    if wing:                                              # 4 m x 10 m, 20 m tall, against the east side
        Vw, Fw = _open_box(4.0, 10.0, 20.0, 0.0)
        Vw[:, 0] += 7.0
        pod = (np.vstack([pod[0], Vw]), np.vstack([pod[1], Fw + len(pod[0])]))
    parts = {"P": pod, "T": tower, "G": flat}
    z0 = {"P": 5.0, "T": 15.0, "G": 5.0}
    dx = {"P": 15.0, "T": 15.0, "G": 0.0}               # off the centre: carved as one of "all"
    meshes, entries = {}, []
    for k, (V, F) in parts.items():
        V = V.copy()
        V[:, 0] += 1000.0 + dx[k]
        V[:, 1] += 1000.0
        V[:, 2] += z0[k]
        grp, folder = ("terrain", "TERRAIN") if k == "G" else ("building", "BUILDING")
        rel = f"11-SW-14A/{folder}/{k}/{k}.fbx"
        f = tmp_path / "data" / rel
        f.parent.mkdir(parents=True)
        f.write_bytes(b"synthetic")
        meshes[str(f)] = (V, F)
        st = f.stat()
        entries.append(Entry(path=rel, sheet="11-SW-14A", folder=folder, group=grp,
                             bbox=[V[:, 0].min(), V[:, 1].min(), V[:, 2].min(), V[:, 0].max(), V[:, 1].max(),
                                   V[:, 2].max()],
                             ntri=len(F), size=st.st_size, mtime=st.st_mtime))
    monkeypatch.setattr(raster, "load_meshes", lambda path: [("b", *map(np.copy, meshes[str(path)]))])
    monkeypatch.setattr(raster, "_MESH_CACHE", raster.MeshCache())
    monkeypatch.setattr(pl, "_sheet_mask", lambda catalog, grid: np.ones((grid.n, grid.n), bool))
    monkeypatch.setattr(Catalog, "coverage_fraction", lambda self, *a: 1.0)
    if not stack:                                         # the old behaviour, to show what this guards
        monkeypatch.setattr(pl, "_parts_below", lambda *a, **k: [])
        # (filling what cannot be seen would also hide it)
        monkeypatch.setattr(pl.printable, "fill_hollows", lambda *a, **k: (None, {}))
    cat = Catalog(root=tmp_path / "data", entries=entries)
    p = ModelParams(e=1000, n=1000, radius_m=25, diameter_mm=50, base_mm=2, detail_mm=0.2,
                    groups=["terrain", "building"], precise_landmark=True, precise_all=True,
                    engrave=False, tile=False, support_free=False, openings_mm=1.0)
    res = build(p, cat, tmp_path / "out", tmp_path / "cache")
    tm = trimesh.load(tmp_path / "out" / res["files"]["stl"])
    assert tm.is_watertight
    # halfway up the podium (10 mPD): the whole 10 m x 10 m podium, not a 1 m ring around the tower
    z = 2.0 + (10.0 - res["stats"]["elev_min_m"]) * 1.0
    import manifold3d as m3d
    man = m3d.Manifold(m3d.Mesh(np.asarray(tm.vertices, np.float32), np.asarray(tm.faces, np.uint32)))
    sec = man.slice(z)
    assert sec.bounds()[0] > 5.0                          # only the podium is cut at this height
    pod = (sec ^ m3d.CrossSection.square((10.0, 10.0), center=True).translate((15.0, 0.0))).area()
    if stack:
        assert pod > 0.9 * 100.0
        # and no higher than the tower's foot under it: the tower's own floors
        # (15 mPD up) are where they were, the wing's 25 mPD roof is not there
        assert man.slice(2.0 + (16.0 - res["stats"]["elev_min_m"])).area() < 0.9 * 100.0 + (40.0 if wing else 0.0)
        assert res["stats"]["landmark"].get("stacked_groups") == 1
    else:
        assert pod < 0.8 * 100.0                          # a third or more of it gone


def test_model_key_follows_the_input_files(tmp_path, monkeypatch):
    from hk3d import pipeline

    cat = _synthetic_area(tmp_path, monkeypatch)
    p = pipeline.ModelParams(e=1000, n=1000, radius_m=10, precise_landmark=True, facade_relief_mm=0.3)
    k = p.key(pipeline.input_signature(cat, p))
    assert k == p.key(pipeline.input_signature(cat, p))
    # a downloaded / rescanned file (new size or time) gives a new key
    e = cat.entries[0]
    e.size += 1
    k2 = p.key(pipeline.input_signature(cat, p))
    assert k2 != k
    # so does a texture replaced beside it, when the relief reads textures
    img = tmp_path / "data" / e.path.rsplit("/", 1)[0] / "B1_001.jpg"
    img.write_bytes(b"one")
    k3 = p.key(pipeline.input_signature(cat, p))
    assert k3 != k2
    img.write_bytes(b"three")
    assert p.key(pipeline.input_signature(cat, p)) != k3
    # files outside the area do not matter
    from hk3d.catalog import Entry
    before = p.key(pipeline.input_signature(cat, p))
    cat.entries.append(Entry(path="elsewhere.fbx", sheet="11-SW-14A", folder="BUILDING", group="building",
                             bbox=[5000, 5000, 0, 5010, 5010, 10], ntri=1, size=1, mtime=0))
    assert p.key(pipeline.input_signature(cat, p)) == before


def test_mesh_cache_rereads_a_replaced_file(tmp_path, monkeypatch):
    import os

    from hk3d import raster

    f = tmp_path / "a.fbx"
    f.write_bytes(b"10")
    monkeypatch.setattr(raster, "load_meshes", lambda path: [
        ("m", np.array([[0, 0, float(open(path).read())]] * 3), np.array([[0, 1, 2]]))])
    c = raster.MeshCache()
    assert c.get(str(f))[0][0][0, 2] == 10
    f.write_bytes(b"200")
    os.utime(f, ns=(f.stat().st_atime_ns, f.stat().st_mtime_ns + 10**9))
    assert c.get(str(f))[0][0][0, 2] == 200
    assert len(c._d) == 1                                 # the old version is dropped


def test_server_builds_again_after_the_data_changes(tmp_path, monkeypatch):
    """The audit's case: build, download + rescan new data, same request."""
    import json
    import time

    from hk3d import server

    cat = _synthetic_area(tmp_path, monkeypatch)
    built = []

    class Holder:
        scanning, scan_progress = False, (0, 0)

        def __init__(self, *a):
            pass

        def get(self):
            return cat

        def peek(self):
            return cat

        def rescan(self):                 # a download replaced the building file
            cat.entries[0].size += 100
            cat.entries[0].mtime += 1.0
            return cat

    def fake_build(params, catalog, out, cache_dir, prog=None):
        built.append(catalog.entries[0].size)
        out.mkdir(parents=True, exist_ok=True)
        res = {"version": catalog.entries[0].size}
        (out / "result.json").write_text(json.dumps(res))
        return res

    monkeypatch.setattr(server, "CatalogHolder", Holder)
    monkeypatch.setattr(server, "build", fake_build)
    monkeypatch.setattr(server.download, "download_sheet", lambda *a, **k: None)
    app = server.create_app(tmp_path / "data", tmp_path / "cache")
    route = {(r.path, m): r.endpoint for r in app.routes for m in getattr(r, "methods", [])}
    jobs_post, job_get = route[("/api/jobs", "POST")], route[("/api/jobs/{job_id}", "GET")]

    def run(make, req):
        j = make(req)
        for _ in range(500):
            v = job_get(j["id"])
            if v["state"] in ("done", "error"):
                return v
            time.sleep(0.01)
        raise AssertionError("job did not finish")

    req = server.JobRequest(e=1000, n=1000, radius_m=10, diameter_mm=20)
    first = run(jobs_post, req)
    again = run(jobs_post, req)
    assert again["message"] == "Done (cached)" and again["result"] == first["result"] and len(built) == 1
    run(route[("/api/downloads", "POST")], server.DownloadRequest(sheets=["11-SW-14A"]))
    after = run(jobs_post, req)
    assert len(built) == 2 and after["result"] != first["result"] and after["key"] != first["key"]


def test_server_queues_a_model_requested_during_the_first_scan(tmp_path, monkeypatch):
    """A request before the first catalogue scan has finished waits for it."""
    import json
    import threading
    import time

    from hk3d import server

    cat = _synthetic_area(tmp_path, monkeypatch)
    scanned = threading.Event()

    class Holder:
        scanning, scan_progress = True, (0, 0)

        def __init__(self, *a):
            pass

        def get(self):
            scanned.wait(5)
            return cat

        def peek(self):
            return cat if scanned.is_set() else None

    def fake_build(params, catalog, out, cache_dir, prog=None):
        out.mkdir(parents=True, exist_ok=True)
        (out / "result.json").write_text(json.dumps({"ok": 1}))
        return {"ok": 1}

    monkeypatch.setattr(server, "CatalogHolder", Holder)
    monkeypatch.setattr(server, "build", fake_build)
    app = server.create_app(tmp_path / "data", tmp_path / "cache")
    route = {(r.path, m): r.endpoint for r in app.routes for m in getattr(r, "methods", [])}
    j = route[("/api/jobs", "POST")](server.JobRequest(e=1000, n=1000, radius_m=10, diameter_mm=20))
    assert j["key"] is None                           # no catalogue yet, so no key
    time.sleep(0.05)
    scanned.set()
    for _ in range(500):
        v = route[("/api/jobs/{job_id}", "GET")](j["id"])
        if v["state"] in ("done", "error"):
            break
        time.sleep(0.01)
    assert v["state"] == "done", v.get("error")
    assert v["key"] and v["result"] == {"ok": 1}


def test_height_map_cache_sees_a_rewrite_within_the_same_second(tmp_path, monkeypatch):
    """Same size, same second, new content, catalogue rescanned: the model key,
    the meshes and the cached height map must all move on together."""
    import os

    from hk3d import pipeline, raster

    cat = _synthetic_area(tmp_path, monkeypatch)
    f = cat.root / cat.entries[0].path
    V, F = _open_box(6, 6, 1, 0.0)
    V[:, :2] += 1000.0

    def load(path):                                      # roof height is the file's content
        W = V.copy()
        W[:, 2] *= float(open(path).read())
        return [("b", W, F.copy())]

    monkeypatch.setattr(raster, "load_meshes", load)
    grid = raster.make_grid(1000, 1000, 10, 0.5)

    def top(t):
        f.write_bytes(t)
        os.utime(f, ns=(0, int(t == b"20") * 800_000_000 + 1000 * 10**9 + 100_000_000))
        st = f.stat()
        cat.entries[0].size, cat.entries[0].mtime = st.st_size, st.st_mtime   # the rescan
        p = pipeline.ModelParams(e=1000, n=1000, radius_m=10)
        H = raster.raster_groups(cat, grid, ["building"], tmp_path / "cache")["building"]
        return pipeline.input_signature(cat, p), float(np.nanmax(H))

    k1, h1 = top(b"10")                                 # mtime 1000.1 s
    k2, h2 = top(b"20")                                 # mtime 1000.9 s, same size
    assert k1 != k2
    assert abs(h1 - 10) < 0.01 and abs(h2 - 20) < 0.01


def test_roads_are_recessed_with_a_tidy_outline():
    """Roads sink one layer below the pavement; the outline drops lanes
    narrower than a printed line and specks, fills car-sized holes with
    ground only, and leaves bridges and buildings alone."""
    from hk3d import pipeline as P

    n, cell_mm = 200, 0.05                                # 0.4 mm line = 8 cells
    top = np.full((n, n), 10.0, np.float32)
    inf = np.full((n, n), np.nan, np.float32)
    inf[80:120, :] = 10.0                                 # a 2 mm wide road across
    inf[95:105, 90:96] = np.nan                           # a car on it: ground there
    inf[20:23, :] = 10.0                                  # a lane 0.15 mm wide: unprintable
    inf[160:163, 40:43] = 10.0                            # a speck of road
    bridge = np.full((n, n), np.nan, np.float32)
    bridge[80:120, 150:160] = 10.0                        # a deck level with the road
    bld = np.full((n, n), np.nan, np.float32)
    bld[60:80, 60:80] = 30.0
    top = np.fmax(top, np.nan_to_num(bld, nan=0))
    H = {"ground": np.full((n, n), 10.0, np.float32), "infrastructure": inf, "bridge": bridge, "building": bld}
    cls = np.zeros((n, n), np.uint8)
    cls[np.isfinite(inf)] = P.CLASS_OF_GROUP["infrastructure"]
    cls[np.isfinite(bridge)] = P.CLASS_OF_GROUP["bridge"]
    cls[np.isfinite(bld)] = P.CLASS_OF_GROUP["building"]
    Z = (3.0 + (top - 10.0) * 0.4).astype(np.float32)
    order = ["ground", "infrastructure", "bridge", "building"]
    Z2, cls2, st = P._recess_roads(Z, cls, H, top, order, np.ones((n, n), bool), 0.15, cell_mm, 0.2, 0.4)
    d = Z - Z2
    assert abs(d[100, 20] - 0.2) < 1e-6 and abs(d[100, 92] - 0.2) < 1e-6   # road, and the car's hole
    assert d[21, 100] == 0 and d[161, 41] == 0                           # lane and speck stay
    assert d[100, 155] == 0                                              # the bridge deck stays
    assert (d[60:80, 60:80] == 0).all()                                  # the building stays
    assert d[50, 100] == 0                                               # pavement stays
    # colours follow the recess exactly
    road_cls = cls2 == P.CLASS_OF_GROUP["infrastructure"]
    deck = np.isfinite(bridge)
    assert ((d > 0) == (road_cls & ~deck)).all()
    assert st["road_recess_mm"] == 0.2 and st["road_dropped_cells"] > 0 and st["road_filled_cells"] >= 60


@pytest.mark.parametrize("clip", [True, False])
def test_support_meets_the_rim_without_a_comb(clip):
    """Support under an overhang that reaches the outline ends on the outline
    itself, not on its 0.2 mm cells (which cut by the rim leave a comb of
    notches along the edge of the print)."""
    import manifold3d as m3d

    from hk3d import printable, solid
    R = 10.0
    disc = m3d.Manifold.cylinder(1.0, R, R, 720).translate((0, 0, 4.0))       # overhang, edge on the rim
    column = m3d.Manifold.cylinder(3.0, 3.0, 3.0, 64).translate((0, 0, 1.0))  # what it leans on
    outline = solid.shape_prism("circle", 2 * R, 2.0).slice(0.0)
    sup, _ = printable.support_fill(disc + column, None, 1.0, 0.2, 0.2, max_bridge_mm=2.0,
                                    outline=outline if clip else None)
    assert sup is not None
    sup = sup ^ solid.shape_prism("circle", 2 * R, 20.0)                      # the model's outline cut
    pts = np.concatenate(sup.slice(3.9).to_polygons())
    ang = np.arctan2(pts[:, 1], pts[:, 0])
    r = np.hypot(pts[:, 0], pts[:, 1])
    # outermost support radius in each 2-degree sector
    sec = ((ang + np.pi) / np.radians(2)).astype(int)
    rmax = np.array([r[sec == k].max() for k in np.unique(sec)])
    if clip:
        assert rmax.min() > R - 0.01                 # the rim all round
    else:
        assert rmax.min() < R - 0.05                 # the cell steps show (the old behaviour)


def _mast_truss(bar_w: float, ang_deg: float = 15.0):
    """Two masts 6 mm apart (print mm, base top at 1.0) joined near the top
    by a bar ``bar_w`` wide in plan and 0.4 mm deep, along ``ang_deg``."""
    import manifold3d as m3d

    a = np.radians(ang_deg)
    d = np.array([np.cos(a), np.sin(a)])
    mast = m3d.Manifold.cube((0.8, 0.8, 12.0), center=True)
    m1 = mast.translate((*(-3.4 * d), 7.0))
    m2 = mast.translate((*(3.4 * d), 7.0))
    bar = m3d.Manifold.cube((6.8, bar_w, 0.4), center=True).rotate((0, 0, ang_deg)).translate((0, 0, 10.2))
    return m1 + m2 + bar


def test_parts_too_thin_to_print_are_widened_corners_and_specks_are_not():
    """A truss member thinner than a printed line in plan (a mast truss seen
    edge-on) is widened to the target width; a sharp facade corner and a
    speck too small to print are left alone."""
    import manifold3d as m3d

    from hk3d import printable
    truss = _mast_truss(0.2)
    # a 35 degree wedge (a facade corner) and a 0.15 mm speck on a stub
    wedge = m3d.Manifold.extrude(m3d.CrossSection([np.array(
        [[10.0, 0.0], [20.0, -10 * np.tan(np.radians(17.5))], [20.0, 10 * np.tan(np.radians(17.5))]])]), 6.0)
    wedge = wedge.translate((0, 0, 1.0))
    speck = m3d.Manifold.cube((0.15, 0.15, 3.0)).translate((0.0, 8.0, 1.0))
    add, st = printable.widen_thin(truss + wedge + speck, 1.0, 0.1, 0.2, 0.6)
    assert add is not None and st["widened_mm3"] > 0
    out = truss + wedge + speck + add
    # the bar: at least 0.55 mm wide all along (it was 0.2)
    bar = out.slice(10.3)
    a = np.radians(15.0)
    for t in np.linspace(-2.0, 2.0, 9):
        cut = bar ^ m3d.CrossSection.square((0.05, 3.0), center=True).rotate(15.0).translate(
            (t * np.cos(a), t * np.sin(a)))
        assert cut.area() / 0.05 > 0.55
    # the corner and the speck are as they were
    assert abs((out ^ m3d.Manifold.cube((30, 30, 10)).translate((9, -15, 0))).volume() - wedge.volume()) < 0.05
    assert abs(out.slice(2.0).area() - (truss + wedge + speck).slice(2.0).area()) < 0.05


def test_a_bar_across_the_grid_between_two_masts_is_bridged():
    """A bar running across the grid at 15 degrees between two masts 6 mm
    apart prints as a bridge: nothing is hung under it.  (Bridges were only
    tried along the rows, the columns and the diagonals, and the cells along
    the bar's stepped sides were not counted with the span beside them, so a
    curtain of support hung from it.)"""
    from hk3d import printable
    sup, st = printable.support_fill(_mast_truss(0.6), None, 1.0, 0.2, 0.2, max_bridge_mm=10.0)
    assert st["support_mm3"] < 0.05 and st["bridged_mm2"] > 2.0


def test_tunnels_are_not_printed(tmp_path):
    """Modelled tunnels (feature code 04 in the infrastructure folder) run
    under the ground and stood as blocks where their portals reach it; they
    are left out, bridges (03) and everything in other folders stay."""
    from hk3d import catalog
    files = ["INFRASTRUCTURE/I343751637404063A0/I343751637404063A0.fbx",    # tunnel
             "INFRASTRUCTURE/I344611610703063A0/I344611610703063A0.fbx",    # flyover
             "BUILDING/B344541627004063A0/B344541627004063A0.fbx"]          # 04 means nothing here
    for f in files:
        p = tmp_path / "11-SW-8B" / f
        p.parent.mkdir(parents=True)
        p.write_bytes(b"")
    found = {p.name for *_, p in catalog._discover(tmp_path)}
    assert found == {"I344611610703063A0.fbx", "B344541627004063A0.fbx"}


@pytest.mark.parametrize("layer", [0.2, 0.08])
def test_what_cannot_be_seen_from_outside_is_filled_what_can_stays_open(layer):
    """The same at 0.08 mm layers: voxels are then lower than they are
    wide, and distances counted in voxels (a 1 mm opening, the depth seen
    into a crack) shrank in height by 2.5 times."""
    from hk3d import printable

    def box(x0, y0, z0, x1, y1, z1):
        return m3d.Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])

    man = (box(-20, -20, 0, 20, 20, 5) + box(-10, -10, 5, 10, 10, 9)
           + box(-5, -5, 9.4, 5, 5, 15.4) + box(-1, -1, 9, 1, 1, 9.4))      # a tower on a pillar, a 0.4 mm crack
    hollow = box(3, 3, 6, 7, 7, 8)                                           # a room behind a 0.4 mm slit
    slit = box(7, 4.8, 6, 10.5, 5.2, 8)
    recess = box(-8, -10.5, 6, -6, -9.6, 8)                                  # a recess in the facade
    court = box(-9, 5, 6, -6, 8, 9.1)                                        # a courtyard open to the sky
    arcade = box(-10.5, -8, 5, 10.5, -5.5, 7.5)                              # a passage through the podium
    for cut in (hollow, slit, recess, court, arcade):
        man = man - cut
    add, st = printable.fill_hollows(man, 5.0, 0.2, layer, 1.0)
    assert add is not None and st["hollow_mm3"] > 0

    def filled(region):
        return (add ^ region).volume() / region.volume()

    crack = box(-4, -4, 9, 4, 4, 9.4) - box(-1, -1, 9, 1, 1, 9.4)
    assert filled(crack) > 0.9
    assert filled(hollow) > 0.9
    assert filled(box(-8, -10, 6.2, -6, -9.6, 7.8)) < 0.05
    assert filled(box(-8.8, 5.2, 6, -6.2, 7.8, 9)) < 0.05
    assert filled(box(-8, -7.8, 5.2, 8, -5.7, 7.3)) < 0.05
    # nothing is added where it would show: above the crack, or outside the podium
    assert ((add - man) ^ box(-20, -20, 9.6, 20, 20, 20)).volume() < 0.01
    assert (add - man - box(-10.1, -10.1, 5, 10.1, 10.1, 9.6)).volume() < 0.01


def test_a_space_seen_only_from_above_at_a_low_angle_stays_open():
    """An undercroft behind a low wall: the wall hides its floor from the
    side and the roof hides it from high above, but it shows between the
    two (V1).  Sight lines with their slope the wrong way round looked at it
    from below and filled it."""
    from hk3d import printable

    def box(x0, y0, z0, x1, y1, z1):
        return m3d.Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])

    man = (box(-12, -12, 0, 12, 12, 5) + box(0, -5, 5, 1, 5, 9) + box(0, -5, 9, 9, 5, 10)
           + box(0, -5, 5, 9, -4.6, 9) + box(0, 4.6, 5, 9, 5, 9) + box(10, -5, 5, 10.4, 5, 7))
    add, st = printable.fill_hollows(man, 5.0, 0.2, 0.2, 1.0)
    floor = box(2, -3, 5, 8, 3, 5.6)
    assert add is None or (add ^ floor).volume() < 0.05 * floor.volume()


def test_a_tip_hanging_from_a_wall_is_held_up():
    """An upside-down cone joined to a wall only at its wide top is one
    watertight solid, and its sloping sides face down less steeply than the
    down-facing test: support looked nowhere near it and its tip started in
    mid-air.  Comparing every layer with the one below finds it."""
    from hk3d import printable
    base = m3d.Manifold.cube((14.0, 14.0, 1.0)).translate((-7.0, -7.0, 0.0))
    wall = m3d.Manifold.cube((2.0, 4.0, 15.0)).translate((-1.0, -2.0, 0.0))
    cone = m3d.Manifold.cylinder(10.0, 0.0, 3.0, 48).translate((3.0, 0.0, 5.0))
    part = base + wall + cone
    _, st = printable.support_fill(part, None, 1.0, 0.2, 0.2, 10.0, 0.4)
    assert st["support_mm3"] == 0.0                     # what it did before
    sup, st = printable.support_fill(part, None, 1.0, 0.2, 0.2, 10.0, 0.4, scan=True)
    assert st["support_mm3"] > 0
    held = part + sup
    # the tip's first layer now stands on something
    assert (held.slice(5.1) ^ held.slice(4.9).offset(0.2, m3d.JoinType.Round)).area() > 0.9 * held.slice(5.1).area()


def test_relief_stops_where_a_wall_is_too_thin_for_it():
    """Relief recesses from both faces of a 0.6 mm wall would meet and cut
    it in two: there the relief stops.  On a 3 mm wall beside it the same
    recesses are cut in full."""
    import manifold3d as m3d

    from hk3d import printable
    base = m3d.Manifold.cube((20.0, 12.0, 1.0)).translate((-10.0, -6.0, 0.0))
    thin = m3d.Manifold.cube((0.6, 10.0, 6.0)).translate((-5.3, -5.0, 1.0))
    thick = m3d.Manifold.cube((3.0, 10.0, 6.0)).translate((3.5, -5.0, 1.0))
    model = base + thin + thick
    # 0.3 mm deep panels on both faces of both walls
    panels = [m3d.Manifold.cube((0.3, 6.0, 3.0)).translate((x, -3.0, 2.5)) for x in (-5.3, -5.0, 3.5, 6.2)]
    cut, st = printable.relief_guard(model, panels, 1.0, 0.2, 2 * 0.3 + 0.4)
    assert len(cut) == 2 and st["relief_panels_kept"] == 2
    out = model - m3d.Manifold.batch_boolean(cut, m3d.OpType.Add)
    # the thin wall is whole: one piece across at every height
    for z in (3.0, 4.0, 5.0):
        w = out.slice(z) ^ m3d.CrossSection.square((2.0, 12.0)).translate((-6.0, -6.0))
        assert abs(w.area() - 0.6 * 10.0) < 0.05
    # the thick wall has its two recesses
    w = out.slice(4.0) ^ m3d.CrossSection.square((5.0, 12.0)).translate((2.5, -6.0))
    assert abs(w.area() - (3.0 * 10.0 - 2 * 0.3 * 6.0)) < 0.05
    assert st["relief_kept_mm3"] > 0


def test_thin_columns_carrying_a_deck_are_widened():
    """Two 0.2 mm columns 5 mm tall under a deck: each layer of them is
    smaller than the width asked for, so they were taken for specks and left
    for a slicer to drop, while the support pass counted them as piers."""
    from hk3d import printable
    base = m3d.Manifold.cube((14.0, 14.0, 1.0)).translate((-7.0, -7.0, 0.0))
    cols = (m3d.Manifold.cube((0.2, 0.2, 5.0)).translate((-3.0, -0.1, 1.0))
            + m3d.Manifold.cube((0.2, 0.2, 5.0)).translate((2.8, -0.1, 1.0)))
    deck = m3d.Manifold.cube((6.0, 0.8, 0.4)).translate((-3.0, -0.4, 6.0))
    add, st = printable.widen_thin(base + cols + deck, 1.0, 0.1, 0.2, 0.6)
    assert add is not None
    w = (base + cols + deck + add).slice(3.0)
    assert w.area() > 2 * 0.5 * 0.5
    # a one-layer speck stays a speck, and so does a post that carries
    # nothing (the tip of a crown fin, a post on a roof): widening those
    # turned IFC2's crown fins into stacks of blocks
    for h in (0.2, 3.0):
        speck = base + m3d.Manifold.cube((0.2, 0.2, h)).translate((0, 0, 1.0))
        add, _ = printable.widen_thin(speck, 1.0, 0.1, 0.2, 0.6)
        assert add is None


def test_a_bridge_patch_is_laid_in_one_direction():
    """An L-shaped span: each arm is a short bridge, but across each other.
    A slicer lays the patch in one direction, so one arm hangs; taking each
    cell's own best direction passed both."""
    from hk3d import printable
    M = np.zeros((40, 40), bool)
    A = np.zeros_like(M)
    M[5:9, 5:30] = True            # arm along the rows, anchored at both ends
    M[5:30, 5:9] = True            # arm along the columns, anchored at both ends
    A[5:9, 30] = A[30, 5:9] = True     # the far end of each arm
    A[5:9, 4] = A[4, 5:9] = True       # and the outer sides of the corner
    br = printable.bridgeable(M, A, 30, directions=4)
    arm_rows = br[5:9, 10:30].mean()
    arm_cols = br[10:30, 5:9].mean()
    assert max(arm_rows, arm_cols) > 0.9 and min(arm_rows, arm_cols) < 0.1


def _layercheck():
    import importlib.util
    spec = importlib.util.spec_from_file_location("layercheck", Path(__file__).resolve().parents[1] / "tools" / "layercheck.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_support_is_never_thinner_than_a_printed_line():
    """The support under an upside-down cone hanging from a wall was a strip
    0.18 mm wide for 11 layers - thinner than the 0.4 mm line it was built
    for, so a slicer may drop it and the cone again starts on nothing."""
    from hk3d import printable
    base = m3d.Manifold.cube((14.0, 14.0, 1.0)).translate((-7.0, -7.0, 0.0))
    wall = m3d.Manifold.cube((2.0, 4.0, 15.0)).translate((-1.0, -2.0, 0.0))
    cone = m3d.Manifold.cylinder(10.0, 0.0, 3.0, 48).translate((3.0, 0.0, 5.0))
    part = base + wall + cone
    sup, st = printable.support_fill(part, None, 1.0, 0.2, 0.2, 10.0, 0.4, scan=True)
    assert sup is not None
    # below the tip, no part of the support (with the wall beside it) is
    # narrower than a line: opening with square corners leaves it whole
    held = part + sup
    M = m3d.JoinType.Miter
    for z in np.arange(1.1, 5.0, 0.2):
        u = held.slice(float(z))
        thin = u - u.offset(-0.19, M).offset(0.19, M)
        assert (thin ^ sup.slice(float(z))).area() < 0.002, z


def test_layercheck_finds_a_thin_rib_on_a_thick_wall():
    """A 0.1 mm rib standing free was reported; the same rib joined to a
    2 mm block was not, because only whole pieces were tested for width."""
    lc = _layercheck()
    base = m3d.Manifold.cube((14.0, 14.0, 1.0)).translate((-7.0, -7.0, 0.0))
    block = m3d.Manifold.cube((2.0, 2.0, 4.0)).translate((-1.0, -1.0, 1.0))
    joined = m3d.Manifold.cube((3.0, 0.1, 3.0)).translate((1.0, -0.05, 1.0))
    free = m3d.Manifold.cube((3.0, 0.1, 3.0)).translate((3.0, 3.0, 1.0))
    for model in (base + block + joined, base + block + free):
        r = lc.check(model, 1.0, 0.2, 0.4, 10.0, 0.0025)
        assert r["thin"]["count"] >= 1
        assert any(m["min_width_mm"] < 0.2 and m["layers"] >= 10 for m in r["thin"]["members"])


def test_layercheck_bridges_need_real_anchors():
    """A 6 x 2 mm deck whose ends rest on its piers for only 0.05 mm was
    passed as a bridge: the first anchor cell along the line was enough."""
    lc = _layercheck()
    base = m3d.Manifold.cube((14.0, 14.0, 1.0)).translate((-7.0, -7.0, 0.0))
    deck = m3d.Manifold.cube((6.0, 2.0, 0.4)).translate((-3.0, -1.0, 3.0))
    for overlap, ok in ((0.05, False), (1.0, True)):
        piers = (m3d.Manifold.cube((1.5, 2.0, 2.0)).translate((-3.0 - 1.5 + overlap, -1.0, 1.0))
                 + m3d.Manifold.cube((1.5, 2.0, 2.0)).translate((3.0 - overlap, -1.0, 1.0)))
        r = lc.check(base + piers + deck, 1.0, 0.2, 0.4, 10.0, 0.0025)
        hung = sum(e["area_mm2"] for e in r["unsupported"]["all"] if 2.9 < e["z"] < 3.2)
        assert (hung < 0.5) == ok, (overlap, hung)


def test_a_failed_landmark_is_reported_not_dressed_up(tmp_path, monkeypatch):
    """When carving the landmark failed, the build still wrote its files
    and said "Done"; the error sat in the stats where nobody saw it."""
    from hk3d import pipeline
    from hk3d.pipeline import ModelParams, build

    cat = _synthetic_area(tmp_path, monkeypatch, w=6.0, h=20.0)
    p = ModelParams(e=1000, n=1000, radius_m=10, diameter_mm=10, base_mm=3, detail_mm=0.2,
                    groups=["building"], precise_landmark=True, engrave=False, tile=False,
                    support_free=False)
    p.validate()
    ok = build(p, cat, tmp_path / "ok", tmp_path / "cache")
    assert ok["status"]["level"] == "generated" and not ok["status"]["slicer_verified"]

    def boom(*a, **k):
        raise RuntimeError("carving exploded")
    monkeypatch.setattr(pipeline, "_carve_landmark", boom)
    res = build(p, cat, tmp_path / "bad", tmp_path / "cache2")
    assert res["status"]["level"] == "needs_review"
    assert any(i["code"] == "landmark_failed" and "exploded" in i["detail"] for i in res["status"]["issues"])


def test_print_base_falls_on_the_slicer_layers():
    from hk3d.pipeline import ModelParams
    assert ModelParams(e=0, n=0, base_mm=5.0, layer_mm=0.2).print_base() == 5.0
    # 0.12 mm layers after a 0.2 mm first layer: 0.2 + 40 x 0.12 = 5.0
    assert ModelParams(e=0, n=0, base_mm=5.0, layer_mm=0.12, first_layer_mm=0.2).print_base() == 5.0
    # 0.12 mm layers from 0.12: the boundary nearest 5 mm is 4.92 + ... = 5.04
    assert abs(ModelParams(e=0, n=0, base_mm=5.0, layer_mm=0.12).print_base() - 5.04) < 1e-9


def test_relief_that_would_cut_a_part_loose_is_not_cut():
    """A panel over a whole facade passed the thin-wall test (five sample
    layers, all through thick wall) while one layer between them was a
    0.2 mm neck: cutting it left the block above floating."""
    from hk3d import printable
    base = m3d.Manifold.cube((14.0, 14.0, 1.0)).translate((-7.0, -7.0, 0.0))
    low = m3d.Manifold.cube((4.0, 4.0, 3.0)).translate((-2.0, -2.0, 1.0))
    neck = m3d.Manifold.cube((0.2, 2.0, 0.4)).translate((1.8, -1.0, 4.0))
    high = m3d.Manifold.cube((4.0, 4.0, 6.6)).translate((-2.0, -2.0, 4.4))
    model = base + low + neck + high
    assert len(model.decompose()) == 1
    panel = m3d.Manifold.cube((0.3, 2.0, 10.0)).translate((1.7, -1.0, 1.0))
    cut, st = printable.relief_guard(model, [panel], 1.0, 0.2, 1.0)
    out = model - m3d.Manifold.batch_boolean(cut, m3d.OpType.Add) if cut else model
    assert len(out.decompose()) == 1
    # not cut round the neck, cut everywhere else (v24 at first left out
    # the whole panel: the relief went from 100 mm of an ICC tower face)
    assert st["relief_panels_trimmed"] >= 1 and st["relief_panels_kept"] == 0
    assert len(cut) == 1
    assert (model ^ cut[0]).volume() > 0.5 * (model ^ panel).volume()
    assert (model ^ cut[0]).bounding_box()[2] < 2.0 and (model ^ cut[0]).bounding_box()[5] > 9.0
    # a panel that cuts nothing loose is still cut
    other = m3d.Manifold.cube((0.3, 2.0, 2.0)).translate((-2.0, -1.0, 6.0))
    cut, _ = printable.relief_guard(model, [other], 1.0, 0.2, 1.0)
    assert len(cut) == 1


def _stagetrace():
    import importlib.util
    spec = importlib.util.spec_from_file_location("stagetrace", Path(__file__).parent.parent / "tools" / "stagetrace.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_stage_dumps_cover_every_step(tmp_path, monkeypatch):
    """With HK3D_DEBUG_STAGES the build keeps the model after each step, so a
    defect in the output can be traced to the step that made it."""
    from hk3d.pipeline import ModelParams, build

    cat = _synthetic_area(tmp_path, monkeypatch, w=6.0, h=20.0)
    monkeypatch.setenv("HK3D_DEBUG_STAGES", str(tmp_path / "stages"))
    p = ModelParams(e=1000, n=1000, radius_m=10, diameter_mm=10, base_mm=3, detail_mm=0.2,
                    groups=["building"], precise_landmark=True, engrave=False, tile=False)
    p.validate()
    build(p, cat, tmp_path / "out", tmp_path / "cache")
    got = {f.stem for f in (tmp_path / "stages").glob("*.stl")}
    for name in ("1_source", "2_trimmed", "3_widened", "4_supported", "6_assembled", "9_standing"):
        assert name in got, (name, got)


def test_stagetrace_tells_where_a_ledge_came_from():
    """A ledge hanging in the air: absent before the step that adds it, hangs
    from there on - unless a later step holds it up."""
    import manifold3d as m3d

    st = _stagetrace()
    post = m3d.Manifold.cube((2.0, 2.0, 6.0))
    ledge = m3d.Manifold.cube((3.0, 2.0, 0.4)).translate((2.0, 0.0, 4.0))
    prop = m3d.Manifold.cube((3.0, 2.0, 4.0)).translate((2.0, 0.0, 0.0))
    d = {"z": 4.1, "x": [2.0, 5.0], "y": [0.0, 2.0]}
    assert st.look(post, d, 0.2, 0.4, 10.0)["state"] == "absent"
    assert st.look(post + ledge, d, 0.2, 0.4, 10.0)["state"] == "hangs"
    assert st.look(post + ledge + prop, d, 0.2, 0.4, 10.0)["state"] == "held"
    float_ = m3d.Manifold.cube((1.0, 1.0, 0.4)).translate((4.0, 0.5, 4.0))
    assert st.look(post + float_, {"z": 4.1, "x": [4.0, 5.0], "y": [0.5, 1.5]}, 0.2, 0.4, 10.0)["state"] == "born"
    S = lambda s: {"state": s}
    assert st.origin([("a", S("absent")), ("b", S("hangs")), ("c", S("hangs"))]) == "b"
    assert st.origin([("a", S("hangs")), ("b", S("held")), ("c", S("hangs"))]) == "c"
    assert st.origin([("a", S("hangs")), ("b", S("held"))]) is None


def test_support_against_the_part_grows_only_under_what_it_holds():
    """A free-standing strut one cell wide is widened to a line; a one-cell
    bracket against a thin wall grows only inside the shadow of the layer
    above, never out over air (growing outwards cascaded at 0.08 mm layers)."""
    from hk3d.printable import _at_least_a_line
    solid_ = np.zeros((24, 24), bool)
    solid_[:, 8] = True                          # a thin wall, one cell
    add = np.zeros_like(solid_)
    add[5:15, 9] = True                          # a bracket one cell wide against it
    add[10, 18] = True                           # a free strut one cell across
    shadow = np.zeros_like(solid_)
    shadow[:, 6:11] = True                       # the layer above: two cells past the bracket
    out = _at_least_a_line(add, solid_, 3, shadow)
    assert out[5:15, 10].all()                   # the bracket grown, under the deck
    assert not out[:, 11:16].any()               # and not out past it over air
    assert out[9:12, 17:20].all()                # the strut grown to a line, shadow or not
