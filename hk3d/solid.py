"""Heightmap -> watertight solids (manifold3d) and mesh exporters."""
from __future__ import annotations

import io
import struct
import zipfile
from xml.sax.saxutils import escape as _escape


def escape(text: str) -> str:
    """XML text, safe inside double-quoted attributes as well."""
    return _escape(str(text), {'"': "&quot;", "'": "&apos;"})

import manifold3d as m3d
import numpy as np


def quad_diagonals(Z: np.ndarray) -> np.ndarray:
    """Per-quad split choice (True = split along NW-SE) that best follows Z."""
    za, zb, zc, zd = Z[:-1, :-1], Z[:-1, 1:], Z[1:, :-1], Z[1:, 1:]
    return (np.abs(za - zd) <= np.abs(zb - zc)).ravel()


def grid_solid(xs: np.ndarray, ys: np.ndarray, Z: np.ndarray, diag: np.ndarray | None = None) -> m3d.Manifold:
    """Closed solid between z=0 and the height field Z.

    xs: (nx,) west->east, ys: (ny,) north->south (descending), Z: (ny, nx) > 0.
    The construction is manifold by design: top grid, four walls, fan bottom.
    ``diag`` (from :func:`quad_diagonals`) lets several solids share one
    triangulation so that booleans between them do not leave slivers.
    """
    ny, nx = Z.shape
    gx, gy = np.meshgrid(xs, ys)
    top = np.stack([gx.ravel(), gy.ravel(), Z.ravel()], 1)
    idx = np.arange(ny * nx).reshape(ny, nx)
    a, b = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
    c, d = idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    diag_ad = quad_diagonals(Z) if diag is None else diag
    t1 = np.where(diag_ad[:, None], np.stack([a, c, d], 1), np.stack([a, c, b], 1))
    t2 = np.where(diag_ad[:, None], np.stack([a, d, b], 1), np.stack([b, c, d], 1))
    ring = np.concatenate([idx[0, :], idx[1:, -1], idx[-1, -2::-1], idx[-2:0:-1, 0]])
    nb = len(ring)
    base0 = len(top)
    bot = top[ring].copy()
    bot[:, 2] = 0.0
    cx, cy = (xs[0] + xs[-1]) / 2, (ys[0] + ys[-1]) / 2
    verts = np.concatenate([top, bot, [[cx, cy, 0.0]]]).astype(np.float32)
    bi = base0 + np.arange(nb)
    bn = np.roll(bi, -1)
    rn = np.roll(ring, -1)
    walls1 = np.stack([ring, bn, bi], 1)
    walls2 = np.stack([ring, rn, bn], 1)
    bottom = np.stack([bi, bn, np.full(nb, base0 + nb)], 1)
    F = np.concatenate([t1, t2, walls1, walls2, bottom]).astype(np.uint32)
    man = m3d.Manifold(m3d.Mesh(vert_properties=verts, tri_verts=F))
    if man.status() != m3d.Error.NoError:
        raise RuntimeError(f"grid solid not manifold: {man.status()}")
    return man


def _frustum(shape: str, size_bottom: float, size_top: float, height: float) -> m3d.Manifold:
    """Prism/frustum of the outline shape. ``size`` = circle diameter, hexagon
    corner-to-corner width (vertices at 0deg, 60deg...), or square side."""
    if shape == "circle":
        segs = int(np.clip(max(size_bottom, size_top) * 4, 128, 720))
        return m3d.Manifold.cylinder(height, size_bottom / 2, size_top / 2, segs)
    if shape == "hexagon":
        return m3d.Manifold.cylinder(height, size_bottom / 2, size_top / 2, 6)
    if shape == "square":
        k = np.sqrt(2) / 2
        return m3d.Manifold.cylinder(height, size_bottom * k, size_top * k, 4).rotate((0, 0, 45))
    raise ValueError(f"unknown shape {shape}")


def shape_prism(shape: str, diameter: float, height: float) -> m3d.Manifold:
    return _frustum(shape, diameter, diameter, height).translate((0, 0, -1.0))


def holder_ring(shape: str, diameter: float, pocket_depth: float, wall: float = 5.0,
                clearance: float = 0.3, floor: float = 2.0, chamfer: float = 0.8) -> m3d.Manifold:
    """Display base the model drops into (print it in a wood/dark filament).

    The pocket floor is at z = ``floor``; the model's top surface edge sits at
    about the rim height when ``pocket_depth`` equals the model's base height.
    """
    inner = diameter + 2 * clearance
    outer = inner + 2 * wall
    h = floor + pocket_depth
    ch = min(chamfer, wall / 3, h / 3)
    body = (_frustum(shape, outer, outer, h - ch)
            + _frustum(shape, outer, outer - 2 * ch, ch).translate((0, 0, h - ch)))
    lead = min(0.6, pocket_depth / 3)
    pocket = (_frustum(shape, inner, inner, pocket_depth + 1).translate((0, 0, floor))
              + _frustum(shape, inner, inner + 2 * lead, lead + 1e-3).translate((0, 0, h - lead)))
    return body - pocket


def to_arrays(man: m3d.Manifold) -> tuple[np.ndarray, np.ndarray]:
    mesh = man.to_mesh()
    V = np.asarray(mesh.vert_properties[:, :3], dtype=np.float64)
    F = np.asarray(mesh.tri_verts, dtype=np.int64)
    V = separate_touching(V, F)
    return V.astype(np.float32), F.astype(np.uint32)


def standing(man: m3d.Manifold, crumb: float = 0.0, floor: float | None = None
             ) -> tuple[m3d.Manifold, list, int]:
    """Split a solid into what stands on its bottom and what does not.

    Sealed air pockets are filled (no printer can make them, and they cost
    slicing time). Separate parts whose lowest point is above ``floor``
    (default: the solid's bottom) would be printed starting in the air;
    they are returned apart, except specks under ``crumb`` mm3, which are
    dropped. Returns (standing solid, loose parts, pockets filled)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    mesh = man.to_mesh()
    V = np.asarray(mesh.vert_properties, np.float64)[:, :3]
    F = np.asarray(mesh.tri_verts, np.int64)
    if not len(F):
        return man, [], 0
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    g = coo_matrix((np.ones(len(e), np.int8), (e[:, 0], e[:, 1])), shape=(len(V), len(V)))
    n, lab = connected_components(g, directed=False)
    if n == 1:
        return man, [], 0
    shell = lab[F[:, 0]]
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    vol = np.bincount(shell, np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0, minlength=n)
    zmin = np.full(n, np.inf)
    np.minimum.at(zmin, lab, V[:, 2])
    up = zmin > (V[:, 2].min() if floor is None else floor) + 0.05
    keep = (vol > 0) & ~up
    pockets = int((vol <= 0).sum())
    if keep.sum() == n:
        return man, [], 0
    V32 = V.astype(np.float32)

    def part(mask):
        return m3d.Manifold(m3d.Mesh(vert_properties=V32, tri_verts=F[mask[shell]].astype(np.uint32)))

    loose = [part(np.arange(n) == i) for i in np.nonzero((vol >= max(crumb, 1e-9)) & up)[0]]
    return part(keep), loose, pockets


def _dup_mask(V: np.ndarray) -> np.ndarray:
    """Vertices whose position collides with another one after rounding to
    1e-5 mm or after the float32 cast that STL/3MF files apply."""
    out = np.zeros(len(V), bool)
    for key in (np.round(V / 1e-5).astype(np.int64), V.astype(np.float32).view(np.int32)):
        _, inv, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
        out |= counts[inv.ravel()] > 1
    return out


def separate_touching(V: np.ndarray, F: np.ndarray, step: float = 0.002) -> np.ndarray:
    """Pull apart vertices that share a position but belong to different sheets.

    manifold3d keeps bodies that touch along an edge, a point or a face
    distinct (separate vertex indices at the same position).  File formats
    like STL lose that, and slicers then see non-manifold edges.  Each copy is
    moved ``step`` mm (default 2 um) into its own body - along the inverse of
    its faces' mean normal, so two sheets touching face to face separate too -
    and the check is repeated, with a larger step, until no position collides.
    """
    if not len(V):
        return V
    dup = _dup_mask(V)
    if not dup.any():
        return V
    T = V[F]
    fn = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])       # area-weighted normals
    cen = T.mean(1)
    nsum = np.zeros_like(V)
    csum = np.zeros_like(V)
    cnt = np.zeros(len(V))
    for k in range(3):
        np.add.at(nsum, F[:, k], fn)
        np.add.at(csum, F[:, k], cen)
        np.add.at(cnt, F[:, k], 1)
    inward = -nsum
    to_c = np.divide(csum, cnt[:, None], out=np.zeros_like(csum), where=cnt[:, None] > 0) - V
    for v in (inward, to_c):
        nv = np.linalg.norm(v, axis=1, keepdims=True)
        v /= np.where(nv > 0, nv, 1.0)
    dirn = inward + 0.5 * to_c
    nd = np.linalg.norm(dirn, axis=1, keepdims=True)
    dirn = np.divide(dirn, nd, out=to_c.copy(), where=nd > 1e-9)
    V = V.copy()
    moved = np.zeros(len(V), bool)
    rng = np.random.default_rng(0)
    for attempt in range(6):
        idx = np.nonzero(dup & (cnt > 0) & ~moved)[0] if attempt == 0 else np.nonzero(dup & (cnt > 0))[0]
        if not len(idx):
            break
        move = dirn[idx]
        if attempt:
            # copies that share a direction never part: nudge each one sideways too
            move = move + 0.7 * rng.standard_normal(move.shape)
            move /= np.maximum(np.linalg.norm(move, axis=1, keepdims=True), 1e-9)
        V[idx] += move * step * (2 ** min(attempt, 3))
        moved[idx] = True
        dup = _dup_mask(V)
        if not dup.any():
            break
    return V


# ----------------------------------------------------------------- exporters

def write_stl(path, V: np.ndarray, F: np.ndarray, name: str = "hk3d") -> None:
    with open(path, "wb") as fh:
        fh.write(stl_bytes(V, F, name))


def write_3mf(path, parts: list[tuple[str, str, np.ndarray, np.ndarray]], title: str = "hk3d") -> None:
    """parts: list of (name, '#rrggbb', V, F).

    Several parts are written as one assembly object made of components, so
    slicers (Bambu Studio, PrusaSlicer, Orca) load them as one object with
    aligned parts that can each get their own filament.  Coordinates are
    the float32 values the STL holds, written with enough digits to read
    back exactly: rounded to 4 decimals, vertices kept apart by a few
    microns (``separate_touching``) merged again and left thousands of
    zero-area triangles (BoC, v22).
    """
    mats = "".join(f'<base name="{escape(n)}" displaycolor="{c.upper()}FF"/>' for n, c, _, _ in parts)
    objs = []
    for i, (name, _c, V, F) in enumerate(parts):
        oid = i + 2
        objs.append(f'<object id="{oid}" name="{escape(name)}" type="model" pid="1" pindex="{i}">'
                    f"<mesh><vertices>{_xml_rows(V.astype(np.float32), 'vertex', ('x', 'y', 'z'), '{:.9g}')}</vertices>"
                    f"<triangles>{_xml_rows(F, 'triangle', ('v1', 'v2', 'v3'), '{:d}')}</triangles></mesh></object>")
    if len(parts) > 1:
        root = len(parts) + 2
        comps = "".join(f'<component objectid="{i + 2}"/>' for i in range(len(parts)))
        objs.append(f'<object id="{root}" name="{escape(title)}" type="model"><components>{comps}</components></object>')
        items = f'<item objectid="{root}"/>'
    else:
        items = '<item objectid="2"/>'
    model = ('<?xml version="1.0" encoding="UTF-8"?>\n'
             '<model unit="millimeter" xml:lang="en-US" '
             'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">'
             f'<metadata name="Title">{escape(title)}</metadata>'
             '<metadata name="Application">hk3d</metadata>'
             '<metadata name="Copyright">3D data: Map from Lands Department, HKSAR Government</metadata>'
             f'<resources><basematerials id="1">{mats}</basematerials>{"".join(objs)}</resources>'
             f'<build>{items}</build></model>')
    ctypes = ('<?xml version="1.0" encoding="UTF-8"?>\n'
              '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
              '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
              '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>'
              '</Types>')
    rels = ('<?xml version="1.0" encoding="UTF-8"?>\n'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Target="/3D/3dmodel.model" Id="rel0" '
            'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>')
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", ctypes)
        z.writestr("_rels/.rels", rels)
        z.writestr("3D/3dmodel.model", model)
        if len(parts) > 1:
            # Bambu Studio / Orca read part names and filament slots from here
            cfg = [f'<?xml version="1.0" encoding="UTF-8"?>\n<config>\n  <object id="{root}">',
                   f'    <metadata key="name" value="{escape(title)}"/>',
                   '    <metadata key="extruder" value="1"/>']
            for i, (name, _c, _V, F) in enumerate(parts):
                cfg += [f'    <part id="{i + 2}" subtype="normal_part">',
                        f'      <metadata key="name" value="{escape(name)}"/>',
                        f'      <metadata key="extruder" value="{i + 1}"/>',
                        '    </part>']
            cfg += ['  </object>', '</config>']
            z.writestr("Metadata/model_settings.config", "\n".join(cfg))


def _xml_rows(arr: np.ndarray, tag: str, attrs: tuple, fmt: str) -> str:
    """Fast XML element rows for large vertex/triangle arrays."""
    row = "<" + tag + " " + " ".join(f'{a}="{fmt}"' for a in attrs) + "/>"
    if "d" in fmt:
        arr = arr.astype(np.int64)
    return "".join(row.format(*r) for r in arr.tolist())


def write_glb(path, parts: list[tuple[str, str, np.ndarray, np.ndarray]]) -> None:
    """Coloured preview GLB (z-up, millimetres)."""
    import trimesh
    from trimesh.visual.material import PBRMaterial

    scene = trimesh.Scene()
    for name, color, V, F in parts:
        if not len(F):
            continue
        rgb = [int(color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        mesh = trimesh.Trimesh(V, F, process=False)
        mesh.remove_unreferenced_vertices()
        mesh.visual = trimesh.visual.TextureVisuals(
            material=PBRMaterial(name=name, baseColorFactor=[*rgb, 1.0], metallicFactor=0.0, roughnessFactor=0.9))
        scene.add_geometry(mesh, node_name=name, geom_name=name)
    data = scene.export(file_type="glb")
    with open(path, "wb") as fh:
        fh.write(data)


def write_zip(path, files: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            z.writestr(name, data)


def stl_bytes(V, F, name="hk3d") -> bytes:
    buf = io.BytesIO()
    tri = V[F].astype(np.float32)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.divide(n, ln, out=np.zeros_like(n), where=ln > 0)
    rec = np.zeros(len(F), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"], rec["v"] = n, tri
    buf.write(name.encode()[:80].ljust(80, b" "))
    buf.write(struct.pack("<I", len(F)))
    buf.write(rec.tobytes())
    return buf.getvalue()
