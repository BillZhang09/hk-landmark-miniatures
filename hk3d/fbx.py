"""Minimal binary FBX reader (geometry + transforms only).

Supports FBX binary 7.x (32-bit and 64-bit node headers). Only the parts
needed to extract triangle meshes in world coordinates are implemented.
"""
from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, field

import numpy as np

_ARRAY_TYPES = {b"f": np.float32, b"d": np.float64, b"l": np.int64, b"i": np.int32, b"b": np.bool_}


@dataclass
class Node:
    name: str
    props: list
    children: list = field(default_factory=list)

    def find(self, name):
        for c in self.children:
            if c.name == name:
                return c
        return None

    def find_all(self, name):
        return [c for c in self.children if c.name == name]


def _read_prop(buf, off):
    t = buf[off:off + 1]
    off += 1
    if t == b"Y":
        return struct.unpack_from("<h", buf, off)[0], off + 2
    if t == b"C":
        return bool(buf[off]), off + 1
    if t == b"I":
        return struct.unpack_from("<i", buf, off)[0], off + 4
    if t == b"F":
        return struct.unpack_from("<f", buf, off)[0], off + 4
    if t == b"D":
        return struct.unpack_from("<d", buf, off)[0], off + 8
    if t == b"L":
        return struct.unpack_from("<q", buf, off)[0], off + 8
    if t in _ARRAY_TYPES:
        n, enc, clen = struct.unpack_from("<III", buf, off)
        off += 12
        raw = buf[off:off + clen]
        off += clen
        if enc == 1:
            raw = zlib.decompress(raw)
        return np.frombuffer(raw, dtype=_ARRAY_TYPES[t], count=n), off
    if t in (b"S", b"R"):
        n = struct.unpack_from("<I", buf, off)[0]
        off += 4
        data = bytes(buf[off:off + n])
        off += n
        if t == b"S":
            return data.decode("utf-8", "replace"), off
        return data, off
    raise ValueError(f"Unknown FBX property type {t!r} at {off - 1}")


def _read_node(buf, off, wide):
    if wide:
        end, nprops, _plen = struct.unpack_from("<QQQ", buf, off)
        off += 24
    else:
        end, nprops, _plen = struct.unpack_from("<III", buf, off)
        off += 12
    nlen = buf[off]
    off += 1
    if end == 0:
        return None, off
    name = bytes(buf[off:off + nlen]).decode("ascii", "replace")
    off += nlen
    props = []
    for _ in range(nprops):
        p, off = _read_prop(buf, off)
        props.append(p)
    node = Node(name, props)
    sentinel = 25 if wide else 13
    while off < end - sentinel or (off < end and end - off > sentinel):
        child, off = _read_node(buf, off, wide)
        if child is None:
            break
        node.children.append(child)
    return node, end


def parse(path_or_bytes) -> Node:
    if isinstance(path_or_bytes, (bytes, bytearray, memoryview)):
        buf = memoryview(path_or_bytes)
    else:
        with open(path_or_bytes, "rb") as fh:
            buf = memoryview(fh.read())
    if bytes(buf[:20]) != b"Kaydara FBX Binary  ":
        raise ValueError("Not a binary FBX file")
    version = struct.unpack_from("<I", buf, 23)[0]
    wide = version >= 7500
    off = 27
    root = Node("root", [])
    while off < len(buf):
        child, off = _read_node(buf, off, wide)
        if child is None:
            break
        root.children.append(child)
    root.props = [version]
    return root


# ---------------------------------------------------------------------------
# Scene extraction
# ---------------------------------------------------------------------------

def _p70(node: Node) -> dict:
    out = {}
    p70 = node.find("Properties70")
    if p70 is None:
        return out
    for p in p70.children:
        if p.props:
            out[p.props[0]] = p.props[4:]
    return out


def _euler_xyz(deg) -> np.ndarray:
    rx, ry, rz = np.radians(np.asarray(deg, dtype=np.float64))
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx  # FBX default eEulerXYZ


def _model_matrix(props: dict) -> np.ndarray:
    def v(key, default):
        val = props.get(key)
        return np.asarray(val[:3] if val else default, dtype=np.float64)

    T = v("Lcl Translation", (0, 0, 0))
    R = v("Lcl Rotation", (0, 0, 0))
    S = v("Lcl Scaling", (1, 1, 1))
    pre = v("PreRotation", (0, 0, 0))
    post = v("PostRotation", (0, 0, 0))
    rot_off = v("RotationOffset", (0, 0, 0))
    rot_piv = v("RotationPivot", (0, 0, 0))
    sc_off = v("ScalingOffset", (0, 0, 0))
    sc_piv = v("ScalingPivot", (0, 0, 0))

    def tr(t):
        m = np.eye(4)
        m[:3, 3] = t
        return m

    def rot(r):
        m = np.eye(4)
        m[:3, :3] = r
        return m

    Sm = np.diag([S[0], S[1], S[2], 1.0])
    Rm = rot(_euler_xyz(pre)) @ rot(_euler_xyz(R)) @ rot(np.linalg.inv(_euler_xyz(post)))
    return (tr(T) @ tr(rot_off) @ tr(rot_piv) @ Rm @ tr(-rot_piv)
            @ tr(sc_off) @ tr(sc_piv) @ Sm @ tr(-sc_piv))


def _geometric_matrix(props: dict) -> np.ndarray:
    m = np.eye(4)
    gt = props.get("GeometricTranslation")
    gr = props.get("GeometricRotation")
    gs = props.get("GeometricScaling")
    if gs:
        m = np.diag([gs[0], gs[1], gs[2], 1.0]) @ m
    if gr:
        r = np.eye(4)
        r[:3, :3] = _euler_xyz(gr[:3])
        m = r @ m
    if gt:
        t = np.eye(4)
        t[:3, 3] = gt[:3]
        m = t @ m
    return m


def _triangulate(pvi: np.ndarray) -> np.ndarray:
    """Fan-triangulate FBX polygon index list. Returns (n,3) vertex indices."""
    pvi = pvi.astype(np.int64)
    ends = np.nonzero(pvi < 0)[0]
    if len(ends) == 0:
        return np.zeros((0, 3), dtype=np.int64)
    idx = np.where(pvi < 0, ~pvi, pvi)
    starts = np.concatenate([[0], ends[:-1] + 1])
    sizes = ends - starts + 1
    # fast path: all triangles
    if np.all(sizes == 3):
        return idx[: ends[-1] + 1].reshape(-1, 3)
    ntri = np.clip(sizes - 2, 0, None)
    total = int(ntri.sum())
    poly_of_tri = np.repeat(np.arange(len(sizes)), ntri)
    local = np.arange(total) - np.repeat(np.cumsum(ntri) - ntri, ntri)
    s = starts[poly_of_tri]
    a = idx[s]
    b = idx[s + 1 + local]
    c = idx[s + 2 + local]
    return np.stack([a, b, c], axis=1)


def _global_scale(root: Node) -> float:
    gs = root.find("GlobalSettings")
    if gs is None:
        return 1.0
    usf = _p70(gs).get("UnitScaleFactor")
    # UnitScaleFactor is "cm per unit"; we want metres.
    return float(usf[0]) / 100.0 if usf else 0.01


def load_meshes(path) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Return list of (model_name, vertices(N,3) float64 world metres, faces(M,3) int64)."""
    root = parse(path)
    objects = root.find("Objects")
    conns = root.find("Connections")
    if objects is None:
        return []
    geoms, models = {}, {}
    for o in objects.children:
        if o.name == "Geometry" and len(o.props) > 2 and o.props[2] == "Mesh":
            geoms[o.props[0]] = o
        elif o.name == "Model":
            models[o.props[0]] = o
    parent = {}
    geom_to_model = {}
    if conns is not None:
        for c in conns.children:
            if c.props and c.props[0] == "OO":
                child, par = c.props[1], c.props[2]
                if child in geoms and par in models:
                    geom_to_model[child] = par
                elif child in models:
                    parent[child] = par
    world_cache: dict = {}

    def world(mid):
        if mid in world_cache:
            return world_cache[mid]
        m = _model_matrix(_p70(models[mid]))
        p = parent.get(mid)
        if p in models:
            m = world(p) @ m
        world_cache[mid] = m
        return m

    scale = _global_scale(root)
    out = []
    for gid, g in geoms.items():
        vn = g.find("Vertices")
        pn = g.find("PolygonVertexIndex")
        if vn is None or pn is None or not len(vn.props[0]):
            continue
        verts = np.asarray(vn.props[0], dtype=np.float64).reshape(-1, 3)
        faces = _triangulate(np.asarray(pn.props[0]))
        mid = geom_to_model.get(gid)
        name = ""
        if mid is not None:
            props = _p70(models[mid])
            M = world(mid) @ _geometric_matrix(props)
            verts = _apply(verts, M)
            name = models[mid].props[1].split("\x00")[0]
        if scale != 1.0:
            verts = verts * scale
        out.append((name, verts, faces))
    return out


def _apply(verts: np.ndarray, M: np.ndarray) -> np.ndarray:
    # Explicit multiply-add: avoids spurious FP warnings from BLAS matmul on
    # macOS Accelerate with large (HK1980 ~8e5) coordinates.
    R, t = M[:3, :3], M[:3, 3]
    if np.allclose(R, np.eye(3)):
        return verts + t
    out = np.empty_like(verts)
    for i in range(3):
        out[:, i] = verts[:, 0] * R[i, 0] + verts[:, 1] * R[i, 1] + verts[:, 2] * R[i, 2] + t[i]
    return out


def _triangulate_corners(pvi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Like ``_triangulate`` but returns polygon-vertex *positions* (into the
    ByPolygonVertex arrays) for each triangle corner, and each triangle's polygon."""
    pvi = np.asarray(pvi).astype(np.int64)
    ends = np.nonzero(pvi < 0)[0]
    if len(ends) == 0:
        return np.zeros((0, 3), np.int64), np.zeros(0, np.int64)
    starts = np.concatenate([[0], ends[:-1] + 1])
    sizes = ends - starts + 1
    ntri = np.clip(sizes - 2, 0, None)
    poly = np.repeat(np.arange(len(sizes)), ntri)
    local = np.arange(int(ntri.sum())) - np.repeat(np.cumsum(ntri) - ntri, ntri)
    s = starts[poly]
    return np.stack([s, s + 1 + local, s + 2 + local], 1), poly


def load_textured(path) -> dict | None:
    """Triangles with texture coordinates, for buildings with a diffuse texture.

    Returns ``{"V": (N,3) world metres, "F": (M,3), "uv": (M,3,2), "tex": (M,)
    index into "files" or -1, "files": [absolute texture paths]}``, or None
    when the file has no UVs.  Only the first UV layer and the diffuse map of
    each material are read.
    """
    from pathlib import Path

    root = parse(path)
    objects, conns = root.find("Objects"), root.find("Connections")
    if objects is None or conns is None:
        return None
    by_id = {o.props[0]: o for o in objects.children if o.props}
    mats_of: dict = {}
    diffuse: dict = {}
    video_of: dict = {}
    for c in conns.children:
        if not c.props:
            continue
        child, par = c.props[1], c.props[2]
        cn, pn = by_id.get(child), by_id.get(par)
        if cn is None or pn is None:
            continue
        if c.props[0] == "OO" and cn.name == "Material" and pn.name == "Model":
            mats_of.setdefault(par, []).append(child)
        elif c.props[0] == "OP" and cn.name == "Texture" and len(c.props) > 3 and c.props[3] == "DiffuseColor":
            diffuse[par] = child
    folder = Path(path).parent

    def tex_file(tid):
        t = by_id[tid]
        for key in ("RelativeFilename", "FileName"):
            n = t.find(key)
            if n is not None and n.props and n.props[0]:
                name = str(n.props[0]).replace("\\", "/").split("/")[-1]
                if (folder / name).is_file():
                    return str(folder / name)
        return None

    meshes = load_meshes(path)  # world vertices, same geometry order
    geoms = [o for o in objects.children if o.name == "Geometry" and len(o.props) > 2 and o.props[2] == "Mesh"
             and o.find("Vertices") is not None and o.find("PolygonVertexIndex") is not None
             and len(o.find("Vertices").props[0])]
    geom_model = {}
    for c in conns.children:
        if c.props and c.props[0] == "OO" and c.props[1] in by_id and by_id[c.props[1]].name == "Geometry":
            geom_model[c.props[1]] = c.props[2]
    files: list[str] = []
    Vs, Fs, UVs, Ts = [], [], [], []
    off = 0
    for g, (_name, V, F) in zip(geoms, meshes):
        corners, poly = _triangulate_corners(g.find("PolygonVertexIndex").props[0])
        uvl = g.find("LayerElementUV")
        if uvl is None or len(corners) != len(F):
            return None
        uv = np.asarray(uvl.find("UV").props[0], np.float64).reshape(-1, 2)
        idx = uvl.find("UVIndex")
        mapping = uvl.find("MappingInformationType").props[0]
        if mapping != "ByPolygonVertex":
            return None
        uvc = uv[np.asarray(idx.props[0])[corners]] if idx is not None else uv[corners]
        tex = np.full(len(F), -1, np.int64)
        mats = mats_of.get(geom_model.get(g.props[0]), [])
        ml = g.find("LayerElementMaterial")
        if ml is not None and mats:
            m = np.asarray(ml.find("Materials").props[0]).astype(np.int64)
            per_tri = m[poly] if ml.find("MappingInformationType").props[0] == "ByPolygon" else np.full(len(F), m[0])
            lut = []
            for mid in mats:
                f = tex_file(diffuse[mid]) if mid in diffuse else None
                if f is not None and f not in files:
                    files.append(f)
                lut.append(files.index(f) if f is not None else -1)
            lut = np.asarray(lut)
            tex = np.where((per_tri >= 0) & (per_tri < len(lut)), lut[np.clip(per_tri, 0, len(lut) - 1)], -1)
        Vs.append(V)
        Fs.append(F + off)
        UVs.append(uvc)
        Ts.append(tex)
        off += len(V)
    if not Vs:
        return None
    return {"V": np.concatenate(Vs), "F": np.concatenate(Fs), "uv": np.concatenate(UVs),
            "tex": np.concatenate(Ts), "files": files}
