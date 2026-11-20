"""Rasterise LandsD meshes into per-group 2.5D heightmaps by top-down raycasting."""
from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .catalog import Catalog, file_version
from .fbx import load_meshes

RAY_Z = 3000.0  # metres; above the highest point in HK (~957 mPD)


@dataclass
class Grid:
    cx: float     # HK1980 easting of the centre
    cy: float     # HK1980 northing of the centre
    cell: float   # metres per cell
    n: int        # cells per side (square grid)

    @property
    def half(self) -> float:
        return self.n * self.cell / 2

    def local_coords(self):
        """x (n,) west->east and y (n,) north->south of cell centres, relative to centre."""
        c = -self.half + (np.arange(self.n) + 0.5) * self.cell
        return c, c[::-1].copy()

    def bbox(self, pad=0.0):
        h = self.half + pad
        return self.cx - h, self.cy - h, self.cx + h, self.cy + h


def make_grid(cx: float, cy: float, radius: float, cell: float, pad_cells: int = 3) -> Grid:
    n = int(np.ceil(2 * radius / cell)) + 2 * pad_cells
    return Grid(cx, cy, cell, n)


class MeshCache:
    """Small LRU of cropped-to-nothing raw meshes (users iterate on one area),
    keyed by path, size and modification time: a file replaced by a download
    is read again."""

    def __init__(self, max_tris: int = 40_000_000):
        self.max_tris = max_tris
        self._d: OrderedDict[tuple, list] = OrderedDict()
        self._tris = 0

    def get(self, path: str):
        try:
            st = os.stat(path)
            key = (path, file_version(st.st_size, st.st_mtime))
        except OSError:
            key = (path, None, None)
        if key in self._d:
            self._d.move_to_end(key)
            return self._d[key]
        for old in [k for k in self._d if k[0] == path]:   # an older version of this file
            self._tris -= sum(len(f) for _, f in self._d.pop(old))
        meshes = [(v.astype(np.float64), f.astype(np.int64)) for _, v, f in load_meshes(path)]
        ntri = sum(len(f) for _, f in meshes)
        self._d[key] = meshes
        self._tris += ntri
        while self._tris > self.max_tris and len(self._d) > 1:
            _, old = self._d.popitem(last=False)
            self._tris -= sum(len(f) for _, f in old)
        return meshes


_MESH_CACHE = MeshCache()


def _crop(V: np.ndarray, F: np.ndarray, bbox) -> tuple[np.ndarray, np.ndarray]:
    xmin, ymin, xmax, ymax = bbox
    tv = V[F]  # (m,3,3)
    keep = ((tv[:, :, 0].max(1) >= xmin) & (tv[:, :, 0].min(1) <= xmax)
            & (tv[:, :, 1].max(1) >= ymin) & (tv[:, :, 1].min(1) <= ymax))
    if not keep.any():
        return V[:0], F[:0]
    F = F[keep]
    used, inv = np.unique(F.ravel(), return_inverse=True)
    return V[used], inv.reshape(-1, 3)


def _group_scene(catalog: Catalog, grid: Grid, group: str,
                 progress: Callable[[float, str], None] | None = None):
    """Ray-casting scene of one group's meshes cropped to the grid (local x, y)."""
    import open3d as o3d

    bbox = grid.bbox(pad=grid.cell)
    entries = catalog.query(*bbox, groups={group})
    scene = o3d.t.geometry.RaycastingScene()
    ntri = 0
    geom_entry: dict[int, int] = {}
    for i, e in enumerate(entries):
        for V, F in _MESH_CACHE.get(str(catalog.root / e.path)):
            V, F = _crop(V, F, bbox)
            if not len(F):
                continue
            Vl = np.empty_like(V, dtype=np.float32)
            Vl[:, 0] = V[:, 0] - grid.cx
            Vl[:, 1] = V[:, 1] - grid.cy
            Vl[:, 2] = V[:, 2]
            gid = scene.add_triangles(o3d.core.Tensor(Vl), o3d.core.Tensor(F.astype(np.uint32)))
            geom_entry[int(gid)] = i
            ntri += len(F)
        if progress:
            progress((i + 1) / max(1, len(entries)) * 0.8, f"{group}: {i + 1}/{len(entries)} files")
    return scene, geom_entry, entries, ntri


def raster_group(catalog: Catalog, grid: Grid, group: str,
                 progress: Callable[[float, str], None] | None = None) -> tuple[np.ndarray, dict]:
    """Return (heights (n,n) float32 mPD with NaN where nothing was hit, stats)."""
    import open3d as o3d

    scene, geom_entry, entries, ntri = _group_scene(catalog, grid, group, progress)
    H = np.full((grid.n, grid.n), np.nan, dtype=np.float32)
    owner = np.full((grid.n, grid.n), -1, dtype=np.int32)
    if ntri:
        xs, ys = grid.local_coords()
        gx, gy = np.meshgrid(xs.astype(np.float32), ys.astype(np.float32))
        rays = np.zeros((grid.n, grid.n, 6), dtype=np.float32)
        rays[..., 0], rays[..., 1], rays[..., 2], rays[..., 5] = gx, gy, RAY_Z, -1.0
        res = scene.cast_rays(o3d.core.Tensor(rays))
        t = res["t_hit"].numpy()
        hit = np.isfinite(t)
        H[hit] = RAY_Z - t[hit]
        gids = res["geometry_ids"].numpy()
        lut = np.full(max(geom_entry) + 2, -1, dtype=np.int32)
        for gid, ei in geom_entry.items():
            lut[gid] = ei
        owner[hit] = lut[np.minimum(gids[hit].astype(np.int64), len(lut) - 1)]
    if progress:
        progress(1.0, f"{group}: done")
    return H, {"files": len(entries), "triangles": ntri, "owner": owner,
               "paths": [e.path for e in entries]}


def raster_up(catalog: Catalog, grid: Grid, group: str, z_from: np.ndarray) -> np.ndarray:
    """Lowest surface of ``group`` above ``z_from`` (mPD) in each cell where
    ``z_from`` is finite - the underside of a bridge deck seen from the road
    below it.  NaN where nothing is above."""
    import open3d as o3d

    out = np.full(z_from.shape, np.nan, dtype=np.float32)
    m = np.isfinite(z_from)
    if not m.any():
        return out
    scene, _, _, ntri = _group_scene(catalog, grid, group)
    if not ntri:
        return out
    xs, ys = grid.local_coords()
    gx, gy = np.meshgrid(xs.astype(np.float32), ys.astype(np.float32))
    rays = np.zeros((int(m.sum()), 6), dtype=np.float32)
    rays[:, 0], rays[:, 1], rays[:, 2], rays[:, 5] = gx[m], gy[m], z_from[m], 1.0
    t = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
    out[m] = z_from[m] + t
    return out


def raster_groups(catalog: Catalog, grid: Grid, groups: list[str], cache_dir: Path | None = None,
                  progress: Callable[[float, str], None] | None = None,
                  owners: dict | None = None) -> dict[str, np.ndarray]:
    """Per-group heightmaps, cached on disk by (grid, group, catalog state).

    If ``owners`` is given, it is filled with {group: (owner, paths)} where
    ``owner[i, j]`` indexes ``paths`` (the FBX file whose surface is the top
    of that cell) or is -1.
    """
    out = {}
    for gi, g in enumerate(groups):
        def sub(f, msg, gi=gi):
            if progress:
                progress((gi + f) / len(groups), msg)

        key = None
        if cache_dir is not None:
            ents = catalog.query(*grid.bbox(pad=grid.cell), groups={g})
            sig = json.dumps([round(grid.cx, 2), round(grid.cy, 2), round(grid.cell, 5), grid.n, g,
                              sorted((e.path, e.version) for e in ents)])
            key = hashlib.sha1(sig.encode()).hexdigest()[:16]
            f = Path(cache_dir) / "rasters" / f"{g}-{key}.npy"
            fo = f.with_name(f"{g}-{key}-owner.npz")
            if f.exists() and (owners is None or fo.exists()):
                out[g] = np.load(f)
                if owners is not None:
                    z = np.load(fo)
                    owners[g] = (z["owner"], [str(x) for x in z["paths"]])
                sub(1.0, f"{g}: cached")
                continue
        H, info = raster_group(catalog, grid, g, sub)
        out[g] = H
        if owners is not None:
            owners[g] = (info["owner"], info["paths"])
        if key is not None:
            f.parent.mkdir(parents=True, exist_ok=True)
            np.save(f, H)
            np.savez_compressed(fo, owner=info["owner"], paths=np.array(info["paths"]))
    return out
