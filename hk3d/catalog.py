"""Index of locally available LandsD 3D Digital Map FBX files.

The data root may contain sheet folders (e.g. ``11-NE-18B``) anywhere below it;
each sheet folder contains layer folders (BUILDING, GENERIC, ...) which in turn
contain one folder per object with the FBX + textures. Every FBX is parsed once
to obtain its bounding box; results are cached in ``catalog.json`` and only
new/changed files are rescanned.
"""
from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from . import sheets
from .layers import file_group, folder_group

CATALOG_VERSION = 2


def _scan_file(path: str) -> dict:
    from .fbx import load_meshes

    try:
        meshes = load_meshes(path)
    except Exception as exc:  # corrupt / unsupported file
        return {"error": f"{type(exc).__name__}: {exc}"}
    if not meshes:
        return {"error": "no geometry"}
    lo = np.min([m[1].min(0) for m in meshes], axis=0)
    hi = np.max([m[1].max(0) for m in meshes], axis=0)
    return {
        "bbox": [round(float(v), 3) for v in (*lo, *hi)],
        "ntri": int(sum(len(m[2]) for m in meshes)),
    }


def file_version(size: int, mtime: float) -> str:
    """The one version of a data file that every cache goes by: its size and
    modification time to the microsecond.  The catalogue, the height-map
    cache, the mesh cache and the model key all compare this, so a file
    rewritten within the same second is never served from an older one."""
    return f"{int(size)}:{int(round(float(mtime) * 1e6))}"


@dataclass
class Entry:
    path: str          # relative to data root
    sheet: str
    folder: str        # original layer folder name
    group: str
    bbox: list         # xmin, ymin, zmin, xmax, ymax, zmax (HK1980 / mPD)
    ntri: int
    size: int
    mtime: float

    @property
    def version(self) -> str:
        return file_version(self.size, self.mtime)


@dataclass
class Catalog:
    root: Path
    entries: list[Entry] = field(default_factory=list)
    errors: dict = field(default_factory=dict)
    _bb: np.ndarray | None = None

    # ------------------------------------------------------------------ query
    def _bbox_array(self) -> np.ndarray:
        if self._bb is None or len(self._bb) != len(self.entries):
            self._bb = np.array([e.bbox for e in self.entries], dtype=np.float64).reshape(-1, 6)
        return self._bb

    def query(self, xmin, ymin, xmax, ymax, groups=None) -> list[Entry]:
        bb = self._bbox_array()
        if not len(bb):
            return []
        hit = (bb[:, 0] <= xmax) & (bb[:, 3] >= xmin) & (bb[:, 1] <= ymax) & (bb[:, 4] >= ymin)
        out = [self.entries[i] for i in np.nonzero(hit)[0]]
        if groups is not None:
            out = [e for e in out if e.group in groups]
        return out

    def _buildings(self) -> tuple[list, np.ndarray]:
        idx = [i for i, e in enumerate(self.entries) if e.group == "building"]
        return idx, self._bbox_array()[idx] if idx else np.zeros((0, 6))

    def landmarks(self, e: float, n: float, radius: float = 600.0, limit: int = 8,
                  min_height: float = 40.0) -> list[dict]:
        """Tallest buildings (by height above their base, so a mid-rise on a
        hillside does not outrank a tower) within ``radius`` m."""
        idx, bb = self._buildings()
        if not idx:
            return []
        cx, cy = (bb[:, 0] + bb[:, 3]) / 2, (bb[:, 1] + bb[:, 4]) / 2
        height = bb[:, 5] - np.maximum(bb[:, 2], 0.0)
        dist = np.hypot(cx - e, cy - n)
        ok = np.nonzero((dist <= radius) & (height >= min_height))[0]
        ok = ok[np.argsort(-height[ok])][:limit]
        return [self._building_info(idx[k], bb[k], e, n) for k in ok]

    def building_at(self, e: float, n: float, margin: float = 3.0) -> dict | None:
        """The building under a point: of those whose bounding box contains it
        (within ``margin`` m), the one with the smallest footprint, so a tower
        wins over a podium whose box happens to cover the click."""
        idx, bb = self._buildings()
        if not idx:
            return None
        hit = np.nonzero((bb[:, 0] - margin <= e) & (bb[:, 3] + margin >= e)
                         & (bb[:, 1] - margin <= n) & (bb[:, 4] + margin >= n))[0]
        if not len(hit):
            return None
        area = (bb[hit, 3] - bb[hit, 0]) * (bb[hit, 4] - bb[hit, 1])
        k = hit[np.lexsort((-bb[hit, 5], area))[0]]
        return self._building_info(idx[k], bb[k], e, n)

    def _building_info(self, i: int, b: np.ndarray, e: float, n: float) -> dict:
        cx, cy = float(b[0] + b[3]) / 2, float(b[1] + b[4]) / 2
        return {"e": round(cx, 1), "n": round(cy, 1), "top_m": round(float(b[5]), 1),
                "height_m": round(float(b[5] - max(b[2], 0.0)), 1),
                "width_m": round(float(b[3] - b[0]), 1), "depth_m": round(float(b[4] - b[1]), 1),
                "dist_m": round(float(np.hypot(cx - e, cy - n)), 1), "sheet": self.entries[i].sheet}

    def sheets(self) -> dict[str, dict]:
        """Sheet name -> {bounds, groups{group: count}, ntri}."""
        out: dict[str, dict] = {}
        for e in self.entries:
            s = out.setdefault(e.sheet, {"groups": {}, "ntri": 0, "files": 0})
            s["groups"][e.group] = s["groups"].get(e.group, 0) + 1
            s["ntri"] += e.ntri
            s["files"] += 1
        for name, s in out.items():
            if sheets.is_sheet_name(name):
                s["bounds"] = sheets.sheet_bounds(name)
            else:  # unknown naming: union of contents
                bb = np.array([e.bbox for e in self.entries if e.sheet == name])
                s["bounds"] = (bb[:, 0].min(), bb[:, 1].min(), bb[:, 3].max(), bb[:, 4].max())
        return out

    def coverage_fraction(self, xmin, ymin, xmax, ymax) -> float:
        """Fraction of a bbox covered by locally available sheets (approx)."""
        total = (xmax - xmin) * (ymax - ymin)
        if total <= 0:
            return 0.0
        covered = 0.0
        for s in self.sheets().values():
            x0, y0, x1, y1 = s["bounds"]
            w = max(0.0, min(x1, xmax) - max(x0, xmin))
            h = max(0.0, min(y1, ymax) - max(y0, ymin))
            covered += w * h
        return min(1.0, covered / total)

    # ---------------------------------------------------------------- persist
    def to_json(self) -> dict:
        return {
            "version": CATALOG_VERSION,
            "root": str(self.root),
            "entries": [e.__dict__ for e in self.entries],
            "errors": self.errors,
        }


def _discover(root: Path) -> list[tuple[str, str, str, Path]]:
    """Find (sheet, folder, group, fbx_path) under root."""
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        p = Path(dirpath)
        rel = p.relative_to(root).parts
        # locate the layer folder within the relative path
        for i, part in enumerate(rel):
            group = folder_group(part)
            if group is None:
                continue
            sheet = rel[i - 1] if i > 0 else root.name
            for fn in filenames:
                if fn.lower().endswith(".fbx") and file_group(part, fn):
                    found.append((sheet, part, group, p / fn))
            break
    return found


def load_or_scan(
    root: str | Path,
    cache_dir: str | Path,
    progress: Callable[[int, int, str], None] | None = None,
    workers: int | None = None,
) -> Catalog:
    root = Path(root).resolve()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "catalog.json"
    old: dict[str, dict] = {}
    if cache_file.exists():
        try:
            data = json.loads(cache_file.read_text())
            if data.get("version") == CATALOG_VERSION and data.get("root") == str(root):
                old = {e["path"]: e for e in data["entries"]}
        except (ValueError, KeyError):
            old = {}

    cat = Catalog(root=root)
    todo = []
    for sheet, folder, group, path in _discover(root):
        rel = str(path.relative_to(root))
        st = path.stat()
        prev = old.get(rel)
        if prev and file_version(prev["size"], prev["mtime"]) == file_version(st.st_size, st.st_mtime):
            ent = Entry(**prev)
            ent.group = group  # mapping may change between versions; no rescan needed
            cat.entries.append(ent)
        else:
            todo.append((sheet, folder, group, path, rel, st))

    if todo:
        # big files first so the pool stays busy
        todo.sort(key=lambda t: -t[5].st_size)
        done = 0
        with ProcessPoolExecutor(max_workers=workers or max(1, (os.cpu_count() or 2) - 1)) as ex:
            futs = {ex.submit(_scan_file, str(t[3])): t for t in todo}
            for fut in as_completed(futs):
                sheet, folder, group, path, rel, st = futs[fut]
                res = fut.result()
                done += 1
                if "error" in res:
                    cat.errors[rel] = res["error"]
                else:
                    cat.entries.append(Entry(rel, sheet, folder, group, res["bbox"], res["ntri"],
                                             st.st_size, st.st_mtime))
                if progress:
                    progress(done, len(todo), rel)
        cat.entries.sort(key=lambda e: e.path)
        # one temporary file per process: builds started together all rescan
        tmp = cache_file.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(cat.to_json()))
        tmp.replace(cache_file)
    return cat


class CatalogHolder:
    """Thread-safe holder so the server can rescan in the background."""

    def __init__(self, root, cache_dir):
        self.root, self.cache_dir = Path(root), Path(cache_dir)
        self._cat: Catalog | None = None
        self._lock = threading.Lock()
        self.scanning = False
        self.scan_progress = (0, 0)

    def get(self) -> Catalog:
        with self._lock:
            if self._cat is None:
                self._cat = load_or_scan(self.root, self.cache_dir, self._on_progress)
            return self._cat

    def peek(self) -> Catalog | None:
        """The catalogue if it is loaded, without waiting for a scan."""
        return self._cat

    def rescan(self) -> Catalog:
        self.scanning = True
        try:
            cat = load_or_scan(self.root, self.cache_dir, self._on_progress)
            with self._lock:
                self._cat = cat
            return cat
        finally:
            self.scanning = False

    def _on_progress(self, done, total, _path):
        self.scan_progress = (done, total)
