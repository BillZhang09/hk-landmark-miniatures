"""Fetch LandsD 3D Digital Map sheets (Open3Dhk download API) into the data root.

URL pattern (Open3Dhk "Download API"):
    https://download.map.gov.hk/api/3d-zip/{format}/{sheet}.zip[?key=...]
Formats: FBX (textured, ~0.5-2 GB per sheet) or FBX0 (no textures, ~50 MB) -
textures are not needed for printing, so FBX0 is the default.

LandsD asks API users to obtain a free key from 3dmap@landsd.gov.hk and to
avoid bulk bursts; set HK3D_LANDSD_KEY to send yours. Downloads are sequential.
Data (c) Lands Department, HKSAR Government ("Map from Lands Department").
"""
from __future__ import annotations

import os
import shutil
import tempfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable

from .layers import FOLDER_GROUP
from .sheets import is_sheet_name

URL = "https://download.map.gov.hk/api/3d-zip/{fmt}/{sheet}.zip"
FORMATS = ("FBX0", "FBX")
USER_AGENT = "hk3d/0.1 (3D print model generator)"


def sheet_url(sheet: str, fmt: str = "FBX0", key: str | None = None) -> str:
    if not is_sheet_name(sheet):
        raise ValueError(f"invalid sheet name {sheet!r}")
    if fmt not in FORMATS:
        raise ValueError(f"format must be one of {FORMATS}")
    url = URL.format(fmt=fmt, sheet=sheet)
    key = key or os.environ.get("HK3D_LANDSD_KEY")
    if key:
        url += "?" + urllib.parse.urlencode({"key": key})
    return url


def remote_size(sheet: str, fmt: str = "FBX0", key: str | None = None) -> int | None:
    req = urllib.request.Request(sheet_url(sheet, fmt, key), method="HEAD", headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return int(r.headers.get("Content-Length") or 0) or None
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def _safe_extract(zf: zipfile.ZipFile, dest: Path) -> None:
    dest = dest.resolve()
    for info in zf.infolist():
        target = (dest / info.filename).resolve()
        if not str(target).startswith(str(dest) + os.sep) and target != dest:
            raise ValueError(f"unsafe path in zip: {info.filename}")
    zf.extractall(dest)


def _find_layer_parent(root: Path) -> Path | None:
    """Directory that directly contains the layer folders (BUILDING, GENERIC...)."""
    for dirpath, dirnames, _ in os.walk(root):
        if any(d.upper() in FOLDER_GROUP for d in dirnames):
            return Path(dirpath)
    return None


TEXTURED_ONLY = ("GENERIC", "INFRASTRUCTURE(TB)", "VEGETATION(TB)", "WATERBODY")


def is_textured(folder: Path) -> bool:
    """True if a local sheet folder came from the full FBX download (has ground layers)."""
    return any((Path(folder) / d).is_dir() for d in TEXTURED_ONLY)


def download_sheet(sheet: str, data_root: Path, fmt: str = "FBX0", key: str | None = None,
                   progress: Callable[[int, int | None], None] | None = None) -> Path:
    """Download + unpack one sheet into ``data_root/<sheet>/``. Returns that folder.

    An existing sheet is left alone, except that asking for ``FBX`` upgrades
    a sheet that only has the FBX0 layers.  The old folder is replaced only
    after the new one has been fully downloaded and unpacked.
    """
    data_root = Path(data_root)
    target = data_root / sheet
    if target.exists() and any(target.iterdir()):
        if not (fmt == "FBX" and not is_textured(target)):
            return target
    url = sheet_url(sheet, fmt, key)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with tempfile.TemporaryDirectory(dir=data_root, prefix=f".dl-{sheet}-") as tmp:
        tmp = Path(tmp)
        zpath = tmp / f"{sheet}.zip"
        try:
            with urllib.request.urlopen(req, timeout=60) as r, open(zpath, "wb") as fh:
                total = int(r.headers.get("Content-Length") or 0) or None
                done = 0
                while chunk := r.read(1 << 20):
                    fh.write(chunk)
                    done += len(chunk)
                    if progress:
                        progress(done, total)
        except urllib.error.HTTPError as e:
            msg = {401: "access denied (an API key may be required: set HK3D_LANDSD_KEY)",
                   404: "sheet not available"}.get(e.code, f"HTTP {e.code}")
            raise RuntimeError(f"{sheet}: {msg}") from e
        out = tmp / "x"
        with zipfile.ZipFile(zpath) as zf:
            _safe_extract(zf, out)
        parent = _find_layer_parent(out)
        if parent is None:
            raise RuntimeError(f"{sheet}: unexpected zip layout (no layer folders found)")
        staging = tmp / sheet
        shutil.move(str(parent), staging)
        if target.exists():
            # swap out the old copy; it is deleted together with tmp
            shutil.move(str(target), tmp / "old")
        shutil.move(str(staging), target)
    return target
