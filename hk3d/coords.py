"""WGS84 <-> HK1980 Grid (EPSG:2326) conversion."""
from __future__ import annotations

from functools import lru_cache

from pyproj import Transformer


@lru_cache(maxsize=1)
def _to_hk():
    return Transformer.from_crs("EPSG:4326", "EPSG:2326", always_xy=True)


@lru_cache(maxsize=1)
def _to_wgs():
    return Transformer.from_crs("EPSG:2326", "EPSG:4326", always_xy=True)


def wgs84_to_hk80(lat: float, lon: float) -> tuple[float, float]:
    e, n = _to_hk().transform(lon, lat)
    return float(e), float(n)


def hk80_to_wgs84(e: float, n: float) -> tuple[float, float]:
    lon, lat = _to_wgs().transform(e, n)
    return float(lat), float(lon)
