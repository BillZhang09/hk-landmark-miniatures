"""FastAPI web app: pick a place on the map, generate, preview and download."""
from __future__ import annotations

import json
import os
import queue
import threading
import time
import traceback
import urllib.parse
import urllib.request
import uuid
from dataclasses import asdict, fields
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import download, sheets
from .catalog import CatalogHolder
from .coords import hk80_to_wgs84, wgs84_to_hk80
from .layers import GROUP_ORDER, GROUPS
from .pipeline import ModelParams, build, input_signature

STATIC = Path(__file__).parent / "static"


class JobRequest(BaseModel):
    lat: float | None = None
    lon: float | None = None
    e: float | None = None
    n: float | None = None
    radius_m: float = 300
    diameter_mm: float = 100
    base_mm: float = 3
    z_scale: float = 1
    detail_mm: float = 0.1
    groups: list[str] = Field(default_factory=lambda: [g for g in GROUP_ORDER if GROUPS[g]["default"]])
    min_feature_mm: float = 0.3
    shape: str = "circle"
    multicolor: bool = False
    skin_mm: float = 0.6
    floor_m: float | None = None
    name: str = ""
    engrave: bool = True
    ground_clean_m: float = 2.5
    spire_mm: float = 0.6
    line_mm: float = 0.4
    layer_mm: float = 0.2
    first_layer_mm: float | None = None
    trim_edge: bool = False
    precise_landmark: bool = False
    precise_all: bool = False
    facade_relief_mm: float = 0.0
    road_recess_layers: float = 1.0
    openings_mm: float = 1.0
    support_free: bool = True
    bridge_mm: float = 10.0
    holder: bool = False
    tile: bool = True
    bed_mm: list[float] = Field(default_factory=lambda: [256.0, 256.0, 256.0])
    pin_mm: float = 1.75


class DownloadRequest(BaseModel):
    sheets: list[str]
    format: str = "FBX0"


class Job:
    def __init__(self, kind: str, payload, key: str | None = None):
        self.id = uuid.uuid4().hex[:10]
        self.kind = kind
        self.payload = payload
        self.key = key
        self.state = "queued"
        self.progress = 0.0
        self.message = "Queued"
        self.result = None
        self.error = None
        self.created = time.time()

    def view(self):
        return {"id": self.id, "kind": self.kind, "state": self.state, "progress": round(self.progress, 3),
                "message": self.message, "result": self.result, "error": self.error, "key": self.key}


def create_app(data_root: str | Path, cache_dir: str | Path) -> FastAPI:
    data_root, cache_dir = Path(data_root).resolve(), Path(cache_dir).resolve()
    jobs_dir = cache_dir / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    holder = CatalogHolder(data_root, cache_dir)
    jobs: dict[str, Job] = {}
    q: queue.Queue[Job] = queue.Queue()

    def worker():
        while True:
            job = q.get()
            job.state = "running"
            try:
                if job.kind == "model":
                    run_model(job)
                elif job.kind == "download":
                    run_download(job)
                job.state, job.progress = "done", 1.0
            except Exception as exc:  # report to UI
                traceback.print_exc()
                job.state, job.error = "error", str(exc)
            finally:
                q.task_done()

    def run_model(job: Job):
        params: ModelParams = job.payload

        def prog(f, msg):
            job.progress, job.message = f, msg

        job.message = "Loading catalogue"
        cat = holder.get()
        # the key covers the input files too: data downloaded or replaced
        # since the request came in gives a new model, not the old one
        job.key = params.key(input_signature(cat, params))
        out = jobs_dir / job.key
        res_file = out / "result.json"
        if res_file.exists():
            job.result, job.message = json.loads(res_file.read_text()), "Done (cached)"
            return
        job.result = build(params, cat, out, cache_dir, prog)

    def run_download(job: Job):
        req: DownloadRequest = job.payload
        n = len(req.sheets)
        for i, s in enumerate(req.sheets):
            def prog(done, total, i=i, s=s):
                frac = (done / total) if total else 0
                job.progress = (i + frac) / n * 0.9
                job.message = f"Downloading {s} ({done / 1e6:.0f}{f'/{total / 1e6:.0f}' if total else ''} MB)"
            download.download_sheet(s, data_root, req.format, progress=prog)
        job.message = "Indexing new sheets"
        holder.rescan()
        job.result = {"sheets": req.sheets}

    threading.Thread(target=worker, daemon=True).start()
    # warm the catalogue in the background so the first request is fast
    threading.Thread(target=holder.get, daemon=True).start()

    app = FastAPI(title="HK 3D Print Model Generator")

    @app.get("/api/info")
    def info():
        return {
            "data_root": str(data_root),
            "groups": [{"key": k, **v} for k, v in GROUPS.items()],
            "defaults": {f.name: f.default for f in fields(ModelParams)
                         if f.name not in ("e", "n", "groups") and not callable(f.default)},
            "scanning": holder.scanning,
            "scan_progress": holder.scan_progress,
            "has_key": bool(os.environ.get("HK3D_LANDSD_KEY")),
        }

    def sheet_feature(name, props):
        x0, y0, x1, y1 = sheets.sheet_bounds(name)
        ring = [hk80_to_wgs84(x, y)[::-1] for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0))]
        return {"type": "Feature", "properties": {"name": name, **props},
                "geometry": {"type": "Polygon", "coordinates": [ring]}}

    @app.get("/api/sheets")
    def local_sheets():
        cat = holder.get()
        feats = [sheet_feature(name, {"local": True, "groups": s["groups"], "files": s["files"]})
                 for name, s in cat.sheets().items() if sheets.is_sheet_name(name)]
        return {"type": "FeatureCollection", "features": feats}

    @app.get("/api/plan")
    def plan(lat: float, lon: float, radius_m: float):
        e, n = wgs84_to_hk80(lat, lon)
        cat = holder.get()
        local = cat.sheets()
        need = sheets.sheets_in_bbox(e - radius_m, n - radius_m, e + radius_m, n + radius_m)
        feats = [sheet_feature(s, {"local": s in local}) for s in need]
        # local sheets from the FBX0 download have no ground/road/water layers
        basic = [s for s in need if s in local
                 and not {"ground", "infrastructure", "water", "vegetation"} & set(local[s]["groups"])]
        return {"e": e, "n": n, "sheets": need, "missing": [s for s in need if s not in local],
                "basic": basic,
                "coverage": cat.coverage_fraction(e - radius_m, n - radius_m, e + radius_m, n + radius_m),
                "geojson": {"type": "FeatureCollection", "features": feats}}

    def _with_wgs(b: dict | None) -> dict | None:
        if b is not None:
            b["lat"], b["lon"] = hk80_to_wgs84(b["e"], b["n"])
        return b

    @app.get("/api/landmarks")
    def landmarks(lat: float, lon: float, radius_m: float = 600, limit: int = 8):
        e, n = wgs84_to_hk80(lat, lon)
        items = holder.get().landmarks(e, n, min(radius_m, 3000), max(1, min(limit, 30)))
        return [_with_wgs(b) for b in items]

    @app.get("/api/snap")
    def snap(lat: float, lon: float):
        e, n = wgs84_to_hk80(lat, lon)
        return {"building": _with_wgs(holder.get().building_at(e, n))}

    @app.get("/api/search")
    def search(q: str):
        # LandsD/OGCIO Address Lookup Service; returns HK1980 + WGS84 coordinates
        url = "https://www.als.gov.hk/lookup?" + urllib.parse.urlencode({"q": q, "n": 8})
        req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": download.USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                data = json.load(r)
        except Exception as exc:
            raise HTTPException(502, f"search service unavailable: {exc}") from exc
        out = []
        for item in data.get("SuggestedAddress", []):
            pa = item["Address"]["PremisesAddress"]
            geo = pa.get("GeospatialInformation") or {}
            if isinstance(geo, list):
                geo = geo[0] if geo else {}
            if not geo.get("Latitude"):
                continue
            eng, chi = pa.get("EngPremisesAddress", {}), pa.get("ChiPremisesAddress", {})

            def line(d, keys):
                parts = []
                for k in keys:
                    v = d.get(k)
                    if isinstance(v, dict):
                        v = v.get("EstateName") or v.get("StreetName") or v.get("DcDistrict") or v.get("LocationName")
                    if v:
                        parts.append(str(v))
                return ", ".join(parts)

            out.append({
                "label": line(chi, ["BuildingName", "ChiStreet", "ChiEstate", "ChiDistrict"]) or
                         line(eng, ["BuildingName", "EngStreet", "EngEstate", "EngDistrict"]),
                "sub": line(eng, ["BuildingName", "EngStreet", "EngDistrict"]),
                "lat": float(geo["Latitude"]), "lon": float(geo["Longitude"]),
            })
        return out

    @app.post("/api/jobs")
    def create_job(req: JobRequest):
        if req.e is None or req.n is None:
            if req.lat is None or req.lon is None:
                raise HTTPException(400, "give lat/lon or e/n")
            req.e, req.n = wgs84_to_hk80(req.lat, req.lon)
        d = req.model_dump()
        d.pop("lat"), d.pop("lon")
        params = ModelParams(**d)
        try:
            params.validate()
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        cat = holder.peek()          # do not wait here for a first scan
        key = params.key(input_signature(cat, params)) if cat is not None else None
        res_file = jobs_dir / key / "result.json" if key else None
        job = Job("model", params, key)
        jobs[job.id] = job
        if res_file is not None and res_file.exists():  # identical request, same data, already built
            job.state, job.progress, job.message = "done", 1.0, "Done (cached)"
            job.result = json.loads(res_file.read_text())
        else:
            q.put(job)
        return job.view()

    @app.post("/api/downloads")
    def create_download(req: DownloadRequest):
        bad = [s for s in req.sheets if not sheets.is_sheet_name(s)]
        if bad or not req.sheets:
            raise HTTPException(400, f"invalid sheets {bad}")
        if len(req.sheets) > 6:
            raise HTTPException(400, "LandsD allows at most 6 sheets per request; download in batches")
        if req.format not in download.FORMATS:
            raise HTTPException(400, "format must be FBX0 or FBX")
        job = Job("download", req)
        jobs[job.id] = job
        q.put(job)
        return job.view()

    @app.post("/api/rescan")
    def rescan():
        threading.Thread(target=holder.rescan, daemon=True).start()
        return {"ok": True}

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(404, "no such job")
        return job.view()

    @app.get("/api/history")
    def history():
        out = []
        for d in sorted(jobs_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)[:30]:
            f = d / "result.json"
            if f.exists():
                r = json.loads(f.read_text())
                out.append({"key": d.name, "lat": r["lat"], "lon": r["lon"], "params": r["params"],
                            "stats": r["stats"], "files": r["files"]})
        return out

    @app.get("/files/{key}/{name}")
    def get_file(key: str, name: str):
        if "/" in key or ".." in key or "/" in name or ".." in name:
            raise HTTPException(400, "bad path")
        path = jobs_dir / key / name
        if not path.is_file():
            raise HTTPException(404, "not found")
        media = {"glb": "model/gltf-binary", "png": "image/png", "json": "application/json"}.get(name.rsplit(".", 1)[-1])
        if media:
            return FileResponse(path, media_type=media)
        return FileResponse(path, filename=name)

    @app.middleware("http")
    async def revalidate_ui(request, call_next):
        # the UI changes with the app; make browsers revalidate (cheap 304s)
        # instead of running a stale app.js after an update
        resp = await call_next(request)
        if not request.url.path.startswith(("/api/", "/files/")):
            resp.headers["Cache-Control"] = "no-cache"
        return resp

    app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
    return app


def _default_app():
    return create_app(os.environ.get("HK3D_DATA", ".."), os.environ.get("HK3D_CACHE", "cache"))
