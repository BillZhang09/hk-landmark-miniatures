# HK3D: 3D-printable Hong Kong from the LandsD 3D Digital Map

HK3D turns the Hong Kong Lands Department (LandsD) **3D Digital Map** (the
Open3Dhk individualised 3D models, FBX) into watertight, 3D-printable
miniatures: a single landmark on a round 80 mm base at about 1:2,500, or a
whole neighbourhood at about 1:6,000. It has a local web app (pick a point on
the map, preview, download) and a command line.

> 把地政總署「三維數碼地圖」（Open3Dhk 個體化模型，FBX）轉換成可直接 3D 打印的水密微縮模型：地標擺設約 1:2,500，城市景觀約 1:6,000。有網頁應用和命令行，並附精度和可打印性的獨立驗證工具。

Map © Lands Department, HKSAR. This project is not affiliated with or endorsed by
the Lands Department.

## What it does

- Works out which 1:1000 map sheets an area needs, and downloads missing ones
  from the Open3Dhk download API.
- Builds a **guaranteed-watertight solid** from LandsD meshes that are never
  watertight themselves (the six validation towers have 246 to 22,676 open
  edges each in the raw data).
- Makes it **printable without supports** by default: overhangs get 45° material
  or are bridged within the span you give, thin spires are thickened, see-through
  openings at building feet and under footbridges are kept when they are big
  enough to print.
- Writes STL, 3MF (optionally multi-colour parts with Bambu Studio / Orca
  filament slots), a GLB preview and a top-view PNG. Models larger than the
  printer bed are cut into pieces along streets, with pin holes and engraved
  piece labels.
- Ships independent checks (`tools/`) that measure how far a print is from the
  raw LandsD mesh and whether every layer really stands on the one below.

## Pipeline

```
pick a point ─▶ sheets needed (1:1000 grid) ─▶ download missing sheets (FBX0 / FBX)
  ─▶ index FBX files (bounding-box catalog)
  ─▶ top-down ray casting per layer (terrain, ground, roads, bridges, water, trees, buildings)
       → one heightmap per layer
  ─▶ ground clean-up, road recess, remove unprintable specks, thicken spires
  ─▶ precise carving of buildings from their facade directions (--precise / --landmark):
       flat facades, see-through gaps, optional facade relief from the texture
  ─▶ watertight solid (manifold3d) ─▶ support-free fill and bridges
  ─▶ boolean cut to circle / hexagon / square ─▶ simplify (≈0.02 mm)
  ─▶ STL / 3MF (multi-colour parts) / pieces ZIP for big models / GLB / PNG
```

The model is a 2.5D heightmap at its core, not stitched raw meshes: ray casting
(Open3D / Embree) gives the top surface of every cell and guarantees a closed
solid. Buildings carved with `--precise` keep facade detail and openings that a
pure heightmap would fill. Results are cached per parameter set in
`cache/jobs/<hash>/`; the hash includes `PIPELINE_VERSION` and the size and
modification time of every input file, so new code or re-downloaded data never
returns a stale model.

### LandsD layers

| LandsD folder | Used for | Default |
|---|---|---|
| `TERRAIN(TB)` | bare-earth DTM, fills holes in other layers | ✓ |
| `GENERIC` | ground reality mesh (no buildings, roads, trees) | ✓ |
| `INFRASTRUCTURE(TB)` | roads (reality mesh) | ✓ |
| `INFRASTRUCTURE` | footbridges and flyovers; tunnels (underground) are skipped | ✓ |
| `WATERBODY` | water | ✓ |
| `VEGETATION(TB)` | trees (reality mesh) | optional |
| `BUILDING` | individual building models | ✓ |

The untextured download (`FBX0`, ~60 MB per sheet) has `BUILDING`,
`INFRASTRUCTURE` and `TERRAIN(TB)` only, which is enough for buildings. The
textured download (`FBX`, ~0.5–2 GB per sheet) adds the ground, roads, trees
and water, enables multi-colour parts, and is needed for facade relief.

## Install

Python 3.10–3.12.

```bash
git clone https://github.com/BillZhang09/hk-landmark-miniatures.git
cd hk-landmark-miniatures
python3.11 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/pytest -q        # tests that need LandsD data are skipped without it
```

With [uv](https://github.com/astral-sh/uv):
`uv venv -p 3.11 .venv && uv pip install -p .venv -e '.[dev]'`.

Put LandsD sheet folders (for example `11-SW-14A/`, each containing `BUILDING/`,
`TERRAIN(TB)/` …, exactly as unzipped from the download) in one folder, and
point `--data` (or `HK3D_DATA`) at it:

```bash
.venv/bin/hk3d serve --data ./data        # open http://127.0.0.1:8765
```

The first start indexes every FBX file (a few seconds for ten sheets) and
caches the result in `cache/catalog.json` (`--cache` or `HK3D_CACHE` to move it).

## Docker

The image holds the dependencies and the code; LandsD data and outputs are
mounted. Running builds in a container lets you cap memory: a landmark build
with facade relief can use many GB, and a capped container is killed cleanly
instead of exhausting the host.

```bash
docker build -f docker/Dockerfile -t hk3d .

# data in ./data, outputs and cache in ./cache, 16 GB cap
HK3D_MEM=16g docker/run.sh index
docker/run.sh build --hk80 834691 815600 --landmark --name BoC --out /cache/BoC
docker/run.sh serve --host 0.0.0.0        # then open http://127.0.0.1:8765
```

`docker/run.sh` reads `HK3D_DATA` (default `./data`, mounted read-only),
`HK3D_CACHE` (default `./cache`), `HK3D_MEM` (default `16g`) and passes
`HK3D_LANDSD_KEY` through when it is set.

## Using your own LandsD key: `HK3D_LANDSD_KEY`

`hk3d download` and the web app's download dialog fetch sheets from the
Open3Dhk download API (`https://download.map.gov.hk/api/3d-zip/{FBX0|FBX}/{sheet}.zip`).
LandsD asks API users to obtain a free key (see the Open3Dhk site,
<https://3d.map.gov.hk/>, for how to apply) and to avoid bulk bursts. Keep the
key in your own environment; it is never stored in this repository or in the
cache:

```bash
export HK3D_LANDSD_KEY=...        # your own key, e.g. in your shell profile
hk3d download --data ./data 11-SW-14A 11-SW-9C
```

When set, the key is appended to the download URL; the web app only reports
whether one is present. Downloads run one sheet at a time. Please respect the
Open3Dhk fair-usage policy.

## Command line

```bash
hk3d index     --data ./data                               # local sheets and their layers
hk3d sheets    --latlon 22.2819 114.1582 --radius 300      # which sheets an area needs
hk3d landmarks --data ./data --latlon 22.2810 114.1595     # tallest buildings nearby (with HK1980 coordinates)
hk3d download  --data ./data 11-SW-14A 11-SW-14B           # untextured FBX0 by default; --format FBX for textured

# a neighbourhood, multi-colour parts
hk3d build --data ./data --latlon 22.3205 114.2225 --radius 300 --diameter 100 --multicolor

# landmark preset: R 100 m, 80 mm (≈1:2,500), 0.05 mm detail, 5 mm base,
# every building carved precisely with facade relief, trimmed edge, display holder ring
hk3d build --data ./data --hk80 834691 815600 --landmark --name "BoC"     # Bank of China Tower (11-SW-14A + 9C)
hk3d build --data ./data --hk80 834456 816274 --landmark --name "IFC2"    # Two IFC (11-SW-8B)
hk3d build --data ./data --hk80 834550 818266 --landmark --name "ICC"     # ICC (11-NW-24C, 23D)
```

Useful `build` options (all have defaults tuned for a 0.4 mm nozzle):

| Option | Meaning |
|---|---|
| `--radius`, `--diameter`, `--shape` | real-world radius (m), printed size (mm), `circle` / `hexagon` / `square` |
| `--base`, `--zscale`, `--detail` | base thickness (mm), vertical exaggeration, heightmap cell on the print (mm) |
| `--nozzle`, `--line-width`, `--layer-height`, `--first-layer` | printer settings; facade ribs are at least one line wide, floor lines two layers tall |
| `--supports`, `--bridge` | you will print with supports (no 45° fill added); longest span your printer bridges |
| `--openings` | keep openings under buildings and bridges at least this big (mm), 0 = fill |
| `--spire`, `--min-feature` | thicken masts to this width; remove ground noise thinner than this |
| `--precise`, `--precise-all`, `--relief` | carve buildings from facade directions; all buildings or the centre one; facade relief depth (needs textured FBX) |
| `--trim-edge`, `--holder` | drop buildings cut by the outline; also output a display ring |
| `--road-recess`, `--ground-clean` | sink roads by N layers so streets read on a one-colour print; remove cars/bushes below this size |
| `--bed`, `--pin`, `--no-tile` | build volume; big models are cut into pieces with pin holes |
| `--multicolor`, `--layers`, `--name`, `--no-engrave`, `--out` | parts, layer list, label, bottom engraving, output folder |

`hk3d build --help` lists everything. The web app accepts links such as
`http://127.0.0.1:8765/?lat=22.2776&lon=114.1658&preset=landmark`.

## Outputs

- **STL**: one watertight solid, millimetres, Z up, origin at the bottom centre.
- **3MF**: one object; with `--multicolor` one object of aligned parts (base,
  buildings, roads, water, trees) with part names and filament slots for
  Bambu Studio / Orca. A ZIP with one STL per part is written too.
- **Engraving** underneath: name, coordinates, radius, scale and the data
  credit (mirrored, 0.6 mm deep; Chinese names work with a CJK font installed).
- **Holder ring** (`--holder`): a display base the model drops into (0.3 mm clearance).
- **Pieces** when the model is bigger than `--bed`: `<name>_pieces.zip`,
  `assembly.png` and `assembly.txt`. Seams follow streets, tall towers are cut
  where the cross-section is large and constant, every piece prints on its own.
- **result.json**: parameters, statistics, data coverage and the carved source
  files, used by the verification tools.

## Verifying accuracy and printability

The generator has rules of its own (sampling grid, support fill, bridging), so
acceptance uses tools that do not share them. All read the model's centre,
scale and base from the `result.json` (or `<name>_result.json`) written beside
the model and stop rather than guess when it is missing.

- **`tools/deviation.py`**: how far the print is from the raw LandsD mesh, in
  real-world metres. It samples 300,000 points on the outer surface of the raw
  building that the build carved, measures each to the output model, and splits
  large deviations into *buried* (a recess or open shell filled solid) and
  *missing* (geometry the print lost). It also reports the elevation fitted from
  upward roofs, so an error in the whole model's height shows up.

  ```bash
  python tools/deviation.py prints/BoC.stl       # a model with BoC_result.json beside it
  python tools/deviation.py <prefix> boc ifc2    # jobs in cache/jobs/<prefix>_<name>/
  ```

- **`tools/layercheck.py`**: an **independent layer-by-layer check**, sliced
  like a slicer at mid-layer heights and compared with the layer below. It reports
  *islands* that start in mid-air, *overhangs* beyond half a line width (each
  tested for a one-direction bridge within `--bridge`, the rest unsupported,
  severe ones listed apart) and *thin parts* narrower than one line width for
  three layers or more, separately for each layer height.

  ```bash
  python tools/layercheck.py BoC.stl --layers 0.08,0.12,0.2 --line 0.4 --bridge 10 --json BoC_layers.json
  ```

- **`tools/stagetrace.py`**: which pipeline step a defect comes from. Build
  with `HK3D_DEBUG_STAGES=<dir>` to save the model after every step, then trace
  the largest defects from a `layercheck` report through the stages:
  `python tools/stagetrace.py <dir> BoC_layers.json --model BoC.stl --layer 0.2`.

- **`tools/printcheck.py <stl> [layer] [bridge]`**: re-runs the generator's own support rule with other
  ray offsets (quick, but with the generator's blind spots; `layercheck` decides).

- Data inspection: `tools/raw_topo.py` (open edges and components of the raw
  meshes), `tools/facade_check.py` (is facade detail real geometry or texture),
  `tools/relief_probe.py`, `tools/relief_sheet.py`, `tools/relief_metrics.py`
  (facade relief on single buildings, with a manifest of code version and input
  SHA-256s so a run can be repeated).

Measured on six Central and Admiralty towers with the landmark preset
(1:2,500, 0.25 m cells, no facade relief), above the lowest 12 % of the height
where buildings meet the ground and podiums:

| Tower | within 0.25 m | within 0.5 m | > 0.5 m, buried (base + above) | > 0.5 m, missing |
|---|---|---|---|---|
| Bank of China Tower | 99 % | 100 % | 0.2 % + 0.2 % | 0.2 % |
| Two IFC | 98 % | 99 % | 0.4 % + 1.0 % | 0.0 % |
| The Center | 99 % | 99 % | 1.1 % + 0.7 % | 0.0 % |
| Cheung Kong Center | 99 % | 100 % | 0.9 % + 0.0 % | 0.1 % |
| HSBC Main Building | 80 % | 92 % | 5.6 % + 5.7 % | 0.5 % |
| Admiralty office tower | 97 % | 99 % | 1.3 % + 1.0 % | 0.3 % |

Deviations are almost all *filled* rather than *lost*: surfaces seen only from
below or at an angle (soffits, gaps between HSBC's exposed structure) become
solid, which also prints more easily. 0.25 m on the ground is 0.1 mm on the print.

## Sample models

Print-ready samples of the Bank of China Tower, Two IFC and ICC (landmark
preset, 80 mm, 1:2,500, 0.4 mm nozzle, 0.2 mm layers, support-free) are
attached to the [GitHub Release `samples-v24`](https://github.com/BillZhang09/hk-landmark-miniatures/releases/tag/samples-v24)
rather than stored in git:

| File | Centre (HK1980) | Tower | Model height |
|---|---|---|---|
| `BoC.3mf` + `BoC_holder.stl` | 834690.8, 815599.9 | 366 m | 152 mm |
| `IFC2.3mf` + `IFC2_holder.stl` | 834455.6, 816273.8 | 414 m | 171 mm |
| `ICC.3mf` + `ICC_holder.stl` | 834550.4, 818265.5 | 463 m | 200 mm |

They are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (credit: BillZhang09, hk-landmark-miniatures) for this project's own contribution; the LandsD data in them stays under the LandsD terms below, so keep both credits when you share them.

## Data source and terms

**Map © Lands Department, HKSAR.** 地圖由地政總署提供 / Map from Lands Department.

The 3D data comes from the Lands Department of the Government of the Hong Kong
Special Administrative Region through **Open3Dhk** (<https://3d.map.gov.hk/>).
The Government owns the intellectual property rights in the data and in all
copies of it. The Open3Dhk *Terms and Conditions of Use* allow browsing,
downloading, distributing, reproducing and printing the data, commercially and
non-commercially, free of charge, on condition that users identify the
Government and Open3Dhk as the source, acknowledge the Government's ownership of
the intellectual property rights, and indemnify the Government against claims
arising from their use. The data is provided "as is". Read the current terms on
the Open3Dhk site and <https://api.portal.hkmapservice.gov.hk/disclaimer> before
using or sharing data or models.

What this means here:

- This repository contains **no LandsD data**. Download sheets yourself.
- Models you generate are derived from LandsD data: keep the credit (it is
  engraved underneath by default and written into the 3MF metadata) and follow
  the terms above when you share them.
- The Apache License 2.0 covers the code only; the LandsD data is not covered by it and remains subject to the Open3Dhk / LandsD (and, where applicable, DATA.GOV.HK) terms.
- This is an independent project; it is not affiliated with or endorsed by the
  Lands Department or the HKSAR Government.

## Licence

- **Code**: [Apache License 2.0](LICENSE), Copyright 2026 BillZhang09. You may
  use it commercially, modify it and ship it in closed-source products. When you
  redistribute it, original or modified, include [`LICENSE`](LICENSE) and
  [`NOTICE`](NOTICE), keep the copyright and attribution notices, and mark the
  files you changed as modified.
- **Sample models** (GitHub Release `samples-v24`): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
  for this project's contribution; credit BillZhang09 and the data source.
- **LandsD data**: not covered by either licence. Map © Lands Department, HKSAR;
  follow the Open3Dhk / LandsD terms in "Data source and terms" above.
