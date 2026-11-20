"""LandsD 3D Digital Map layer folders -> semantic groups used by the pipeline."""
from __future__ import annotations

# Folder name (as delivered by LandsD) -> group key.
FOLDER_GROUP = {
    "TERRAIN(TB)": "terrain",          # bare-earth DTM, coarse but gap-free
    "GENERIC": "ground",               # photo-mesh of ground surface (no buildings/roads/trees)
    "INFRASTRUCTURE(TB)": "infrastructure",  # photo-mesh of roads / paved areas
    "INFRASTRUCTURE": "bridge",              # modelled bridges, footbridges, flyovers
    "WATERBODY": "water",
    "VEGETATION(TB)": "vegetation",
    "BUILDING": "building",
}

# Ordered bottom -> top. "label" is shown in the UI.
GROUPS = {
    "terrain": {"label": "地形 Terrain (DTM)", "default": True, "color": "#cbbfa6"},
    "ground": {"label": "地面細節 Ground detail", "default": True, "color": "#cbbfa6"},
    "infrastructure": {"label": "道路 Roads", "default": True, "color": "#8c8c8c"},
    "bridge": {"label": "天橋及行人橋 Bridges & footbridges", "default": True, "color": "#8c8c8c"},
    "water": {"label": "水體 Water bodies", "default": True, "color": "#6aa6dc"},
    "vegetation": {"label": "植被 Vegetation", "default": False, "color": "#86b06f"},
    "building": {"label": "建築物 Buildings", "default": True, "color": "#f7f5f0"},
}
GROUP_ORDER = list(GROUPS)

# Groups printed as their own body in multi-colour mode, and how:
#  "column": body goes from the ground surface up to the top (buildings, trees)
#  "skin":   only a thin top layer of this colour (roads, water)
PART_MODE = {"building": "column", "vegetation": "column", "infrastructure": "skin", "water": "skin"}
# Groups that form the (cleanable) ground surface, as opposed to objects on it.
SURFACE_GROUPS = ("terrain", "ground", "infrastructure", "water")
BASE_COLOR = "#cbbfa6"


def folder_group(name: str) -> str | None:
    return FOLDER_GROUP.get(name.upper())


# Feature codes (characters 12-13 of a LandsD model ID, e.g. I343751637404063A0)
# of modelled infrastructure that is not printed.  04 is a tunnel: every
# such file runs below the ground it passes under (to -33 mPD at Central)
# and only its portals and roof slabs reach the surface, where they stood
# as blocks in the street.
SKIP_FEATURES = {"INFRASTRUCTURE": ("04",)}


def file_group(folder: str, filename: str) -> str | None:
    """Group of one model file in a layer folder, or None to leave it out."""
    group = folder_group(folder)
    skip = SKIP_FEATURES.get(folder.upper())
    if group and skip and len(filename) >= 13 and filename[11:13] in skip:
        return None
    return group
