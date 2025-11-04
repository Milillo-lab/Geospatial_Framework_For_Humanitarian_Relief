## B) Calculation of Damaged Buildings
# =============================== B0) Imports ===============================
import os
from pathlib import Path

import numpy as np
import geopandas as gpd

import rasterio as rio
from rasterio.mask import mask
from rasterio.features import shapes
from rasterio.windows import from_bounds

from shapely.geometry import shape, box
from shapely.ops import unary_union
from shapely.prepared import prep

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from IPython.display import display, Image
import yaml
# ============== B1) Data, Outputs, and User Parameters ===================

# Directories
DATA_DIR   = r"./Data"
OUT_BASE     = r"./Outputs"

# Fallback to Section A outputs if memory is missing
A_OUT_DIR    = os.path.join(OUT_BASE, "A")

# This section's (B) outputs
OUT_DIR      = os.path.join(OUT_BASE, "B")
IMAGES_DIR   = os.path.join(OUT_BASE, "Images")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# Data
DPM_PATH = Path(os.path.join(DATA_DIR, require("DPM_FILE", str)))

# Outputs
EVAL_OUT_SHP      = Path(os.path.join(OUT_DIR, "Evaluated_Buildings.shp"))
DAMAGED_OUT_SHP   = Path(os.path.join(OUT_DIR, "Damaged_Buildings.shp"))
UNDAMAGED_OUT_SHP = Path(os.path.join(OUT_DIR, "Undamaged_Buildings.shp"))
DAMAGED_POLY_PATH = Path(os.path.join(OUT_DIR, "damaged_polygons_DPM.shp"))  # for plotting
DAMAGED_MASK_TIF  = Path(os.path.join(OUT_DIR, "AOI_DPM_DamagedMask_ge84.tif"))

# Threshold
DPM_THRESHOLD = require("DPM_THRESHOLD", float)

# ==================== B2) Load AOI & Buildings =================

# AOI polygon (EPSG:4326) — prefer memory from Section A
if 'aoi_gdf_ll' not in globals():
    # Load from Section A outputs
    aoi_path = Path(os.path.join(A_OUT_DIR, "region_boundary.geojson"))
    if not aoi_path.exists():
        raise FileNotFoundError("AOI not in memory and region_boundary.geojson not found in Outputs_Final\\A.")
    aoi_gdf_ll = gpd.read_file(aoi_path).to_crs(4326)

# Buildings — prefer memory
if 'bld_3857' in globals() and bld_3857 is not None and not bld_3857.empty:
    buildings = bld_3857.copy()
else:
    buildings_path = Path(os.path.join(A_OUT_DIR, "buildings_selected_region.shp"))
    if not buildings_path.exists():
        raise FileNotFoundError("Buildings not in memory and buildings_selected_region.shp not found in Outputs_Final\\A.")
    buildings = gpd.read_file(buildings_path)
    if buildings.crs is None:
        raise ValueError("Loaded buildings have no CRS; please re-run Section A.")

# Clean geometries
buildings = buildings[~buildings.geometry.is_empty & buildings.geometry.notnull()].copy()

# ========================= B3) Clip DPM raster to AOI =========================
with rio.open(DPM_PATH) as src:
    AOI_RCRS = aoi_gdf_ll.to_crs(src.crs)
    aoi_geom = [AOI_RCRS.union_all().__geo_interface__]
    out_img, out_transform = mask(src, aoi_geom, crop=True, all_touched=True)
    out_meta = src.meta.copy()
    out_meta.update({
        "height": out_img.shape[1],
        "width" : out_img.shape[2],
        "transform": out_transform,
        "driver": "GTiff",
    })
    nd   = src.nodata
    rcrs = src.crs

tif_clip = Path(os.path.join(OUT_DIR, "AOI_DPM_Clipped.tif"))
with rio.open(tif_clip, "w", **out_meta) as dst:
    dst.write(out_img)

print(f" Clipped DPM saved → {tif_clip}")

# ========== B4) Threshold → damaged mask GeoTIFF → polygons =========
with rio.open(tif_clip) as src:
    arr = src.read(1)
    nd  = src.nodata
    t   = src.transform
    rcrs = src.crs

valid = (~np.isnan(arr)) if nd is None else (arr != nd)
damage_mask = (valid) & (arr >= DPM_THRESHOLD)
damage_mask_u8 = damage_mask.astype("uint8")     # 1 = damaged, 0 = not

# Save binary damaged mask
mask_meta = out_meta.copy()
mask_meta.update({"count": 1, "dtype": "uint8", "nodata": 0})
with rio.open(DAMAGED_MASK_TIF, "w", **mask_meta) as dst:
    dst.write(damage_mask_u8, 1)
print(f" Damaged binary mask saved → {DAMAGED_MASK_TIF}")

# Vectorize damaged pixels for plotting
dam_polys = [
    shape(geom)
    for geom, val in shapes(damage_mask_u8, mask=damage_mask_u8.astype(bool), transform=t)
    if val == 1
]
damaged_gdf = gpd.GeoDataFrame(geometry=dam_polys, crs=rcrs) if len(dam_polys) else gpd.GeoDataFrame(geometry=[], crs=rcrs)
damaged_gdf.to_file(DAMAGED_POLY_PATH)
print(f" Damaged polygons (for plotting) saved → {DAMAGED_POLY_PATH}")

# =========== B5) Tiled Labeling ============

# Tiling params
TILE_M    = require("TILE_M", float)     # tile side in meters
OVERLAP_M = require("OVERLAP_M", float)  # overlap to avoid boundary misses
DISSOLVE_IN_TILE = require("DISSOLVE_IN_TILE", bool)

# Open AOI-clipped raster; get CRS/bounds
with rio.open(tif_clip) as ds:
    rcrs     = ds.crs
    r_bounds = ds.bounds  # (minx, miny, maxx, maxy)
    nd       = ds.nodata

# Reproject buildings to raster CRS
buildings_r = buildings.to_crs(rcrs) if buildings.crs != rcrs else buildings.copy()
buildings_r = buildings_r[~buildings_r.geometry.is_empty & buildings_r.geometry.notnull()].copy()
try:
    invalid_mask = ~buildings_r.is_valid
    if invalid_mask.any():
        buildings_r.loc[invalid_mask, "geometry"] = buildings_r.loc[invalid_mask, "geometry"].buffer(0)
except Exception:
    pass

# Build tile grid with overlap
minx, miny, maxx, maxy = r_bounds
xs = np.arange(minx, maxx, TILE_M - OVERLAP_M)
ys = np.arange(miny, maxy, TILE_M - OVERLAP_M)

damaged_mask_buildings = np.zeros(len(buildings_r), dtype=bool)
_ = buildings_r.sindex  # build spatial index once

def _vectorize_damaged_in_bounds(bounds):
    """Return damaged polygons (GeoDataFrame in rcrs) within 'bounds'. Empty if none."""
    with rio.open(tif_clip) as ds:
        win = from_bounds(*bounds, transform=ds.transform)
        win = win.intersection(rio.windows.Window(0, 0, ds.width, ds.height))
        if win.width <= 0 or win.height <= 0:
            return gpd.GeoDataFrame(geometry=[], crs=rcrs)

        arr = ds.read(1, window=win)
        sub_transform = ds.window_transform(win)
        valid = (~np.isnan(arr)) if ds.nodata is None else (arr != ds.nodata)
        dmg = (valid) & (arr >= DPM_THRESHOLD)
        if not dmg.any():
            return gpd.GeoDataFrame(geometry=[], crs=rcrs)

        dam_polys = [shape(geom) for geom, val in shapes(dmg.astype(np.uint8),
                                                         mask=dmg, transform=sub_transform) if val == 1]
        if not dam_polys:
            return gpd.GeoDataFrame(geometry=[], crs=rcrs)

        if DISSOLVE_IN_TILE:
            u = unary_union(dam_polys)
            geoms = list(u.geoms) if getattr(u, "geom_type", "") == "MultiPolygon" else [u]
        else:
            geoms = dam_polys

        return gpd.GeoDataFrame(geometry=geoms, crs=rcrs)

# Iterate tiles → vectorize damage per tile → test only local buildings
for x in xs:
    x1 = min(x + TILE_M, maxx)
    if x1 <= x:
        continue
    for y in ys:
        y1 = min(y + TILE_M, maxy)
        if y1 <= y:
            continue

        tb = (x, y, x1, y1)
        dmg_tile = _vectorize_damaged_in_bounds(tb)
        if dmg_tile.empty:
            continue

        # Candidate buildings intersecting this tile bbox
        try:
            L, R = buildings_r.sindex.query_bulk(gpd.GeoSeries([box(*tb)], crs=rcrs), predicate="intersects")
            if R.size == 0:
                continue
            cand = buildings_r.iloc[np.unique(R)]
        except Exception:
            cand = buildings_r[buildings_r.intersects(box(*tb))]
            if cand.empty:
                continue

        p = prep(dmg_tile.unary_union)
        hit = cand.geometry.apply(p.intersects).to_numpy()
        if hit.any():
            damaged_mask_buildings[cand.index.values[hit]] = True

# Final labels and splits
buildings_r["Damaged"] = np.where(damaged_mask_buildings, "yes", "no")
damaged_only   = buildings_r.loc[buildings_r["Damaged"] == "yes"].copy()
undamaged_only = buildings_r.loc[buildings_r["Damaged"] == "no"].copy()

# Save
buildings_r.to_file(EVAL_OUT_SHP)
damaged_only.to_file(DAMAGED_OUT_SHP)
undamaged_only.to_file(UNDAMAGED_OUT_SHP)

print(f" Evaluated buildings: {len(buildings_r)}")
print(f"  Damaged   (≥ {DPM_THRESHOLD}): {len(damaged_only)}")
print(f"  Undamaged (<  {DPM_THRESHOLD}): {len(undamaged_only)}")
print(" Saved:")
print("  ", EVAL_OUT_SHP)
print("  ", DAMAGED_OUT_SHP)
print("  ", UNDAMAGED_OUT_SHP)

# =================== B6) PLOT: BUILDINGS + DAMAGED PIXELS ===================
P = {
    # ---- Output ----
    "OUT_NAME": "Buildings_with_DamagedPixels_LOCAL_UTM.png",  # output PNG filename

    # ---- Optional manual zoom in EPSG:4326 (set all to None for autoscale) ----
    "lon_min": 36.080,             # min longitude of custom zoom (None = auto)
    "lon_max": 36.220,             # max longitude (None = auto)
    "lat_min": 36.150,             # min latitude  (None = auto)
    "lat_max": 36.270,             # max latitude  (None = auto)

    # ---- Figure & labels ----
    "FIGSIZE": (10, 9),            # figure size in inches (w, h)
    "DPI": 300,                    # export resolution
    "TITLE": "Buildings + Damaged Pixels (Local UTM, ≥ threshold)",
    "XLABEL": "Easting (m)",
    "YLABEL": "Northing (m)",

    # ---- Buildings style ----
    "BLDG_FACE": "#9e9e9e",        # fill color for all buildings
    "BLDG_EDGE": "none",           # outline color
    "BLDG_LW": 0.2,                # outline width (if not 'none')
    "BLDG_ALPHA": 1.0,             # layer transparency (0..1)

    # ---- Damaged polygons style ----
    "DMG_FACE": "red",             # fill color for damaged polygons
    "DMG_EDGE": "none",            # outline color
    "DMG_ALPHA": 0.5,              # layer transparency (0..1)

    # ---- AOI boundary (optional if available) ----
    "DRAW_AOI": True,              # draw AOI if aoi_gdf_ll exists
    "AOI_COLOR": "black",          # AOI boundary color
    "AOI_LW": 1.5,                 # AOI boundary width

    # ---- North arrow ----
    "ADD_NORTH_ARROW": True,       # draw north arrow
    "NA_X": 0.05,                  # arrow x-position (axes fraction)
    "NA_Y": 0.15,                  # arrow y-position (axes fraction)
    "NA_LEN": 0.08,                # arrow length  (axes fraction)
    "NA_LABEL": "N",               # arrow label
    "NA_COLOR": "black",           # arrow color
    "NA_LW": 2,                    # arrow line width
    "NA_FONTSIZE": 14,             # 'N' font size

    # ---- Scalebar ----
    "ADD_SCALEBAR": True,          # draw scalebar
    "SB_UNITS": "m",               # scalebar units
    "SB_LOC": "lower right",       # scalebar location
    "SB_BOX_ALPHA": 0.8,           # scalebar box transparency

    # ---- Legend ----
    "LEGEND_LOC": "upper right",   # legend location
    "LBL_BUILDINGS": "Buildings",  # legend label for buildings
    "LBL_DAMAGED": "Damaged pixels", # legend label for damaged polygons
    "LBL_AOI": "AOI Boundary"      # legend label for AOI boundary
}

# ----- Methodology -----

# Local UTM from damaged polygons
utm_crs = damaged_gdf.to_crs(4326).estimate_utm_crs()

# Reproject to local UTM
b_all_utm = buildings_r.to_crs(utm_crs)
dmg_utm   = damaged_gdf.to_crs(utm_crs)

# Optional lon/lat bbox → project to UTM and set frame
use_bbox = all(P[k] is not None for k in ["lon_min","lon_max","lat_min","lat_max"])
if use_bbox:
    aoi_ll  = gpd.GeoDataFrame(
        geometry=[box(min(P["lon_min"], P["lon_max"]),
                      min(P["lat_min"], P["lat_max"]),
                      max(P["lon_min"], P["lon_max"]),
                      max(P["lat_min"], P["lat_max"]))],
        crs=4326
    )
    xmin, ymin, xmax, ymax = aoi_ll.to_crs(utm_crs).total_bounds
else:
    bounds = [b_all_utm.total_bounds, dmg_utm.total_bounds]
    xmin = min(b[0] for b in bounds); ymin = min(b[1] for b in bounds)
    xmax = max(b[2] for b in bounds); ymax = max(b[3] for b in bounds)

# Plot
fig, ax = plt.subplots(figsize=P["FIGSIZE"])

b_all_utm.plot(ax=ax, color=P["BLDG_FACE"], edgecolor=P["BLDG_EDGE"],
               linewidth=P["BLDG_LW"], alpha=P["BLDG_ALPHA"], zorder=1)
dmg_utm.plot(ax=ax, facecolor=P["DMG_FACE"], edgecolor=P["DMG_EDGE"],
             alpha=P["DMG_ALPHA"], zorder=2)

# Optional AOI boundary
if P["DRAW_AOI"] and 'aoi_gdf_ll' in globals():
    aoi_gdf_ll.to_crs(utm_crs).boundary.plot(ax=ax, color=P["AOI_COLOR"],
                                             linewidth=P["AOI_LW"], zorder=5)

ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)

# North arrow
if P["ADD_NORTH_ARROW"]:
    ax.annotate(P["NA_LABEL"], xy=(P["NA_X"], P["NA_Y"]), xytext=(P["NA_X"], P["NA_Y"] - P["NA_LEN"]),
                xycoords="axes fraction", textcoords="axes fraction",
                ha="center", va="center", fontsize=P["NA_FONTSIZE"], fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", lw=P["NA_LW"], color=P["NA_COLOR"]),
                clip_on=False, zorder=20)

# Scalebar
if P["ADD_SCALEBAR"]:
    ax.add_artist(ScaleBar(1, P["SB_UNITS"], location=P["SB_LOC"], box_alpha=P["SB_BOX_ALPHA"]))

ax.set_aspect("equal"); ax.ticklabel_format(style="plain")
ax.set_xlabel(P["XLABEL"]); ax.set_ylabel(P["YLABEL"])
ax.set_title(P["TITLE"])

# Legend
legend_elements = [
    Patch(facecolor=P["BLDG_FACE"], edgecolor="none", label=P["LBL_BUILDINGS"]),
    Patch(facecolor=P["DMG_FACE"], edgecolor="none", alpha=P["DMG_ALPHA"], label=P["LBL_DAMAGED"]),
]
if P["DRAW_AOI"] and 'aoi_gdf_ll' in globals():
    legend_elements.append(Line2D([0], [0], color=P["AOI_COLOR"], lw=P["AOI_LW"], label=P["LBL_AOI"]))
ax.legend(handles=legend_elements, loc=P["LEGEND_LOC"], frameon=True)

out_png = Path(IMAGES_DIR) / P["OUT_NAME"]
fig.savefig(out_png, dpi=P["DPI"], bbox_inches="tight")
plt.close(fig)
display(Image(filename=out_png))

# ========== B7) PLOT: DAMAGED vs UNDAMAGED (LOCAL UTM) =========
P = {
    # ---- Output ----
    "OUT_NAME": "Damaged_vs_Undamaged_Buildings_LocalUTM.png",  # output PNG filename

    # ---- Manual zoom in EPSG:4326; set all to None for auto extent ----
    "lon_min": 36.080,             # min longitude of custom zoom (None = auto)
    "lon_max": 36.220,             # max longitude (None = auto)
    "lat_min": 36.150,             # min latitude  (None = auto)
    "lat_max": 36.270,               # max latitude  (None = auto)

    # ---- Figure & labels ----
    "FIGSIZE": (10, 9),              # figure size (w, h) inches
    "DPI": 300,                      # export resolution
    "TITLE": "Buildings: Damaged (red) vs Undamaged (gray) — Local UTM",
    "XLABEL": "Easting (m)",
    "YLABEL": "Northing (m)",

    # ---- Undamaged style ----
    "UND_FACE": "#bdbdbd",           # fill color
    "UND_EDGE": "none",              # outline color
    "UND_LW": 0.2,                   # outline width
    "UND_ALPHA": 1.0,                # transparency (0..1)
    "LBL_UND": "Undamaged",          # legend label

    # ---- Damaged style ----
    "DAM_FACE": "#d73027",           # fill color
    "DAM_EDGE": "none",              # outline color
    "DAM_LW": 0.2,                   # outline width
    "DAM_ALPHA": 1.0,                # transparency (0..1)
    "LBL_DAM": "Damaged",            # legend label

    # ---- AOI boundary (optional) ----
    "DRAW_AOI": True,                # draw AOI boundary (expects aoi_gdf_ll upstream)
    "AOI_COLOR": "black",            # AOI boundary color
    "AOI_LW": 1.5,                   # AOI boundary width
    "LBL_AOI": "AOI Boundary",       # legend label

    # ---- North arrow ----
    "ADD_NORTH_ARROW": True,         # draw north arrow
    "NA_X": 0.05,                    # arrow x-position (axes fraction)
    "NA_Y": 0.15,                    # arrow y-position (axes fraction)
    "NA_LEN": 0.08,                  # arrow length (axes fraction)
    "NA_LABEL": "N",                 # arrow text
    "NA_COLOR": "black",             # arrow color
    "NA_LW": 2,                      # arrow line width
    "NA_FONTSIZE": 14,               # 'N' font size

    # ---- Scalebar ----
    "ADD_SCALEBAR": True,            # draw scalebar
    "SB_UNITS": "m",                 # scalebar units
    "SB_LOC": "lower right",         # scalebar location
    "SB_BOX_ALPHA": 0.8,             # scalebar box transparency

    # ---- Legend ----
    "LEGEND_LOC": "upper right"      # legend placement
}
# ---------------------------------------------------------------------------

# Choose UTM from labeled layers
seed = damaged_only if not damaged_only.empty else undamaged_only
utm_crs = seed.to_crs(4326).estimate_utm_crs()

# Reproject
dam_utm = damaged_only.to_crs(utm_crs)
und_utm = undamaged_only.to_crs(utm_crs)

# Optional lon/lat bbox → UTM frame
use_bbox = all(P[k] is not None for k in ("lon_min", "lon_max", "lat_min", "lat_max"))
if use_bbox:
    aoi_ll = gpd.GeoDataFrame(
        geometry=[box(min(P["lon_min"], P["lon_max"]),
                      min(P["lat_min"], P["lat_max"]),
                      max(P["lon_min"], P["lon_max"]),
                      max(P["lat_min"], P["lat_max"]))],
        crs=4326
    )
    xmin, ymin, xmax, ymax = aoi_ll.to_crs(utm_crs).total_bounds
else:
    bounds = [und_utm.total_bounds, dam_utm.total_bounds]
    xmin = min(b[0] for b in bounds); ymin = min(b[1] for b in bounds)
    xmax = max(b[2] for b in bounds); ymax = max(b[3] for b in bounds)

# Plot
fig, ax = plt.subplots(figsize=P["FIGSIZE"])

if not und_utm.empty:
    und_utm.plot(ax=ax, color=P["UND_FACE"], edgecolor=P["UND_EDGE"],
                 linewidth=P["UND_LW"], alpha=P["UND_ALPHA"], zorder=1)
if not dam_utm.empty:
    dam_utm.plot(ax=ax, color=P["DAM_FACE"], edgecolor=P["DAM_EDGE"],
                 linewidth=P["DAM_LW"], alpha=P["DAM_ALPHA"], zorder=2)

# Optional AOI boundary
if P["DRAW_AOI"] and 'aoi_gdf_ll' in globals():
    aoi_gdf_ll.to_crs(utm_crs).boundary.plot(ax=ax, color=P["AOI_COLOR"], linewidth=P["AOI_LW"], zorder=5)

ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)

# North arrow
if P["ADD_NORTH_ARROW"]:
    ax.annotate(P["NA_LABEL"], xy=(P["NA_X"], P["NA_Y"]), xytext=(P["NA_X"], P["NA_Y"] - P["NA_LEN"]),
                xycoords="axes fraction", textcoords="axes fraction",
                ha="center", va="center", fontsize=P["NA_FONTSIZE"], fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", lw=P["NA_LW"], color=P["NA_COLOR"]),
                clip_on=False, zorder=20)

# Scalebar
if P["ADD_SCALEBAR"]:
    ax.add_artist(ScaleBar(1, P["SB_UNITS"], location=P["SB_LOC"], box_alpha=P["SB_BOX_ALPHA"]))

ax.set_aspect("equal"); ax.ticklabel_format(style="plain")
ax.set_xlabel(P["XLABEL"]); ax.set_ylabel(P["YLABEL"])
ax.set_title(P["TITLE"])

# Legend
legend_elements = [
    Patch(facecolor=P["UND_FACE"], edgecolor="none", label=P["LBL_UND"]),
    Patch(facecolor=P["DAM_FACE"], edgecolor="none", label=P["LBL_DAM"]),
]
if P["DRAW_AOI"] and 'aoi_gdf_ll' in globals():
    legend_elements.append(Line2D([0], [0], color=P["AOI_COLOR"], lw=P["AOI_LW"], label=P["LBL_AOI"]))
ax.legend(handles=legend_elements, loc=P["LEGEND_LOC"], frameon=True)

out_png = Path(IMAGES_DIR) / P["OUT_NAME"]
fig.savefig(out_png, dpi=P["DPI"], bbox_inches="tight")
plt.close(fig)
display(Image(filename=out_png))

# ============================== B8) Summary ==============================
def _crs_str(crs):
    try:
        return str(crs) if crs is not None else "(no CRS)"
    except Exception:
        return "(unknown CRS)"

# Datasets
_b  = buildings_r
_bd = damaged_only
_bu = undamaged_only

# OSM nodes/edges
if 'nodes' in globals() and nodes is not None and not nodes.empty:
    _nodes = nodes
else:
    npath = Path(os.path.join(A_OUT_DIR, "osm_nodes.shp"))
    _nodes = gpd.read_file(npath) if npath.exists() else None

if 'edges' in globals() and edges is not None and not edges.empty:
    _edges = edges
else:
    epath = Path(os.path.join(A_OUT_DIR, "osm_edges.shp"))
    _edges = gpd.read_file(epath) if epath.exists() else None

# Damaged pixels
dam_pixels = 0
if DAMAGED_MASK_TIF.exists():
    with rio.open(DAMAGED_MASK_TIF) as src:
        arr = src.read(1)
        dam_pixels = int(np.count_nonzero(arr == 1))
        mask_crs = src.crs
else:
    mask_crs = None

print("===== Section B Summary (AOI only) =====")
print(f"Total buildings:           {len(_b):,}")
print(f"  Damaged (>= threshold):  {len(_bd):,}")
print(f"  Undamaged:               {len(_bu):,}")
print(f"Total OSM nodes:           {len(_nodes) if _nodes is not None else 0:,}")
print(f"Total OSM edges:           {len(_edges) if _edges is not None else 0:,}")
print(f"Total damaged pixels:      {dam_pixels:,}")

print("\n===== Output files (Section B) =====")
print("All Buildings (labeled):   ", EVAL_OUT_SHP)
print("Damaged Buildings:         ", DAMAGED_OUT_SHP)
print("Undamaged Buildings:       ", UNDAMAGED_OUT_SHP)
print("Damaged Pixels (mask):     ", DAMAGED_MASK_TIF)
print("Damaged Polygons:          ", DAMAGED_POLY_PATH)
