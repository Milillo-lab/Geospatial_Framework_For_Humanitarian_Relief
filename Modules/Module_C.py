## C) Affected Population Calculation

# =============================== C0) Imports ===============================
import os
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

import rasterio as rio
from rasterio.mask import mask
from rasterio.features import shapes
from rasterio.windows import from_bounds  # used later in grid flow if needed

from shapely.geometry import shape, box

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from IPython.display import Image, display
import yaml

# ============== C1) User Options, Inputs, Outputs (memory-first) =============

# Unified directories
INPUTS_DIR = r"./Inputs"
OUT_BASE   = r"./Outputs"

# Prior-section fallbacks
A_OUT_DIR  = os.path.join(OUT_BASE, "A")
B_OUT_DIR  = os.path.join(OUT_BASE, "B")

# This section's outputs
OUT_DIR    = os.path.join(OUT_BASE, "C")
IMAGES_DIR = os.path.join(OUT_BASE, "Images")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# Population raster (from config)
POP_RASTER = os.path.join(INPUTS_DIR, require("POP_FILE", str))

# Evaluated buildings from Section B (memory first)
EVAL_OUT_SHP = (
    EVAL_OUT_SHP
    if "EVAL_OUT_SHP" in globals()
    else Path(os.path.join(B_OUT_DIR, "Evaluated_Buildings.shp"))
)

# Outputs for this section
OUT_SHP_VOL      = Path(os.path.join(OUT_DIR, "Evaluated_Buildings_withVol.shp"))
OUT_SHP_VOL_POP  = Path(os.path.join(OUT_DIR, "Evaluated_Buildings_withVol_Pop.shp"))

# Height settings
HEIGHT_COL     = "height"                  # keep as-is unless you prefer to read from config
DEFAULT_HEIGHT = optional("DEFAULT_HEIGHT")  # e.g., 3.0 to force a default height (meters)

# ======== C2) Load AOI & Evaluated Buildings (memory-first, else fallback) ========

# AOI polygon (EPSG:4326)
if "aoi_gdf_ll" not in globals() or aoi_gdf_ll is None or aoi_gdf_ll.empty:
    aoi_path = Path(os.path.join(A_OUT_DIR, "region_boundary.geojson"))
    if not aoi_path.exists():
        raise FileNotFoundError("AOI not in memory and region_boundary.geojson not found in Outputs_Final\\A.")
    aoi_gdf_ll = gpd.read_file(aoi_path).to_crs(4326)

# Evaluated buildings
if "buildings_r" in globals() and buildings_r is not None and not buildings_r.empty:
    b_eval = buildings_r.copy()
else:
    b_eval = gpd.read_file(EVAL_OUT_SHP)
    if b_eval.empty:
        raise RuntimeError("Evaluated_Buildings.shp is empty.")
    if b_eval.crs is None:
        raise ValueError("Evaluated_Buildings.shp has no CRS defined.")

# ====================== C3) Compute Volume (m³) and Save =====================

# Ensure AOI strictness (usually already AOI-limited)
try:
    b_eval = gpd.overlay(b_eval, aoi_gdf_ll.to_crs(b_eval.crs), how="intersection")
except Exception:
    b_eval = gpd.sjoin(b_eval, aoi_gdf_ll[["geometry"]].to_crs(b_eval.crs),
                       predicate="intersects", how="inner").drop(columns="index_right")

# Compute area in m² using a projected CRS (local UTM if needed)
if not getattr(b_eval.crs, "is_projected", False):
    utm_crs = b_eval.to_crs(4326).estimate_utm_crs()
    b_area = b_eval.to_crs(utm_crs)
else:
    b_area = b_eval

area_m2 = b_area.geometry.area.values  # numpy array (m²)

# Heights
if HEIGHT_COL in b_eval.columns:
    h = pd.to_numeric(b_eval[HEIGHT_COL], errors="coerce").fillna(0.0).to_numpy()
elif DEFAULT_HEIGHT is not None:
    h = np.full(len(b_eval), float(DEFAULT_HEIGHT), dtype=float)
else:
    raise ValueError(
        f"Height column '{HEIGHT_COL}' not found. "
        f"Either set HEIGHT_COL correctly or provide DEFAULT_HEIGHT."
    )

# Volume
b_eval["Volume"] = area_m2 * h

# Save
b_eval.to_file(OUT_SHP_VOL)
print(f" Added 'Volume' (m³) and saved → {OUT_SHP_VOL}")
print(f" Count: {len(b_eval):,} | CRS: {b_eval.crs}")

# === C4) AOI Population → Allocate to Buildings by Volume (m³) ===

# Ensure we have the Volume-enabled buildings (memory-first)
b = b_eval.copy() if "Volume" in b_eval.columns else gpd.read_file(OUT_SHP_VOL)
if b.empty:
    raise RuntimeError("Volume-enabled buildings are empty.")
if "Volume" not in b.columns:
    raise ValueError("Expected 'Volume' column in buildings.")

# Clip population raster to AOI and sum
with rio.open(POP_RASTER) as src:
    aoi_in_rcrs = aoi_gdf_ll.to_crs(src.crs)
    aoi_geom = [aoi_in_rcrs.union_all().__geo_interface__]  # shapely 2.x union
    pop_clip, _ = mask(src, aoi_geom, crop=True, all_touched=True)
    pop_arr = pop_clip[0].astype(float)
    nodata  = src.nodata
    if nodata is not None:
        pop_arr = np.where(pop_arr == nodata, 0.0, pop_arr)
    pop_arr = np.nan_to_num(pop_arr, nan=0.0)
    total_population = int(round(pop_arr.sum()))

print(f" Total population within AOI: {total_population:,}")

# Allocate ∝ Volume
vol = pd.to_numeric(b["Volume"], errors="coerce").to_numpy(dtype=float)
vol[~np.isfinite(vol)] = 0.0
vol_sum = float(vol.sum())
if vol_sum <= 0:
    raise ValueError("Total building Volume is zero/invalid; cannot allocate population.")

alloc   = (vol / vol_sum) * total_population
pop_int = np.floor(alloc).astype(int)
remainder = int(total_population - int(pop_int.sum()))
if remainder > 0:
    frac = alloc - pop_int
    top_idx = np.argsort(frac)[::-1][:remainder]
    pop_int[top_idx] += 1

b["Pop_Est"] = pop_int  # integer population per building
b.to_file(OUT_SHP_VOL_POP)
print(f" Saved buildings with Volume + Pop_Est → {OUT_SHP_VOL_POP}")
print(f"   Buildings: {len(b):,} | CRS: {b.crs}")
print(f"   Sum Pop_Est (should match AOI total): {int(b['Pop_Est'].sum()):,}")

# ========== C5) Plot: Damaged buildings colored by population bins ==========
# --------------------------- USER CONTROLS (edit here) ---------------------------
P = {
    # ---- Output ----
    "OUT_NAME": "Affected_Population_by_Building.png",  # output PNG filename

    # ---- Manual zoom in EPSG:4326 (set all to None for auto extent) ----
    "lon_min": 36.920,                 # min longitude (None = auto)
    "lon_max": 36.940,                 # max longitude (None = auto)
    "lat_min": 37.575,                 # min latitude  (None = auto)
    "lat_max": 37.590,                 # max latitude  (None = auto)

    # ---- Figure & labels ----
    "FIGSIZE": (10, 9),                # figure size (w, h) inches
    "DPI": 300,                        # export resolution
    "TITLE": "Affected Population by Building",
    "XLABEL": "Easting (m)",
    "YLABEL": "Northing (m)",

    # ---- Binning (0–10–…–100, and 100+ by default) ----
    "BIN_MIN": 0,                      # lower bound for bins
    "BIN_MAX": 100,                    # upper bound included in last numeric bin
    "BIN_STEP": 10,                    # step between bins
    "PALETTE": [                       # colors per bin; last color used for the "+" bin
        "#fee5d9", "#fcbba1", "#fc9272", "#fb6a4a",
        "#ef3b2c", "#cb181d", "#a50f15", "#67000d",
        "#4a0010", "#2e0006", "#1a0004"
    ],

    # ---- Undamaged style (background outlines) ----
    "UND_FACE": "white",               # undamaged fill
    "UND_EDGE": "black",               # undamaged outline
    "UND_LW": 0.3,                     # undamaged outline width
    "UND_ALPHA": 1.0,                  # undamaged transparency (0..1)

    # ---- Damaged style (bins) ----
    "DAM_EDGE": "black",               # damaged outline color
    "DAM_LW": 0.2,                     # damaged outline width
    "DAM_ALPHA": 1.0,                  # damaged transparency (0..1)

    # ---- AOI boundary (optional if available upstream) ----
    "DRAW_AOI": True,                  # draw AOI boundary if aoi_gdf_ll exists
    "AOI_COLOR": "black",
    "AOI_LW": 1.0,

    # ---- North arrow & scalebar ----
    "ADD_NORTH_ARROW": True,           # draw north arrow
    "NA_X": 0.05, "NA_Y": 0.15,        # arrow position (axes fraction)
    "NA_LEN": 0.08,                    # arrow length (axes fraction)
    "NA_LABEL": "N",                   # arrow text
    "NA_COLOR": "black", "NA_LW": 2, "NA_FONTSIZE": 14,
    "ADD_SCALEBAR": True,              # draw scalebar (ScaleBar imported in section header)
    "SB_UNITS": "m",
    "SB_LOC": "lower right",
    "SB_BOX_ALPHA": 0.8,               # scalebar box transparency

    # ---- Legend ----
    "LEGEND_LOC": "upper right"        # legend location
}
# ------------------------------------------------------------------------------

# Preconditions (unchanged)
if b.empty:
    raise RuntimeError("No buildings to plot.")
if "Damaged" not in b.columns or "Pop_Est" not in b.columns:
    raise ValueError("Required columns 'Damaged' and 'Pop_Est' missing.")

# UTM & splits
utm_crs = b.to_crs(4326).estimate_utm_crs()
b_utm   = b.to_crs(utm_crs)
dam_utm = b_utm[b_utm["Damaged"] == "yes"].copy()
und_utm = b_utm[b_utm["Damaged"] == "no"].copy()

# Bins & labels
edges = np.arange(P["BIN_MIN"], P["BIN_MAX"] + P["BIN_STEP"], P["BIN_STEP"])
edges_ext = np.append(edges, np.inf)
labels = [f"{edges[i]}–{edges[i+1]}" for i in range(len(edges)-1)] + [f"{P['BIN_MAX']}+"]

dam_utm["pop_bin"] = pd.cut(
    dam_utm["Pop_Est"].astype(float),
    bins=edges_ext, right=False, labels=labels, include_lowest=True
)

palette = P["PALETTE"][:len(labels)]

# Optional lon/lat zoom → UTM frame
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
    xmin, ymin, xmax, ymax = b_utm.total_bounds

# Plot
fig, ax = plt.subplots(figsize=P["FIGSIZE"])
legend_elems = []

# Undamaged base
if not und_utm.empty:
    und_utm.plot(ax=ax, facecolor=P["UND_FACE"], edgecolor=P["UND_EDGE"],
                 linewidth=P["UND_LW"], alpha=P["UND_ALPHA"], zorder=1)
    legend_elems.append(Patch(facecolor=P["UND_FACE"], edgecolor=P["UND_EDGE"], label="Undamaged"))

# Damaged by population bin
for lab, color in zip(labels, palette):
    sub = dam_utm[dam_utm["pop_bin"] == lab]
    if not sub.empty:
        sub.plot(ax=ax, facecolor=color, edgecolor=P["DAM_EDGE"],
                 linewidth=P["DAM_LW"], alpha=P["DAM_ALPHA"], zorder=2)
        legend_elems.append(Patch(facecolor=color, edgecolor=P["DAM_EDGE"], label=f"Affected Pop. {lab}"))

# Optional AOI boundary
if P["DRAW_AOI"] and 'aoi_gdf_ll' in globals():
    aoi_gdf_ll.to_crs(utm_crs).boundary.plot(ax=ax, color=P["AOI_COLOR"], linewidth=P["AOI_LW"], zorder=4)
    legend_elems.append(Line2D([0], [0], color=P["AOI_COLOR"], lw=P["AOI_LW"], label="AOI Boundary"))

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
ax.legend(handles=legend_elems, loc=P["LEGEND_LOC"], frameon=True)

out_png = Path(IMAGES_DIR) / P["OUT_NAME"]
fig.savefig(out_png, dpi=P["DPI"], bbox_inches="tight")
plt.show()
print(f" Saved: {out_png}")

# === C6) Affected Population Density Grid (Rectangular, fast) ===
CELL_SIZE = require("CELL_SIZE", float)  # meters
OUT_GRID_SHP = Path(os.path.join(OUT_DIR, f"Affected_population_density_{int(CELL_SIZE)}_meters.shp"))

if b.empty:
    raise RuntimeError("Building dataset is empty.")
if "Damaged" not in b.columns or "Pop_Est" not in b.columns:
    raise ValueError("Columns 'Damaged' and 'Pop_Est' required.")

utm_crs = b.to_crs(4326).estimate_utm_crs()
b_utm   = b.to_crs(utm_crs)
affected = b_utm[(b_utm["Damaged"] == "yes") & (b_utm["Pop_Est"] > 0)].copy()
if affected.empty:
    raise RuntimeError("No damaged buildings with population > 0 found.")

centroids = gpd.GeoDataFrame(
    affected[["Pop_Est"]],
    geometry=affected.geometry.centroid,
    crs=b_utm.crs
)

aoi_utm = aoi_gdf_ll.to_crs(utm_crs)
minx, miny, maxx, maxy = aoi_utm.total_bounds

x_coords = np.arange(minx, maxx, CELL_SIZE)
y_coords = np.arange(miny, maxy, CELL_SIZE)

grid_cells = [box(x, y, x + CELL_SIZE, y + CELL_SIZE) for x in x_coords for y in y_coords]
grid = gpd.GeoDataFrame(geometry=grid_cells, crs=utm_crs)
print(f" Created grid: {len(grid):,} cells ({int(CELL_SIZE)} m)")

join = gpd.sjoin(centroids, grid, how="left", predicate="within")
pop_sum = join.groupby("index_right")["Pop_Est"].sum()
grid["Affected_Pop"] = grid.index.map(pop_sum).fillna(0).astype(int)

grid.to_file(OUT_GRID_SHP)
print(f" Saved affected population grid → {OUT_GRID_SHP}")
print(f" Total affected population (check): {int(grid['Affected_Pop'].sum()):,}")

# === C7) Plot: Affected Population Density Grid (Nonzero Cells, Yellow→Red) ===
# --------------------------- USER CONTROLS (edit here) ---------------------------
P = {
    # ---- Output ----
    "OUT_NAME": f"Affected_PopDensity_Nonzero_{CELL_SIZE}m.png",  # output PNG filename

    # ---- Manual zoom in EPSG:4326 (set all to None for auto extent) ----
    "lon_min": 36.920,                 # min longitude (None = auto)
    "lon_max": 36.940,                 # max longitude (None = auto)
    "lat_min": 37.575,                 # min latitude  (None = auto)
    "lat_max": 37.590,                 # max latitude  (None = auto)

    # ---- Figure & labels ----
    "FIGSIZE": (10, 9),                # figure size (w, h) inches
    "DPI": 300,                        # export resolution
    "TITLE": "Affected Population Density",  # figure title
    "XLABEL": "Easting (m)",           # x-axis label (projected meters)
    "YLABEL": "Northing (m)",          # y-axis label (projected meters)

    # ---- Buildings context style ----
    "BLDG_FACE": "white",              # building fill color
    "BLDG_EDGE": "black",              # building outline color
    "BLDG_LW": 0.3,                    # outline width
    "BLDG_ALPHA": 1.0,                 # transparency

    # ---- Grid cell style ----
    "GRID_ALPHA": 0.5,                 # grid cell transparency (0..1)
    "PALETTE": [                       # color palette for bins (light→dark)
        "#ffffcc", "#ffeda0", "#fed976", "#feb24c",
        "#fd8d3c", "#fc4e2a", "#e31a1c", "#bd0026",
        "#800026", "#4a0010"
    ],

    # ---- AOI boundary ----
    "DRAW_AOI": True,                  # draw AOI boundary (expects aoi_gdf_ll upstream)
    "AOI_COLOR": "black",              # AOI line color
    "AOI_LW": 1.0,                     # AOI line width
    "LBL_AOI": "AOI Boundary",         # legend label for AOI

    # ---- North arrow & scalebar ----
    "ADD_NORTH_ARROW": True,           # draw north arrow
    "NA_X": 0.05, "NA_Y": 0.15,        # arrow position (axes fraction)
    "NA_LEN": 0.08,                    # arrow length (axes fraction)
    "NA_LABEL": "N",                   # arrow text
    "NA_COLOR": "black",               # arrow color
    "NA_LW": 2,                        # arrow line width
    "NA_FONTSIZE": 14,                 # 'N' font size
    "ADD_SCALEBAR": True,              # draw scalebar (ScaleBar imported in section header)
    "SB_UNITS": "m",                   # scalebar units
    "SB_LOC": "lower right",           # scalebar location

    # ---- Legend ----
    "LBL_BUILDINGS": "Buildings",      # legend label for buildings
    "LEGEND_LOC": "upper right"        # legend placement
}
# ------------------------------------------------------------------------------

# Preconditions
if grid.empty:
    raise RuntimeError("Grid dataset is empty.")
if "Affected_Pop" not in grid.columns:
    raise ValueError("Grid missing column 'Affected_Pop'.")

# Keep only nonzero cells
grid = grid[grid["Affected_Pop"] > 0].copy()

# Reproject context layers to grid CRS
b_utm_all = b.to_crs(grid.crs)
aoi_utm   = aoi_gdf_ll.to_crs(grid.crs)

# Optional lon/lat zoom → grid CRS frame
use_bbox = all(P[k] is not None for k in ("lon_min","lon_max","lat_min","lat_max"))
if use_bbox:
    aoi_ll_zoom = gpd.GeoDataFrame(
        geometry=[box(min(P["lon_min"], P["lon_max"]),
                      min(P["lat_min"], P["lat_max"]),
                      max(P["lon_min"], P["lon_max"]),
                      max(P["lat_min"], P["lat_max"]))],
        crs=4326
    )
    xmin, ymin, xmax, ymax = aoi_ll_zoom.to_crs(grid.crs).total_bounds
else:
    xmin, ymin, xmax, ymax = aoi_utm.total_bounds

# Dynamic bins
max_pop = grid["Affected_Pop"].max()
if max_pop <= 10:
    bins = [0, 1, 5, 10]
elif max_pop <= 100:
    bins = list(np.arange(0, 110, 10))
else:
    bins = list(np.arange(0, 110, 10)) + [np.inf]
labels = [f"{bins[i]}–{bins[i+1]}" for i in range(len(bins)-1)]
labels[-1] = "100+"

grid["pop_bin"] = pd.cut(grid["Affected_Pop"], bins=bins, right=False,
                         labels=labels, include_lowest=True)
palette = P["PALETTE"][:len(labels)]

# Plot
fig, ax = plt.subplots(figsize=P["FIGSIZE"])
legend_elems = []

# Buildings (context)
if not b_utm_all.empty:
    b_utm_all.plot(ax=ax, facecolor=P["BLDG_FACE"], edgecolor=P["BLDG_EDGE"],
                   linewidth=P["BLDG_LW"], alpha=P["BLDG_ALPHA"], zorder=1)
    legend_elems.append(Patch(facecolor=P["BLDG_FACE"], edgecolor=P["BLDG_EDGE"], label=P["LBL_BUILDINGS"]))

# Grid by population bin
for lab, color in zip(labels, palette):
    sub = grid[grid["pop_bin"] == lab]
    if not sub.empty:
        sub.plot(ax=ax, facecolor=color, edgecolor="none", alpha=P["GRID_ALPHA"], zorder=2)
        legend_elems.append(Patch(facecolor=color, edgecolor="none", label=f"Affected Pop {lab}"))

# AOI boundary (optional)
if P["DRAW_AOI"]:
    aoi_utm.boundary.plot(ax=ax, color=P["AOI_COLOR"], linewidth=P["AOI_LW"], zorder=4)
    legend_elems.append(Line2D([0], [0], color=P["AOI_COLOR"], lw=P["AOI_LW"], label=P["LBL_AOI"]))

# Frame & decor
ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)

if P["ADD_NORTH_ARROW"]:
    ax.annotate(P["NA_LABEL"], xy=(P["NA_X"], P["NA_Y"]), xytext=(P["NA_X"], P["NA_Y"] - P["NA_LEN"]),
                xycoords="axes fraction", textcoords="axes fraction",
                ha="center", va="center", fontsize=P["NA_FONTSIZE"], fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", lw=P["NA_LW"], color=P["NA_COLOR"]),
                clip_on=False, zorder=20)

if P["ADD_SCALEBAR"]:
    ax.add_artist(ScaleBar(1, P["SB_UNITS"], location=P["SB_LOC"]))

ax.set_aspect("equal"); ax.ticklabel_format(style="plain")
ax.set_xlabel(P["XLABEL"]); ax.set_ylabel(P["YLABEL"])
ax.set_title(P["TITLE"])

ax.legend(handles=legend_elems, loc=P["LEGEND_LOC"], frameon=True)

out_png = Path(IMAGES_DIR) / P["OUT_NAME"]
fig.savefig(out_png, dpi=P["DPI"], bbox_inches="tight")
plt.show()
print(f" Saved: {out_png}")

# =============================== C8) Summary ===============================
if "Damaged" in b.columns and "Pop_Est" in b.columns:
    affected_pop   = int(b.loc[b["Damaged"] == "yes", "Pop_Est"].sum())
    unaffected_pop = int(b.loc[b["Damaged"] == "no",  "Pop_Est"].sum())
    total_pop      = affected_pop + unaffected_pop

    print("\n Population Summary (within AOI):")
    print(f"   Affected population (Damaged buildings):   {affected_pop:,}")
    print(f"   Unaffected population (Undamaged buildings): {unaffected_pop:,}")
    print(f"   Total population (check):                  {total_pop:,}")
else:
    print(" Columns 'Damaged' and/or 'Pop_Est' not found in building dataset.")
