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
from matplotlib_scalebar.scalebar import ScaleBar

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

# ========== C5) Plot: Affected Population by Building ==========
# Preconditions
if b.empty:
    raise RuntimeError("No buildings to plot.")
if "Damaged" not in b.columns or "Pop_Est" not in b.columns:
    raise ValueError("Required columns 'Damaged' and 'Pop_Est' missing.")

# --------------------------- USER CONTROLS ---------------------------
P = {
    "OUT_NAME": "Affected_Population_by_Building.png",

    # Manual zoom in EPSG:4326 (set all to None for auto extent)
    "lon_min": None,
    "lon_max": None,
    "lat_min": None,
    "lat_max": None,

    # Figure & labels
    "FIGSIZE": (10, 9),
    "DPI": 300,
    "TITLE": "Affected Population by Building",
    "XLABEL": "Easting (m)",
    "YLABEL": "Northing (m)",

    # Binning (0–10 … 100+) and palette (11 colors → 11 bins)
    "BIN_MIN": 0,
    "BIN_MAX": 100,
    "BIN_STEP": 10,
    "PALETTE": [
        "#fee5d9","#fcbba1","#fc9272","#fb6a4a",
        "#ef3b2c","#cb181d","#a50f15","#67000d",
        "#4a0010","#2e0006","#1a0004"
    ],

    # Styles
    "UND_FACE": "white", "UND_EDGE": "black", "UND_LW": 0.3, "UND_ALPHA": 1.0,
    "DAM_EDGE": "black", "DAM_LW": 0.2, "DAM_ALPHA": 1.0,

    # AOI boundary
    "DRAW_AOI": True, "AOI_COLOR": "black", "AOI_LW": 1.0,

    # North arrow & scalebar
    "ADD_NORTH_ARROW": True, "NA_X": 0.05, "NA_Y": 0.15, "NA_LEN": 0.08,
    "NA_LABEL": "N", "NA_COLOR": "black", "NA_LW": 2, "NA_FONTSIZE": 14,
    "ADD_SCALEBAR": True, "SB_UNITS": "m", "SB_LOC": "lower right",

    # Legend
    "LEGEND_LOC": "upper right"
}
# --------------------------------------------------------------------

# Reproject to local UTM for plotting in meters
utm_crs = b.to_crs(4326).estimate_utm_crs()
b_utm   = b.to_crs(utm_crs)
dam_utm = b_utm[b_utm["Damaged"] == "yes"].copy()
und_utm = b_utm[b_utm["Damaged"] == "no"].copy()

# Create equal-width bins up to BIN_MAX and a 100+ bin
edges      = np.arange(P["BIN_MIN"], P["BIN_MAX"] + P["BIN_STEP"], P["BIN_STEP"])
edges_ext  = np.append(edges, np.inf)
labels     = [f"{edges[i]}–{edges[i+1]}" for i in range(len(edges)-1)] + [f"{P['BIN_MAX']}+"]

dam_utm["pop_bin"] = pd.cut(
    dam_utm["Pop_Est"].astype(float),
    bins=edges_ext, right=False, labels=labels, include_lowest=True
)
palette = P["PALETTE"][:len(labels)]

# Optional zoom window (lon/lat → UTM)
use_bbox = all(P[k] is not None for k in ("lon_min", "lon_max", "lat_min", "lat_max"))
if use_bbox:
    zoom_ll = gpd.GeoDataFrame(
        geometry=[box(min(P["lon_min"], P["lon_max"]),
                      min(P["lat_min"], P["lat_max"]),
                      max(P["lon_min"], P["lon_max"]),
                      max(P["lat_min"], P["lat_max"]))],
        crs=4326
    )
    xmin, ymin, xmax, ymax = zoom_ll.to_crs(utm_crs).total_bounds
else:
    xmin, ymin, xmax, ymax = b_utm.total_bounds

# Plot
fig, ax = plt.subplots(figsize=P["FIGSIZE"])
legend_elems = []

# Undamaged outlines
if not und_utm.empty:
    und_utm.plot(ax=ax, facecolor=P["UND_FACE"], edgecolor=P["UND_EDGE"],
                 linewidth=P["UND_LW"], alpha=P["UND_ALPHA"], zorder=1)
    legend_elems.append(Patch(facecolor=P["UND_FACE"], edgecolor=P["UND_EDGE"], label="Undamaged"))

# Damaged buildings colored by pop_bin
for lab, color in zip(labels, palette):
    sub = dam_utm[dam_utm["pop_bin"] == lab]
    if not sub.empty:
        sub.plot(ax=ax, facecolor=color, edgecolor=P["DAM_EDGE"],
                 linewidth=P["DAM_LW"], alpha=P["DAM_ALPHA"], zorder=2)
        legend_elems.append(Patch(facecolor=color, edgecolor=P["DAM_EDGE"], label=f"Affected Pop. {lab}"))

# AOI boundary
if P["DRAW_AOI"] and 'aoi_gdf_ll' in globals():
    aoi_gdf_ll.to_crs(utm_crs).boundary.plot(ax=ax, color=P["AOI_COLOR"], linewidth=P["AOI_LW"], zorder=4)
    legend_elems.append(Line2D([0], [0], color=P["AOI_COLOR"], lw=P["AOI_LW"], label="AOI Boundary"))

# Frame & decor
ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)

if P["ADD_NORTH_ARROW"]:
    ax.annotate(P["NA_LABEL"], xy=(P["NA_X"], P["NA_Y"]),
                xytext=(P["NA_X"], P["NA_Y"] - P["NA_LEN"]),
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

# =============================== C6) Summary ===============================
if "Damaged" in b.columns and "Pop_Est" in b.columns:
    affected_pop   = int(b.loc[b["Damaged"] == "yes", "Pop_Est"].sum())
    unaffected_pop = int(b.loc[b["Damaged"] == "no",  "Pop_Est"].sum())
    total_pop      = affected_pop + unaffected_pop

    print("\n Population Summary (within AOI):")
    print(f"   Affected population (Damaged buildings):     {affected_pop:,}")
    print(f"   Unaffected population (Undamaged buildings): {unaffected_pop:,}")
    print(f"   Total population (check):                    {total_pop:,}")
else:
    print(" Columns 'Damaged' and/or 'Pop_Est' not found in building dataset.")


