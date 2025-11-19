## E) Buffer and Damaged Edge Calculation
# =============================== E0) Imports ===============================
import os
import numpy as np
import geopandas as gpd
from pathlib import Path

from shapely.geometry import box
from shapely.ops import unary_union
from shapely.prepared import prep

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from IPython.display import Image, display
import yaml

# =============================== E1) User Options & Paths ===============================
OUT_BASE   = r"./Outputs"
A_OUT_DIR  = os.path.join(OUT_BASE, "A")
B_OUT_DIR  = os.path.join(OUT_BASE, "B")
C_OUT_DIR  = os.path.join(OUT_BASE, "C")
E_OUT_DIR  = os.path.join(OUT_BASE, "E")
IMAGES_DIR = os.path.join(OUT_BASE, "Images")
os.makedirs(E_OUT_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# --- User-controllable numerical parameters from Configuration ---
HEIGHT_COL        = require("HEIGHT_COL", str)      # building height column
BUFFER_FALLBACK_M = require("BUFFER_FALLBACK_M", float)  # meters when height missing
SIMPLIFY_TOL_M    = require("SIMPLIFY_TOL_M", float)     # simplify buffers for speed
TILE_M            = require("TILE_M_E", float)      # tile side in meters (E section)
OVERLAP_M         = require("OVERLAP_M_E", float)   # overlap between tiles

# --- Data: damaged buildings ---
if 'damaged_only' in globals() and damaged_only is not None and not damaged_only.empty:
    bldg_damaged_src = damaged_only.copy()
elif 'b' in globals() and b is not None and not b.empty and ("Damaged" in b.columns):
    bldg_damaged_src = b[b["Damaged"].astype(str).str.lower().eq("yes")].copy()
elif 'buildings_r' in globals() and buildings_r is not None and not buildings_r.empty and ("Damaged" in buildings_r.columns):
    bldg_damaged_src = buildings_r[buildings_r["Damaged"].astype(str).str.lower().eq("yes")].copy()
else:
    c_eval_pop = os.path.join(C_OUT_DIR, "Evaluated_Buildings_withVol_Pop.shp")
    if os.path.exists(c_eval_pop):
        _tmp = gpd.read_file(c_eval_pop)
        if "Damaged" not in _tmp.columns:
            raise RuntimeError("Fallback buildings file lacks 'Damaged' column.")
        bldg_damaged_src = _tmp[_tmp["Damaged"].astype(str).str.lower().eq("yes")].copy()
    else:
        raise RuntimeError("No damaged buildings found (memory/C outputs unavailable).")

if bldg_damaged_src.empty:
    raise SystemExit("No damaged buildings; nothing to buffer.")

# --- Data: OSM edges ---
if 'edges' in globals() and isinstance(edges, gpd.GeoDataFrame) and not edges.empty:
    edges_src = edges.copy()
else:
    edges_path = os.path.join(A_OUT_DIR, "osm_edges.shp")
    if not os.path.exists(edges_path):
        raise RuntimeError("OSM edges not found in memory or Outputs_Final\\A.")
    edges_src = gpd.read_file(edges_path)
    if edges_src is None or edges_src.empty:
        raise RuntimeError("Loaded edges are empty.")

# --- Output file paths ---
EVAL_EDGES_SHP        = Path(os.path.join(E_OUT_DIR, "Evaluated_Edges.shp"))
DAMAGED_EDGES_SHP     = Path(os.path.join(E_OUT_DIR, "Damaged_Edges.shp"))
UNDAMAGED_EDGES_SHP   = Path(os.path.join(E_OUT_DIR, "Undamaged_Edges.shp"))
BUFFER_ISLANDS_GPKG   = Path(os.path.join(E_OUT_DIR, "Buffer_Damaged_Buildings.gpkg"))
PER_BUILDING_BUFF_GPKG= Path(os.path.join(E_OUT_DIR, "PerBuilding_Buffers.gpkg"))

# ======== E2) Local UTM =========
seed_wgs84 = bldg_damaged_src.to_crs(4326)
utm_crs = seed_wgs84.estimate_utm_crs()
print("Local UTM for buffers & edge tests →", utm_crs)

bldg_damaged = bldg_damaged_src.to_crs(utm_crs)
edges_utm    = edges_src.to_crs(utm_crs)

# Cheap validity repairs
try:
    inv_b = ~bldg_damaged.is_valid
    if inv_b.any():
        bldg_damaged.loc[inv_b, "geometry"] = bldg_damaged.loc[inv_b, "geometry"].buffer(0)
    inv_e = ~edges_utm.is_valid
    if inv_e.any():
        edges_utm.loc[inv_e, "geometry"] = edges_utm.loc[inv_e, "geometry"].buffer(0)
except Exception:
    pass

# ============== E3) Per-building buffers  ==============

if HEIGHT_COL in bldg_damaged.columns:
    h_m = gpd.pd.to_numeric(bldg_damaged[HEIGHT_COL], errors='coerce')
else:
    h_m = gpd.pd.Series(np.nan, index=bldg_damaged.index, dtype=float)

h_m = h_m.fillna(BUFFER_FALLBACK_M).astype(float)
bldg_damaged["buf_m"] = h_m.values
print(f"Buffer stats (m): min={h_m.min():.2f}, median={h_m.median():.2f}, max={h_m.max():.2f}, n={len(h_m)}")

# Geometry-by-geometry buffers
per_buffers = [geom.buffer(dist) for geom, dist in zip(bldg_damaged.geometry, bldg_damaged["buf_m"])]
per_buffer_gdf = gpd.GeoDataFrame(
    bldg_damaged.drop(columns="geometry").copy(),
    geometry=per_buffers,
    crs=utm_crs
)
print(f"Per-building buffers created: {len(per_buffer_gdf)}")

# ============== E4) Dissolve buffers → islands ==============
dissolved = unary_union(per_buffer_gdf.geometry.values)
islands = list(dissolved.geoms) if getattr(dissolved, "geom_type", "") == "MultiPolygon" else [dissolved]

buffer_islands = gpd.GeoDataFrame(
    {"island_id": range(len(islands))},
    geometry=islands,
    crs=utm_crs
)
print(f"Dissolved islands: {len(buffer_islands)}")

# Optional light simplify for speed
buffers_simpl = buffer_islands.copy()
buffers_simpl["geometry"] = buffers_simpl.geometry.simplify(SIMPLIFY_TOL_M, preserve_topology=True)

# Save QA layers
if not buffer_islands.empty: buffer_islands.to_file(BUFFER_ISLANDS_GPKG, driver="GPKG")
if not per_buffer_gdf.empty: per_buffer_gdf.to_file(PER_BUILDING_BUFF_GPKG, driver="GPKG")

# ===================== E5) Tile the AOI and classify edges ====================
if buffers_simpl.empty:
    raise SystemExit("No buffer islands exist; cannot evaluate edges.")

minx, miny, maxx, maxy = buffers_simpl.total_bounds
xs = np.arange(minx, maxx, TILE_M - OVERLAP_M)
ys = np.arange(miny, maxy, TILE_M - OVERLAP_M)

# Pre-build spatial index
_ = edges_utm.sindex
edge_damaged_mask = np.zeros(len(edges_utm), dtype=bool)

for x in xs:
    x1 = min(x + TILE_M, maxx)
    if x1 <= x: continue
    for y in ys:
        y1 = min(y + TILE_M, maxy)
        if y1 <= y: continue

        tile_box_geom = box(x, y, x1, y1)

        # Candidate islands
        try:
            Lb, Rb = buffers_simpl.sindex.query_bulk(
                gpd.GeoSeries([tile_box_geom], crs=utm_crs), predicate="intersects"
            )
            if Rb.size == 0: 
                continue
            isl_tile = buffers_simpl.iloc[np.unique(Rb)]
        except Exception:
            isl_tile = buffers_simpl[buffers_simpl.intersects(tile_box_geom)]
            if isl_tile.empty: 
                continue

        p = prep(isl_tile.union_all())

        # Candidate edges
        try:
            Le, Re = edges_utm.sindex.query_bulk(
                gpd.GeoSeries([tile_box_geom], crs=utm_crs), predicate="intersects"
            )
            if Re.size == 0:
                continue
            cand = edges_utm.iloc[np.unique(Re)]
        except Exception:
            cand = edges_utm[edges_utm.intersects(tile_box_geom)]
            if cand.empty: 
                continue

        # Any intersection → damaged
        hit = cand.geometry.apply(p.intersects).to_numpy()
        if hit.any():
            edge_damaged_mask[cand.index.values[hit]] = True

# ============== E6) Build evaluated / damaged / undamaged edge layers ==============
edges_eval = edges_utm.copy()
edges_eval["Damaged"] = np.where(edge_damaged_mask, "yes", "no")

edges_dmg = edges_eval.loc[edges_eval["Damaged"] == "yes"].copy()
edges_ok  = edges_eval.loc[edges_eval["Damaged"] == "no"].copy()

# Keep geometry-only in the split outputs
keep_cols = []
if keep_cols:
    keep_cols = [c for c in keep_cols if c in edges_eval.columns]
    edges_dmg = gpd.GeoDataFrame(edges_dmg[keep_cols].copy(), geometry=edges_dmg.geometry, crs=edges_dmg.crs)
    edges_ok  = gpd.GeoDataFrame(edges_ok[keep_cols].copy(),  geometry=edges_ok.geometry,  crs=edges_ok.crs)
else:
    edges_dmg = gpd.GeoDataFrame(geometry=edges_dmg.geometry, crs=edges_dmg.crs)
    edges_ok  = gpd.GeoDataFrame(geometry=edges_ok.geometry,  crs=edges_ok.crs)

# =============================== E7) Save edges ===============================
edges_eval.to_file(EVAL_EDGES_SHP)
edges_dmg.to_file(DAMAGED_EDGES_SHP)
edges_ok.to_file(UNDAMAGED_EDGES_SHP)

print(f" Evaluated edges: {len(edges_eval)}")
print(f"  Damaged   (touches buffer): {len(edges_dmg)}")
print(f"  Undamaged:                  {len(edges_ok)}")
print(" Saved:")
print("  ", EVAL_EDGES_SHP)
print("  ", DAMAGED_EDGES_SHP)
print("  ", UNDAMAGED_EDGES_SHP)

# ============ E8) Plot: Buffers, Buildings, Damaged (Local UTM) ============
P = {
    # ---- Output ----
    "OUT_NAME": "Buffer_Zones.png",        # output PNG filename (saved under IMAGES_DIR)

    # ---- Manual zoom in EPSG:4326 (set all to None for auto extent) ----
    "lon_min": 36.900,             # min longitude of custom zoom (None = auto)
    "lon_max": 36.930,             # max longitude (None = auto)
    "lat_min": 37.570,             # min latitude  (None = auto)
    "lat_max": 37.600,             # max latitude  (None = auto)

    # ---- Figure & labels ----
    "FIGSIZE": (10, 9),                    # figure size (w, h) inches
    "DPI": 300,                            # export resolution
    "TITLE": "Buffer Zones, Buildings, and Damaged Buildings",
    "XLABEL": "Easting (m)",               # x-axis label (projected meters)
    "YLABEL": "Northing (m)",              # y-axis label (projected meters)

    # ---- Buffer polygons style ----
    "BUF_FACE": "black",                   # buffer fill color
    "BUF_EDGE": "none",                    # buffer outline color
    "BUF_ALPHA": 0.5,                      # buffer transparency

    # ---- All buildings (context) ----
    "BLDG_FACE": "white",                  # building fill color
    "BLDG_EDGE": "black",                  # building outline color
    "BLDG_LW": 0.3,                        # building outline width
    "BLDG_ALPHA": 1.0,                     # building transparency

    # ---- Damaged buildings ----
    "DAM_FACE": "#d73027",                 # damaged fill color
    "DAM_EDGE": "black",                   # damaged outline color
    "DAM_LW": 0.2,                         # damaged outline width
    "DAM_ALPHA": 1.0,                      # damaged transparency

    # ---- AOI boundary (optional) ----
    "DRAW_AOI": True,                      # draw AOI boundary (expects aoi_gdf_ll upstream)
    "AOI_COLOR": "black",                  # AOI line color
    "AOI_LW": 1.0,                         # AOI line width
    "LBL_AOI": "AOI Boundary",             # legend label

    # ---- North arrow & scalebar ----
    "ADD_NORTH_ARROW": True,               # draw north arrow
    "NA_X": 0.05, "NA_Y": 0.15,            # arrow position (axes fraction)
    "NA_LEN": 0.08,                        # arrow length (axes fraction)
    "NA_LABEL": "N",                       # arrow text
    "NA_COLOR": "black",                   # arrow color
    "NA_LW": 2,                            # arrow line width
    "NA_FONTSIZE": 14,                     # 'N' font size
    "ADD_SCALEBAR": True,                  # draw scalebar
    "SB_UNITS": "m",                       # scalebar units
    "SB_LOC": "lower right",               # scalebar location

    # ---- Legend ----
    "LBL_BUFFERS": "Buffer zones",
    "LBL_BUILDINGS": "Buildings",
    "LBL_DAMAGED": "Damaged buildings",
    "LEGEND_LOC": "upper right"            # legend placement
}
# ------------------------------------------------------------------------------

# Buildings "Damaged"
if 'b' in globals() and isinstance(b, gpd.GeoDataFrame) and not b.empty and ("Damaged" in b.columns):
    b_src = b.copy()
elif 'buildings_r' in globals() and isinstance(buildings_r, gpd.GeoDataFrame) and not buildings_r.empty and ("Damaged" in buildings_r.columns):
    b_src = buildings_r.copy()
else:
    eval_b = os.path.join(B_OUT_DIR, "Evaluated_Buildings.shp")
    if not os.path.exists(eval_b):
        raise RuntimeError("No evaluated buildings found for plotting.")
    b_src = gpd.read_file(eval_b)
    if "Damaged" not in b_src.columns:
        raise ValueError("Loaded buildings file has no 'Damaged' column.")

# Buffers fallback to per-building
buffers_src = buffer_islands if 'buffer_islands' in globals() else per_buffer_gdf
if buffers_src is None or buffers_src.empty:
    raise RuntimeError("No buffer polygons found for plotting.")

# Optional AOI
aoi_src = aoi_gdf_ll if 'aoi_gdf_ll' in globals() else None

# Reproject to local UTM
seed = buffers_src if not buffers_src.empty else b_src
utm_plot    = seed.to_crs(4326).estimate_utm_crs()
b_utm       = b_src.to_crs(utm_plot)
buffers_utm = buffers_src.to_crs(utm_plot)
aoi_utm     = aoi_src.to_crs(utm_plot) if isinstance(aoi_src, gpd.GeoDataFrame) else None
dam_utm     = b_utm[b_utm["Damaged"].astype(str).str.lower().eq("yes")].copy()

# Frame
use_bbox = all(P[k] is not None for k in ("lon_min","lon_max","lat_min","lat_max"))
if use_bbox:
    aoi_ll_zoom = gpd.GeoDataFrame(
        geometry=[box(min(P["lon_min"], P["lon_max"]),
                      min(P["lat_min"], P["lat_max"]),
                      max(P["lon_min"], P["lon_max"]),
                      max(P["lat_min"], P["lat_max"]))],
        crs=4326
    )
    xmin, ymin, xmax, ymax = aoi_ll_zoom.to_crs(utm_plot).total_bounds
else:
    bounds = []
    if not b_utm.empty:       bounds.append(b_utm.total_bounds)
    if not buffers_utm.empty: bounds.append(buffers_utm.total_bounds)
    if aoi_utm is not None and not aoi_utm.empty: bounds.append(aoi_utm.total_bounds)
    xmin = min(b[0] for b in bounds); ymin = min(b[1] for b in bounds)
    xmax = max(b[2] for b in bounds); ymax = max(b[3] for b in bounds)

# Plot
fig, ax = plt.subplots(figsize=P["FIGSIZE"])
legend_elems = []

# Buffers
buffers_utm.plot(ax=ax, facecolor=P["BUF_FACE"], edgecolor=P["BUF_EDGE"],
                 alpha=P["BUF_ALPHA"], zorder=1)
legend_elems.append(Patch(facecolor=P["BUF_FACE"], edgecolor=P["BUF_EDGE"],
                          alpha=P["BUF_ALPHA"], label=P["LBL_BUFFERS"]))

# All buildings
b_utm.plot(ax=ax, facecolor=P["BLDG_FACE"], edgecolor=P["BLDG_EDGE"],
           linewidth=P["BLDG_LW"], alpha=P["BLDG_ALPHA"], zorder=2)
legend_elems.append(Patch(facecolor=P["BLDG_FACE"], edgecolor=P["BLDG_EDGE"],
                          label=P["LBL_BUILDINGS"]))

# Damaged buildings
dam_utm.plot(ax=ax, facecolor=P["DAM_FACE"], edgecolor=P["DAM_EDGE"],
             linewidth=P["DAM_LW"], alpha=P["DAM_ALPHA"], zorder=3)
legend_elems.append(Patch(facecolor=P["DAM_FACE"], edgecolor=P["DAM_EDGE"],
                          label=P["LBL_DAMAGED"]))

# AOI boundary
if P["DRAW_AOI"] and aoi_utm is not None and not aoi_utm.empty:
    aoi_utm.boundary.plot(ax=ax, color=P["AOI_COLOR"], linewidth=P["AOI_LW"], zorder=4)
    legend_elems.append(Line2D([0],[0], color=P["AOI_COLOR"], lw=P["AOI_LW"], label=P["LBL_AOI"]))

# Frame + decor
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

# Save + show
out_png = Path(IMAGES_DIR) / P["OUT_NAME"]
fig.savefig(out_png, dpi=P["DPI"], bbox_inches="tight")
plt.close(fig)
print(f" Saved: {out_png}")
display(Image(filename=str(out_png)))

# =============================== E9) Summary ===============================
n_eval   = len(edges_eval) if 'edges_eval' in globals() else 0
n_dmg    = len(edges_dmg)  if 'edges_dmg'  in globals() else 0
n_ok     = len(edges_ok)   if 'edges_ok'   in globals() else 0
n_island = len(buffer_islands) if 'buffer_islands' in globals() else 0
avg_buf  = float(per_buffer_gdf["buf_m"].mean()) if ('per_buffer_gdf' in globals() and "buf_m" in per_buffer_gdf.columns) else float("nan")

print("\n===== Section E Summary =====")
print(f"Total edges evaluated:        {n_eval:,}")
print(f"  Damaged edges (touching):   {n_dmg:,}")
print(f"  Undamaged edges:            {n_ok:,}")
print(f"Buffer islands (dissolved):   {n_island:,}")
print(f"Per-building buffer (m):      mean ≈ {avg_buf:.2f}")
print("\nOutputs:")
print(f"  Evaluated_Edges:            {EVAL_EDGES_SHP}")
print(f"  Damaged_Edges:              {DAMAGED_EDGES_SHP}")
print(f"  Undamaged_Edges:            {UNDAMAGED_EDGES_SHP}")
print(f"  Buffer islands (QA, gpkg):  {BUFFER_ISLANDS_GPKG}")
print(f"  Per-building buffers (gpkg):{PER_BUILDING_BUFF_GPKG}")
print(f"  Figure:                     {Path(IMAGES_DIR) / 'Buffer_Zones.png'}")
print("========================================")
