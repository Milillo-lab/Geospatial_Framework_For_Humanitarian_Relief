## F) Tunnels and Bridges Damage Evaluation

# =============================== F0) Imports ===============================
import os
import numpy as np
import geopandas as gpd
import rasterio as rio
from rasterio.features import shapes
from shapely.geometry import shape, box
from shapely.prepared import prep
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from IPython.display import Image, display
import yaml

# ====================== F1) User Options & Paths ======================
OUT_BASE   = r"./Outputs"
A_OUT_DIR  = os.path.join(OUT_BASE, "A")
B_OUT_DIR  = os.path.join(OUT_BASE, "B")
C_OUT_DIR  = os.path.join(OUT_BASE, "C")
E_OUT_DIR  = os.path.join(OUT_BASE, "E")
F_OUT_DIR  = os.path.join(OUT_BASE, "F")
IMAGES_DIR = os.path.join(OUT_BASE, "Images")
os.makedirs(F_OUT_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# --- Inputs (memory-first; fallback to disk) ---
# Evaluated edges (with Damaged yes/no) from Section E
if 'edges_eval' in globals() and isinstance(edges_eval, gpd.GeoDataFrame) and not edges_eval.empty:
    edges_eval_src = edges_eval.copy()
elif 'EVAL_EDGES_SHP' in globals() and Path(EVAL_EDGES_SHP).exists():
    edges_eval_src = gpd.read_file(EVAL_EDGES_SHP)
else:
    # final fallback from Outputs_Final\E
    _eval_e = Path(E_OUT_DIR) / "Evaluated_Edges.shp"
    if _eval_e.exists():
        edges_eval_src = gpd.read_file(_eval_e)
    else:
        raise RuntimeError("Evaluated edges not found. Run Section E first.")

# Damaged pixels vector (prefer memory from Section B; else rasterize mask)
# NOTE: only used to override tunnels/bridges when they intersect damaged pixels.
if 'damaged_gdf' in globals() and isinstance(damaged_gdf, gpd.GeoDataFrame) and not damaged_gdf.empty:
    dmg_vec_pref = damaged_gdf.copy()
else:
    dmg_vec_pref = None

# Binary damaged mask GeoTIFF from Section B (fallback for vectorization)
DAMAGED_MASK_TIF = globals().get("DAMAGED_MASK_TIF", Path(C_OUT_DIR) / "AOI_DPM_DamagedMask_ge84.tif")

# --- Output file paths (for Section F) ---
EVAL_EDGES_TB_SHP       = Path(F_OUT_DIR) / "Evaluated_Edges_withTunnels_Bridges.shp"
DAMAGED_EDGES_TB_SHP    = Path(F_OUT_DIR) / "Damaged_Edges_withTunnels_Bridges.shp"
UNDAMAGED_EDGES_TB_SHP  = Path(F_OUT_DIR) / "Undamaged_Edges_withTunnels_Bridges.shp"

# =========== F2) Damaged pixels vector (memory → raster fallback) ===========
if dmg_vec_pref is not None and not dmg_vec_pref.empty:
    dmg_vec = dmg_vec_pref.to_crs(edges_eval_src.crs) if dmg_vec_pref.crs != edges_eval_src.crs else dmg_vec_pref.copy()
else:
    if not (isinstance(DAMAGED_MASK_TIF, (str, Path)) and Path(DAMAGED_MASK_TIF).exists()):
        # No damage layer available at all → short-circuit (reuse evaluated edges)
        print("No damaged pixels available; skipping TB override.")
        edges_eval_tb = edges_eval_src.copy()
        edges_dmg_tb  = edges_eval_tb.loc[edges_eval_tb["Damaged"].astype(str).str.lower().eq("yes")].copy()
        edges_ok_tb   = edges_eval_tb.loc[edges_eval_tb["Damaged"].astype(str).str.lower().eq("no")].copy()

        edges_eval_tb.to_file(EVAL_EDGES_TB_SHP)
        gpd.GeoDataFrame(geometry=edges_dmg_tb.geometry, crs=edges_dmg_tb.crs).to_file(DAMAGED_EDGES_TB_SHP)
        gpd.GeoDataFrame(geometry=edges_ok_tb.geometry,  crs=edges_ok_tb.crs).to_file(UNDAMAGED_EDGES_TB_SHP)

        print(f" Evaluated edges: {len(edges_eval_tb)}")
        print(f"  Damaged   (buffer OR TB override): {len(edges_dmg_tb)}")
        print(f"  Undamaged:                         {len(edges_ok_tb)}")
        print(" Saved:\n   ", EVAL_EDGES_TB_SHP, "\n   ", DAMAGED_EDGES_TB_SHP, "\n   ", UNDAMAGED_EDGES_TB_SHP)
        # You can proceed to F6 plot using these; skip F3–F5.
    else:
        with rio.open(DAMAGED_MASK_TIF) as ds:
            arr = ds.read(1)
            t   = ds.transform
            mask = (arr == 1)
            if not mask.any():
                dmg_vec = gpd.GeoDataFrame(geometry=[], crs=ds.crs)
            else:
                dam_polys = [shape(geom) for geom, val in shapes(mask.astype(np.uint8), mask=mask, transform=t) if val == 1]
                dmg_vec = gpd.GeoDataFrame(geometry=dam_polys, crs=ds.crs)

        if dmg_vec.crs is not None and dmg_vec.crs != edges_eval_src.crs:
            dmg_vec = dmg_vec.to_crs(edges_eval_src.crs)

# =========== F3) Identify tunnel/bridge edges (robust yes/no parsing) ===========
def _yes_like(series):
    s = series.astype(str).str.lower().str.strip()
    return s.isin(["yes", "true", "1", "y", "t"])

has_bridge = _yes_like(edges_eval_src["bridge"]) if "bridge" in edges_eval_src.columns else gpd.pd.Series(False, index=edges_eval_src.index)
has_tunnel = _yes_like(edges_eval_src["tunnel"]) if "tunnel" in edges_eval_src.columns else gpd.pd.Series(False, index=edges_eval_src.index)
tb_mask = has_bridge | has_tunnel
tb_edges = edges_eval_src.loc[tb_mask].copy()

# ======= F4) TB∩damage intersects → indices to force as Damaged =======
if dmg_vec is None or dmg_vec.empty or tb_edges.empty:
    print("No TB edges and/or no damaged pixels; using original edge classifications.")
    tb_hit_idx = gpd.pd.Index([])
else:
    # quick validity repair
    try:
        inv_tb = ~tb_edges.is_valid
        if inv_tb.any():
            tb_edges.loc[inv_tb, "geometry"] = tb_edges.loc[inv_tb, "geometry"].buffer(0)
        inv_d = ~dmg_vec.is_valid
        if inv_d.any():
            dmg_vec.loc[inv_d, "geometry"] = dmg_vec.loc[inv_d, "geometry"].buffer(0)
    except Exception:
        pass

    _ = dmg_vec.sindex  # build index
    hits = gpd.sjoin(tb_edges[["geometry"]], dmg_vec[["geometry"]], how="inner", predicate="intersects")
    tb_hit_idx = hits.index.unique()

print(f"Tunnel/Bridge edges total: {len(tb_edges)} | intersecting damaged pixels: {len(tb_hit_idx)}")

# ================== F5) Apply overrides → split → save ==================
edges_eval_tb = edges_eval_src.copy()
if len(tb_hit_idx) > 0:
    edges_eval_tb.loc[tb_hit_idx, "Damaged"] = "yes"

edges_dmg_tb = edges_eval_tb.loc[edges_eval_tb["Damaged"].astype(str).str.lower().eq("yes")].copy()
edges_ok_tb  = edges_eval_tb.loc[edges_eval_tb["Damaged"].astype(str).str.lower().eq("no")].copy()

# Save (Evaluated keeps attrs; split outputs as geometry-only for lean files)
edges_eval_tb.to_file(EVAL_EDGES_TB_SHP)
gpd.GeoDataFrame(geometry=edges_dmg_tb.geometry, crs=edges_dmg_tb.crs).to_file(DAMAGED_EDGES_TB_SHP)
gpd.GeoDataFrame(geometry=edges_ok_tb.geometry,  crs=edges_ok_tb.crs).to_file(UNDAMAGED_EDGES_TB_SHP)

print(f" Evaluated edges: {len(edges_eval_tb)}")
print(f"  Damaged   (buffer OR TB override): {len(edges_dmg_tb)}")
print(f"  Undamaged:                         {len(edges_ok_tb)}")
print(" Saved:")
print("   ", EVAL_EDGES_TB_SHP)
print("   ", DAMAGED_EDGES_TB_SHP)
print("   ", UNDAMAGED_EDGES_TB_SHP)

# ================== F6) Plot: Damaged Edges, TB, Buildings ==================
# --------------------------- USER CONTROLS (edit here) ---------------------------
P = {
    # ---- Output ----
    "OUT_NAME": "Damaged_Edges_Tunnels_Bridges.png",  # output PNG name (saved to IMAGES_DIR)

    # ---- Manual zoom in EPSG:4326 (set all to None for auto extent) ----
    "lon_min": 36.920,      # min longitude (None = auto)
    "lon_max": 36.940,      # max longitude (None = auto)
    "lat_min": 37.575,      # min latitude  (None = auto)
    "lat_max": 37.590,      # max latitude  (None = auto)

    # ---- Figure & labels ----
    "FIGSIZE": (11, 10),    # figure size (w, h) inches
    "DPI": 300,             # export resolution
    "TITLE": "Damaged Edges, Tunnels/Bridges, and Buildings",
    "XLABEL": "Easting (m)",   # projected meters
    "YLABEL": "Northing (m)",  # projected meters

    # ---- Base edges (all) ----
    "EDGE_ALL_COLOR": "#9e9e9e",  # color for all edges
    "EDGE_ALL_LW": 0.6,           # linewidth
    "EDGE_ALL_ALPHA": 1.0,        # transparency
    "LBL_EDGE_ALL": "All edges",  # legend label

    # ---- Damaged edges ----
    "EDGE_DMG_COLOR": "black",
    "EDGE_DMG_LW": 1.0,
    "EDGE_DMG_ALPHA": 1.0,
    "LBL_EDGE_DMG": "Damaged edges",

    # ---- Damaged tunnels/bridges subset ----
    "TB_COLOR": "#1f78b4",
    "TB_LW": 1.6,
    "TB_ALPHA": 1.0,
    "LBL_TB": "Damaged tunnels/bridges",

    # ---- Buildings (all) ----
    "BLDG_FACE": "white",
    "BLDG_EDGE": "black",
    "BLDG_LW": 0.3,
    "BLDG_ALPHA": 1.0,
    "LBL_BLDG": "Buildings",

    # ---- Damaged buildings ----
    "BLDG_DAM_FACE": "#d73027",
    "BLDG_DAM_EDGE": "black",
    "BLDG_DAM_LW": 0.2,
    "BLDG_DAM_ALPHA": 1.0,
    "LBL_BLDG_DAM": "Damaged buildings",

    # ---- AOI boundary (optional if provided upstream) ----
    "DRAW_AOI": True,
    "AOI_COLOR": "black",
    "AOI_LW": 1.0,
    "LBL_AOI": "AOI boundary",

    # ---- North arrow & scalebar ----
    "ADD_NORTH_ARROW": True,
    "NA_X": 0.05, "NA_Y": 0.15,   # arrow position (axes fraction)
    "NA_LEN": 0.08,               # arrow length (axes fraction)
    "NA_LABEL": "N",
    "NA_COLOR": "black", "NA_LW": 2, "NA_FONTSIZE": 14,
    "ADD_SCALEBAR": True,         # requires ScaleBar imported in section header
    "SB_UNITS": "m",
    "SB_LOC": "lower right",      # scalebar location

    # ---- Legend ----
    "LEGEND_LOC": "upper right"   # legend placement
}
# ------------------------------------------------------------------------------

# Edges (prefer TB-evaluated, then evaluated, then basic evaluated set)
if 'edges_eval_tb' in globals() and isinstance(edges_eval_tb, gpd.GeoDataFrame) and not edges_eval_tb.empty:
    e_src = edges_eval_tb.copy()
elif 'edges_eval_src' in globals() and isinstance(edges_eval_src, gpd.GeoDataFrame) and not edges_eval_src.empty:
    e_src = edges_eval_src.copy()
elif EVAL_EDGES_TB_SHP.exists():
    e_src = gpd.read_file(EVAL_EDGES_TB_SHP)
else:
    e_src = gpd.read_file(EVAL_EDGES_SHP) if 'EVAL_EDGES_SHP' in globals() and Path(EVAL_EDGES_SHP).exists() else None
    if e_src is None:
        raise RuntimeError("No evaluated edges available for plotting.")
if "Damaged" not in e_src.columns:
    raise ValueError("Edges layer must include 'Damaged'.")

# Buildings (must have Damaged)
if 'b' in globals() and isinstance(b, gpd.GeoDataFrame) and not b.empty and ("Damaged" in b.columns):
    b_src = b.copy()
elif 'buildings_r' in globals() and isinstance(buildings_r, gpd.GeoDataFrame) and not buildings_r.empty and ("Damaged" in buildings_r.columns):
    b_src = buildings_r.copy()
elif 'EVAL_OUT_SHP' in globals() and Path(EVAL_OUT_SHP).exists():
    b_src = gpd.read_file(EVAL_OUT_SHP)
    if "Damaged" not in b_src.columns:
        raise ValueError("Loaded buildings file has no 'Damaged' column.")
else:
    _b_eval = Path(B_OUT_DIR) / "Evaluated_Buildings.shp"
    if not _b_eval.exists():
        raise RuntimeError("No evaluated buildings found for plotting.")
    b_src = gpd.read_file(_b_eval)

aoi_src = aoi_gdf_ll if 'aoi_gdf_ll' in globals() else None

# Common local UTM
seed   = b_src if not b_src.empty else e_src
utm_crs = seed.to_crs(4326).estimate_utm_crs()
e_utm   = e_src.to_crs(utm_crs)
b_utm   = b_src.to_crs(utm_crs)
aoi_utm = aoi_src.to_crs(utm_crs) if isinstance(aoi_src, gpd.GeoDataFrame) else None

# Subsets
edge_is_dmg   = e_utm["Damaged"].astype(str).str.lower().eq("yes")
edges_all_utm = e_utm
edges_dmg_utm = e_utm[edge_is_dmg].copy()

def _yes_like(series):
    s = series.astype(str).str.lower().str.strip()
    return s.isin(["yes", "true", "1", "y", "t"])

has_bridge = _yes_like(e_utm["bridge"]) if "bridge" in e_utm.columns else gpd.pd.Series(False, index=e_utm.index)
has_tunnel = _yes_like(e_utm["tunnel"]) if "tunnel" in e_utm.columns else gpd.pd.Series(False, index=e_utm.index)
tb_dmg_utm = e_utm[(edge_is_dmg) & (has_bridge | has_tunnel)].copy()

b_dmg = b_utm[b_utm["Damaged"].astype(str).str.lower().eq("yes")].copy()

# Optional lon/lat zoom → UTM frame
use_bbox = all(P[k] is not None for k in ("lon_min","lon_max","lat_min","lat_max"))
if use_bbox:
    aoi_ll_zoom = gpd.GeoDataFrame(
        geometry=[box(min(P["lon_min"], P["lon_max"]),
                      min(P["lat_min"], P["lat_max"]),
                      max(P["lon_min"], P["lon_max"]),
                      max(P["lat_min"], P["lat_max"]))],
        crs=4326
    )
    xmin, ymin, xmax, ymax = aoi_ll_zoom.to_crs(utm_crs).total_bounds
else:
    bounds = []
    if not edges_all_utm.empty: bounds.append(edges_all_utm.total_bounds)
    if not b_utm.empty:         bounds.append(b_utm.total_bounds)
    if aoi_utm is not None and not aoi_utm.empty: bounds.append(aoi_utm.total_bounds)
    xmin = min(b[0] for b in bounds); ymin = min(b[1] for b in bounds)
    xmax = max(b[2] for b in bounds); ymax = max(b[3] for b in bounds)

# Plot
fig, ax = plt.subplots(figsize=P["FIGSIZE"])
legend_items = []

# All edges
edges_all_utm.plot(ax=ax, color=P["EDGE_ALL_COLOR"], linewidth=P["EDGE_ALL_LW"],
                   alpha=P["EDGE_ALL_ALPHA"], zorder=1)
legend_items.append(Line2D([0],[0], color=P["EDGE_ALL_COLOR"], lw=2.5, label=P["LBL_EDGE_ALL"]))

# Buildings (all)
b_utm.plot(ax=ax, facecolor=P["BLDG_FACE"], edgecolor=P["BLDG_EDGE"],
           linewidth=P["BLDG_LW"], alpha=P["BLDG_ALPHA"], zorder=2)
legend_items.append(Patch(facecolor=P["BLDG_FACE"], edgecolor=P["BLDG_EDGE"], label=P["LBL_BLDG"]))

# Damaged buildings
b_dmg.plot(ax=ax, facecolor=P["BLDG_DAM_FACE"], edgecolor=P["BLDG_DAM_EDGE"],
           linewidth=P["BLDG_DAM_LW"], alpha=P["BLDG_DAM_ALPHA"], zorder=3)
legend_items.append(Patch(facecolor=P["BLDG_DAM_FACE"], edgecolor=P["BLDG_DAM_EDGE"], label=P["LBL_BLDG_DAM"]))

# Damaged edges
edges_dmg_utm.plot(ax=ax, color=P["EDGE_DMG_COLOR"], linewidth=P["EDGE_DMG_LW"],
                   alpha=P["EDGE_DMG_ALPHA"], zorder=4)
legend_items.append(Line2D([0],[0], color=P["EDGE_DMG_COLOR"], lw=3, label=P["LBL_EDGE_DMG"]))

# Damaged tunnels/bridges
tb_dmg_utm.plot(ax=ax, color=P["TB_COLOR"], linewidth=P["TB_LW"],
                alpha=P["TB_ALPHA"], zorder=5)
legend_items.append(Line2D([0],[0], color=P["TB_COLOR"], lw=3, label=P["LBL_TB"]))

# AOI boundary (optional)
if P["DRAW_AOI"] and aoi_utm is not None and not aoi_utm.empty:
    aoi_utm.boundary.plot(ax=ax, color=P["AOI_COLOR"], linewidth=P["AOI_LW"], zorder=6)
    legend_items.append(Line2D([0],[0], color=P["AOI_COLOR"], lw=P["AOI_LW"], label=P["LBL_AOI"]))

# Frame + decor
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
    ax.add_artist(ScaleBar(1, P["SB_UNITS"], location=P["SB_LOC"]))

ax.set_aspect("equal"); ax.ticklabel_format(style="plain")
ax.set_xlabel(P["XLABEL"]); ax.set_ylabel(P["YLABEL"])
ax.set_title(P["TITLE"])

# Legend
ax.legend(handles=legend_items, loc=P["LEGEND_LOC"], frameon=True)

# Save + show
out_png = Path(IMAGES_DIR) / P["OUT_NAME"]
fig.savefig(out_png, dpi=P["DPI"], bbox_inches="tight")
plt.close(fig)
print(f" Saved: {out_png}")
display(Image(filename=str(out_png)))

# =============================== F7) Summary ===============================
n_eval = len(edges_eval_tb) if 'edges_eval_tb' in globals() else len(edges_eval_src)
n_dmg  = len(edges_dmg_tb)  if 'edges_dmg_tb'  in globals() else int((edges_eval_src["Damaged"].astype(str).str.lower()=="yes").sum())
n_ok   = len(edges_ok_tb)   if 'edges_ok_tb'   in globals() else int((edges_eval_src["Damaged"].astype(str).str.lower()=="no").sum())
n_tb   = int(tb_edges.shape[0]) if 'tb_edges' in globals() else 0
n_tb_hit = int(len(tb_hit_idx)) if 'tb_hit_idx' in globals() else 0

print("\n===== Section F Summary =====")
print(f"Evaluated edges (total):                 {n_eval:,}")
print(f"  Damaged edges (after TB override):     {n_dmg:,}")
print(f"  Undamaged edges:                       {n_ok:,}")
print(f"Tunnel/Bridge edges (OSM):               {n_tb:,}")
print(f"  TB edges intersecting damaged pixels:  {n_tb_hit:,}")
print("Outputs:")
print(f"  Evaluated_Edges_withTunnels_Bridges:   {EVAL_EDGES_TB_SHP}")
print(f"  Damaged_Edges_withTunnels_Bridges:     {DAMAGED_EDGES_TB_SHP}")
print(f"  Undamaged_Edges_withTunnels_Bridges:   {UNDAMAGED_EDGES_TB_SHP}")
print(f"  Figure:                                 {Path(IMAGES_DIR) / 'Damaged_Edges_Tunnels_Bridges.png'}")
print("=========================================================")
