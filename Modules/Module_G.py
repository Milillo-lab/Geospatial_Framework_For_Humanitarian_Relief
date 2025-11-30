## G) Damaged Node Calculation
# ====================== G0) Imports ======================
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import box
from matplotlib.lines import Line2D
from matplotlib_scalebar.scalebar import ScaleBar
from IPython.display import Image
# ====================== G1) Paths ======================

print("=== Section G: Damaged Node Calculation ===")

# Base output dirs
OUT_BASE   = r"./Outputs"
A_OUT_DIR  = os.path.join(OUT_BASE, "A")
E_OUT_DIR  = os.path.join(OUT_BASE, "E")
F_OUT_DIR  = os.path.join(OUT_BASE, "F")
G_OUT_DIR  = os.path.join(OUT_BASE, "G")
IMAGES_DIR = os.path.join(OUT_BASE, "Images")

os.makedirs(G_OUT_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# Edges: memory candidates and disk fallbacks
EDGE_VARS_MEM = ["edges_eval_tb", "edges_eval"]
EDGE_PATHS_FS = [
    os.path.join(F_OUT_DIR, "Evaluated_Edges_withTunnels_Bridges.shp"),
    os.path.join(E_OUT_DIR, "Evaluated_Edges.shp"),
]

# Nodes: memory + disk
NODES_VAR_MEM = "nodes"
NODES_PATH_FS = os.path.join(A_OUT_DIR, "osm_nodes.shp")

# Node outputs
EVAL_NODES_SHP     = Path(G_OUT_DIR) / "Evaluated_Nodes.shp"
DAMAGED_NODES_SHP  = Path(G_OUT_DIR) / "Damaged_Nodes.shp"
UNDAMAGED_NODES_SHP = Path(G_OUT_DIR) / "Undamaged_Nodes.shp"

# ====================== G2) Load Data (edges & nodes) ======================

# ---- Edges with 'Damaged' ----
edges_candidates = []

# 1) From memory
for var in EDGE_VARS_MEM:
    if var in globals():
        g = globals()[var]
        if isinstance(g, gpd.GeoDataFrame) and not g.empty and ("Damaged" in g.columns):
            edges_candidates.append(g.copy())
            print(f"G2: Using edges from memory variable '{var}'")
            break  # prefer first in priority list

# 2) From disk (if nothing in memory)
if not edges_candidates:
    for p in EDGE_PATHS_FS:
        pth = Path(p)
        if pth.exists():
            g = gpd.read_file(pth)
            if not g.empty and ("Damaged" in g.columns):
                edges_candidates.append(g)
                print(f"G2: Using edges from file '{pth}'")
                break

if not edges_candidates:
    raise RuntimeError("G2: No edges with a 'Damaged' column found (memory or disk).")

edges_in = edges_candidates[0].copy()
print(f"G2: Loaded {len(edges_in)} edges.")

# Require u/v columns
if not {"u", "v"}.issubset(edges_in.columns):
    raise RuntimeError("G2: edges must have 'u' and 'v' columns.")

# ---- Nodes ----
if NODES_VAR_MEM in globals() and isinstance(globals()[NODES_VAR_MEM], gpd.GeoDataFrame):
    nodes_src = globals()[NODES_VAR_MEM].copy()
    print(f"G2: Using nodes from memory variable '{NODES_VAR_MEM}'")
elif Path(NODES_PATH_FS).exists():
    nodes_src = gpd.read_file(NODES_PATH_FS)
    print(f"G2: Using nodes from file '{NODES_PATH_FS}'")
else:
    raise RuntimeError("G2: OSM nodes not found (memory or Outputs/A).")

# Ensure index is osmid
if "osmid" in nodes_src.columns:
    nodes_src = nodes_src.set_index("osmid", drop=False)
elif nodes_src.index.name == "osmid":
    pass
else:
    # fallback: assume index already carries osmid-like IDs
    nodes_src["osmid"] = nodes_src.index
    nodes_src = nodes_src.set_index("osmid", drop=False)

print(f"G2: Loaded {len(nodes_src)} nodes.")
print("G2: Edges CRS:", edges_in.crs)
print("G2: Nodes CRS:", nodes_src.crs)

# ====================== G3) Compute Damaged Nodes ======================

# Normalize edge damaged flag
dam_flag = edges_in["Damaged"].astype(str).str.lower().str.strip()
dam_mask = dam_flag.isin(["yes", "true", "1"])

edges_all = edges_in.dropna(subset=["u", "v"])
edges_dmg = edges_all[dam_mask]

# Count all incident edges per node (using u, v)
u_all = edges_all["u"].value_counts()
v_all = edges_all["v"].value_counts()
total_count = u_all.add(v_all, fill_value=0)

# Count damaged incident edges per node
u_dmg = edges_dmg["u"].value_counts()
v_dmg = edges_dmg["v"].value_counts()
damaged_count = u_dmg.add(v_dmg, fill_value=0)

# Align indices (nodes with at least 1 edge)
damaged_count_aligned = damaged_count.reindex(total_count.index, fill_value=0)

# Node damaged if ALL its incident edges are damaged
mask = (damaged_count_aligned == total_count) & (total_count > 0)
candidate_ids = total_count.index[mask]

# Restrict to nodes we actually have geometry for
damaged_nodes = [nid for nid in candidate_ids if nid in nodes_src.index]

print(f"G3: Nodes with at least 1 edge: {len(total_count):,}")
print(f"G3: Damaged nodes (by rule):   {len(damaged_nodes):,}")

# ====================== G4) Build Node GeoDataFrames ======================

nodes_eval = nodes_src.copy()
nodes_eval["Damaged"] = "no"
nodes_eval.loc[damaged_nodes, "Damaged"] = "yes"

nodes_dmg = nodes_eval[nodes_eval["Damaged"] == "yes"].copy()
nodes_ok  = nodes_eval[nodes_eval["Damaged"] != "yes"].copy()

print(f"G4: Nodes total     = {len(nodes_eval):,}")
print(f"G4: Damaged nodes   = {len(nodes_dmg):,}")
print(f"G4: Undamaged nodes = {len(nodes_ok):,}")

# ====================== G5) Save Node Shapefiles ======================

def _safe_write_nodes(gdf, path):
    gdf2 = gdf.copy()
    # Avoid index-name / osmid duplication on save
    gdf2.reset_index(drop=True, inplace=True)
    gdf2.to_file(path)

_safe_write_nodes(nodes_eval,    EVAL_NODES_SHP)
_safe_write_nodes(nodes_dmg,     DAMAGED_NODES_SHP)
_safe_write_nodes(nodes_ok,      UNDAMAGED_NODES_SHP)

print("G5: Saved node shapefiles:")
print(f"  Evaluated nodes: {EVAL_NODES_SHP}")
print(f"  Damaged nodes:   {DAMAGED_NODES_SHP}")
print(f"  Undamaged nodes: {UNDAMAGED_NODES_SHP}")

# ====================== G6) Plot: Damaged & Undamaged Nodes ======================

P = {
    "OUT_NAME": "Damaged_Nodes.png",

    # Manual zoom in EPSG:4326 (set to None for auto)
    "lon_min": 36.920,             # min longitude of custom zoom (None = auto)
    "lon_max": 36.940,             # max longitude (None = auto)
    "lat_min": 37.580,             # min latitude  (None = auto)
    "lat_max": 37.590,             # max latitude  (None = auto)

    "FIGSIZE": (11, 10),
    "DPI": 300,
    "TITLE": "Damaged Nodes (black) & Edges (red) with Network Context",
    "XLABEL": "Easting (m)",
    "YLABEL": "Northing (m)",

    "EDGE_ALL_COLOR": "#b0b0b0",
    "EDGE_ALL_LW": 0.6,
    "EDGE_ALL_ALPHA": 1.0,
    "EDGE_DMG_COLOR": "#d73027",
    "EDGE_DMG_LW": 1.2,
    "EDGE_DMG_ALPHA": 1.0,

    "NODE_OK_COLOR": "#1f78b4",
    "NODE_OK_SIZE": 12,
    "NODE_OK_ALPHA": 1.0,
    "NODE_DMG_COLOR": "black",
    "NODE_DMG_SIZE": 18,
    "NODE_DMG_ALPHA": 1.0,

    "DRAW_AOI": True,
    "AOI_COLOR": "black",
    "AOI_LW": 1.0,

    "ADD_NORTH_ARROW": True,
    "NA_X": 0.05, "NA_Y": 0.15,
    "NA_LEN": 0.08,
    "NA_LABEL": "N",
    "NA_COLOR": "black", "NA_LW": 2, "NA_FONTSIZE": 14,

    "ADD_SCALEBAR": True,
    "SB_UNITS": "m",
    "SB_LOC": "lower right",

    "LBL_EDGE_OK": "Undamaged edges",
    "LBL_EDGE_DMG": "Damaged edges",
    "LBL_NODE_OK": "Undamaged nodes",
    "LBL_NODE_DMG": "Damaged nodes",
    "LEGEND_LOC": "upper right",
}

# Data for plotting
e_src = edges_in.copy()
n_src = nodes_eval.copy()
aoi_src = aoi_gdf_ll if "aoi_gdf_ll" in globals() else None

# CRS: use edges CRS (already UTM in F)
utm_crs   = e_src.crs if e_src.crs is not None else e_src.to_crs(4326).estimate_utm_crs()
edges_utm = e_src.to_crs(utm_crs)
nodes_utm = n_src.to_crs(utm_crs)
aoi_utm   = aoi_src.to_crs(utm_crs) if isinstance(aoi_src, gpd.GeoDataFrame) else None

# Split edges/nodes
dam_col = edges_utm["Damaged"].astype(str).str.lower().str.strip()
e_all = edges_utm
e_dmg = edges_utm[dam_col.eq("yes")].copy()

n_dmg = nodes_utm[nodes_utm["Damaged"] == "yes"].copy()
n_ok  = nodes_utm[nodes_utm["Damaged"] != "yes"].copy()

# Extent
use_bbox = all(P[k] is not None for k in ("lon_min","lon_max","lat_min","lat_max"))

if use_bbox:
    aoi_ll_zoom = gpd.GeoDataFrame(
        geometry=[box(min(P["lon_min"], P["lon_max"]),
                      min(P["lat_min"], P["lat_max"]),
                      max(P["lon_min"], P["lon_max"]),
                      max(P["lat_min"], P["lat_max"]))],
        crs=4326,
    )
    xmin, ymin, xmax, ymax = aoi_ll_zoom.to_crs(utm_crs).total_bounds
else:
    bounds = []
    if not e_all.empty:
        bounds.append(e_all.total_bounds)
    if not nodes_utm.empty:
        bounds.append(nodes_utm.total_bounds)
    if aoi_utm is not None and not aoi_utm.empty:
        bounds.append(aoi_utm.total_bounds)
    xmin = min(b[0] for b in bounds); ymin = min(b[1] for b in bounds)
    xmax = max(b[2] for b in bounds); ymax = max(b[3] for b in bounds)

# Plot
fig, ax = plt.subplots(figsize=P["FIGSIZE"])
legend_items = []

# Edges (all)
e_all.plot(ax=ax, color=P["EDGE_ALL_COLOR"], linewidth=P["EDGE_ALL_LW"],
           alpha=P["EDGE_ALL_ALPHA"], zorder=1)
legend_items.append(Line2D([0],[0], color=P["EDGE_ALL_COLOR"], lw=2,
                           label=P["LBL_EDGE_OK"]))

# Damaged edges
e_dmg.plot(ax=ax, color=P["EDGE_DMG_COLOR"], linewidth=P["EDGE_DMG_LW"],
           alpha=P["EDGE_DMG_ALPHA"], zorder=2)
legend_items.append(Line2D([0],[0], color=P["EDGE_DMG_COLOR"], lw=2.5,
                           label=P["LBL_EDGE_DMG"]))

# Nodes (ok)
if not n_ok.empty:
    n_ok.plot(ax=ax, color=P["NODE_OK_COLOR"], markersize=P["NODE_OK_SIZE"],
              alpha=P["NODE_OK_ALPHA"], zorder=3)
    legend_items.append(Line2D([0],[0], marker="o", linestyle="None",
                               markerfacecolor=P["NODE_OK_COLOR"],
                               markeredgecolor="none", markersize=8,
                               label=P["LBL_NODE_OK"]))

# Nodes (damaged)
if not n_dmg.empty:
    n_dmg.plot(ax=ax, color=P["NODE_DMG_COLOR"], markersize=P["NODE_DMG_SIZE"],
               alpha=P["NODE_DMG_ALPHA"], zorder=4)
    legend_items.append(Line2D([0],[0], marker="o", linestyle="None",
                               markerfacecolor=P["NODE_DMG_COLOR"],
                               markeredgecolor="none", markersize=8,
                               label=P["LBL_NODE_DMG"]))

# AOI boundary
if P["DRAW_AOI"] and aoi_utm is not None and not aoi_utm.empty:
    aoi_utm.boundary.plot(ax=ax, color=P["AOI_COLOR"],
                          linewidth=P["AOI_LW"], zorder=5)

ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
ax.set_xlabel(P["XLABEL"]); ax.set_ylabel(P["YLABEL"])
ax.set_title(P["TITLE"])

# North arrow
if P["ADD_NORTH_ARROW"]:
    ax.annotate(P["NA_LABEL"],
                xy=(P["NA_X"], P["NA_Y"]),
                xytext=(P["NA_X"], P["NA_Y"] - P["NA_LEN"]),
                xycoords="axes fraction",
                textcoords="axes fraction",
                ha="center", va="center",
                fontsize=P["NA_FONTSIZE"], fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", lw=P["NA_LW"], color=P["NA_COLOR"]),
                clip_on=False, zorder=20)

# Scalebar
if P["ADD_SCALEBAR"]:
    ax.add_artist(ScaleBar(1, P["SB_UNITS"], location=P["SB_LOC"]))

ax.set_aspect("equal")
ax.legend(handles=legend_items, loc=P["LEGEND_LOC"], frameon=True)

out_png = Path(IMAGES_DIR) / P["OUT_NAME"]
fig.savefig(out_png, dpi=P["DPI"], bbox_inches="tight")
plt.close(fig)

print(f"G6: Saved figure: {out_png}")
display(Image(filename=str(out_png)))

# =============================== G7) Summary ===============================

n_edges_total = len(edges_in)
edge_dam_mask = edges_in["Damaged"].astype(str).str.lower().str.strip().isin(["yes", "true", "1"])
n_edges_dmg   = int(edge_dam_mask.sum())

n_nodes_total = len(nodes_eval)
n_nodes_dmg   = len(nodes_dmg)
n_nodes_ok    = len(nodes_ok)

print("\n===== Section G Summary =====")
print(f"Edges evaluated (total):                 {n_edges_total:,}")
print(f"  Damaged edges:                         {n_edges_dmg:,}")
print(f"Nodes evaluated (total):                 {n_nodes_total:,}")
print(f"  Damaged nodes (all incident damaged):  {n_nodes_dmg:,}")
print(f"  Undamaged nodes:                       {n_nodes_ok:,}")
print("Outputs:")
print(f"  Evaluated Nodes:                       {EVAL_NODES_SHP}")
print(f"  Damaged Nodes:                         {DAMAGED_NODES_SHP}")
print(f"  Undamaged Nodes:                       {UNDAMAGED_NODES_SHP}")
print(f"  Figure:                                {Path(IMAGES_DIR) / 'Damaged_Nodes.png'}")
print("=======================================================")
