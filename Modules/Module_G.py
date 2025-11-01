## G) Damaged Node Calculation

# =============================== G0) Imports ===============================
import os
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from shapely.geometry import box
from IPython.display import Image, display
import yaml

# ====================== G1) User Options & Paths ======================
OUT_BASE   = r"./Outputs"
A_OUT_DIR  = os.path.join(OUT_BASE, "A")
E_OUT_DIR  = os.path.join(OUT_BASE, "E")
F_OUT_DIR  = os.path.join(OUT_BASE, "F")
G_OUT_DIR  = os.path.join(OUT_BASE, "G")
IMAGES_DIR = os.path.join(OUT_BASE, "Images")
os.makedirs(G_OUT_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# Preferred evaluated-edges inputs (must include a 'Damaged' column)
EDGE_VARS_MEM = ["edges_eval_tb", "edges_eval_src", "edges_eval"]          # memory-first
EDGE_PATHS_FS = [os.path.join(F_OUT_DIR, "Evaluated_Edges_withTunnels_Bridges.shp"),
                 os.path.join(E_OUT_DIR, "Evaluated_Edges.shp")]           # disk fallbacks

# Preferred nodes inputs (raw OSM nodes; no labels required)
NODES_VAR_MEM = "nodes"
NODES_PATH_FS = os.path.join(A_OUT_DIR, "osm_nodes.shp")

# Output files for this section
EVAL_NODES_TB_SHP   = Path(G_OUT_DIR) / "Evaluated_Nodes_withTunnels_Bridges.shp"
DAMAGED_NODES_TB_SHP= Path(G_OUT_DIR) / "Damaged_Nodes_withTunnels_Bridges.shp"
UNDAM_NODES_TB_SHP  = Path(G_OUT_DIR) / "Undamaged_Nodes_withTunnels_Bridges.shp"

# ====================== G2) Load inputs (edges & nodes) ======================
# 1) Edges with 'Damaged'
edges_eval_candidates = []

for var in EDGE_VARS_MEM:
    if var in globals():
        g = globals()[var]
        if isinstance(g, gpd.GeoDataFrame) and not g.empty and ("Damaged" in g.columns):
            edges_eval_candidates.append(g.copy())

for p in EDGE_PATHS_FS:
    if Path(p).exists():
        g = gpd.read_file(p)
        if not g.empty and ("Damaged" in g.columns):
            edges_eval_candidates.append(g)

if not edges_eval_candidates:
    raise RuntimeError("No evaluated edges with a 'Damaged' column found.")

edges_eval_in = edges_eval_candidates[0]

# 2) Nodes (OSM)
if NODES_VAR_MEM in globals() and isinstance(globals()[NODES_VAR_MEM], gpd.GeoDataFrame) and not globals()[NODES_VAR_MEM].empty:
    nodes_src = globals()[NODES_VAR_MEM].copy()
elif Path(NODES_PATH_FS).exists():
    nodes_src = gpd.read_file(NODES_PATH_FS)
else:
    raise RuntimeError("No OSM nodes found (memory or Outputs_Final\\A).")

def ensure_uv_columns(edge_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Ensure columns 'u' and 'v' exist. If MultiIndex (u,v,key), reset index and rename.
    """
    if edge_gdf.empty:
        return edge_gdf
    if {'u','v'}.issubset(edge_gdf.columns):
        return edge_gdf
    if isinstance(edge_gdf.index, pd.MultiIndex):
        df = edge_gdf.reset_index()
        idx_names = list(edge_gdf.index.names)
        rename_map = {}
        if len(idx_names) >= 1:
            rename_map[idx_names[0] if idx_names[0] else 'level_0'] = 'u'
        if len(idx_names) >= 2:
            rename_map[idx_names[1] if idx_names[1] else 'level_1'] = 'v'
        df = df.rename(columns=rename_map)
        if 'u' not in df.columns or 'v' not in df.columns:
            raise KeyError("Could not map MultiIndex to 'u'/'v'. Inspect edges.")
        return df
    raise KeyError("Edges lack 'u'/'v' and are not a MultiIndex—inspect edges.")

edges_u = ensure_uv_columns(edges_eval_in)

# ============== G3) Classify nodes: all-incident-edges-damaged rule ==============
# Normalize 'Damaged' yes/no on edges
edge_dam = edges_u["Damaged"].astype(str).str.lower().str.strip().eq("yes")
edges_dmg = edges_u[edge_dam].copy()

print(f" Evaluated edges (input): {len(edges_u):,}")
print(f"   Damaged edges:           {len(edges_dmg):,}")
print(f"   Undamaged edges:         {len(edges_u) - len(edges_dmg):,}")

def node_degree_counts(edge_gdf: gpd.GeoDataFrame) -> pd.Series:
    if edge_gdf.empty:
        return pd.Series(dtype=int)
    # degree = number of incident edges (counts of u and v)
    return pd.concat([edge_gdf['u'], edge_gdf['v']]).value_counts()

deg_total   = node_degree_counts(edges_u)
deg_damaged = node_degree_counts(edges_dmg)

deg_df = (
    pd.DataFrame({'deg_total': deg_total})
      .join(deg_damaged.rename('deg_damaged'), how='left')
      .fillna({'deg_damaged': 0})
      .astype(int)
)

# Rule: node is "yes" only if ALL incident edges are damaged (and degree ≥ 1)
node_is_damaged = (deg_df['deg_damaged'] == deg_df['deg_total']) & (deg_df['deg_total'] >= 1)
deg_df['Damaged'] = node_is_damaged.map({True: "yes", False: "no"})

# Join onto nodes (index should be node IDs)
nodes_eval = nodes_src.join(deg_df, how='left')

# Isolated nodes (no incident edges) → undamaged by definition
nodes_eval[['deg_total','deg_damaged']] = nodes_eval[['deg_total','deg_damaged']].fillna(0).astype(int)
nodes_eval['Damaged'] = nodes_eval['Damaged'].fillna("no").astype(str)

nodes_dmg = nodes_eval[nodes_eval['Damaged'].str.lower().eq("yes")].copy()
nodes_ok  = nodes_eval[nodes_eval['Damaged'].str.lower().ne("yes")].copy()

print(f" Evaluated nodes:                        {len(nodes_eval):,}")
print(f"   Damaged nodes (all incident damaged):  {len(nodes_dmg):,}")
print(f"   Undamaged nodes:                       {len(nodes_ok):,}")

# =============================== G4) Save nodes ===============================
if not nodes_eval.empty: nodes_eval.to_file(EVAL_NODES_TB_SHP)
if not nodes_dmg.empty:  nodes_dmg.to_file(DAMAGED_NODES_TB_SHP)
if not nodes_ok.empty:   nodes_ok.to_file(UNDAM_NODES_TB_SHP)

print(" Saved:")
print("  ", EVAL_NODES_TB_SHP)
print("  ", DAMAGED_NODES_TB_SHP)
print("  ", UNDAM_NODES_TB_SHP)

# ======= G5) Plot: Nodes (Damaged BLACK, Undamaged BLUE) + Edges (Gray/Red) =======
# --------------------------- USER CONTROLS (edit here) ---------------------------
P = {
    # ---- Output ----
    "OUT_NAME": "Damaged_Nodes.png",      # output PNG name (saved to IMAGES_DIR)

    # ---- Manual zoom in EPSG:4326 (set all to None for auto extent) ----
    "lon_min": 36.920,                    # min longitude (None = auto)
    "lon_max": 36.940,                    # max longitude (None = auto)
    "lat_min": 37.575,                    # min latitude  (None = auto)
    "lat_max": 37.590,                    # max latitude  (None = auto)

    # ---- Figure & labels ----
    "FIGSIZE": (11, 10),                  # figure size (w, h) inches
    "DPI": 300,                           # export resolution
    "TITLE": "Damaged Nodes (black) & Edges (red) with Network Context",
    "XLABEL": "Easting (m)",              # projected meters
    "YLABEL": "Northing (m)",             # projected meters

    # ---- Edges style ----
    "EDGE_ALL_COLOR": "#b0b0b0",          # undamaged edges color (gray)
    "EDGE_ALL_LW": 0.6,                   # linewidth
    "EDGE_ALL_ALPHA": 1.0,                # transparency
    "EDGE_DMG_COLOR": "#d73027",          # damaged edges color (red)
    "EDGE_DMG_LW": 1.2,
    "EDGE_DMG_ALPHA": 1.0,
    "LBL_EDGE_OK": "Undamaged edges",     # legend label
    "LBL_EDGE_DMG": "Damaged edges",      # legend label

    # ---- Nodes style ----
    "NODE_OK_COLOR": "#1f78b4",           # undamaged nodes (blue)
    "NODE_OK_SIZE": 12,
    "NODE_OK_ALPHA": 1.0,
    "NODE_DMG_COLOR": "black",            # damaged nodes (black)
    "NODE_DMG_SIZE": 18,
    "NODE_DMG_ALPHA": 1.0,
    "LBL_NODE_OK": "Undamaged nodes",     # legend label
    "LBL_NODE_DMG": "Damaged nodes",      # legend label

    # ---- AOI boundary (optional if provided upstream) ----
    "DRAW_AOI": True,
    "AOI_COLOR": "black",
    "AOI_LW": 1.0,

    # ---- North arrow & scalebar ----
    "ADD_NORTH_ARROW": True,
    "NA_X": 0.05, "NA_Y": 0.15,           # position (axes fraction)
    "NA_LEN": 0.08,                       # arrow length (axes fraction)
    "NA_LABEL": "N",
    "NA_COLOR": "black", "NA_LW": 2, "NA_FONTSIZE": 14,
    "ADD_SCALEBAR": True,                 # assumes ScaleBar imported in section header
    "SB_UNITS": "m",
    "SB_LOC": "lower right",              # scalebar location

    # ---- Legend ----
    "LEGEND_LOC": "upper right"           # legend placement
}
# ------------------------------------------------------------------------------

# Reuse evaluated edges/nodes from memory in this section (unchanged logic)
e_src = edges_u.copy()
n_src = nodes_eval.copy()
aoi_src = aoi_gdf_ll if 'aoi_gdf_ll' in globals() else None

# Common CRS (local UTM from edges)
utm_crs   = e_src.to_crs(4326).estimate_utm_crs()
edges_utm = e_src.to_crs(utm_crs)
nodes_utm = n_src.to_crs(utm_crs)
aoi_utm   = aoi_src.to_crs(utm_crs) if isinstance(aoi_src, gpd.GeoDataFrame) else None

# Splits
dam_col = edges_utm["Damaged"].astype(str).str.lower().str.strip()
e_all   = edges_utm
e_dmg   = edges_utm[dam_col.eq("yes")].copy()
n_dmg   = nodes_utm[nodes_utm["Damaged"].astype(str).str.lower().eq("yes")].copy()
n_ok    = nodes_utm[nodes_utm["Damaged"].astype(str).str.lower().ne("yes")].copy()

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
    if not e_all.empty:     bounds.append(e_all.total_bounds)
    if not nodes_utm.empty: bounds.append(nodes_utm.total_bounds)
    if aoi_utm is not None and not aoi_utm.empty: bounds.append(aoi_utm.total_bounds)
    xmin = min(b[0] for b in bounds); ymin = min(b[1] for b in bounds)
    xmax = max(b[2] for b in bounds); ymax = max(b[3] for b in bounds)

# Plot
fig, ax = plt.subplots(figsize=P["FIGSIZE"])
legend_items = []

# Edges (ok + damaged)
e_all.plot(ax=ax, color=P["EDGE_ALL_COLOR"], linewidth=P["EDGE_ALL_LW"],
           alpha=P["EDGE_ALL_ALPHA"], zorder=1)
legend_items.append(Line2D([0],[0], color=P["EDGE_ALL_COLOR"], lw=2, label=P["LBL_EDGE_OK"]))

e_dmg.plot(ax=ax, color=P["EDGE_DMG_COLOR"], linewidth=P["EDGE_DMG_LW"],
           alpha=P["EDGE_DMG_ALPHA"], zorder=2)
legend_items.append(Line2D([0],[0], color=P["EDGE_DMG_COLOR"], lw=2.5, label=P["LBL_EDGE_DMG"]))

# Nodes (ok + damaged)
if not n_ok.empty:
    n_ok.plot(ax=ax, color=P["NODE_OK_COLOR"], markersize=P["NODE_OK_SIZE"],
              alpha=P["NODE_OK_ALPHA"], zorder=3)
    legend_items.append(Line2D([0],[0], marker="o", linestyle="None",
                               markerfacecolor=P["NODE_OK_COLOR"], markeredgecolor="none",
                               markersize=8, label=P["LBL_NODE_OK"]))
if not n_dmg.empty:
    n_dmg.plot(ax=ax, color=P["NODE_DMG_COLOR"], markersize=P["NODE_DMG_SIZE"],
               alpha=P["NODE_DMG_ALPHA"], zorder=4)
    legend_items.append(Line2D([0],[0], marker="o", linestyle="None",
                               markerfacecolor=P["NODE_DMG_COLOR"], markeredgecolor="none",
                               markersize=8, label=P["LBL_NODE_DMG"]))

# AOI boundary (optional)
if P["DRAW_AOI"] and aoi_utm is not None and not aoi_utm.empty:
    aoi_utm.boundary.plot(ax=ax, color=P["AOI_COLOR"], linewidth=P["AOI_LW"], zorder=5)

# Frame + decor
ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)

# North arrow
if P["ADD_NORTH_ARROW"]:
    ax.annotate(P["NA_LABEL"], xy=(P["NA_X"], P["NA_Y"]), xytext=(P["NA_X"], P["NA_Y"] - P["NA_LEN"]),
                xycoords="axes fraction", textcoords="axes fraction",
                ha="center", va="center", fontsize=P["NA_FONTSIZE"], fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", lw=P["NA_LW"], color=P["NA_COLOR"]),
                clip_on=False, zorder=20)

# Scalebar (assumes ScaleBar imported in the section header)
if P["ADD_SCALEBAR"]:
    ax.add_artist(ScaleBar(1, P["SB_UNITS"], location=P["SB_LOC"]))

ax.set_aspect("equal"); ax.ticklabel_format(style="plain")
ax.set_xlabel(P["XLABEL"]); ax.set_ylabel(P["YLABEL"])
ax.set_title(P["TITLE"])

ax.legend(handles=legend_items, loc=P["LEGEND_LOC"], frameon=True)

out_png = Path(IMAGES_DIR) / P["OUT_NAME"]
fig.savefig(out_png, dpi=P["DPI"], bbox_inches="tight")
plt.close(fig)
print(f" Saved: {out_png}")
display(Image(filename=str(out_png)))

# =============================== G6) Summary ===============================
n_edges_total = len(edges_u)
n_edges_dmg   = int(edge_dam.sum())
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
print(f"  Evaluated Nodes:                       {EVAL_NODES_TB_SHP}")
print(f"  Damaged Nodes:                         {DAMAGED_NODES_TB_SHP}")
print(f"  Undamaged Nodes:                       {UNDAM_NODES_TB_SHP}")
print(f"  Figure:                                {Path(IMAGES_DIR) / 'Damaged_Nodes.png'}")
print("=======================================================")
