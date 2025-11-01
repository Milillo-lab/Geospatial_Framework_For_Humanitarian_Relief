## I) Utility 2: Betweenness Centrality

# ========================= I0) Imports  =========================
import os
import math
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import osmnx as ox

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap, ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

from shapely.geometry import Point, box
from IPython.display import Image, display

# Optional: scalebar (safe import)
try:
    from matplotlib_scalebar.scalebar import ScaleBar
    _have_scalebar = True
except Exception:
    _have_scalebar = False
import yaml

# ====================== I1) User Options & Paths ======================
# --- Spatial subset for BC (lon/lat degrees). Set USE_BBOX=False to use full area.
USE_BBOX = require("USE_BBOX", bool)
lon_min, lon_max = require("lon_min", float), require("lon_max", float)
lat_min, lat_max = require("lat_min", float), require("lat_max", float)

# --- Betweenness centrality options ---
USE_APPROX_BC = require("USE_APPROX_BC", bool)     # True: randomized approx (fast), False: exact (slow)
K_SAMPLES     = require("K_SAMPLES", int)          # used only if USE_APPROX_BC=True (clipped to #nodes)
BC_WEIGHT     = require("BC_WEIGHT", str)          # edge weight for BC
BC_NORMALIZED = require("BC_NORMALIZED", bool)     # normalize BC values to [0,1] scale

# --- Project output structure (consistent with earlier sections) ---
OUT_BASE   = r"./Outputs"
A_OUT_DIR  = os.path.join(OUT_BASE, "A")
E_OUT_DIR  = os.path.join(OUT_BASE, "E")
F_OUT_DIR  = os.path.join(OUT_BASE, "F")
I_OUT_DIR  = os.path.join(OUT_BASE, "I")
IMAGES_DIR = os.path.join(OUT_BASE, "Images")
os.makedirs(I_OUT_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# ============== I2) Gather UNDAMAGED nodes/edges (memory first) ===============
# Try ready-made undamaged from previous section F
nodes_ok_src = globals().get("nodes_ok", None)
edges_ok_src = globals().get("edges_ok", None)

# Else: take evaluated and filter Damaged=="no"
if (nodes_ok_src is None or getattr(nodes_ok_src, "empty", True)) and ("nodes_eval" in globals()):
    n_ = globals()["nodes_eval"]
    if isinstance(n_, gpd.GeoDataFrame) and not n_.empty and ("Damaged" in n_.columns):
        nodes_ok_src = n_[n_["Damaged"].astype(str).str.lower().eq("no")].copy()

if (edges_ok_src is None or getattr(edges_ok_src, "empty", True)) and ("edges_eval" in globals()):
    e_ = globals()["edges_eval"]
    if isinstance(e_, gpd.GeoDataFrame) and not e_.empty and ("Damaged" in e_.columns):
        edges_ok_src = e_[e_["Damaged"].astype(str).str.lower().eq("no")].copy()

# Else: fall back to disk (prefer TB-adjusted edges from F)
if (edges_ok_src is None or getattr(edges_ok_src, "empty", True)):
    cand_edges = [
        Path(F_OUT_DIR) / "Evaluated_Edges_withTunnels_Bridges.shp",
        Path(E_OUT_DIR) / "Evaluated_Edges.shp",
        Path(A_OUT_DIR) / "osm_edges.shp",
    ]
    for p in cand_edges:
        if p.exists():
            g = gpd.read_file(p)
            if "Damaged" in g.columns:
                edges_ok_src = g[g["Damaged"].astype(str).str.lower().eq("no")].copy()
            else:
                # last resort: take all as undamaged if no column
                edges_ok_src = g.copy()
            if not edges_ok_src.empty:
                break

if (nodes_ok_src is None or getattr(nodes_ok_src, "empty", True)):
    cand_nodes = [
        Path(F_OUT_DIR) / "Evaluated_Nodes_withTunnels_Bridges.shp",
        Path(A_OUT_DIR) / "osm_nodes.shp",
    ]
    for p in cand_nodes:
        if p.exists():
            n = gpd.read_file(p)
            # if a Damaged column exists we can filter, else keep all
            if "Damaged" in n.columns:
                nodes_ok_src = n[n["Damaged"].astype(str).str.lower().eq("no")].copy()
            else:
                nodes_ok_src = n.copy()
            if not nodes_ok_src.empty:
                break

if nodes_ok_src is None or edges_ok_src is None or nodes_ok_src.empty or edges_ok_src.empty:
    raise RuntimeError("Could not assemble undamaged nodes/edges for BC.")

print(f"UNDAMAGED for BC → nodes: {len(nodes_ok_src):,} | edges: {len(edges_ok_src):,}")

# ================= I3) Normalize schema & ensure lengths =================
def _ensure_nodes_index_xy(nodes_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    n = nodes_gdf.copy()
    if "osmid" not in n.columns:
        n = n.reset_index().rename(columns={"index":"osmid"})
    if n.index.name != "osmid":
        n = n.set_index("osmid", drop=False)
    if ("x" not in n.columns) or ("y" not in n.columns):
        if n.crs and getattr(n.crs, "is_projected", False):
            n_ll = n.to_crs(4326)
            n["x"] = n_ll.geometry.x
            n["y"] = n_ll.geometry.y
        else:
            n["x"] = n.geometry.x
            n["y"] = n.geometry.y
    return n

def _ensure_edges_uvk(edges_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    e = edges_gdf.copy()
    if ("u" not in e.columns) or ("v" not in e.columns):
        if isinstance(e.index, pd.MultiIndex):
            e = e.reset_index()
        else:
            raise ValueError("Edges must have columns 'u' and 'v' or MultiIndex (u,v,key).")
    if "key" not in e.columns:
        e["key"] = e.groupby(["u","v"]).cumcount()
    e = e.set_index(["u","v","key"], drop=False)
    e = e.drop(columns=[c for c in ("u","v","key") if c in e.columns])
    return e

def _ensure_lengths(edges_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    e = edges_gdf.copy()
    if "length" in e.columns and e["length"].notna().all():
        return e
    try:
        utm = e.estimate_utm_crs()
        e["length"] = e.to_crs(utm).geometry.length
    except Exception:
        from math import radians, sin, asin, sqrt, cos
        def hav_m(geom):
            try:
                coords = list(geom.coords)
            except Exception:
                return 0.0
            R = 6371008.8
            s = 0.0
            for (x1,y1),(x2,y2) in zip(coords[:-1], coords[1:]):
                dlat = radians(y2-y1); dlon = radians(x2-x1)
                a = sin(dlat/2)**2 + cos(radians(y1))*cos(radians(y2))*sin(dlon/2)**2
                s += 2*R*asin(sqrt(a))
            return s
        e["length"] = e.geometry.apply(hav_m)
    return e

nodes_bc = _ensure_nodes_index_xy(nodes_ok_src)
edges_bc = _ensure_edges_uvk(edges_ok_src)

# dtype align node index with edge u/v index level
u_dtype = edges_bc.index.get_level_values(0).dtype
if nodes_bc.index.dtype != u_dtype:
    nodes_bc.index = nodes_bc.index.astype(u_dtype)
    if "osmid" in nodes_bc.columns:
        nodes_bc["osmid"] = nodes_bc["osmid"].astype(u_dtype)

# ensure CRS
if nodes_bc.crs is None:
    nodes_bc = nodes_bc.set_crs(4326, allow_override=True)
if edges_bc.crs is None:
    edges_bc = edges_bc.set_crs(nodes_bc.crs, allow_override=True)

edges_bc = _ensure_lengths(edges_bc)

print(" Schema OK for OSMnx")
print("  Nodes index:", nodes_bc.index.name, "| dtype:", nodes_bc.index.dtype)
print("  Edges index levels:", list(edges_bc.index.names))
print("  Length present:", "length" in edges_bc.columns)

# ========================= I4) Optional lon/lat bbox clip (FIXED) =========================
def _valid(x): return x is not None and isinstance(x, (int, float))
use_bbox = USE_BBOX and all(_valid(v) for v in [lon_min, lon_max, lat_min, lat_max])

if use_bbox:
    # nodes → lon/lat
    nodes_ll = nodes_bc.to_crs(4326).copy()

    # edges → make sure u/v/key are columns before to_crs/filters
    edges_ll = edges_bc.copy()
    if isinstance(edges_ll.index, pd.MultiIndex):
        edges_ll = edges_ll.reset_index()   # brings 'u','v','key' back as columns
    edges_ll = edges_ll.to_crs(4326)

    xmin, xmax = min(lon_min, lon_max), max(lon_min, lon_max)
    ymin, ymax = min(lat_min, lat_max), max(lat_min, lat_max)
    rect = box(xmin, ymin, xmax, ymax)

    # keep nodes strictly inside bbox
    nodes_clip = nodes_ll[nodes_ll.geometry.within(rect)].copy()
    keep_ids = set(nodes_clip["osmid"].tolist())

    # keep edges whose BOTH endpoints are inside
    if not {"u", "v"}.issubset(edges_ll.columns):
        raise ValueError("Edges must expose 'u' and 'v' columns at this step.")
    edges_clip = edges_ll[edges_ll["u"].isin(keep_ids) & edges_ll["v"].isin(keep_ids)].copy()

    # back to original CRS
    nodes_use = nodes_clip.to_crs(nodes_bc.crs)
    edges_use = edges_clip.to_crs(edges_bc.crs)

    # restore MultiIndex(u,v,key) for downstream steps
    for col in ("u", "v", "key"):
        if col not in edges_use.columns:
            raise ValueError(f"Column '{col}' missing after bbox clip; cannot restore MultiIndex.")
    edges_use = edges_use.set_index(["u", "v", "key"], drop=False)
    # drop duplicate u/v/key columns to keep graph_from_gdfs happy later
    edges_use = edges_use.drop(columns=[c for c in ("u", "v", "key") if c in edges_use.columns])

    print(f" BBox filter applied → nodes: {len(nodes_use):,} | edges: {len(edges_use):,}")
else:
    nodes_use = nodes_bc
    edges_use = edges_bc
    print(f" Using full undamaged network → nodes: {len(nodes_use):,} | edges: {len(edges_use):,}")

if nodes_use.empty or edges_use.empty:
    raise SystemExit("No nodes/edges in the selected extent for BC.")

# ========================= I5) Finalize OSMnx-friendly indices (robust) =========================
import pandas as pd
import geopandas as gpd

# --- Nodes: ensure index=osmid and x/y present ---
nodes_g = nodes_use.copy()
if 'osmid' not in nodes_g.columns:
    nodes_g = nodes_g.reset_index().rename(columns={'index': 'osmid'})
if nodes_g.index.name != 'osmid':
    nodes_g = nodes_g.set_index('osmid', drop=False)

if ('x' not in nodes_g.columns) or ('y' not in nodes_g.columns):
    if nodes_g.crs and getattr(nodes_g.crs, "is_projected", False):
        _n_ll = nodes_g.to_crs(4326)
        nodes_g['x'] = _n_ll.geometry.x
        nodes_g['y'] = _n_ll.geometry.y
    else:
        nodes_g['x'] = nodes_g.geometry.x
        nodes_g['y'] = nodes_g.geometry.y

# --- Edges: make sure u/v/key are columns, then restore canonical MultiIndex ---
edges_g = edges_use.copy()

# If u/v/key live in the index, bring them out as columns.
if isinstance(edges_g.index, pd.MultiIndex):
    idx_names = list(edges_g.index.names)
    edges_g = edges_g.reset_index()
    # rename index levels to exactly 'u','v','key' if needed
    rename_map = {}
    if len(idx_names) >= 1 and idx_names[0] and idx_names[0] != 'u':   rename_map[idx_names[0]] = 'u'
    if len(idx_names) >= 2 and idx_names[1] and idx_names[1] != 'v':   rename_map[idx_names[1]] = 'v'
    if len(idx_names) >= 3 and idx_names[2] and idx_names[2] != 'key': rename_map[idx_names[2]] = 'key'
    if rename_map:
        edges_g = edges_g.rename(columns=rename_map)

# Verify we have u/v; create key if missing.
missing_uv = [c for c in ('u','v') if c not in edges_g.columns]
if missing_uv:
    raise KeyError(f"Edges must have 'u' and 'v' columns; missing: {missing_uv}")
if 'key' not in edges_g.columns:
    edges_g['key'] = edges_g.groupby(['u','v']).cumcount()

# Restore MultiIndex (u,v,key) for graph_from_gdfs; then drop duplicate cols
edges_g = edges_g.set_index(['u','v','key'], drop=False)
edges_g = edges_g.drop(columns=[c for c in ('u','v','key') if c in edges_g.columns])

# ========================= I6) Build graph → undirected → largest CC =========================

# Ensure compatible CRS tags
if nodes_g.crs is None:
    nodes_g = nodes_g.set_crs(4326, allow_override=True)
if edges_g.crs is None:
    edges_g = edges_g.set_crs(nodes_g.crs, allow_override=True)

# Build MultiDiGraph from GDFs
G_multi = ox.convert.graph_from_gdfs(nodes_g, edges_g)

# Make undirected weighted by 'length' (already computed earlier)
G_und = ox.convert.to_undirected(G_multi)

# Largest connected component for stable BC
if G_und.number_of_nodes() == 0:
    raise SystemExit("Graph has no nodes after preprocessing.")
largest_cc = max(nx.connected_components(G_und), key=len)
G = G_und.subgraph(largest_cc).copy()

print(f"Graph for BC → nodes: {G.number_of_nodes():,} | edges: {G.number_of_edges():,}")

# ========================= I7) Betweenness centrality (exact or approx) =========================
# Expect these user controls from earlier (defaults if missing)
USE_APPROX_BC  = globals().get('USE_APPROX_BC', True)
K_SAMPLES      = int(globals().get('K_SAMPLES', 2000))
BC_WEIGHT      = globals().get('BC_WEIGHT', 'length')
BC_NORMALIZED  = bool(globals().get('BC_NORMALIZED', True))

if USE_APPROX_BC:
    k = min(K_SAMPLES, G.number_of_nodes())
    bc_dict = nx.betweenness_centrality(G, k=k, seed=0, weight=BC_WEIGHT, normalized=BC_NORMALIZED)
    print(f"Computed APPROX betweenness centrality with k={k}.")
else:
    bc_dict = nx.betweenness_centrality(G, weight=BC_WEIGHT, normalized=BC_NORMALIZED)
    print("Computed EXACT betweenness centrality.")

# Quick peek
if bc_dict:
    top_node = max(bc_dict, key=bc_dict.get)
    print(f" Top BC node: {top_node} | BC={bc_dict[top_node]:.6f}")
else:
    print(" Empty BC result (degenerate component?).")

# ========================= I8) Attach BC back to nodes dataset =========================

# Start from your broader nodes_bc (undamaged nodes before any bbox), not only the subgraph
nodes_bc = nodes_bc.copy()
if 'osmid' not in nodes_bc.columns:
    nodes_bc = nodes_bc.reset_index().rename(columns={'index': 'osmid'}).set_index('osmid', drop=False)

# Map BC values where available, fill 0 elsewhere
bc_series = pd.Series(bc_dict, name='bc', dtype=float)
nodes_bc['bc'] = nodes_bc['osmid'].map(bc_series).fillna(0.0).astype(float)

print(f"nodes_bc w/ BC assigned: {len(nodes_bc):,} rows | bc>0 count: {(nodes_bc['bc']>0).sum():,}")

# ========================= I9) Plot BC map (dark theme, single image) =========================
# --------------------------- USER CONTROLS (edit here) ---------------------------
P = {
    # ---- Output ----
    "OUT_NAME": "Betweenness_Centrality_Map.png",   # filename for the exported PNG

    # ---- Data / threshold ----
    "BC_THRESHOLD": 0.15,                           # show nodes whose betweenness centrality >= this value

    # ---- Figure & theme ----
    "FIGSIZE": (8, 7),                              # figure size (width, height) in inches
    "DPI": 300,                                     # export resolution (dots per inch)
    "BG_COLOR": "black",                            # background color for figure/axes
    "FG_COLOR": "white",                            # foreground color for ticks/labels/spines

    # ---- Base layers ----
    "EDGE_COLOR": "#808080",                        # line color for the base network edges
    "EDGE_LW": 0.7,                                 # linewidth for base network edges
    "EDGE_ALPHA": 1.0,                              # transparency for base network edges (0..1)
    "AOI_COLOR": "white",                           # color for AOI boundary
    "AOI_LW": 1.2,                                  # linewidth for AOI boundary
    "AOI_ALPHA": 1.0,                               # transparency for AOI boundary (0..1)
    "NODE_SIZE": 12,                                # marker size for plotted nodes (above threshold)
    "NODE_EDGEWIDTH": 0,                            # outline width for node markers (0 = none)
    "CMAP": "plasma",                               # colormap used to color nodes by BC value

    # ---- Colorbar ----
    "ADD_COLORBAR": True,                           # toggle colorbar visibility
    "CB_LABEL": "Betweenness Centrality",           # colorbar label text
    "CB_FRACTION": 0.03,                            # colorbar size fraction relative to axes
    "CB_PAD": 0.02,                                 # padding between axes and colorbar
    "CB_LABEL_COLOR": "white",                      # color for colorbar label text
    "CB_TICK_COLOR": "white",                       # color for colorbar tick labels

    # ---- Axes labels & title ----
    "XLABEL": "Easting (m)",                        # x-axis label
    "YLABEL": "Northing (m)",                       # y-axis label
    "TITLE": "Betweenness Centrality",              # plot title text
    "TITLE_COLOR": "white",                         # plot title color

    # ---- Extent selection ----
    "USE_BBOX": True,                               # True: use bbox below; False: auto extent from data
    "LON_MIN": 36.650, "LON_MAX": 37.150,           # longitude bounds (EPSG:4326) for map extent
    "LAT_MIN": 37.450, "LAT_MAX": 37.750,           # latitude bounds (EPSG:4326) for map extent

    # ---- North arrow ----
    "ADD_NORTH_ARROW": True,                        # toggle north arrow
    "NA_X": 0.05, "NA_Y": 0.15,                     # arrow position in axes fraction (0..1)
    "NA_LEN": 0.08,                                 # arrow length in axes fraction
    "NA_LABEL": "N",                                # text label shown at arrow head
    "NA_COLOR": "white", "NA_LW": 2, "NA_FONTSIZE": 14,  # arrow color, line width, font size

    # ---- Scalebar ----
    "ADD_SCALEBAR": True,                           # toggle scalebar (requires matplotlib-scalebar installed/imported)
    "SB_UNITS": "m",                                # scalebar units label
    "SB_LOC": "lower right",                        # scalebar location on axes
    "SB_BOX_ALPHA": 0.8,                            # scalebar box transparency (0..1)
    "SB_COLOR": "black",                            # scalebar text/line color

    # ---- Legend ----
    "ADD_LEGEND": True,                             # toggle legend visibility
    "LEGEND_LOC": "upper right",                    # legend placement (matplotlib code)
    "LEGEND_FACE": "white",                         # legend box facecolor
    "LEGEND_EDGE": "black",                         # legend box edgecolor
    "LEGEND_FRAME": True,                           # draw legend frame box
    "LBL_EDGES": "Undamaged edges",                 # legend label: base network line
    "LBL_NODES": "Nodes (BC ≥ {thr:.2f})",          # legend label: BC nodes (format with BC_THRESHOLD)
    "LBL_AOI": "AOI boundary",                      # legend label: AOI boundary
    "LEG_EDGES_LW": 2.0,                            # legend line width for edges item
    "LEG_AOI_LW": 1.2,                              # legend line width for AOI item
    "LEG_NODE_MARKERSIZE": 7                        # legend marker size for node item
}

# ---------------------------------------------------------------------------------------------

# Prepare UTM layers
os.makedirs(IMAGES_DIR, exist_ok=True)
seed    = edges_bc if not edges_bc.empty else nodes_bc
utm_crs = seed.to_crs(4326).estimate_utm_crs()
edges_m = edges_bc.to_crs(utm_crs)
nodes_m = nodes_bc.to_crs(utm_crs)
aoi_m   = aoi_gdf_ll.to_crs(utm_crs) if 'aoi_gdf_ll' in globals() else None

# Threshold + color scale
max_bc_val = float(nodes_m["bc"].max()) if len(nodes_m) else 0.0
cmap = get_cmap(P["CMAP"])
norm = Normalize(vmin=0.0, vmax=max_bc_val if max_bc_val > 0 else 1.0)
nodes_thr = nodes_m[nodes_m["bc"] >= P["BC_THRESHOLD"]].copy()

# Compute extent
def _bounds_from_ll(lon_min, lon_max, lat_min, lat_max, crs):
    ll = gpd.GeoDataFrame(
        geometry=[box(min(lon_min, lon_max), min(lat_min, lat_max),
                      max(lon_min, lon_max), max(lat_min, lat_max))],
        crs=4326
    )
    return ll.to_crs(crs).total_bounds

if P["USE_BBOX"]:
    xmin, ymin, xmax, ymax = _bounds_from_ll(P["LON_MIN"], P["LON_MAX"], P["LAT_MIN"], P["LAT_MAX"], utm_crs)
else:
    bounds = []
    if not edges_m.empty: bounds.append(edges_m.total_bounds)
    if not nodes_m.empty: bounds.append(nodes_m.total_bounds)
    if aoi_m is not None and not aoi_m.empty: bounds.append(aoi_m.total_bounds)
    xmin = min(b[0] for b in bounds); ymin = min(b[1] for b in bounds)
    xmax = max(b[2] for b in bounds); ymax = max(b[3] for b in bounds)

# --- Plot ---
fig, ax = plt.subplots(figsize=P["FIGSIZE"], facecolor=P["BG_COLOR"])
ax.set_facecolor(P["BG_COLOR"])

# Base network & AOI
if not edges_m.empty:
    edges_m.plot(ax=ax, color=P["EDGE_COLOR"], linewidth=P["EDGE_LW"], alpha=P["EDGE_ALPHA"], zorder=1)
if aoi_m is not None and not aoi_m.empty:
    aoi_m.boundary.plot(ax=ax, color=P["AOI_COLOR"], linewidth=P["AOI_LW"], alpha=P["AOI_ALPHA"], zorder=2)

# Nodes (BC ≥ threshold)
if not nodes_thr.empty:
    ax.scatter(nodes_thr.geometry.x, nodes_thr.geometry.y,
               c=nodes_thr["bc"], cmap=cmap, norm=norm,
               s=P["NODE_SIZE"], linewidths=P["NODE_EDGEWIDTH"], zorder=3)

# Colorbar
if P["ADD_COLORBAR"]:
    sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=P["CB_FRACTION"], pad=P["CB_PAD"])
    cbar.set_label(P["CB_LABEL"], color=P["CB_LABEL_COLOR"])
    cbar.ax.yaxis.set_tick_params(color=P["CB_TICK_COLOR"])
    plt.setp(cbar.ax.get_yticklabels(), color=P["CB_TICK_COLOR"])

# Frame + decor
ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
if P["ADD_NORTH_ARROW"]:
    ax.annotate(P["NA_LABEL"], xy=(P["NA_X"], P["NA_Y"]), xytext=(P["NA_X"], P["NA_Y"] - P["NA_LEN"]),
                xycoords="axes fraction", textcoords="axes fraction",
                ha="center", va="center", fontsize=P["NA_FONTSIZE"], fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", lw=P["NA_LW"], color=P["NA_COLOR"]),
                color=P["NA_COLOR"], clip_on=False, zorder=20)
if P["ADD_SCALEBAR"]:
    ax.add_artist(ScaleBar(1, P["SB_UNITS"], location=P["SB_LOC"],
                           box_alpha=P["SB_BOX_ALPHA"], color=P["SB_COLOR"]))

ax.set_aspect("equal"); ax.ticklabel_format(style="plain")
ax.tick_params(colors=P["FG_COLOR"])
for sp in ax.spines.values(): sp.set_edgecolor(P["FG_COLOR"])
ax.set_xlabel(P["XLABEL"], color=P["FG_COLOR"])
ax.set_ylabel(P["YLABEL"], color=P["FG_COLOR"])
ax.set_title(P["TITLE"], color=P["TITLE_COLOR"])

# Legend
if P["ADD_LEGEND"]:
    legend_items = [
        Line2D([0],[0], color=P["EDGE_COLOR"], lw=P["LEG_EDGES_LW"], label=P["LBL_EDGES"]),
        Line2D([0],[0], marker="o", color="none", markerfacecolor=P["FG_COLOR"],
               markersize=P["LEG_NODE_MARKERSIZE"], label=P["LBL_NODES"].format(thr=P["BC_THRESHOLD"])),
        Line2D([0],[0], color=P["AOI_COLOR"], lw=P["LEG_AOI_LW"], label=P["LBL_AOI"]),
    ]
    leg = ax.legend(handles=legend_items, loc=P["LEGEND_LOC"], frameon=P["LEGEND_FRAME"],
                    facecolor=P["LEGEND_FACE"], edgecolor=P["LEGEND_EDGE"])
    leg.set_zorder(1000)

# Save + show
out_path = Path(IMAGES_DIR) / P["OUT_NAME"]
fig.savefig(out_path, dpi=P["DPI"], bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f" Saved: {out_path}")
display(Image(filename=out_path))

# ========================= I10) Export BC table (node id, lon, lat, BC) =========================
# Where: nodes_bc['bc'] holds the values. Export lon/lat in EPSG:4326.
import pandas as pd
from pathlib import Path

# Ensure required columns and WGS84 coordinates
_nodes = nodes_bc.copy()
if 'osmid' not in _nodes.columns:
    _nodes = _nodes.reset_index().rename(columns={'index':'osmid'})

nodes_ll = _nodes.to_crs(4326) if getattr(_nodes.crs, "to_epsg", lambda: None)() != 4326 else _nodes

bc_df = pd.DataFrame({
    "node_id": nodes_ll["osmid"].astype(object),
    "lon": nodes_ll.geometry.x.astype(float),
    "lat": nodes_ll.geometry.y.astype(float),
    "BC": nodes_ll["bc"].astype(float)
}).sort_values("BC", ascending=False)

# Save CSV to Outputs_Final/I
bc_csv = Path(I_OUT_DIR) / "Betweenness_Centrality_nodes.csv"
bc_df.to_csv(bc_csv, index=False)
print(f" Saved BC CSV → {bc_csv}")
try:
    display(bc_df.head(10))
except Exception:
    pass
