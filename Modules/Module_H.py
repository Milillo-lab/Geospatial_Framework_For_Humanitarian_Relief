## H) Utility 1: Shortest Path
# =============================== H0) Imports ===============================
import os, json, math
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import osmnx as ox

from shapely.geometry import Point, box
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from IPython.display import Image, display
import yaml
from matplotlib_scalebar.scalebar import ScaleBar
# ====================== H1) User Options & Paths ======================
# --- Origin/Destination in configuration
orig_node = optional("orig_node")                 # e.g., 157370362
dest_node = optional("dest_node")                 # e.g., 13081265302
orig_lonlat = tuple(require("orig_lonlat"))       # (lon, lat) used if node IDs are None
dest_lonlat = tuple(require("dest_lonlat"))

# How many distinct shortest routes to collect in configuration
ROUTE_TARGET = require("ROUTE_TARGET", int)

# --- Output ---
OUT_BASE   = r"./Outputs"
A_OUT_DIR  = os.path.join(OUT_BASE, "A")
E_OUT_DIR  = os.path.join(OUT_BASE, "E")
F_OUT_DIR  = os.path.join(OUT_BASE, "F")
H_OUT_DIR  = os.path.join(OUT_BASE, "H")
IMAGES_DIR = os.path.join(OUT_BASE, "Images")

SP_OUT_DIR = os.path.join(H_OUT_DIR, "Shortest_Path")
os.makedirs(SP_OUT_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

print("Shortest-path outputs →", SP_OUT_DIR)

# ======================== H2) Helper Functions ========================
def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371008.8
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2
         + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2)
    return 2 * R * math.asin(math.sqrt(a))

def ensure_nodes_index_xy(nodes_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    For routing:
      - Ensure index is 'osmid' (matching edge u/v IDs),
      - Ensure CRS is EPSG:4326 (lon/lat),
      - Force columns 'x' and 'y' to be geometry.x / geometry.y in that CRS.
    """
    n = nodes_gdf.copy()

    # --- Index: use OSMID if present ---
    if "osmid" in n.columns:
        if n.index.name != "osmid":
            n = n.set_index("osmid", drop=False)
    else:
        # fallback: create osmid from existing index
        if n.index.name != "osmid":
            n = n.reset_index().rename(columns={"index": "osmid"})
            n = n.set_index("osmid", drop=False)

    # --- CRS: make sure geometries are in WGS84 lon/lat ---
    if n.crs is None:
        n = n.set_crs(4326, allow_override=True)
    if n.crs.to_epsg() != 4326:
        n = n.to_crs(4326)

    # --- x, y: ALWAYS recompute from geometry (ignore any old x/y) ---
    n["x"] = n.geometry.x  # longitude
    n["y"] = n.geometry.y  # latitude

    return n

def ensure_edges_index(edges_gdf):
    e = edges_gdf.copy()
    for must in ("u","v"):
        if must not in e.columns:
            raise ValueError(f"Edges missing required column '{must}'.")
    if "key" not in e.columns:
        e["key"] = e.groupby(["u","v"]).cumcount()
    if (not isinstance(e.index, pd.MultiIndex)) or (list(e.index.names) != ["u","v","key"]):
        e = e.set_index(["u","v","key"], drop=False)
    drop_cols = [c for c in ("u","v","key") if c in e.columns]
    if drop_cols:
        e = e.drop(columns=drop_cols)
    return e

def ensure_lengths(edges_gdf):
    e = edges_gdf.copy()
    if "length" in e.columns and e["length"].notna().all():
        return e
    try:
        utm = e.estimate_utm_crs()
        e["length"] = e.to_crs(utm).geometry.length
    except Exception:
        def approx_len_m(geom):
            try:
                coords = list(geom.coords)
            except Exception:
                return 0.0
            s = 0.0
            for (x1,y1),(x2,y2) in zip(coords[:-1], coords[1:]):
                s += haversine_m(y1,x1,y2,x2)
            return s
        e["length"] = e.geometry.apply(approx_len_m)
    return e

def build_graph(nodes_gdf, edges_gdf):
    if nodes_gdf.crs is None:
        nodes_gdf = nodes_gdf.set_crs(4326, allow_override=True)
    if edges_gdf.crs is None:
        edges_gdf = edges_gdf.set_crs(nodes_gdf.crs, allow_override=True)
    return ox.convert.graph_from_gdfs(nodes_gdf, edges_gdf)

def nearest_node_from_lonlat(G, lon, lat):
    """
    Find the closest graph node to (lon, lat) using explicit haversine
    distance between your (lon, lat) and each node's (x, y) = (lon, lat).
    """
    best_n, best_d = None, float("inf")
    for n, d in G.nodes(data=True):
        x = d.get("x", None)  # longitude
        y = d.get("y", None)  # latitude
        if x is None or y is None:
            continue
        dist = haversine_m(lat, lon, y, x)  # (lat1, lon1, lat2, lon2)
        if dist < best_d:
            best_d, best_n = dist, n
    return best_n

def get_route_length(G, route):
    if not route or len(route) < 2:
        return 0.0
    total = 0.0
    for u, v in zip(route[:-1], route[1:]):
        data = G.get_edge_data(u, v)
        if not data:
            continue
        total += min(d.get("length", 0.0) for d in data.values())
    return total

def export_nodes_csv_rows(G, route):
    rows = []
    if not route:
        return rows
    try:
        step_lengths = ox.utils_graph.get_route_edge_attributes(G, route, "length")
    except Exception:
        step_lengths = []
        for u, v in zip(route[:-1], route[1:]):
            data = G.get_edge_data(u, v) or {}
            step_lengths.append(min(d.get("length", 0.0) for d in data.values()) if data else 0.0)
    cum = 0.0
    for seq, n in enumerate(route):
        if seq > 0 and seq - 1 < len(step_lengths):
            cum += step_lengths[seq - 1]
        lon = G.nodes[n].get("x"); lat = G.nodes[n].get("y")
        rows.append({"seq": seq, "node": n, "lon": lon, "lat": lat, "cum_length_m": cum})
    return rows

def _route_signature(route):
    return tuple(route) if route else tuple()

def add_north_arrow(ax, x=0.05, y=0.15, length=0.08):
    ax.annotate("N", xy=(x, y), xytext=(x, y - length),
                xycoords="axes fraction", textcoords="axes fraction",
                ha="center", va="center", fontsize=12, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", lw=2, color="black"),
                clip_on=False, zorder=20)

# ========================= H3) Load Data =========================
if 'nodes' in globals() and isinstance(nodes, gpd.GeoDataFrame) and not nodes.empty:
    nodes_raw = nodes.copy()
else:
    nodes_path = Path(A_OUT_DIR) / "osm_nodes.shp"
    if nodes_path.exists():
        nodes_raw = gpd.read_file(nodes_path)

    else:
        raise RuntimeError("Nodes layer not found (memory or Outputs_Final\\A).")

# Evaluated edges
edges_eval_candidates = []
for _var in ("edges_eval_tb", "edges_eval_src", "edges_eval"):
    if _var in globals():
        _g = globals()[_var]
        if isinstance(_g, gpd.GeoDataFrame) and not _g.empty and ("Damaged" in _g.columns):
            edges_eval_candidates.append(_g.copy())

for _p in (Path(F_OUT_DIR) / "Evaluated_Edges_withTunnels_Bridges.shp",
           Path(E_OUT_DIR) / "Evaluated_Edges.shp"):
    if _p.exists():
        g = gpd.read_file(_p)
        if not g.empty and ("Damaged" in g.columns):
            edges_eval_candidates.append(g)

if not edges_eval_candidates:
    raise RuntimeError("Evaluated edges with 'Damaged' column not found (memory or disk).")

edges_eval_tb = edges_eval_candidates[0]

# Keep ONLY undamaged edges for routing
dam_col = edges_eval_tb["Damaged"].astype(str).str.lower().str.strip()
edges_ok = edges_eval_tb[~dam_col.eq("yes")].copy()
print(f"Routing over {len(edges_ok):,} undamaged edges (of {len(edges_eval_tb):,} total).")

# ===================== H4) Normalize & Build Graph =====================
nodes_norm  = ensure_nodes_index_xy(nodes_raw)
edges_norm  = ensure_edges_index(edges_ok)
edges_norm  = ensure_lengths(edges_norm)

G = build_graph(nodes_norm, edges_norm)

# Resolve Origin /Destination nodes (ID first; else nearest by lon/lat)
if orig_node is None:
    orig_node = nearest_node_from_lonlat(G, *orig_lonlat)
if dest_node is None:
    dest_node = nearest_node_from_lonlat(G, *dest_lonlat)

orig_x = G.nodes[orig_node].get("x")
orig_y = G.nodes[orig_node].get("y")
dest_x = G.nodes[dest_node].get("x")
dest_y = G.nodes[dest_node].get("y")

print(f"Graph built → nodes: {G.number_of_nodes():,} | edges: {G.number_of_edges():,}")
print(f"Requested origin lon/lat:      {orig_lonlat}")
print(f"Requested destination lon/lat: {dest_lonlat}")
print(f"Snapped origin node:      {orig_node}  → ({orig_x:.6f}, {orig_y:.6f})")
print(f"Snapped destination node: {dest_node}  → ({dest_x:.6f}, {dest_y:.6f})")

# =============== H5) Shortest Paths (expanding-radius variants) ===============
# First: direct shortest
try:
    route0 = ox.routing.shortest_path(G, orig_node, dest_node, weight="length")
except Exception:
    try:
        route0 = nx.shortest_path(G, orig_node, dest_node, weight="length", method="dijkstra")
    except Exception:
        route0 = None

routes_list, sigset = [], set()
if route0:
    routes_list.append(route0); sigset.add(_route_signature(route0))
    print(f"Direct path found: {len(route0)} nodes, {get_route_length(G, route0):.1f} m")
else:
    print("No direct path found yet—will try expanding-radius search.")

# Precompute single-source distances from origin
try:
    lengths, paths = nx.single_source_dijkstra(G, source=orig_node, weight="length")
except Exception as e:
    lengths, paths = {}, {}
    print(" single_source_dijkstra failed:", e)

# Node GeoDataFrame in lon/lat and meters for radius filtering
node_ids, node_pts_ll = [], []
for n, d in G.nodes(data=True):
    if "x" in d and "y" in d:
        node_ids.append(n); node_pts_ll.append(Point(d["x"], d["y"]))
nodes_ll = gpd.GeoDataFrame({"node": node_ids}, geometry=node_pts_ll, crs=4326)

try:
    utm = nodes_ll.estimate_utm_crs()
except Exception:
    utm = 3857
nodes_m = nodes_ll.to_crs(utm)

# Centers in meters
start_lon, start_lat = G.nodes[orig_node]["x"], G.nodes[orig_node]["y"]
dest_lon,  dest_lat  = G.nodes[dest_node]["x"],  G.nodes[dest_node]["y"]
start_m = gpd.GeoSeries([Point(start_lon, start_lat)], crs=4326).to_crs(utm).iloc[0]
dest_m  = gpd.GeoSeries([Point(dest_lon,  dest_lat)],  crs=4326).to_crs(utm).iloc[0]
d_m = start_m.distance(dest_m)
base_step = max(25.0, 0.01 * d_m)  # 1% of straight-line distance, min 25 m

has_sindex = getattr(nodes_m, "sindex", None) is not None
R = base_step
max_radius = max(d_m, 15000.0)
iters = 0

while len(routes_list) < ROUTE_TARGET and R <= max_radius:
    iters += 1
    circle = dest_m.buffer(R)

    if has_sindex:
        idx = list(nodes_m.sindex.query(circle.envelope))
        cand = nodes_m.iloc[idx]
        cand = cand[cand.geometry.within(circle)]
    else:
        cand = nodes_m[nodes_m.geometry.distance(dest_m) <= R]

    reachable = []
    if not cand.empty and lengths:
        reachable = [nid for nid in cand["node"].tolist() if nid in lengths]

    reachable_sorted = sorted(reachable, key=lambda nid: lengths[nid]) if reachable else []

    new_routes = 0
    for nid in reachable_sorted:
        rpath = paths.get(nid)
        if not rpath:
            continue
        sig = _route_signature(rpath)
        if sig in sigset:
            continue
        routes_list.append(rpath)
        sigset.add(sig)
        new_routes += 1
        if len(routes_list) >= ROUTE_TARGET:
            break

    pct = (R / d_m * 100.0) if d_m > 0 else float("inf")
    print(f"[Iter {iters}] R={R:.0f} m ({pct:.1f}%) | cand={0 if cand.empty else len(cand)} | "
          f"reach={len(reachable)} | new={new_routes} | total={len(routes_list)}")

    if len(routes_list) >= ROUTE_TARGET:
        break
    R += base_step

print(f"\n Total iterations: {iters}")
print(f" Routes collected: {len(routes_list)} (target {ROUTE_TARGET})")

# =============== H6) Export CSV + PNGs ===============
P = {
    # ---- Output names ----
    "CSV_NAME": "All_Routes_Combined.csv",      # filename for the combined CSV of all routes
    "OUT_NAME_FMT": "Shortest_path_{i}.png",    # per-route PNG name; {i} will be replaced by route index

    # ---- Manual zoom in EPSG:4326 (set all to None for auto extent) ----
    "lon_min": 36.900,             # min longitude of custom zoom (None = auto)
    "lon_max": 36.930,             # max longitude (None = auto)
    "lat_min": 37.570,             # min latitude  (None = auto)
    "lat_max": 37.600,             # max latitude  (None = auto)

    # ---- Figure & export ----
    "FIGSIZE": (7.5, 6.5),                      # figure size (width, height) in inches
    "DPI": 200,                                  # PNG resolution; lower = smaller files
    "XLABEL": "Easting (m)",                    # x-axis label
    "YLABEL": "Northing (m)",                   # y-axis label
    "LEGEND_LOC": "upper right",                # legend location (matplotlib position code)

    # ---- Base road network style ----
    "BASE_COLOR": "#cccccc",                    # base network line color
    "BASE_LW": 0.7,                             # base network line width
    "BASE_ALPHA": 1.0,                          # base network transparency (0..1)
    "LBL_BASE": "Road network",                 # legend label for base network

    # ---- AOI boundary (optional if provided upstream) ----
    "DRAW_AOI": True,                           # draw AOI boundary if available
    "AOI_COLOR": "black",                       # AOI boundary color
    "AOI_LW": 1.0,                              # AOI boundary line width
    "AOI_ALPHA": 1.0,                           # AOI boundary transparency
    "LBL_AOI": "AOI boundary",                  # legend label for AOI boundary
    "LEG_AOI_LW": 1.2,                          # legend line width for AOI item

    # ---- Route style ----
    "ROUTE_COLOR": "red",                       # route line color
    "ROUTE_LW": 2.6,                            # route line width
    "ROUTE_ALPHA": 1.0,                         # route transparency
    "LBL_ROUTE": "Route",                       # legend label for route

    # ---- Start/Destination markers ----
    "START_COLOR": "#1b9e77",                   # start marker color
    "START_SIZE": 48,                           # start marker size
    "DEST_COLOR": "black",                      # destination marker color
    "DEST_SIZE": 40,                            # destination marker size
    "LBL_START": "Start",                       # legend label for start
    "LBL_DEST": "Destination",                  # legend label for destination
    "LEG_MARKER_SIZE": 8,                       # legend marker size for start/dest items

    # ---- North arrow ----
    "ADD_NORTH_ARROW": True,                    # toggle north arrow
    "NA_X": 0.05, "NA_Y": 0.15,                 # arrow position (axes fraction)
    "NA_LEN": 0.08,                             # arrow length (axes fraction)
    "NA_LABEL": "N",                            # arrow label
    "NA_COLOR": "black", "NA_LW": 2, "NA_FONTSIZE": 14,  # arrow color/line width/font size

    # ---- Scalebar ----
    "ADD_SCALEBAR": True,                       # toggle scalebar
    "SB_UNITS": "m",                            # scalebar units
    "SB_LOC": "lower right",                    # scalebar location
    "SB_BOX_ALPHA": 0.8,                        # scalebar box transparency

    # ---- Title ----
    "TITLE_PREFIX": "Shortest Path"             # title prefix → "<prefix> i — length = Xm"
}
# ------------------------------------------------------------------------------

def _route_to_gdfs_ll(G, route):
    res = ox.routing.route_to_gdf(G, route, weight="length")
    if isinstance(res, tuple):
        return res[0], res[1]
    xs = [G.nodes[n]["x"] for n in route]
    ys = [G.nodes[n]["y"] for n in route]
    nodes_ll = gpd.GeoDataFrame({"node": route, "seq": range(len(route))},
                                geometry=gpd.points_from_xy(xs, ys), crs=4326)
    return res, nodes_ll

# Base network in meters (CRS from edges_ok)
base_utm = edges_ok.estimate_utm_crs()
base_m   = edges_ok.to_crs(base_utm)

# Optional AOI (in same meters CRS)
aoi_m = aoi_gdf_ll.to_crs(base_utm) if P["DRAW_AOI"] and 'aoi_gdf_ll' in globals() else None

# --- Save combined CSV for all routes (unchanged methodology) ---
all_rows = []
for i, r in enumerate(routes_list[:ROUTE_TARGET]):
    for row in export_nodes_csv_rows(G, r):
        row["route_id"] = i
        all_rows.append(row)

csv_path = Path(SP_OUT_DIR) / P["CSV_NAME"]
pd.DataFrame(all_rows).to_csv(csv_path, index=False)
print(f" Saved CSV → {csv_path}")

# --- Plot & save (smaller images; one PNG per route) ---
def _plot_route_png(i, r):
    edges_ll, nodes_ll = _route_to_gdfs_ll(G, r)
    edges_m = edges_ll.to_crs(base_utm)
    nodes_m = nodes_ll.to_crs(base_utm)

    # Start/End markers (project to meters CRS)
    start_lon, start_lat = G.nodes[orig_node]["x"], G.nodes[orig_node]["y"]
    dest_lon,  dest_lat  = G.nodes[dest_node]["x"],  G.nodes[dest_node]["y"]
    start_m = gpd.GeoSeries([Point(start_lon, start_lat)], crs=4326).to_crs(base_utm)
    dest_m  = gpd.GeoSeries([Point(dest_lon,  dest_lat)],  crs=4326).to_crs(base_utm)

    # Frame (zoom if user provided bbox; else auto from layers)
    if all(P[k] is not None for k in ("lon_min","lon_max","lat_min","lat_max")):
        zoom_ll = gpd.GeoDataFrame(
            geometry=[box(min(P["lon_min"], P["lon_max"]),
                          min(P["lat_min"], P["lat_max"]),
                          max(P["lon_min"], P["lon_max"]),
                          max(P["lat_min"], P["lat_max"]))],
            crs=4326
        ).to_crs(base_utm)
        xmin, ymin, xmax, ymax = zoom_ll.total_bounds
    else:
        bounds = []
        if not base_m.empty:  bounds.append(base_m.total_bounds)
        if not edges_m.empty: bounds.append(edges_m.total_bounds)
        if aoi_m is not None and not aoi_m.empty: bounds.append(aoi_m.total_bounds)
        xmin = min(b[0] for b in bounds); ymin = min(b[1] for b in bounds)
        xmax = max(b[2] for b in bounds); ymax = max(b[3] for b in bounds)

    # Plot
    fig, ax = plt.subplots(figsize=P["FIGSIZE"])
    if not base_m.empty:
        base_m.plot(ax=ax, color=P["BASE_COLOR"], linewidth=P["BASE_LW"],
                    alpha=P["BASE_ALPHA"], zorder=1)
    if aoi_m is not None and not aoi_m.empty:
        aoi_m.boundary.plot(ax=ax, color=P["AOI_COLOR"], linewidth=P["AOI_LW"],
                            alpha=P["AOI_ALPHA"], zorder=2)

    edges_m.plot(ax=ax, color=P["ROUTE_COLOR"], linewidth=P["ROUTE_LW"],
                 alpha=P["ROUTE_ALPHA"], zorder=3)
    if len(nodes_m) >= 2:
        nodes_m.iloc[[0, -1]].plot(ax=ax, color=P["ROUTE_COLOR"],
                                   markersize=max(P["ROUTE_LW"]*5, 10), zorder=4)

    start_m.plot(ax=ax, color=P["START_COLOR"], markersize=P["START_SIZE"], marker="o", zorder=5)
    dest_m.plot(ax=ax,  color=P["DEST_COLOR"],  markersize=P["DEST_SIZE"],  marker="o", zorder=5)

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
    length_m = get_route_length(G, r)
    ax.set_title(f"{P['TITLE_PREFIX']} {i} — length = {length_m:.1f} m")

    legend_items = [
        Line2D([0],[0], color=P["BASE_COLOR"], lw=3, label=P["LBL_BASE"]),
        Line2D([0],[0], color=P["ROUTE_COLOR"], lw=3, label=P["LBL_ROUTE"]),
        Line2D([0],[0], marker="o", color="none", markerfacecolor=P["START_COLOR"],
               markersize=P["LEG_MARKER_SIZE"], label=P["LBL_START"]),
        Line2D([0],[0], marker="o", color="none", markerfacecolor=P["DEST_COLOR"],
               markersize=P["LEG_MARKER_SIZE"], label=P["LBL_DEST"]),
        Line2D([0],[0], color=P["AOI_COLOR"], lw=P["LEG_AOI_LW"], label=P["LBL_AOI"])
    ]
    ax.legend(handles=legend_items, loc=P["LEGEND_LOC"], frameon=True)

    out_png_i = Path(SP_OUT_DIR) / P["OUT_NAME_FMT"].format(i=i)
    fig.savefig(out_png_i, dpi=P["DPI"], bbox_inches="tight")
    plt.close(fig)
    print(f" Saved: {out_png_i}")
    display(Image(filename=str(out_png_i)))

# Render all requested routes
for i, r in enumerate(routes_list[:ROUTE_TARGET]):
    _plot_route_png(i, r)
