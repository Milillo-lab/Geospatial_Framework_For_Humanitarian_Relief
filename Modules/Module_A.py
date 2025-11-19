## A. Data Download and Display
# =============================== A0) Imports ===============================
import os
import io
import time
from pathlib import Path

import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, box

import osmnx as ox
from owslib.wfs import WebFeatureService

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib_scalebar.scalebar import ScaleBar

from IPython.display import display
import yaml

# ============== A1) Data, Outputs, and User Parameters ===================

# ---- Directories ----
DATA_DIR   = r"./Data"
OUT_DIR_BASE = r"./Outputs"

# Section-scoped outputs (this is Section A)
OUT_DIR      = os.path.join(OUT_DIR_BASE, "A")
IMAGES_DIR   = os.path.join(OUT_DIR_BASE, "Images")

# Ensure output directories exist
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# ---- AOI selection mode: choose ONE of "place", "coords", or "shapefile"
MODE = require("MODE", str)       # "place" | "coords" | "shapefile"

# If MODE == "place": the region name to query from OSM
PLACE_NAME = require("PLACE_NAME", str)

# Historical OSM snapshot date (optional in Configuration)
target_dt = require("target_dt", str)
ox.settings.overpass_settings   = f'[out:json][timeout:180][date:"{target_dt}"]'
ox.settings.overpass_rate_limit = True

# WFS server/layer
WFS_URL   = require("WFS_URL", str)
WFS_LAYER = require("WFS_LAYER", str)

# Output filenames (written under OUT_DIR)
OUT_BLDG      = "buildings_selected_region.shp"
OUT_NODES     = "osm_nodes.shp"
OUT_EDGES     = "osm_edges.shp"
OUT_CITY_POLY = "region_boundary.geojson"   # saved when MODE == "place"

# Grid splitting for WFS
GRID_PARTS = require("GRID_PARTS", int)   # e.g., 4x4

# Optional: pad the AOI bounding box used for WFS (degrees)
BBOX_PAD_DEG = require("BBOX_PAD_DEG", float)

# If MODE == "coords": manual AOI polygon coordinates (lon, lat)
AOI_COORDS = [tuple(p) for p in require("AOI_COORDS")]

# If MODE == "shapefile": path to a polygon shapefile (inside DATA_DIR)
SHP_PATH = os.path.join(DATA_DIR, require("SHP_NAME", str))

# ============================ A2) Helper Functions =========================
def split_polygon(gdf_in: gpd.GeoDataFrame, n_parts: int = 16) -> gpd.GeoDataFrame:
    """
    Make an equal grid over the AOI and intersect to get sub-polygons.
    For n_parts=16, creates a 4x4 grid over the AOI extent.
    """
    if gdf_in.empty:
        return gpd.GeoDataFrame(geometry=[], crs=gdf_in.crs)

    minx, miny, maxx, maxy = gdf_in.total_bounds
    cols = int(n_parts ** 0.5)
    rows = int(n_parts / cols)
    dx = (maxx - minx) / cols
    dy = (maxy - miny) / rows

    cells = []
    for i in range(cols):
        for j in range(rows):
            x1, y1 = minx + i * dx, miny + j * dy
            x2, y2 = x1 + dx, y1 + dy
            cells.append(box(x1, y1, x2, y2))

    grid = gpd.GeoDataFrame(geometry=cells, crs=gdf_in.crs)
    # Intersect grid with AOI to keep only overlapping parts
    try:
        out = gpd.overlay(grid, gdf_in, how="intersection", keep_geom_type=True)
    except Exception:
        out = grid
    return out


def download_buildings_chunk(
    wfs: WebFeatureService,
    bbox_3857: tuple,
    typename: str,
    srsname: str = "urn:x-ogc:def:crs:EPSG:3857",
    max_retries: int = 3,
    wait_sec: int = 5
) -> gpd.GeoDataFrame:
    """
    Download buildings using a rectangular bbox (EPSG:3857).
    """
    for attempt in range(1, max_retries + 1):
        try:
            response = wfs.getfeature(
                typename=typename,
                bbox=bbox_3857,
                srsname=srsname,
                outputFormat="application/json",
            )
            gdf = gpd.read_file(io.BytesIO(response.read()))
            if gdf.crs is None:
                gdf.set_crs("EPSG:3857", inplace=True)
            return gdf
        except Exception as e:
            print(f"  Error on attempt {attempt}/{max_retries} for bbox={bbox_3857}: {e}")
            if attempt < max_retries:
                time.sleep(wait_sec)

    return gpd.GeoDataFrame(geometry=[], crs="EPSG:3857")

# ===== A3) OSM download ===========
# ===== Warning Suppression =====
import warnings
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message="The 'unary_union' attribute is deprecated, use the 'union_all\\(\\)' method instead."
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Column names longer than 10 characters will be truncated when saved to ESRI Shapefile."
)
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message="Normalized/laundered field name:.*"
)

# Downloading
print(f"[OSM Download Mode] {MODE!r}")

if MODE == "place":
    print(f"Fetching OSM admin polygon for: {PLACE_NAME}")
    region_boundary_gdf = ox.geocode_to_gdf(PLACE_NAME).to_crs("EPSG:4326")

    # Save region boundary polygon
    region_poly_path = os.path.join(OUT_DIR, OUT_CITY_POLY)
    region_boundary_gdf.to_file(region_poly_path, driver="GeoJSON")
    print(f"Saved region boundary → {region_poly_path}")

    # Download OSM network within the exact region polygon
    print("Downloading OSM network within region polygon ...")
    region_geom = region_boundary_gdf.unary_union
    G_city = ox.graph_from_polygon(region_geom, network_type="all")
    nodes, edges = ox.graph_to_gdfs(G_city)

    # AOI polygon for later use in WFS
    aoi_polygon_ll = region_boundary_gdf.unary_union
    aoi_gdf_ll = gpd.GeoDataFrame(geometry=[aoi_polygon_ll], crs="EPSG:4326")

elif MODE == "shapefile":
    print(f"Loading AOI from shapefile: {SHP_PATH}")
    aoi_gdf_ll = gpd.read_file(SHP_PATH).to_crs("EPSG:4326")
    aoi_geom = aoi_gdf_ll.unary_union

    print("Downloading OSM network within AOI polygon ...")
    G_city = ox.graph_from_polygon(aoi_geom, network_type="all")
    nodes, edges = ox.graph_to_gdfs(G_city)

    # Also save the region boundary in this mode:
    region_boundary_gdf = aoi_gdf_ll.copy()
    region_poly_path = os.path.join(OUT_DIR, OUT_CITY_POLY)
    region_boundary_gdf.to_file(region_poly_path, driver="GeoJSON")
    print(f"Saved region boundary (from shapefile) → {region_poly_path}")

elif MODE == "coords":
    print("Building AOI polygon from manual coordinates ...")
    polygon_ll = Polygon(AOI_COORDS)
    aoi_gdf_ll = gpd.GeoDataFrame(geometry=[polygon_ll], crs="EPSG:4326")

    print("Downloading OSM network within AOI polygon ...")
    G_city = ox.graph_from_polygon(aoi_gdf_ll.unary_union, network_type="all")
    nodes, edges = ox.graph_to_gdfs(G_city)

    # Also save the region boundary in this mode:
    region_boundary_gdf = aoi_gdf_ll.copy()
    region_poly_path = os.path.join(OUT_DIR, OUT_CITY_POLY)
    region_boundary_gdf.to_file(region_poly_path, driver="GeoJSON")
    print(f"Saved region boundary (from coords) → {region_poly_path}")

else:
    raise ValueError("MODE must be one of: 'place', 'coords', 'shapefile'")

# Save OSM nodes and edges into Outputs\A
nodes_path = os.path.join(OUT_DIR, OUT_NODES)
edges_path = os.path.join(OUT_DIR, OUT_EDGES)
nodes.to_file(nodes_path)
edges.to_file(edges_path)
print(f"Saved OSM nodes → {nodes_path}")
print(f"Saved OSM edges → {edges_path}")

# === A4) WFS download using AOI bbox → clip to AOI polygon → save ==========
# Compute lon/lat min/max from AOI polygon
minx, miny, maxx, maxy = aoi_gdf_ll.total_bounds
if BBOX_PAD_DEG and BBOX_PAD_DEG > 0:
    minx -= BBOX_PAD_DEG; miny -= BBOX_PAD_DEG
    maxx += BBOX_PAD_DEG; maxy += BBOX_PAD_DEG

print("AOI bbox (lon/lat):")
print(f"  lon: {minx:.6f} .. {maxx:.6f}")
print(f"  lat: {miny:.6f} .. {maxy:.6f}")

# Convert AOI to EPSG:3857 and split into smaller polygons for WFS
aoi_gdf_3857 = aoi_gdf_ll.to_crs("EPSG:3857")
sub_polys_3857 = split_polygon(aoi_gdf_3857, n_parts=GRID_PARTS)
print(f"AOI split into {len(sub_polys_3857)} parts for WFS.\n")

# Connect to WFS
print("Connecting to WFS ...")
wfs = WebFeatureService(url=WFS_URL, version="1.1.0", timeout=120)
print("Connected.\n")

# Download per grid cell via BBOX
all_chunks = []
for i, row in sub_polys_3857.iterrows():
    part_id = i + 1
    cell_bounds = tuple(row.geometry.bounds)
    print(f"Downloading buildings via WFS: part {part_id}/{len(sub_polys_3857)}")
    gdf_part = download_buildings_chunk(
        wfs=wfs,
        bbox_3857=cell_bounds,
        typename=WFS_LAYER,
        srsname="urn:x-ogc:def:crs:EPSG:3857",
        max_retries=3,
        wait_sec=5
    )
    if gdf_part.empty:
        print(f"  No buildings for part {part_id}.")
        continue

    # Clip to the exact cell polygon
    cell_gdf = gpd.GeoDataFrame(geometry=[row.geometry], crs=sub_polys_3857.crs)
    gdf_clip_cell = gpd.overlay(gdf_part, cell_gdf, how="intersection")
    if not gdf_clip_cell.empty:
        all_chunks.append(gdf_clip_cell)

if not all_chunks:
    raise SystemExit("No buildings were downloaded from WFS for the AOI.")

# Merge and clip to the full AOI polygon
print("\nMerging WFS parts ...")
total_chunks = len(all_chunks)
print(f"  Total non-empty chunks to merge: {total_chunks}")

merged_list = []
for idx, gdf_chunk in enumerate(all_chunks, start=1):
    merged_list.append(gdf_chunk)
    # lightweight "progress bar" in the same line
    print(f"  Merging chunks: {idx}/{total_chunks}", end="\r")

# finalize the progress line
print()

bld_3857_raw = gpd.GeoDataFrame(pd.concat(merged_list, ignore_index=True), crs="EPSG:3857")

# Drop duplicate geometries
before_n = len(bld_3857_raw)
print("  Dropping duplicate geometries ...")
bld_3857_raw = bld_3857_raw.drop_duplicates(subset="geometry")
after_n = len(bld_3857_raw)
removed_n = before_n - after_n
print(f"  Kept {after_n} unique buildings (removed {removed_n} duplicates).")

print("  Clipping merged buildings to full AOI polygon ...")
aoi_full_3857 = aoi_gdf_ll.to_crs("EPSG:3857")
bld_3857 = gpd.overlay(bld_3857_raw, aoi_full_3857, how="intersection")

# Add centroid lon/lat (correct: centroid in projected CRS, then to EPSG:4326)
print("  Computing centroids ...")
centroids_3857 = bld_3857.geometry.centroid          # in EPSG:3857
centroids_ll = centroids_3857.to_crs("EPSG:4326")    # convert to lon/lat

bld_3857["longitude"] = centroids_ll.x
bld_3857["latitude"]  = centroids_ll.y

# Save buildings shapefile into Outputs\A
bldg_path = os.path.join(OUT_DIR, OUT_BLDG)
bld_3857.to_file(bldg_path)
print(f"\nExported AOI-clipped building shapefile → {bldg_path}")
print(f"Total buildings saved (inside AOI): {len(bld_3857)}")

# ======================= A5) Quick inspection (CRS + samples) ===============
datasets = {
    "OSM_Nodes": globals().get("nodes", None),
    "OSM_Edges": globals().get("edges", None),
    "Region_Boundary": globals().get("region_boundary_gdf", None),  # if available
}

# Prefer previously created buildings if present; otherwise use fresh result
if "buildings_r" in globals() and (globals()["buildings_r"] is not None) and (not globals()["buildings_r"].empty):
    buildings_gdf = globals()["buildings_r"]
elif "bldg" in globals() and (globals()["bldg"] is not None) and (not globals()["bldg"].empty):
    buildings_gdf = globals()["bldg"]
elif "bld_3857" in globals() and (globals()["bld_3857"] is not None) and (not globals()["bld_3857"].empty):
    buildings_gdf = globals()["bld_3857"]
else:
    buildings_gdf = globals().get("bld_3857", None)

datasets["Buildings"] = buildings_gdf

# --- CRS report ---
print(" Coordinate Reference Systems (CRS) of current GeoDataFrames:\n")
for name, gdf in datasets.items():
    if gdf is not None and hasattr(gdf, "crs"):
        print(f"{name:20s}: {gdf.crs}")
    else:
        print(f"{name:20s}: (not found or no CRS assigned)")

# --- Example rows ---
print("\n Example rows from loaded GeoDataFrames:\n")
for name, gdf in datasets.items():
    if gdf is not None and hasattr(gdf, "empty") and (not gdf.empty):
        print(f"── {name} ───────────────────────────────────────────────")
        try:
            display(gdf.drop(columns="geometry").head(5))
        except Exception:
            display(pd.DataFrame(gdf.head(5)))
    else:
        print(f"── {name}: (not found or empty)\n")

# =============== A6) Plotting  ===============
P = {
    # ===== General Drawing Toggles =====
    "DRAW_EDGES": True,                 # draw OSM road edges
    "DRAW_NODES": False,                # draw OSM network nodes
    "DRAW_BUILDINGS": True,             # draw building footprints
    "DRAW_REGION": True,                # draw AOI boundary polygon

    # ===== Manual Zoom (EPSG:4326) =====
    "lon_min": None,                    # min longitude for custom zoom (None = auto)
    "lon_max": None,                    # max longitude for custom zoom (None = auto)
    "lat_min": None,                    # min latitude for custom zoom (None = auto)
    "lat_max": None,                    # max latitude for custom zoom (None = auto)

    # ===== Figure & Labels =====
    "FIGSIZE": (11, 10),                # figure size (width, height) in inches
    "DPI": 300,                         # export resolution (dots per inch)
    "TITLE": "AOI, Edges and Buildings",  # plot title
    "TITLE_FONTSIZE": 13,               # title font size (pt)
    "XLABEL": "Easting (m)",            # x-axis label (projected meters)
    "YLABEL": "Northing (m)",           # y-axis label (projected meters)

    # ===== Region Boundary Style =====
    "REGION_COLOR": "black",            # region boundary color
    "REGION_LW": 1.2,                   # region boundary line width
    "REGION_ALPHA": 1.0,                # region boundary transparency (0..1)
    "REGION_LABEL": "Region Boundary",  # legend label for region

    # ===== Road / Edge Style =====
    "EDGE_COLOR": "#9e9e9e",            # edge color
    "EDGE_LW": 0.6,                     # edge line width
    "EDGE_ALPHA": 1.0,                  # edge transparency (0..1)
    "EDGE_LABEL": "Edges",              # legend label for edges

    # ===== Building Style =====
    "BLDG_FACE": "black",               # building fill color
    "BLDG_EDGE": "none",                # building outline color ('none' = no outline)
    "BLDG_LW": 0.2,                     # building outline width (if not 'none')
    "BLDG_ALPHA": 1.0,                  # building fill transparency (0..1)
    "BLDG_LABEL": "Buildings",          # legend label for buildings

    # ===== Node Style =====
    "NODE_COLOR": "#1f78b4",            # node color
    "NODE_SIZE": 1,                     # node marker size
    "NODE_ALPHA": 1.0,                  # node transparency (0..1)
    "NODE_MAX_PLOT": 150000,            # sample cap to avoid over-plotting
    "NODE_LABEL": "Nodes",              # legend label for nodes

    # ===== North Arrow =====
    "ADD_NORTH_ARROW": True,            # show north arrow
    "NA_X": 0.08,                       # arrow x-position (axes fraction 0–1)
    "NA_Y": 0.12,                       # arrow y-position (axes fraction 0–1)
    "NA_LEN": 0.08,                     # arrow length (axes fraction)
    "NA_LABEL": "N",                    # arrow text
    "NA_COLOR": "black",                # arrow color
    "NA_LW": 2,                         # arrow line width
    "NA_FONTSIZE": 14,                  # 'N' font size

    # ===== Scalebar =====
    "ADD_SCALEBAR": True,               # show scalebar
    "SB_DX": 1,                         # units-per-pixel (1 when plotting in meters/UTM)
    "SB_UNITS": "m",                    # units label for scalebar
    "SB_LOC": "lower right",            # scalebar location
    "SB_BOX_ALPHA": 0.8,                # scalebar box transparency (0..1)
    "SB_COLOR": "black",                # scalebar text/line color

    # ===== Legend =====
    "SHOW_LEGEND": True,                # show legend
    "LEGEND_LOC": "upper right",        # legend location
    "LEGEND_FRAME": True,               # draw legend frame box
    "LEGEND_FACE": "white",             # legend box facecolor
    "LEGEND_EDGE": "black",             # legend box edgecolor

    # ===== Output =====
    "OUT_NAME": "Region_UTM_Map_Datasets.png"  # output PNG filename
}

# --- minimal helpers ---
def add_north(ax, x, y, length, color, lw, fs, label):
    ax.annotate(label, xy=(x, y), xytext=(x, y - length),
                xycoords="axes fraction", textcoords="axes fraction",
                ha="center", va="center", fontsize=fs, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", lw=lw, color=color),
                clip_on=False, zorder=20)

def add_scalebar(ax, dx, units, loc, alpha, color):
    ax.add_artist(ScaleBar(dx, units, location=loc, box_alpha=alpha, color=color))

# --- choose a CRS source ---
cand = None
if P["DRAW_BUILDINGS"] and buildings_gdf is not None and not buildings_gdf.empty:
    cand = buildings_gdf
elif P["DRAW_EDGES"] and datasets["OSM_Edges"] is not None and not datasets["OSM_Edges"].empty:
    cand = datasets["OSM_Edges"]
elif P["DRAW_NODES"] and datasets["OSM_Nodes"] is not None and not datasets["OSM_Nodes"].empty:
    cand = datasets["OSM_Nodes"]
elif P["DRAW_REGION"] and datasets["Region_Boundary"] is not None and not datasets["Region_Boundary"].empty:
    cand = datasets["Region_Boundary"]

utm_crs = cand.to_crs(4326).estimate_utm_crs()

# --- reproject only what we draw ---
edges_utm  = datasets["OSM_Edges"].to_crs(utm_crs)       if P["DRAW_EDGES"]   and datasets["OSM_Edges"]   is not None else None
nodes_utm  = datasets["OSM_Nodes"].to_crs(utm_crs)       if P["DRAW_NODES"]   and datasets["OSM_Nodes"]   is not None else None
bldg_utm   = buildings_gdf.to_crs(utm_crs)               if P["DRAW_BUILDINGS"] and buildings_gdf is not None       else None
region_utm = datasets["Region_Boundary"].to_crs(utm_crs) if P["DRAW_REGION"]  and datasets["Region_Boundary"] is not None else None

# --- optional zoom (EPSG:4326 → UTM bounds) ---
use_zoom = all(v is not None for v in [P["lon_min"], P["lon_max"], P["lat_min"], P["lat_max"]])
if use_zoom:
    zbox = gpd.GeoDataFrame(
        geometry=[box(min(P["lon_min"], P["lon_max"]),
                      min(P["lat_min"], P["lat_max"]),
                      max(P["lon_min"], P["lon_max"]),
                      max(P["lat_min"], P["lat_max"]))],
        crs=4326
    ).to_crs(utm_crs).total_bounds

# --- plot ---
fig, ax = plt.subplots(figsize=P["FIGSIZE"])
legend_items = []

if region_utm is not None and P["DRAW_REGION"]:
    region_utm.boundary.plot(ax=ax, color=P["REGION_COLOR"], linewidth=P["REGION_LW"],
                             alpha=P["REGION_ALPHA"], zorder=3)
    if P["SHOW_LEGEND"]:
        legend_items.append(Line2D([0],[0], color=P["REGION_COLOR"], lw=P["REGION_LW"],
                                   label=P["REGION_LABEL"]))

if edges_utm is not None and P["DRAW_EDGES"]:
    edges_utm.plot(ax=ax, color=P["EDGE_COLOR"], linewidth=P["EDGE_LW"],
                   alpha=P["EDGE_ALPHA"], zorder=1)
    if P["SHOW_LEGEND"]:
        legend_items.append(Line2D([0],[0], color=P["EDGE_COLOR"], lw=P["EDGE_LW"]*5,
                                   label=P["EDGE_LABEL"]))

if bldg_utm is not None and P["DRAW_BUILDINGS"]:
    bldg_utm.plot(ax=ax, color=P["BLDG_FACE"], edgecolor=P["BLDG_EDGE"],
                  linewidth=P["BLDG_LW"], alpha=P["BLDG_ALPHA"], zorder=2)
    if P["SHOW_LEGEND"]:
        legend_items.append(Patch(facecolor=P["BLDG_FACE"], edgecolor=P["BLDG_EDGE"],
                                  label=P["BLDG_LABEL"]))

if nodes_utm is not None and P["DRAW_NODES"]:
    nshow = nodes_utm if len(nodes_utm) <= P["NODE_MAX_PLOT"] else nodes_utm.sample(P["NODE_MAX_PLOT"], random_state=0)
    nshow.plot(ax=ax, color=P["NODE_COLOR"], markersize=P["NODE_SIZE"],
               alpha=P["NODE_ALPHA"], zorder=4)
    if P["SHOW_LEGEND"]:
        legend_items.append(Line2D([0],[0], marker="o", linestyle="None",
                                   markersize=max(4, int((P["NODE_SIZE"]**0.5)/1.5)),
                                   markerfacecolor=P["NODE_COLOR"], markeredgecolor="none",
                                   label=P["NODE_LABEL"]))

if use_zoom:
    xmin, ymin, xmax, ymax = zbox
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)

ax.set_aspect("equal")
ax.set_xlabel(P["XLABEL"]); ax.set_ylabel(P["YLABEL"])
ax.set_title(P["TITLE"], fontsize=P["TITLE_FONTSIZE"])

if P["ADD_NORTH_ARROW"]:
    add_north(ax, x=P["NA_X"], y=P["NA_Y"], length=P["NA_LEN"],
              color=P["NA_COLOR"], lw=P["NA_LW"], fs=P["NA_FONTSIZE"], label=P["NA_LABEL"])

if P["ADD_SCALEBAR"]:
    add_scalebar(ax, dx=P["SB_DX"], units=P["SB_UNITS"], loc=P["SB_LOC"],
                 alpha=P["SB_BOX_ALPHA"], color=P["SB_COLOR"])

if P["SHOW_LEGEND"] and legend_items:
    ax.legend(handles=legend_items, loc=P["LEGEND_LOC"],
              frameon=P["LEGEND_FRAME"], facecolor=P["LEGEND_FACE"],
              edgecolor=P["LEGEND_EDGE"])

plt.show()

# --- save ---
out_img_dir = Path(r"./Outputs/Images"); out_img_dir.mkdir(parents=True, exist_ok=True)
out_png = out_img_dir / P["OUT_NAME"]
fig.savefig(out_png, dpi=P["DPI"], bbox_inches="tight")
print(f" Figure saved to: {out_png}")

# =============================== A7) Summary ===============================

def _count(name):
    obj = globals().get(name, None)
    return 0 if (obj is None or getattr(obj, "empty", False)) else len(obj)

print("\n===== Section A Summary =====")
print(f"Buildings (downloaded RAW, pre-clip): {_count('bld_3857_raw')}")
print(f"Buildings (clipped to AOI):           {_count('bld_3857')}")
print(f"OSM Edges:                              {_count('edges')}")
print(f"OSM Nodes:                              {_count('nodes')}")
print("==========================================\n")
