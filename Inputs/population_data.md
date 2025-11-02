# High Resolution Population Density (HRSL) — Download & Use Guide

This guide explains **how to obtain and use** the High Resolution Population Density maps (HRSL) provided by **Data for Good at Meta** (formerly Facebook). You’ll find step-by-step instructions to download rasters, export via Google Earth Engine (GEE), and load them in QGIS/Python.

> **What is HRSL?** Country-level population density rasters derived from high-resolution imagery and census data, typically ~30 m resolution (varies by country). Values represent estimated **people per pixel**.

---

## 1) Download via HDX (Humanitarian Data Exchange)

**Best for**: quick, per-country GeoTIFFs/ZIPs.

1. Go to the **High Resolution Population Density Maps** page on HDX.
2. Use the search box to find your **country** (e.g., *Nepal*, *Turkey*, *Kenya*).
3. Open the country dataset page and review the **Files** section (usually GeoTIFF or a ZIP containing `.tif`).
4. Click **Download**. (You may need to sign in to HDX to accept terms.)
5. Unzip if needed; you should have a `.tif` raster.

**Notes**

* File sizes can be large (hundreds of MB).
* Some countries provide multiple layers (e.g., density, population counts, masks); grab the one you need.

---

## 2) Export via Google Earth Engine (GEE)

**Best for**: programmatic export, clipping to AOI, reprojection.

> You can use a community-hosted HRSL collection in GEE (country mosaics), then clip/export to Google Drive or Cloud Storage.

### Example (JavaScript, GEE Code Editor)

```js
// ---------------------------
// HRSL via community asset
// ---------------------------
var hrsl = ee.ImageCollection('projects/sat-io/open-datasets/hrsl'); 
// TIP: Inspect the collection; it contains per-country images.

// 1) Pick a country by ISO code (example: Nepal 'NPL')
var country = ee.FeatureCollection('FAO/GAUL/2015/level0')
  .filter(ee.Filter.eq('ADM0_NAME', 'Nepal')); // change country name

// 2) Filter HRSL for the same country; assets are named by ISO code
// If your country isn't matched by name, browse the collection in the Inspector.
var hrslNPL = hrsl.filter(ee.Filter.eq('country', 'NPL')).mosaic();

// 3) Clip to country (or your AOI)
var clipped = hrslNPL.clip(country);

// 4) Visualize (stretch values for display)
Map.centerObject(country, 7);
Map.addLayer(clipped, {min:0, max:500, palette:['white','yellow','red']}, 'HRSL');

// 5) Export GeoTIFF to Google Drive
Export.image.toDrive({
  image: clipped,
  description: 'HRSL_Nepal',
  fileNamePrefix: 'hrsl_nepal',
  region: country.geometry(),
  scale: 30,         // adjust if needed; some HRSL are ~30m, some coarser
  maxPixels: 1e13
});
```

**Tips**

* If your country isn’t found by name, filter the collection by the `system:index` or `country` property visible in the Inspector.
* For a smaller file, replace `region: country.geometry()` with your custom AOI polygon and/or set `scale` appropriately.

---

## 3) Load in QGIS

**Best for**: quick viewing and basic GIS operations.

1. Open **QGIS** → **Layer** → **Add Layer** → **Add Raster Layer…**
2. Browse to your downloaded HRSL `.tif` → **Add**.
3. Right-click the layer → **Properties** → **Symbology** → set **Singleband pseudocolor** and choose a color ramp.
4. (Optional) Reproject to your project CRS via **Raster** → **Projections** → **Warp (Reproject)**.

---

## 4) Python (GeoPandas/Rasterio) — Quick Start

```python
import rasterio
from rasterio.mask import mask
import geopandas as gpd
import numpy as np

# Paths
hrsl_tif = "path/to/hrsl_country.tif"
aoi = gpd.read_file("path/to/aoi.shp").to_crs("EPSG:4326")

# Clip HRSL to AOI
with rasterio.open(hrsl_tif) as src:
    aoi_geom = [aoi.geometry.unary_union.__geo_interface__]
    out_img, out_transform = mask(src, aoi_geom, crop=True)
    out = out_img[0].astype(np.float32)
    nodata = src.nodata

# Basic stats (exclude nodata)
if nodata is not None:
    out = np.where(out == nodata, np.nan, out)

print("HRSL stats (AOI):")
print("  min:", np.nanmin(out))
print("  max:", np.nanmax(out))
print("  sum (people):", np.nansum(out))
```

---

## 5) Interpreting Values

* **Pixel value** ≈ **people per pixel** (not per km²).
* For per-square-kilometer density, convert using pixel area (depends on CRS & resolution).
* Some countries have ~30 m pixels; others may be coarser (e.g., ~100 m). Always verify the **metadata**.

---

## 6) Common Pitfalls & Fixes

* **Huge files**: Clip to an AOI in GEE or QGIS before downloading/exporting.
* **CRS mismatch**: Reproject layers to a common CRS (e.g., EPSG:4326 or your local UTM).
* **Nodata handling**: Check `nodata` in the raster and treat those values as missing.
* **Attribution**: Follow the licensing and attribution text shown on the download page for your country/version.

---

## 7) Suggested Citation / Attribution

When you publish results or share maps, include attribution similar to:

> *“High Resolution Population Density Maps (HRSL) from Data for Good at Meta (Facebook) and partners. Accessed [Month Year].”*
> *(Add the source page and any country-specific license text from the download page.)*

---

## 8) Folder Structure (Example)

```
data/
  hrsl/
    hrsl_nepal.tif
    hrsl_turkey.tif
  aoi/
    aoi_nepal.shp
outputs/
  hrsl_stats.csv
  hrsl_clipped_nepal.tif
README.md
```

---

## 9) FAQ

**Q: I can’t find my country on HDX.**
A: Try alternative spellings or check for regional packs. If not available, consider the GEE approach.

**Q: The values look too small/large.**
A: Confirm whether the layer is **population per pixel** vs **density per km²**, and check if you need to resample or aggregate.

**Q: The export fails in GEE due to maxPixels.**
A: Increase `maxPixels`, export a smaller AOI, or reduce the resolution (`scale`).
