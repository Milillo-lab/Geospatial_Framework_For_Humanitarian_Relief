# 1. Damage Proxy Map (DPM) Data

This pipeline requires a Damage Proxy Map (DPM) raster to identify damaged areas after a disaster. A DPM is a single-band GeoTIFF produced from remote sensing change analysis (commonly SAR), where higher pixel values represent greater likelihood of damage.

## Requirements
- File format: **GeoTIFF (.tif)**
- **Single band** (grayscale values) in between 0 and 1
- Higher values = more likely damaged
- Must cover the analysis area (AOI)
- Must include valid spatial reference (CRS)

## File Location
Place the file inside your project data directory (./Data)
Then change the file name in configuration file (config.yaml)

## Example
The DPM used in this example is from; https://zenodo.org/records/15369579
For use, download the DPM and paste it into your data directory.


# 2. High Resolution Population Density (HRSL) — Download & Use Guide

This guide explains **how to obtain and use** the High Resolution Population Density maps (HRSL) provided by **Data for Good at Meta** (formerly Facebook). You’ll find step-by-step instructions to download rasters, and load them in QGIS/Python.

**What is HRSL?** Country-level population density rasters derived from high-resolution imagery and census data, typically ~30 m resolution (varies by country). Values represent estimated **people per pixel**.

---

## Download via HDX (Humanitarian Data Exchange)

**Best for**: quick, per-country GeoTIFFs/ZIPs.

1. Go to the **High Resolution Population Density Maps** page on HDX. [Link](https://data.humdata.org/organization/meta?q=population%20density&sort=if(gt(last_modified%2Creview_date)%2Clast_modified%2Creview_date )
2. Use the search box to find your **country** (e.g., *Nepal*, *Turkey*, *Kenya*).
3. Open the country dataset page and review the **Files** section (usually GeoTIFF or a ZIP containing `.tif`).
4. Click **Download**. (You may need to sign in to HDX to accept terms.)
5. Unzip if needed; you should have a `.tif` raster.

**Notes**

* File sizes can be large (hundreds of MB).
* Some countries provide multiple layers (e.g., density, population counts, masks); grab the one you need.

---

## After you download the population raster from HDX, you can now view them either using QGIS or Python

    ## 1) Load in QGIS

    **Best for**: quick viewing and basic GIS operations.

    1. Open **QGIS** → **Layer** → **Add Layer** → **Add Raster Layer…**
    2. Browse to your downloaded HRSL `.tif` → **Add**.
    3. Right-click the layer → **Properties** → **Symbology** → set **Singleband pseudocolor** and choose a color ramp.
    4. (Optional) Reproject to your project CRS via **Raster** → **Projections** → **Warp (Reproject)**.

    ---

    ## 2) Python (GeoPandas/Rasterio) — Quick Start

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

## Interpreting Values

* **Pixel value** ≈ **people per pixel** (not per km²).
* For per-square-kilometer density, convert using pixel area (depends on CRS & resolution).
* Some countries have ~30 m pixels; others may be coarser (e.g., ~100 m). Always verify the **metadata**.

---

## Common Pitfalls & Fixes

* **Huge files**: Clip to an AOI in GEE or QGIS before downloading/exporting.
* **CRS mismatch**: Reproject layers to a common CRS (e.g., EPSG:4326 or your local UTM).
* **Nodata handling**: Check `nodata` in the raster and treat those values as missing.
* **Attribution**: Follow the licensing and attribution text shown on the download page for your country/version.

---

## Suggested Citation / Attribution

If you use the **High-Resolution Population Maps** or related datasets in your research, please cite the following work:

> **Tiecke, Tobias G., Liu, Xianming, Zhang, Amy, Gros, Andreas, Li, Nan, Yetman, Gregory, Kilic, Talip, Murray, Siobhan, Blankespoor, Brian, Prydz, Espen B., & Dang, Hai-Anh H. (2017).**  
> *Mapping the world population one building at a time.*  
> arXiv preprint [arXiv:1712.05839](https://arxiv.org/abs/1712.05839)

**BibTeX:**
```bibtex
@misc{tiecke2017mappingworldpopulationbuilding,
  title={Mapping the world population one building at a time},
  author={Tobias G. Tiecke and Xianming Liu and Amy Zhang and Andreas Gros and Nan Li and Gregory Yetman and Talip Kilic and Siobhan Murray and Brian Blankespoor and Espen B. Prydz and Hai-Anh H. Dang},
  year={2017},
  eprint={1712.05839},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/1712.05839}
}

```
# 3. Global Building Atlas Dataset (Manual Download)

The code in this repository normally downloads building polygons automatically through the **Global Building Atlas WFS service**.  
However, if the server is unavailable or returns errors (e.g., 502 Bad Gateway), you can manually download the building data and use it instead.

---

### Requirements for Manual Input
To ensure correct processing in the pipeline, the manually downloaded dataset must satisfy:

- **File format:** ESRI Shapefile (`.shp`) or GeoPackage (`.gpkg`)  
- Should contain **building footprint polygons** (LoD1)  
- Must include **valid spatial reference (CRS)**  
- The layer must **cover your Area of Interest (AOI)**  
- Optional but recommended fields:
  - `height` column (for buffer/volume calculations)
  - Unique building IDs

---
#### Full Global Download (Recommended for manual use)
**Download from mediaTUM:**  
https://mediatum.ub.tum.de/1782307

This download contains the full worldwide LoD1 building dataset as vector files (shapefiles and geopackages).  
You can extract only the region you need using **QGIS** or **GeoPandas**.

After downloading the building data:

Put it inside ./Outputs/A for the pipeline to read it from there in subsequent runs.


```
## Suggested Citation / Attribution

If you use the **Global Building Dataset** or related datasets in your research, please cite the following work:

> **Zhu, Xiao Xiang and  Chen, Sining and  Zhang, Fahong and  Shi, Yilei and  Wang, Yuanyuan**  
> *Complete Dataset of Building Polygons*  
> arXiv preprint [arXiv:2506.04106]((https://arxiv.org/abs/2506.04106))

**BibTeX:**
```bibtex
@misc{zhu2025globalbuildingatlasopenglobalcomplete,
      title={GlobalBuildingAtlas: An Open Global and Complete Dataset of Building Polygons, Heights and LoD1 3D Models}, 
      author={Xiao Xiang Zhu and Sining Chen and Fahong Zhang and Yilei Shi and Yuanyuan Wang},
      year={2025},
      eprint={2506.04106},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.04106}, 
}

```

---

## FAQ

**Q: I can’t find my country on HDX.**
A: Try alternative spellings or check for regional packs.

**Q: The values look too small/large.**
A: Confirm whether the layer is **population per pixel** vs **density per km²**, and check if you need to resample or aggregate.







