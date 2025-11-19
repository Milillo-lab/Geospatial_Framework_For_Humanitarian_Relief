# Urban Damage and Infrastructure Analysis Pipeline
---

<p align="center">
  <strong>
    Ashutosh Lamsal<sup>1,2</sup>, 
    Demirkan Orhun Oral<sup>1,2</sup>, 
    Pietro Milillo<sup>1,2,3</sup>
  </strong>
</p>

<p align="center">
  <sup>1</sup> Department of Civil and Environmental Engineering, University of Houston,<br>
  Houston, TX, USA<br>
  <sup>2</sup> National Center for Airborne Laser Mapping (NCALM), University of Houston,<br>
  Houston, TX, USA<br>
  <sup>3</sup> German Aerospace Centre (DLR), Microwaves and Radar Institute,<br>
  Munich, Germany
</p>

<p align="center"><em>CIVE 6381 - Applied Geospatial Computations (Fall/2025)</em></p>

---



## 1. Aim of the Project

This project provides a comprehensive, automated framework for assessing urban damage, population exposure, and accessibility following an earthquake or similar disaster.

It integrates global open datasets with analytical tools to measure the extent and impact of physical damage, identify critical road networks, and estimate humanitarian relief requirements.
The project automatically:

- Detects and classifies damaged vs. undamaged buildings and roads
- Estimates affected population within damaged regions
- Calculates relief supply requirements (water, food, shelter, medical kits) based on international humanitarian guidelines
- Identifies critical road segments that sustain network connectivity
- Produces visual and quantitative summaries for disaster response and recovery planning

The overall goal is to support rapid, data-driven decision-making in disaster-struck regions, helping authorities, researchers, and humanitarian teams understand:

- Where the greatest damage has occurred
- How many people are likely affected
- Which routes remain functional or most vital for aid delivery

The entire codebase is modular, meaning each section (A–I) can be executed independently.
Once the full pipeline has been run at least once, users can re-run only specific sections (for example, just the population estimation or shortest-path analysis) without needing to repeat the entire workflow.
This makes it easy to update parts of the analysis, compare scenarios, or test different parameters without starting from scratch.

The workflow is built entirely in Python, using libraries such as OSMnx, GeoPandas, Shapely, and Matplotlib, but is designed so that non-programmers can operate it easily by following this README and adjusting only a few basic parameters.


---

## 2. Code Structure (Table of Contents)

A) Base Layers – AOI, OSM Buildings & Roads  
B) Building Damage Evaluation (DPM Overlay)  
C) Population and Volume Estimation  
D) Relief and Supply Estimation  
E) Edge Buffering and Damage Classification  
F) Tunnel/Bridge Overrides and Node Damage  
G) Damaged Node Calculation  
H) Shortest Path Analysis  
I) Betweenness Centrality Analysis  

---

## 3. Required and Optional Data Inputs

The workflow automatically downloads or generates most layers internally.  
Only two dataset inputs are strictly required (Read "How to Download" sections for help), while others are optional aids for defining or refining the study area.
Put the related datasets that you want to include in ./Data folder.

### **Required Data**

**DPM Raster (Damage Proxy Map)**  
Grayscale or single-band raster used to detect damaged areas.  
→ Must cover the study area. Defined in section B1.

**Population Raster**  
Gridded population dataset (e.g., WorldPop, GHSL, or national census raster).  
→ Used to allocate estimated population to buildings (section C).

### **Optional Data**

**AOI Boundary (Polygon)**  
Optional shapefile/GeoJSON defining the boundary of the study area.  
→ If omitted, the study area is derived using the hierarchy below.

**Longitude / Latitude Bounds**  
Optional manual coordinate limits for a custom study area.

**Place Mode (Recommended for quick start)**  
If mode == "place" and a place_name is provided (e.g., Kahramanmaras), the study area is automatically set to the OSM administrative polygon downloaded in the pipeline.  
→ This overrides AOI and lon/lat bounds unless you explicitly change the mode.

**Outputs Folder Path**  
Root directory where all results (A–I folders + Images) are saved.  
Default:  
`~/../your_path/Outputs`

**Study Area Selection Order (from highest to fallback)**  
1. If mode == "place" and place_name is set → OSM place polygon  
2. Else if AOI polygon provided → AOI polygon  
3. Else if lon/lat bounds provided → coordinate bounds  
4. Else → DPM raster extent

### Config.yaml -> Configuration file

Designed to ease the use of user-led desicions throughout the code, divided for each section A to I.
Follow the explanations while changing the config file to avoid any problems.

---

## 4. Deliverables

Once the entire pipeline is executed, the workflow produces a full set of geospatial and analytical outputs organized into labeled sections (A–I) under the main output directory.  
Each section represents a distinct analytical step and generates its own intermediate or final outputs.  

Below is an overview of what the user can expect after running the code:

### **A. Setup & Initialization**
- Defines user options, paths, and directories.  
- Initializes the environment and output folder structure (A–I + Images).

### **B. DPM and Population Processing**
- Loads and prepares the Damage Proxy Map (DPM) and population raster.  
- Applies masking, resampling, and normalization to align both datasets for analysis.

### **C. Building Extraction and AOI Handling**
- Downloads or extracts building footprints within the area of interest.  
- Clips them to the AOI or to user-defined bounds.  
- Counts buildings and prepares per-pixel population distribution.

### **D. Damage and Humanitarian Analysis**
- Classifies damaged vs. undamaged buildings using DPM thresholds.  
- Estimates affected and unaffected populations.  
- Calculates humanitarian supply needs (water, food, shelter, medical kits).  
- Produces summary tables and bar-chart visualizations saved to `/Images`.

### **E. Road Network Evaluation**
- Downloads the OSM road network for the area.  
- Processes attributes such as tunnels and bridges for improved routing accuracy.

### **F. Edge & Node Evaluation**
- Evaluates road segments and nodes for connectivity and possible disruptions.  
- Stores “evaluated” network layers for further routing and graph analyses.

### **G. Buffer and Impact Zones**
- Generates buffer zones around damaged buildings proportional to height or default distance.  
- Identifies buildings and roads within buffer zones (impact radius).

### **H. Shortest Path Analysis**
- Finds shortest paths between selected origin and destination points.  
- Exports route maps and CSVs with cumulative distances.  
- Produces both full-extent and zoomed maps in `/Images/Shortest_Path/`.

### **I. Betweenness Centrality (Network Importance)**
- Computes betweenness centrality on the undamaged road network.  
- Outputs:
  - Colored BC map (full + zoomed view) → saved in `/Images/`.  
  - BC table (Node ID, Lon, Lat, BC value) → saved in `/I/`.

---

## 5. Known Limitations & Best Practices

This workflow provides a complete DPM–population–infrastructure analysis pipeline but has several practical limits and assumptions.

### **A. General Notes**
- Damage detection is binary: buildings and areas are classified as damaged or undamaged based on a single DPM threshold.  
  → The workflow does not estimate severity levels (e.g., moderate, severe).  
- Results depend on DPM quality and coverage — noisy rasters may misclassify damage.  
- Population data are treated as static pre-event distributions (no migration or temporal updates).  
- Large AOIs may increase processing time and memory usage.

### **B. Data & Projection**
- All layers are automatically reprojected to a local UTM CRS for accurate distance and area calculations.  
- DPM and population rasters must contain valid CRS metadata.  
- It’s best to pre-clip large rasters to the AOI before analysis.

### **C. Humanitarian Estimates**
Based on global humanitarian guidelines (Sphere Project, WHO).

Default assumptions:
- 20 L water / person / day  
- 2,100 kcal / person / day  
- 4 m² shelter / person  
- IEHK = 10,000 people / 90 days  
- A 10 % buffer is added automatically to food and water.

### **D. Network & Routing**
- Only undamaged edges are used for pathfinding and BC (betweenness centrality).  
- Network topology comes from OSM; local inaccuracies may exist.  
- BC calculations can be heavy on large graphs — use `USE_APPROX_BC=True` for faster runs.

---

## 6. How to Run the Workflow

This project is designed to run from start to finish in a **Jupyter Notebook**, with each section labeled from **A to I**.

### **A. Before You Begin**
- Install requirements with the "0. Necessities" code.
- Confirm that you have the required data files in the correct folders.

### **B. Running the Code**
- Run the notebook cell by cell from top to bottom.  
- Each section (A, B, C … I) runs in order.  
- When you see `# === A)`, `# === B)`, etc., that marks the start of a new step.    

### **C. What You’ll See**  
- Map results and charts will appear directly inside the notebook.  
- When finished, all outputs will be stored automatically in the outputs folder.

### **D. After Running**
When all sections are executed successfully:
- Maps and charts will appear under `/Images/`  
- Data summaries and CSVs will appear under their respective section folders (A–I)  

---

## 7. User Options and Variable Selection

The user is invited to change and choose necessary variables from the configuration file.

---

## 8. Plotting

All plotting codes have explanations for each option / variable in the sections provided above them.
