## Damage Proxy Map (DPM) Input

This pipeline requires a Damage Proxy Map (DPM) raster to identify damaged areas after a disaster. A DPM is a single-band GeoTIFF produced from remote sensing change analysis (commonly SAR), where higher pixel values represent greater likelihood of damage.

### Requirements
- File format: **GeoTIFF (.tif)**
- **Single band** (grayscale values)
- Higher values = more likely damaged
- Must cover the analysis area (AOI)
- Must include valid spatial reference (CRS)

### File Location
Place the file inside your project inputs directory (./Inputs)
Then change the file name in configuration file (config.yaml)

### Example
The DPM used in this example is from; https://zenodo.org/records/15369579
For use, download the DPM and paste it into your inputs directory.


## Population.tif File Input