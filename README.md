# All-Weather Satellite Vision

## Project
All-Weather Satellite Vision

## Objective
A future three-stage satellite image restoration and multispectral-to-RGB pipeline.

## Current Status
Phase 2 — Preprocessing & Band Alignment (Completed)

## Planned Architecture
Sentinel-2 multispectral input
→ Atmospheric Restoration
→ Mask-Guided Reconstruction
→ Multispectral-to-RGB Translation

## Tech Stack
* Python (>=3.11)
* PyTorch
* NumPy
* Rasterio
* PyYAML
* Pytest

---

## Phase 0 — Project Foundation
Established basic project layout, deterministic seeding, YAML configuration parser, logging utility, model-agnostic checkpoint serializer, and automated tests.

---

## Phase 1 — Sentinel-2 Acquisition
Implements a reliable, reproducible, and scalable search, catalog, download, and validation pipeline.

---

## Phase 2 — Preprocessing & Band Alignment
Converts raw Sentinel-2 Level-2A products into clean, spatially aligned, geospatially correct float32 surface reflectance GeoTIFF representations.

### Preprocessing Specifications
* **Raw vs Processed**: Raw products under `data/raw/` remain strictly untouched and immutable. All output files are generated under `data/processed/sentinel2/<scene_id>/`.
* **Selected Bands**: B02 (Blue), B03 (Green), B04 (Red), B08 (NIR).
* **Target Grid & Resolution**: Derived from actual geospatial metadata of B02 at native 10 m resolution. Target grid shape, projection (CRS), and geotransform are strictly matched.
* **Geospatial Resampling**: Uses Rasterio `reproject` (GDAL-compatible warp engine). bilinear resampling is applied by default (configurable).
* **Reflectance Scaling**: Reads baseline metadata XML (`MTD_MSIL2A.xml`) to dynamically extract the quantification value and radiometric offset (`BOA_ADD_OFFSET`), computing bottom-of-atmosphere (BOA) reflectance:
  $$\text{Reflectance} = \frac{\text{DN} + \text{BOA\_ADD\_OFFSET}}{\text{QUANTIFICATION\_VALUE}}$$
  Values are saved in `float32`.
* **Invalid-pixel & Nodata Handling**: Raw pixels equal to nodata (DN = 0) are converted to `np.nan` in the output float32 files. Invalid or out-of-bound pixels (e.g. DN < 0 or DN >= 65535) are likewise masked.
* **RGB Channel Mapping**: `rgb.tif` is saved with channels `[R, G, B]` corresponding directly to bands `[B04, B03, B02]`.
* **Quality Validation**: Automatically checks structural shape, band count, coordinate systems (CRS), bounding boxes, transform arrays, and numeric counts (NaN, Inf, Nodata counts). If checks fail, processing is aborted.

*Note: Phase 2 does not perform cloud detection, cloud removal, synthetic degradation, or machine-learning normalization.*

### Output Directory Structure
For each processed scene under `data/processed/sentinel2/<scene_id>/`:
* `multispectral.tif`: 4-band float32 GeoTIFF containing B02, B03, B04, B08.
* `rgb.tif`: 3-band float32 GeoTIFF containing B04, B03, B02.
* `metadata.json`: Machine-readable processing metadata (width, height, bounds, CRS).
* `quality.json`: Validity stats (nan count, valid percentages, status).

### CLI Command Reference

#### 1. Preprocess Scenes
Process a single scene by its ID:
```bash
PYTHONPATH=. venv/bin/python -m src.data.preprocessing --scene-id S2_L2A_20230907_T32TNS_abcdefgh
```

Or process all validated scenes:
```bash
PYTHONPATH=. venv/bin/python -m src.data.preprocessing --all
```

#### 2. Run Preprocessing Dry Run
Preview what will happen without writing any output files:
```bash
PYTHONPATH=. venv/bin/python -m src.data.preprocessing --scene-id S2_L2A_20230907_T32TNS_abcdefgh --dry-run
```

---

## Future Phases
* Phase 3 — Synthetic Degradation
* Phase 4 — Dataset Construction
* Phase 5 — Multispectral-to-RGB Baseline
* Phase 6 — Atmospheric Restoration
* Phase 7 — Mask-Guided Reconstruction
* Phase 8 — Pipeline Integration
* Phase 9 — End-to-End Fine-Tuning
* Phase 10 — Evaluation & Ablation
* Phase 11 — Robustness & Real-World Validation
* Phase 12 — Deployment
