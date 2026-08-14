# All-Weather Satellite Vision

## Project
All-Weather Satellite Vision

## Objective
A future three-stage satellite image restoration and multispectral-to-RGB pipeline.

## Current Status
Phase 1 — Sentinel-2 Acquisition (Completed)

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

### Core Acquisition Principles
* **Primary Data Source**: Sentinel-2 MSI Level-2A Surface Reflectance.
* **Initial Bands**: B02 (Blue), B03 (Green), B04 (Red), B08 (NIR) - native 10m spatial resolution.
* **Official Registry**: Copernicus Data Space Ecosystem (CDSE) is the official provider via OData and STAC APIs.
* **Cloud Cover Filtering**: Product-level cloud cover is used only as a coarse filtering criterion. Pixel-level masks are deferred to later preprocessing.
* **Data Immutability**: All files in `data/raw/` are treated as immutable inputs. Downstream preprocessing reads raw files and writes to processed directories.
* **No Preprocessing**: No band resampling, alignment, normalization, or tiling is performed during this phase.

### Setup & Credentials
To connect to the CDSE API, copy `.env.example` to `.env` (note: `.env` is ignored by git) and add your Copernicus account details:
```bash
CDSE_USERNAME=your_username
CDSE_PASSWORD=your_password
WANDB_API_KEY=your_wandb_key
```

### CLI Command Reference

#### 1. Search Scenes
Query the STAC catalog for available Sentinel-2 Level-2A products over a specified spatio-temporal boundary:
```bash
PYTHONPATH=. venv/bin/python -m src.data.scene_search \
  --start-date 2023-09-01 \
  --end-date 2023-09-10 \
  --bbox 12.4 41.8 12.5 41.9 \
  --max-cloud-cover 10 \
  --max-results 3
```

#### 2. Run Infrastructure Smoke Test
Verify Phase 0 device initialization, seed consistency, optimization, and state dict checkpointing:
```bash
PYTHONPATH=. venv/bin/python src/smoke_test.py
```

#### 3. Run Acquisition Smoke Test
Perform a catalog search, download, validation, and metadata registry test:
```bash
PYTHONPATH=. venv/bin/python src/data/smoke_test_acquisition.py
```

---

## Future Phases
* Phase 2 — Preprocessing & Band Alignment
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
