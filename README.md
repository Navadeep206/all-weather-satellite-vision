# All-Weather Satellite Vision

## Project
All-Weather Satellite Vision

## Objective
A future three-stage satellite image restoration and multispectral-to-RGB pipeline.

## Current Status
Phase 4 — Dataset Construction (Completed)

## Planned Architecture
Sentinel-2 multispectral input
→ Atmospheric Restoration (Stage 1)
→ Mask-Guided Reconstruction (Stage 2)
→ Multispectral-to-RGB Translation (Stage 3)

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

---

## Phase 3 — Synthetic Degradation Engine
Establishes a scientifically controlled synthetic degradation engine that simulates atmospheric haze scattering and spatial occlusion regions while preserving the original clean image as ground truth.

---

## Phase 4 — Dataset Construction
Constructs three separate, reproducible datasets corresponding to the neural-network stages:
* **Stage 1 (Atmospheric Restoration)**: Inputs degraded hazy multispectral patch ($4 \times 256 \times 256$) and targets clean multispectral patch ($4 \times 256 \times 256$).
* **Stage 2 (Mask-Guided Reconstruction)**: Inputs degraded (hazy/occluded) patch ($4 \times 256 \times 256$) and spatial mask ($1 \times 256 \times 256$), targeting clean ground-truth patch ($4 \times 256 \times 256$).
* **Stage 3 (Multispectral-to-RGB Translation)**: Inputs clean multispectral patch ($4 \times 256 \times 256$) and targets RGB composite patch ($3 \times 256 \times 256$).

### Key Dataset Construction Specifications
* **Scene-level Geographic Splitting**: Scenes are divided into train (70%), validation (15%), and test (15%) splits deterministically based on seed (default: 42).
* **Leakage Prevention**: Synthetic variants are generated **AFTER** scene assignment to guarantee that no scene or any of its degraded variants span across different splits.
* **Manifest-based Architecture**: CSV manifests (`data/dataset/manifests/`) are created containing metadata and paths relative to the project root. This avoids massive file duplication.
* **Patch Index System**: Spatial coordinates of eligible 256x256 tiles with minimal nodata values are indexed within the CSV. PyTorch dataset classes resolve and read these windows on-the-fly.
* **Geospatial Integrity**: Validates that transform coordinates, CRS, shapes, and band counts match across inputs and targets.
* **PyTorch Dataset Classes**: Implements `SatelliteStage1Dataset`, `SatelliteStage2Dataset`, and `SatelliteStage3Dataset` using Rasterio windowed reads.
* **Storage Policy**: Generated splits (`data/dataset/`) and raw/processed/degraded images are excluded from Git tracing (listed in `.gitignore`).

### Output Dataset Structure
```text
data/dataset/
├── splits/
│   ├── train.txt
│   ├── val.txt
│   └── test.txt
├── manifests/
│   ├── stage1_train.csv
│   ├── stage1_val.csv
│   ├── stage1_test.csv
│   ├── stage2_train.csv
│   ├── stage2_val.csv
│   ├── stage2_test.csv
│   ├── stage3_train.csv
│   ├── stage3_val.csv
│   └── stage3_test.csv
└── metadata/
    ├── split_metadata.json
    ├── dataset_metadata.json
    └── generation_summary.json
```

### CLI Command Reference

#### 1. Build Datasets
```bash
PYTHONPATH=. venv/bin/python -m src.data.build_dataset --validate
```

#### 2. Run Dry-run Split Simulation
```bash
PYTHONPATH=. venv/bin/python -m src.data.build_dataset --dry-run
```

---

## Future Phases
* Phase 5 — Multispectral-to-RGB Baseline
* Phase 6 — Atmospheric Restoration
* Phase 7 — Mask-Guided Reconstruction
* Phase 8 — Pipeline Integration
* Phase 9 — End-to-End Fine-Tuning
* Phase 10 — Evaluation & Ablation
* Phase 11 — Robustness & Real-World Validation
* Phase 12 — Deployment
