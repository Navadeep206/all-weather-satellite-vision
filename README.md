# All-Weather Satellite Vision

## Project
All-Weather Satellite Vision

## Objective
A future three-stage satellite image restoration and multispectral-to-RGB pipeline.

## Current Status
Phase 3 — Synthetic Degradation Engine (Completed)

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

---

## Phase 3 — Synthetic Degradation Engine
Establishes a scientifically controlled synthetic degradation engine that simulates atmospheric haze scattering and spatial occlusion regions while preserving the original clean image as ground truth.

### Key Degradation Specifications
* **Atmospheric Scattering Model**: Operates per spectral channel using:
  $$I(x) = J(x)t(x) + A(1 - t(x))$$
  where $J(x)$ is clean reflectance, $t(x)$ is the transmission map, and $A$ is the atmospheric light per channel.
* **Transmission Map**: Generates spatially varying, smooth transmission maps using Gaussian-smoothed random noise maps scaled deterministically.
* **Haze Severity Levels**: Controls scattering coefficient $\beta$ range:
  * `low`: $\beta \in [0.1, 0.4]$
  * `medium`: $\beta \in [0.4, 1.0]$
  * `high`: $\beta \in [1.0, 2.2]$
  * `extreme`: $\beta \in [2.2, 3.5]$
* **Spatial Occlusion Generation**:
  * `1` = valid pixel, `0` = occluded pixel.
  * **Cloud-like Masks**: Procedural low-frequency Gaussian noise thresholded to simulate irregular cloud boundaries.
  * **Irregular Masks**: Fragmented connected missing regions using higher-frequency noise.
  * **Rectangular Masks**: Simple sensor-style missing blocks.
* **Occlusion Coverage Levels**:
  * `low`: 5% to 15%
  * `medium`: 15% to 35%
  * `high`: 35% to 60%
  * `extreme`: 60% to 85%
* **Combined Degradation**: Sequences operations in a strict order:
  1. Apply atmospheric degradation.
  2. Apply spatial occlusion.
* **Ground Truth Preservation**: The original clean image remains strictly unchanged as ground truth (`clean.tif`).
* **Value Safety**: Implements post-scattering checks clipping float32 outputs to $[0.0, 1.0]$ and logging pre/post-clipping metrics.
* **Reproducibility**: Local, seeded NumPy Generators guarantee deterministic sample recreation.

*Scientific Caution: Synthetic masks are procedural approximations of spatial occlusion and are not a substitute for physically observed cloud masks. Synthetic haze is a controlled atmospheric-degradation model and may not reproduce every real-world atmospheric condition.*

### Output Directory Structure
For each generated sample under `data/degraded/combined/<sample_id>/`:
* `clean.tif`: 4-band float32 GeoTIFF representing original clean reflectance.
* `degraded.tif`: 4-band float32 GeoTIFF representing simulated degraded observation.
* `mask.tif`: 1-band uint8 GeoTIFF representing spatial occlusion mask (if applicable).
* `metadata.json`: Machine-readable parameters (seed, beta, actual coverage, clipping stats).

### CLI Command Reference

#### 1. Generate Combined Sample
```bash
PYTHONPATH=. venv/bin/python -m src.data.degradation_cli \
  --input-scene S2_DUMMY \
  --combined \
  --severity medium \
  --mask-type cloud_like \
  --seed 42 \
  --max-samples 1
```

#### 2. Generate Haze-only or Occlusion-only Samples
* **Haze-only**:
  ```bash
  PYTHONPATH=. venv/bin/python -m src.data.degradation_cli --input-scene S2_DUMMY --haze-only --severity high
  ```
* **Occlusion-only**:
  ```bash
  PYTHONPATH=. venv/bin/python -m src.data.degradation_cli --input-scene S2_DUMMY --occlusion-only --severity extreme --mask-type irregular
  ```

---

## Future Phases
* Phase 4 — Dataset Construction
* Phase 5 — Multispectral-to-RGB Baseline
* Phase 6 — Atmospheric Restoration
* Phase 7 — Mask-Guided Reconstruction
* Phase 8 — Pipeline Integration
* Phase 9 — End-to-End Fine-Tuning
* Phase 10 — Evaluation & Ablation
* Phase 11 — Robustness & Real-World Validation
* Phase 12 — Deployment
