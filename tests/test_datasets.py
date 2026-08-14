import pytest
import torch
import numpy as np
import rasterio
from pathlib import Path
from src.data.datasets import SatelliteStage1Dataset, SatelliteStage2Dataset, SatelliteStage3Dataset
from src.data.manifest import write_manifest
from src.utils.config import get_project_root

@pytest.fixture
def mock_dataset_files(tmp_path):
    # Set up folders relative to project root
    project_root = get_project_root()
    rel_raw = Path("data/processed/sentinel2/S2_TEST_DATASET_SCENE")
    clean_dir = project_root / rel_raw
    clean_dir.mkdir(parents=True, exist_ok=True)
    
    clean_path = clean_dir / "multispectral.tif"
    rgb_path = clean_dir / "rgb.tif"
    
    # Degraded files
    rel_deg = Path("data/degraded/haze/SMP_S2_TEST_DATASET_SCENE_haze")
    deg_dir = project_root / rel_deg
    deg_dir.mkdir(parents=True, exist_ok=True)
    deg_path = deg_dir / "degraded.tif"
    mask_path = deg_dir / "mask.tif"
    
    # Write mock GeoTIFF profiles
    profile_ms = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": 4,
        "width": 512,
        "height": 512,
        "crs": "EPSG:32632",
        "transform": rasterio.transform.from_origin(600000, 5000000, 10, 10),
        "nodata": np.nan
    }
    profile_rgb = profile_ms.copy()
    profile_rgb.update({"count": 3})
    profile_mask = profile_ms.copy()
    profile_mask.update({"count": 1, "dtype": "uint8", "nodata": 0})
    
    # Write arrays
    with rasterio.open(clean_path, "w", **profile_ms) as dst:
        dst.write(np.ones((4, 512, 512), dtype=np.float32) * 0.4)
        
    with rasterio.open(rgb_path, "w", **profile_rgb) as dst:
        dst.write(np.ones((3, 512, 512), dtype=np.float32) * 0.4)
        
    with rasterio.open(deg_path, "w", **profile_ms) as dst:
        dst.write(np.ones((4, 512, 512), dtype=np.float32) * 0.2)
        
    with rasterio.open(mask_path, "w", **profile_mask) as dst:
        dst.write(np.ones((1, 512, 512), dtype=np.uint8))
        
    # Write mock manifests
    manifest_csv = tmp_path / "stage1_train.csv"
    rows_s1 = [{
        "sample_id": "SMP_001_s1",
        "scene_id": "S2_TEST_DATASET_SCENE",
        "split": "train",
        "clean_path": str(clean_path.relative_to(project_root)),
        "degraded_path": str(deg_path.relative_to(project_root)),
        "patch_y": 0,
        "patch_x": 0,
        "degradation_type": "haze"
    }]
    write_manifest(manifest_csv, rows_s1)
    
    manifest_csv2 = tmp_path / "stage2_train.csv"
    rows_s2 = [{
        "sample_id": "SMP_001_s2",
        "scene_id": "S2_TEST_DATASET_SCENE",
        "split": "train",
        "clean_path": str(clean_path.relative_to(project_root)),
        "degraded_path": str(deg_path.relative_to(project_root)),
        "mask_path": str(mask_path.relative_to(project_root)),
        "patch_y": 0,
        "patch_x": 256,
        "degradation_type": "combined"
    }]
    write_manifest(manifest_csv2, rows_s2)
    
    manifest_csv3 = tmp_path / "stage3_train.csv"
    rows_s3 = [{
        "sample_id": "SMP_001_s3",
        "scene_id": "S2_TEST_DATASET_SCENE",
        "split": "train",
        "clean_path": str(clean_path.relative_to(project_root)),
        "rgb_path": str(rgb_path.relative_to(project_root)),
        "patch_y": 256,
        "patch_x": 0,
        "degradation_type": "none"
    }]
    write_manifest(manifest_csv3, rows_s3)
    
    yield manifest_csv, manifest_csv2, manifest_csv3
    
    # Cleanup files
    for p in [clean_path, rgb_path, deg_path, mask_path]:
        if p.exists():
            p.unlink()
    # Cleanup folders if empty
    for d in [clean_dir, deg_dir]:
        if d.exists():
            try: d.rmdir()
            except Exception: pass

def test_datasets_loading(mock_dataset_files):
    manifest_s1, manifest_s2, manifest_s3 = mock_dataset_files
    
    # 1. Stage 1 loading
    ds1 = SatelliteStage1Dataset(manifest_s1, patch_size=256)
    assert len(ds1) == 1
    x, y, meta = ds1[0]
    assert isinstance(x, torch.Tensor)
    assert x.shape == (4, 256, 256)
    assert y.shape == (4, 256, 256)
    assert meta["sample_id"] == "SMP_001_s1"
    
    # 2. Stage 2 loading
    ds2 = SatelliteStage2Dataset(manifest_s2, patch_size=256)
    assert len(ds2) == 1
    x, mask, y, meta = ds2[0]
    assert x.shape == (4, 256, 256)
    assert mask.shape == (1, 256, 256)
    assert y.shape == (4, 256, 256)
    
    # 3. Stage 3 loading
    ds3 = SatelliteStage3Dataset(manifest_s3, patch_size=256)
    assert len(ds3) == 1
    x, y, meta = ds3[0]
    assert x.shape == (4, 256, 256)
    assert y.shape == (3, 256, 256)
