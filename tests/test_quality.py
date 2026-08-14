import pytest
import numpy as np
import rasterio
from pathlib import Path
from src.data.quality import compute_raster_statistics, verify_processed_scene

def test_compute_raster_statistics_all_valid():
    arr = np.ones((4, 10, 10), dtype=np.float32)
    stats = compute_raster_statistics(arr)
    assert stats["valid_pixel_percentage"] == 100.0
    assert stats["invalid_pixel_percentage"] == 0.0
    assert stats["nan_count"] == 0
    assert stats["inf_count"] == 0

def test_compute_raster_statistics_mixed():
    arr = np.ones((10, 10), dtype=np.float32)
    arr[0, 0] = np.nan
    arr[1, 1] = np.inf
    arr[2, 2] = -np.inf
    
    stats = compute_raster_statistics(arr)
    # total 100 pixels. 1 NaN, 2 Inf
    assert stats["nan_count"] == 1
    assert stats["inf_count"] == 2
    assert stats["nodata_count"] == 1
    assert stats["valid_pixel_percentage"] == 97.0
    assert stats["invalid_pixel_percentage"] == 3.0

def test_compute_raster_statistics_empty():
    arr = np.array([], dtype=np.float32)
    stats = compute_raster_statistics(arr)
    assert stats["valid_pixel_percentage"] == 0.0
    assert stats["invalid_pixel_percentage"] == 100.0

def create_tiny_raster(path: Path, bands: int, width: int, height: int, crs="EPSG:4326", transform=rasterio.transform.from_origin(0, 0, 1, 1)) -> None:
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": bands,
        "width": width,
        "height": height,
        "crs": crs,
        "transform": transform,
        "nodata": np.nan
    }
    with rasterio.open(path, "w", **profile) as dst:
        for i in range(1, bands + 1):
            dst.write(np.ones((height, width), dtype=np.float32) * i, i)

def test_verify_processed_scene_valid(tmp_path):
    ms_path = tmp_path / "multispectral.tif"
    rgb_path = tmp_path / "rgb.tif"
    
    create_tiny_raster(ms_path, bands=4, width=10, height=10)
    create_tiny_raster(rgb_path, bands=3, width=10, height=10)
    
    success, msg = verify_processed_scene(ms_path, rgb_path, expected_bands=4)
    assert success
    assert "integrity validation passed" in msg

def test_verify_processed_scene_dimension_mismatch(tmp_path):
    ms_path = tmp_path / "multispectral.tif"
    rgb_path = tmp_path / "rgb.tif"
    
    create_tiny_raster(ms_path, bands=4, width=10, height=10)
    # RGB has size 5x5 instead of 10x10
    create_tiny_raster(rgb_path, bands=3, width=5, height=5)
    
    success, msg = verify_processed_scene(ms_path, rgb_path, expected_bands=4)
    assert not success
    assert "Dimension mismatch" in msg

def test_verify_processed_scene_crs_mismatch(tmp_path):
    ms_path = tmp_path / "multispectral.tif"
    rgb_path = tmp_path / "rgb.tif"
    
    create_tiny_raster(ms_path, bands=4, width=10, height=10, crs="EPSG:4326")
    create_tiny_raster(rgb_path, bands=3, width=10, height=10, crs="EPSG:3857")
    
    success, msg = verify_processed_scene(ms_path, rgb_path, expected_bands=4)
    assert not success
    assert "CRS mismatch" in msg
