import pytest
import numpy as np
import rasterio
from src.data.dataset_builder import generate_patch_indices

def test_generate_patch_indices_boundaries(tmp_path):
    clean_path = tmp_path / "boundary_clean.tif"
    
    # Raster dimensions: 600 x 600 (not exactly divisible by 256)
    # Patches at y=0, x=0, 256 are valid.
    # Patches at y=512, x=512 exceed 600 so they should be discarded.
    # Grid should yield (0,0), (0,256), (256,0), (256,256).
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": 1,
        "width": 600,
        "height": 600,
        "crs": "EPSG:32632",
        "transform": rasterio.transform.from_origin(600000, 5000000, 10, 10),
        "nodata": np.nan
    }
    
    with rasterio.open(clean_path, "w", **profile) as dst:
        dst.write(np.ones((1, 600, 600), dtype=np.float32))
        
    coords = generate_patch_indices(clean_path, patch_size=256, overlap=0, min_valid_fraction=0.9)
    
    assert len(coords) == 4
    assert (0, 0) in coords
    assert (0, 256) in coords
    assert (256, 0) in coords
    assert (256, 256) in coords
    assert not any(y > 256 or x > 256 for y, x in coords)

def test_generate_patch_indices_filtering(tmp_path):
    clean_path = tmp_path / "filter_clean.tif"
    
    # Raster size: 256 x 256
    # Fill half of the raster with NaNs (nodata)
    # Valid fraction will be 0.50, which is below 0.90, so the patch should be discarded.
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": 1,
        "width": 256,
        "height": 256,
        "crs": "EPSG:32632",
        "transform": rasterio.transform.from_origin(600000, 5000000, 10, 10),
        "nodata": np.nan
    }
    
    data = np.ones((1, 256, 256), dtype=np.float32)
    data[0, :, 128:] = np.nan  # 50% NaNs
    
    with rasterio.open(clean_path, "w", **profile) as dst:
        dst.write(data)
        
    # Minimum valid threshold: 0.90
    coords = generate_patch_indices(clean_path, patch_size=256, overlap=0, min_valid_fraction=0.90)
    assert len(coords) == 0  # Discarded
    
    # Minimum valid threshold: 0.40
    coords_low = generate_patch_indices(clean_path, patch_size=256, overlap=0, min_valid_fraction=0.40)
    assert len(coords_low) == 1  # Kept
    assert coords_low == [(0, 0)]
