import pytest
import numpy as np
import rasterio
import json
from pathlib import Path
from src.data.degradation import (
    generate_transmission_map,
    apply_atmospheric_degradation,
    apply_occlusion,
    generate_sample
)
from src.data.degradation_cli import save_degraded_sample

def test_generate_transmission_map():
    shape = (50, 50)
    beta = 1.5
    seed = 42
    
    t = generate_transmission_map(shape, beta, seed)
    assert t.shape == shape
    
    # Transmission must be in range [exp(-beta), 1.0]
    expected_min = np.exp(-beta)
    assert np.all(t >= expected_min - 1e-7)
    assert np.all(t <= 1.0 + 1e-7)

def test_apply_atmospheric_degradation():
    clean_image = np.ones((4, 50, 50), dtype=np.float32) * 0.5
    transmission = np.ones((50, 50), dtype=np.float32) * 0.8
    A = np.array([0.8, 0.75, 0.7, 0.6], dtype=np.float32)
    
    degraded, stats = apply_atmospheric_degradation(clean_image, transmission, A)
    
    assert degraded.shape == clean_image.shape
    assert not np.any(np.isnan(degraded))
    assert not np.any(np.isinf(degraded))
    assert np.all(degraded >= 0.0)
    assert np.all(degraded <= 1.0)
    
    # Check formula: I = J*t + A*(1-t)
    # Channel 0: 0.5 * 0.8 + 0.8 * 0.2 = 0.4 + 0.16 = 0.56
    assert np.allclose(degraded[0], 0.56)
    
    assert "pre_clipping_min" in stats
    assert "clipping_percentage" in stats

def test_apply_occlusion():
    image = np.ones((4, 10, 10), dtype=np.float32) * 0.5
    mask = np.ones((10, 10), dtype=np.uint8)
    mask[2:5, 2:5] = 0 # Occlude block
    
    occluded = apply_occlusion(image, mask)
    
    # Valid parts remain unchanged (0.5)
    assert np.all(occluded[:, mask == 1] == 0.5)
    # Occluded parts are 0
    assert np.all(occluded[:, mask == 0] == 0.0)

def test_generate_sample():
    clean_image = np.ones((4, 50, 50), dtype=np.float32) * 0.3
    config = {
        "haze": {
            "enabled": True,
            "atmospheric_light": [0.8, 0.8, 0.8, 0.8],
            "severity_levels": {
                "medium": {"beta_min": 0.5, "beta_max": 1.0}
            }
        },
        "occlusion": {
            "enabled": True,
            "coverage_levels": {
                "medium": {"min_fraction": 0.15, "max_fraction": 0.35}
            }
        }
    }
    
    degraded, mask, metadata = generate_sample(
        clean_image=clean_image,
        config=config,
        seed=100,
        haze_severity="medium",
        occlusion_severity="medium",
        mask_type="cloud_like"
    )
    
    assert degraded.shape == clean_image.shape
    assert mask is not None
    assert mask.shape == (50, 50)
    assert "haze" in metadata
    assert "occlusion" in metadata
    assert metadata["degradation_order"] == ["haze", "occlusion"]
    
    # Check J (clean_image) is NOT modified
    assert np.all(clean_image == 0.3)

def test_synthetic_integration_pipeline(tmp_path):
    # 1. Create source TIFF file
    src_path = tmp_path / "source.tif"
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": 4,
        "width": 20,
        "height": 20,
        "crs": "EPSG:32632",
        "transform": rasterio.transform.from_origin(600000, 5000000, 10, 10),
        "nodata": np.nan
    }
    
    clean_data = np.ones((4, 20, 20), dtype=np.float32) * 0.4
    with rasterio.open(src_path, "w", **profile) as dst:
        for i in range(4):
            dst.write(clean_data[i], i + 1)
            
    # 2. Run degradation setup
    config = {
        "haze": {
            "enabled": True,
            "atmospheric_light": [0.8, 0.8, 0.8, 0.8],
            "severity_levels": {
                "medium": {"beta_min": 0.5, "beta_max": 1.0}
            }
        },
        "occlusion": {
            "enabled": True,
            "coverage_levels": {
                "medium": {"min_fraction": 0.15, "max_fraction": 0.35}
            }
        }
    }
    
    degraded, mask, metadata = generate_sample(
        clean_image=clean_data,
        config=config,
        seed=42,
        haze_severity="medium",
        occlusion_severity="medium",
        mask_type="irregular"
    )
    
    # 3. Write files using the CLI save utility
    output_dir = tmp_path / "sample_out"
    save_degraded_sample(
        sample_id="SMP_TEST",
        output_dir=output_dir,
        clean_arr=clean_data,
        degraded_arr=degraded,
        mask_arr=mask,
        metadata=metadata,
        profile=profile
    )
    
    # 4. Reload and validate
    assert (output_dir / "clean.tif").exists()
    assert (output_dir / "degraded.tif").exists()
    assert (output_dir / "mask.tif").exists()
    assert (output_dir / "metadata.json").exists()
    
    with rasterio.open(output_dir / "degraded.tif") as ds:
        assert ds.count == 4
        assert ds.width == 20
        assert ds.height == 20
        assert ds.crs.to_epsg() == 32632
        
    with rasterio.open(output_dir / "mask.tif") as ds:
        assert ds.count == 1
        assert ds.dtypes[0] == "uint8"
