import pytest
import numpy as np
from src.data.degradation import generate_sample

def test_reproducibility_same_seed():
    clean_image = np.ones((4, 40, 40), dtype=np.float32) * 0.25
    config = {
        "haze": {
            "enabled": True,
            "atmospheric_light": [0.8, 0.8, 0.8, 0.8],
            "severity_levels": {
                "high": {"beta_min": 1.0, "beta_max": 2.0}
            }
        },
        "occlusion": {
            "enabled": True,
            "coverage_levels": {
                "high": {"min_fraction": 0.35, "max_fraction": 0.5}
            }
        }
    }
    
    # Run 1 with seed=42
    deg1, mask1, meta1 = generate_sample(
        clean_image=clean_image,
        config=config,
        seed=42,
        haze_severity="high",
        occlusion_severity="high",
        mask_type="cloud_like"
    )
    
    # Run 2 with seed=42
    deg2, mask2, meta2 = generate_sample(
        clean_image=clean_image,
        config=config,
        seed=42,
        haze_severity="high",
        occlusion_severity="high",
        mask_type="cloud_like"
    )
    
    # Assert outputs are exactly identical
    assert np.array_equal(deg1, deg2)
    assert np.array_equal(mask1, mask2)
    assert meta1["haze"]["beta"] == meta2["haze"]["beta"]
    assert meta1["occlusion"]["target_coverage_fraction"] == meta2["occlusion"]["target_coverage_fraction"]

def test_reproducibility_different_seeds():
    clean_image = np.ones((4, 40, 40), dtype=np.float32) * 0.25
    config = {
        "haze": {
            "enabled": True,
            "atmospheric_light": [0.8, 0.8, 0.8, 0.8],
            "severity_levels": {
                "high": {"beta_min": 1.0, "beta_max": 2.0}
            }
        },
        "occlusion": {
            "enabled": True,
            "coverage_levels": {
                "high": {"min_fraction": 0.35, "max_fraction": 0.5}
            }
        }
    }
    
    # Run 1 with seed=42
    deg1, mask1, meta1 = generate_sample(
        clean_image=clean_image,
        config=config,
        seed=42,
        haze_severity="high",
        occlusion_severity="high",
        mask_type="cloud_like"
    )
    
    # Run 2 with seed=43
    deg2, mask2, meta2 = generate_sample(
        clean_image=clean_image,
        config=config,
        seed=43,
        haze_severity="high",
        occlusion_severity="high",
        mask_type="cloud_like"
    )
    
    # Assert outputs are different (with very high probability due to random factors)
    assert not np.array_equal(deg1, deg2)
    assert not np.array_equal(mask1, mask2)
    assert meta1["haze"]["beta"] != meta2["haze"]["beta"]
