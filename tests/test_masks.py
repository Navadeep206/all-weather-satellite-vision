import pytest
import numpy as np
from src.data.masks import (
    generate_cloud_mask,
    generate_irregular_mask,
    generate_rectangular_mask,
    calculate_mask_coverage
)

def test_calculate_mask_coverage():
    # 50% mask coverage
    mask = np.array([
        [1, 1, 0, 0],
        [1, 1, 0, 0]
    ], dtype=np.uint8)
    assert calculate_mask_coverage(mask) == 0.5
    
    # 0% coverage
    mask_all_valid = np.ones((4, 4), dtype=np.uint8)
    assert calculate_mask_coverage(mask_all_valid) == 0.0

def test_generate_cloud_mask():
    shape = (100, 100)
    target_coverage = 0.25
    seed = 42
    
    mask = generate_cloud_mask(shape, target_coverage, seed)
    
    # Check shape
    assert mask.shape == shape
    # Check binary values
    assert np.all((mask == 0) | (mask == 1))
    
    # Check coverage is close to target (procedural smoothing might shift it slightly, 
    # but percentile-based thresholding should make it exactly matched!)
    actual_coverage = calculate_mask_coverage(mask)
    assert np.isclose(actual_coverage, target_coverage, atol=0.01)
    
    # Verify spatial smoothness (cloud-like is not random pixel white noise)
    # Check correlation: adjacent pixels should be highly similar
    # For white noise, diffs will be large; for smooth noise, diffs will be small.
    diff_h = np.abs(mask[:-1, :].astype(int) - mask[1:, :].astype(int))
    mean_diff = np.mean(diff_h)
    # Average difference for pure random binary thresholded noise is around 0.35-0.5
    # For smoothed cloud shapes, boundaries are rare, so mean diff is much lower (<0.1)
    assert mean_diff < 0.1

def test_generate_irregular_mask():
    shape = (100, 100)
    target_coverage = 0.30
    seed = 42
    
    mask = generate_irregular_mask(shape, target_coverage, seed)
    
    assert mask.shape == shape
    assert np.all((mask == 0) | (mask == 1))
    actual_coverage = calculate_mask_coverage(mask)
    assert np.isclose(actual_coverage, target_coverage, atol=0.01)

def test_generate_rectangular_mask():
    shape = (100, 100)
    target_coverage = 0.15
    seed = 42
    
    mask = generate_rectangular_mask(shape, target_coverage, seed)
    
    assert mask.shape == shape
    assert np.all((mask == 0) | (mask == 1))
    
    # Rectangles are added iteratively, so coverage might be slightly above target,
    # but should be reasonably close
    actual_coverage = calculate_mask_coverage(mask)
    assert actual_coverage >= target_coverage
    assert actual_coverage <= target_coverage + 0.15 # shouldn't exceed by too much
