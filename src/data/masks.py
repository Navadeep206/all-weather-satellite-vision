import numpy as np
from scipy.ndimage import gaussian_filter
from typing import Tuple, Union

def calculate_mask_coverage(mask: np.ndarray) -> float:
    """Calculates the fraction of occluded (0) pixels in the mask.
    
    Args:
        mask (np.ndarray): Binary mask where 1 is valid, 0 is occluded.
        
    Returns:
        float: Coverage fraction between 0.0 and 1.0.
    """
    total_pixels = mask.size
    if total_pixels == 0:
        return 0.0
    occluded_pixels = np.sum(mask == 0)
    return float(occluded_pixels / total_pixels)

def generate_cloud_mask(
    shape: Tuple[int, int],
    target_coverage: float,
    seed: int
) -> np.ndarray:
    """Generates an irregular, smooth cloud-like mask using low-frequency Gaussian noise.
    
    Args:
        shape (Tuple[int, int]): Height and width dimensions of the mask.
        target_coverage (float): Target fraction of occluded pixels (0.0 to 1.0).
        seed (int): Random seed for reproducibility.
        
    Returns:
        np.ndarray: Binary mask (1 = valid, 0 = occluded) of dtype np.uint8.
    """
    h, w = shape
    rng = np.random.default_rng(seed)
    
    # Generate random white noise
    noise = rng.normal(size=(h, w))
    
    # Smooth with a large sigma to create low-frequency structures (cloud shapes)
    sigma = min(h, w) / 10.0
    smooth_noise = gaussian_filter(noise, sigma=sigma)
    
    # Determine threshold based on target coverage using percentile
    threshold = np.percentile(smooth_noise, target_coverage * 100.0)
    
    # Create mask: N > threshold is valid (1), N <= threshold is occluded (0)
    mask = (smooth_noise > threshold).astype(np.uint8)
    return mask

def generate_irregular_mask(
    shape: Tuple[int, int],
    target_coverage: float,
    seed: int
) -> np.ndarray:
    """Generates fragmented, irregular missing region masks using higher-frequency noise.
    
    Args:
        shape (Tuple[int, int]): Height and width dimensions of the mask.
        target_coverage (float): Target fraction of occluded pixels (0.0 to 1.0).
        seed (int): Random seed for reproducibility.
        
    Returns:
        np.ndarray: Binary mask (1 = valid, 0 = occluded) of dtype np.uint8.
    """
    h, w = shape
    rng = np.random.default_rng(seed)
    
    # Generate noise
    noise = rng.normal(size=(h, w))
    
    # Use smaller sigma for more localized/fragmented regions
    sigma = min(h, w) / 40.0
    smooth_noise = gaussian_filter(noise, sigma=sigma)
    
    # Threshold
    threshold = np.percentile(smooth_noise, target_coverage * 100.0)
    mask = (smooth_noise > threshold).astype(np.uint8)
    return mask

def generate_rectangular_mask(
    shape: Tuple[int, int],
    target_coverage: float,
    seed: int
) -> np.ndarray:
    """Generates sensor-style rectangular block masks matching target coverage.
    
    Args:
        shape (Tuple[int, int]): Height and width dimensions of the mask.
        target_coverage (float): Target fraction of occluded pixels (0.0 to 1.0).
        seed (int): Random seed for reproducibility.
        
    Returns:
        np.ndarray: Binary mask (1 = valid, 0 = occluded) of dtype np.uint8.
    """
    h, w = shape
    rng = np.random.default_rng(seed)
    
    # Initialize all valid (1)
    mask = np.ones((h, w), dtype=np.uint8)
    
    total_pixels = h * w
    target_pixels = int(total_pixels * target_coverage)
    occluded_pixels = 0
    
    # Iteratively place random rectangles until target coverage is met
    # Limit iterations to avoid infinite loop
    max_rects = 20
    for _ in range(max_rects):
        if occluded_pixels >= target_pixels:
            break
            
        # Draw random dimensions for the rectangle (between 10% and 40% of grid size)
        rect_h = rng.integers(int(h * 0.1), int(h * 0.4) + 1)
        rect_w = rng.integers(int(w * 0.1), int(w * 0.4) + 1)
        
        y = rng.integers(0, h - rect_h)
        x = rng.integers(0, w - rect_w)
        
        # Apply rectangle
        mask[y:y+rect_h, x:x+rect_w] = 0
        
        # Re-evaluate coverage
        occluded_pixels = np.sum(mask == 0)
        
    return mask
