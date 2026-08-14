import numpy as np
from scipy.ndimage import gaussian_filter
from typing import Dict, Any, Tuple, Optional, List
from src.data.masks import (
    generate_cloud_mask,
    generate_irregular_mask,
    generate_rectangular_mask,
    calculate_mask_coverage
)
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

def generate_transmission_map(
    shape: Tuple[int, int],
    beta: float,
    seed: int
) -> np.ndarray:
    """Generates a smooth, spatially varying transmission map based on low-frequency noise.
    
    Formula: t(x) = exp(-beta * N(x)) where N(x) is a normalized smooth random field.
    
    Args:
        shape (Tuple[int, int]): Height and width dimensions of the map.
        beta (float): Haze scattering coefficient.
        seed (int): Random seed.
        
    Returns:
        np.ndarray: Smooth transmission map in range [exp(-beta), 1.0].
    """
    h, w = shape
    rng = np.random.default_rng(seed)
    
    # Generate white noise
    noise = rng.normal(size=(h, w))
    
    # Smooth to create low-frequency spatial variation (representing haze thickness)
    sigma = min(h, w) / 10.0
    smooth_noise = gaussian_filter(noise, sigma=sigma)
    
    # Normalize to [0.0, 1.0]
    n_min = smooth_noise.min()
    n_max = smooth_noise.max()
    if n_max > n_min:
        n_normalized = (smooth_noise - n_min) / (n_max - n_min)
    else:
        n_normalized = np.zeros_like(smooth_noise)
        
    # Compute transmission map
    transmission = np.exp(-beta * n_normalized)
    return transmission

def apply_atmospheric_degradation(
    clean_image: np.ndarray,
    transmission: np.ndarray,
    A: np.ndarray
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Applies the atmospheric scattering model: I(x) = J(x)t(x) + A(1 - t(x)) per channel.
    
    Also implements value safety checks, clipping to [0.0, 1.0] when necessary,
    and returns clipping statistics.
    
    Args:
        clean_image (np.ndarray): Clean multispectral image of shape (C, H, W).
        transmission (np.ndarray): Transmission map of shape (H, W).
        A (np.ndarray): Atmospheric light per channel of shape (C,).
        
    Returns:
        Tuple[np.ndarray, dict]: (degraded_image, clipping_stats)
    """
    c, h, w = clean_image.shape
    degraded = np.zeros_like(clean_image)
    
    # Apply scattering formula per channel
    for ch in range(c):
        degraded[ch] = clean_image[ch] * transmission + A[ch] * (1.0 - transmission)
        
    # Perform value safety checks and clip values to [0.0, 1.0]
    pre_min = float(np.nanmin(degraded))
    pre_max = float(np.nanmax(degraded))
    
    # Clip and record metrics
    clipped = np.clip(degraded, 0.0, 1.0)
    clipped_pixels = int(np.sum((degraded < 0.0) | (degraded > 1.0)))
    total_pixels = degraded.size
    clipping_percentage = (clipped_pixels / total_pixels) * 100.0
    
    stats = {
        "pre_clipping_min": pre_min,
        "pre_clipping_max": pre_max,
        "post_clipping_min": float(np.nanmin(clipped)),
        "post_clipping_max": float(np.nanmax(clipped)),
        "clipping_percentage": clipping_percentage
    }
    
    return clipped, stats

def apply_occlusion(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Applies spatial occlusion (binary mask) on the image.
    
    Args:
        image (np.ndarray): Image of shape (C, H, W).
        mask (np.ndarray): Mask of shape (H, W) or (1, H, W) where 1 = valid, 0 = occluded.
        
    Returns:
        np.ndarray: Occluded image of shape (C, H, W).
    """
    # Multiply channels by mask
    return image * mask

def generate_sample(
    clean_image: np.ndarray,
    config: Dict[str, Any],
    seed: int,
    haze_severity: Optional[str] = None,
    occlusion_severity: Optional[str] = None,
    mask_type: Optional[str] = None
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    """High-level function that applies procedurally controlled haze and occlusion to a clean scene.
    
    Args:
        clean_image (np.ndarray): Clean multispectral image array of shape (C, H, W).
        config (dict): Degradation configuration containing severity parameters.
        seed (int): Local seed for the random state generator.
        haze_severity (str, optional): 'low', 'medium', 'high', 'extreme'.
        occlusion_severity (str, optional): 'low', 'medium', 'high', 'extreme'.
        mask_type (str, optional): 'cloud_like', 'irregular', 'rectangular'.
        
    Returns:
        Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]: 
            (degraded_image, binary_mask, sample_metadata)
    """
    rng = np.random.default_rng(seed)
    
    c, h, w = clean_image.shape
    degraded = clean_image.copy()
    mask = None
    
    # Ground truth validation checksum
    clean_checksum = float(np.nanmean(clean_image))
    
    metadata = {
        "seed": seed,
        "degradations": [],
        "degradation_order": []
    }
    
    # 1. Apply Atmospheric Haze
    if haze_severity and haze_severity.lower() != "none" and config["haze"]["enabled"]:
        severity = haze_severity.lower()
        levels = config["haze"]["severity_levels"]
        if severity in levels:
            beta_min = levels[severity]["beta_min"]
            beta_max = levels[severity]["beta_max"]
            beta = float(rng.uniform(beta_min, beta_max))
        else:
            beta = 0.5
            
        A_config = config["haze"]["atmospheric_light"]
        # Ensure atmospheric light fits channels count (B02, B03, B04, B08)
        A = np.array(A_config[:c], dtype=np.float32)
        
        # Create map and apply haze
        transmission = generate_transmission_map((h, w), beta, seed=int(rng.integers(0, 100000)))
        degraded, clip_stats = apply_atmospheric_degradation(degraded, transmission, A)
        
        metadata["degradations"].append("haze")
        metadata["degradation_order"].append("haze")
        metadata["haze"] = {
            "severity": severity,
            "beta": beta,
            "atmospheric_light": A.tolist(),
            "transmission_min": float(transmission.min()),
            "transmission_max": float(transmission.max()),
            "clipping": clip_stats
        }
        
    # 2. Apply Spatial Occlusion
    if occlusion_severity and occlusion_severity.lower() != "none" and config["occlusion"]["enabled"]:
        severity = occlusion_severity.lower()
        levels = config["occlusion"]["coverage_levels"]
        
        if severity in levels:
            min_frac = levels[severity]["min_fraction"]
            max_frac = levels[severity]["max_fraction"]
            target_coverage = float(rng.uniform(min_frac, max_frac))
        else:
            target_coverage = 0.2
            
        # Determine mask type
        m_type = mask_type.lower() if mask_type else "cloud_like"
        mask_seed = int(rng.integers(0, 100000))
        
        if m_type == "rectangular":
            mask = generate_rectangular_mask((h, w), target_coverage, mask_seed)
        elif m_type == "irregular":
            mask = generate_irregular_mask((h, w), target_coverage, mask_seed)
        else:
            mask = generate_cloud_mask((h, w), target_coverage, mask_seed)
            
        # Apply mask: degraded = degraded * mask
        degraded = apply_occlusion(degraded, mask)
        
        actual_coverage = calculate_mask_coverage(mask)
        
        metadata["degradations"].append("occlusion")
        metadata["degradation_order"].append("occlusion")
        metadata["occlusion"] = {
            "type": m_type,
            "severity": severity,
            "target_coverage_fraction": target_coverage,
            "actual_coverage_fraction": actual_coverage
        }
        
    # Verify ground truth checksum matches to ensure J is not modified
    post_checksum = float(np.nanmean(clean_image))
    assert np.isclose(clean_checksum, post_checksum), "Ground truth image was modified during degradation!"
    
    return degraded, mask, metadata
