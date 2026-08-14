import numpy as np
import rasterio
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

def compute_raster_statistics(array: np.ndarray) -> Dict[str, Any]:
    """Computes numeric statistics over a raster array including NaN, Inf, and Nodata counts.
    
    Expected array dtype is float32 where nodata pixels are represented by np.nan.
    
    Args:
        array (np.ndarray): The numpy array containing raster values.
        
    Returns:
        dict: Numeric statistics including nan_count, inf_count, valid_percentage.
    """
    total_pixels = int(array.size)
    if total_pixels == 0:
        return {
            "valid_pixel_percentage": 0.0,
            "invalid_pixel_percentage": 100.0,
            "nan_count": 0,
            "inf_count": 0,
            "nodata_count": 0
        }
        
    nan_mask = np.isnan(array)
    inf_mask = np.isinf(array)
    
    nan_count = int(np.sum(nan_mask))
    inf_count = int(np.sum(inf_mask))
    
    # We treat NaNs as our nodata representation for float32 data
    nodata_count = nan_count
    invalid_count = nan_count + inf_count
    
    valid_count = total_pixels - invalid_count
    valid_percentage = (valid_count / total_pixels) * 100.0
    invalid_percentage = (invalid_count / total_pixels) * 100.0
    
    return {
        "valid_pixel_percentage": float(valid_percentage),
        "invalid_pixel_percentage": float(invalid_percentage),
        "nan_count": nan_count,
        "inf_count": inf_count,
        "nodata_count": nodata_count
    }

def verify_processed_scene(
    multispectral_path: Path,
    rgb_path: Path,
    expected_bands: int = 4
) -> Tuple[bool, str]:
    """Performs structural and geospatial alignment checks between the multispectral and RGB outputs.
    
    Args:
        multispectral_path (Path): Path to the generated multispectral GeoTIFF.
        rgb_path (Path): Path to the generated RGB GeoTIFF.
        expected_bands (int): Expected band count in the multispectral file (default: 4).
        
    Returns:
        Tuple[bool, str]: (passed_validation, validation_reason)
    """
    if not multispectral_path.exists():
        return False, f"Multispectral output missing at {multispectral_path}"
    if not rgb_path.exists():
        return False, f"RGB output missing at {rgb_path}"
        
    try:
        with rasterio.open(multispectral_path) as ms, rasterio.open(rgb_path) as rgb:
            # 1. Structural Checks
            if ms.count != expected_bands:
                return False, f"Multispectral file has {ms.count} bands, expected {expected_bands}"
            if rgb.count != 3:
                return False, f"RGB file has {rgb.count} bands, expected 3"
                
            # 2. Geospatial Checks
            # Verify CRS matches
            if ms.crs != rgb.crs:
                return False, f"CRS mismatch: MS is {ms.crs}, RGB is {rgb.crs}"
                
            # Verify Dimensions match
            if ms.width != rgb.width or ms.height != rgb.height:
                return False, f"Dimension mismatch: MS is {ms.width}x{ms.height}, RGB is {rgb.width}x{rgb.height}"
                
            # Verify Transform matches (with small float tolerance)
            for i, (t_ms, t_rgb) in enumerate(zip(ms.transform, rgb.transform)):
                if abs(t_ms - t_rgb) > 1e-7:
                    return False, f"Transform mismatch at index {i}: MS is {t_ms}, RGB is {t_rgb}"
                    
            # Verify bounds match (with small float tolerance)
            for key in ["left", "bottom", "right", "top"]:
                val_ms = getattr(ms.bounds, key)
                val_rgb = getattr(rgb.bounds, key)
                if abs(val_ms - val_rgb) > 1e-5:
                    return False, f"Bounds mismatch for {key}: MS is {val_ms}, RGB is {val_rgb}"
                    
    except Exception as e:
        return False, f"Validation check failed due to exception: {e}"
        
    return True, "Geospatial alignment and integrity validation passed."
