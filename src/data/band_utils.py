import re
import zipfile
import xml.etree.ElementTree as ET
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Map string name to rasterio Resampling enum
RESAMPLING_MAP = {
    "nearest": Resampling.nearest,
    "bilinear": Resampling.bilinear,
    "cubic": Resampling.cubic,
    "cubic_spline": Resampling.cubic_spline,
    "lanczos": Resampling.lanczos,
    "average": Resampling.average,
    "mode": Resampling.mode
}

def find_band_members(zip_path: Path, band_names: List[str]) -> Dict[str, str]:
    """Finds the member file paths inside the Sentinel-2 L2A ZIP archive for specified bands.
    
    Args:
        zip_path (Path): Path to the Sentinel-2 product ZIP archive.
        band_names (list of str): List of bands to locate (e.g. ['B02', 'B03', 'B04', 'B08']).
        
    Returns:
        Dict[str, str]: Map of band name to member path inside the ZIP file.
        
    Raises:
        FileNotFoundError: If the zip file is missing.
        ValueError: If a required band cannot be located inside the archive.
    """
    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP product not found at {zip_path}")
        
    band_map = {}
    with zipfile.ZipFile(zip_path, "r") as zf:
        namelist = zf.namelist()
        for band in band_names:
            member_path = None
            # Search for JP2 band files inside the GRANULE folder
            for name in namelist:
                # Standard pattern: ends with _B02_10m.jp2 or similar
                # Also support 20m/60m bands (e.g. _B05_20m.jp2)
                if (name.endswith(f"_{band}_10m.jp2") or 
                    name.endswith(f"_{band}_20m.jp2") or
                    name.endswith(f"_{band}_60m.jp2") or
                    (f"/R10m/" in name and f"_{band}_" in name) or
                    (f"/R20m/" in name and f"_{band}_" in name) or
                    (f"/R60m/" in name and f"_{band}_" in name)):
                    member_path = name
                    break
            
            if member_path:
                band_map[band] = member_path
            else:
                raise ValueError(f"Required band {band} could not be located inside ZIP archive: {zip_path.name}")
                
    return band_map

def parse_metadata_from_zip(zip_path: Path) -> Tuple[float, Dict[str, float]]:
    """Parses MTD_MSIL2A.xml in the product ZIP archive to extract scaling and offset values.
    
    Args:
        zip_path (Path): Path to the Sentinel-2 product ZIP archive.
        
    Returns:
        Tuple[float, Dict[str, float]]: (quantification_value, band_offsets)
    """
    quantification_value = 10000.0
    band_offsets = {}
    
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            xml_member = None
            for name in zf.namelist():
                if name.endswith("MTD_MSIL2A.xml"):
                    xml_member = name
                    break
                    
            if not xml_member:
                logger.warning(f"Metadata file MTD_MSIL2A.xml not found in ZIP {zip_path.name}. Using default scale and offset.")
                return quantification_value, band_offsets
                
            xml_content = zf.read(xml_member)
            root = ET.fromstring(xml_content)
            
            # Use wildcard namespace selector to avoid ElementTree namespace complexity
            quant_el = root.find(".//{*}QUANTIFICATION_VALUE")
            if quant_el is not None:
                quantification_value = float(quant_el.text)
                
            # Parse radiometric offsets for bands
            for offset_el in root.findall(".//{*}BOA_ADD_OFFSET"):
                band_name = offset_el.get("name")
                if band_name:
                    band_offsets[band_name] = float(offset_el.text)
                    
    except Exception as e:
        logger.warning(f"Error parsing metadata xml in ZIP {zip_path.name}: {e}. Using defaults.")
        
    return quantification_value, band_offsets

def get_reference_grid(zip_path: Path, ref_member: str) -> Dict[str, Any]:
    """Retrieves geospatial metadata from a reference band inside the ZIP to define the target grid.
    
    Args:
        zip_path (Path): Path to the Sentinel-2 product ZIP archive.
        ref_member (str): Reference band member path inside the ZIP.
        
    Returns:
        Dict[str, Any]: Dictionary containing target grid CRS, transform, width, and height.
    """
    # Construct rasterio virtual zip URI
    zip_uri = f"zip://{zip_path}!{ref_member}"
    
    with rasterio.open(zip_uri) as src:
        profile = src.profile.copy()
        grid = {
            "crs": src.crs,
            "transform": src.transform,
            "width": src.width,
            "height": src.height,
            "bounds": src.bounds,
            "res": src.res
        }
    return grid

def read_and_resample_band(
    zip_path: Path,
    member_name: str,
    target_crs: Any,
    target_transform: Any,
    target_width: int,
    target_height: int,
    resampling_method: str = "bilinear"
) -> Tuple[np.ndarray, Optional[float]]:
    """Reads a band from inside the ZIP file and warps/resamples it to match the target grid.
    
    Args:
        zip_path (Path): Path to the Sentinel-2 product ZIP archive.
        member_name (str): Member path inside the ZIP file.
        target_crs: Destination Coordinate Reference System.
        target_transform: Destination Affine transform.
        target_width (int): Destination pixel width.
        target_height (int): Destination pixel height.
        resampling_method (str): Name of resampling algorithm (default: 'bilinear').
        
    Returns:
        Tuple[np.ndarray, float]: (resampled_data_array, source_nodata_value)
    """
    zip_uri = f"zip://{zip_path}!{member_name}"
    resampling = RESAMPLING_MAP.get(resampling_method.lower(), Resampling.bilinear)
    
    with rasterio.open(zip_uri) as src:
        source_nodata = src.nodata
        
        # Read source data (first band)
        src_data = src.read(1)
        src_dtype = src.dtypes[0]
        
        # Output destination array
        dst_data = np.empty((target_height, target_width), dtype=src_dtype)
        
        # Reproject source band to target grid
        reproject(
            source=src_data,
            destination=dst_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=target_transform,
            dst_crs=target_crs,
            resampling=resampling,
            src_nodata=source_nodata,
            dst_nodata=source_nodata
        )
        
    return dst_data, source_nodata
