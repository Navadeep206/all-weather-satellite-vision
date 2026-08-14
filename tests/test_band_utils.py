import pytest
import zipfile
import xml.etree.ElementTree as ET
import numpy as np
import rasterio
from pathlib import Path
from src.data.band_utils import (
    find_band_members,
    parse_metadata_from_zip,
    get_reference_grid,
    read_and_resample_band
)

@pytest.fixture
def dummy_sentinel2_zip(tmp_path):
    zip_path = tmp_path / "dummy_s2_product.zip"
    
    # 1. Create a dummy XML metadata file
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
    <n1:Level-2A_User_Product xmlns:n1="https://psd-14.sentinel2.eo.esa.int/PSD/S2_PDI_Level-2A_Tile_Metadata.xsd">
        <General_Info>
            <Product_Image_Characteristics>
                <QUANTIFICATION_VALUE>10000</QUANTIFICATION_VALUE>
                <BOA_ADD_OFFSET name="B02">-1000</BOA_ADD_OFFSET>
                <BOA_ADD_OFFSET name="B03">-1000</BOA_ADD_OFFSET>
                <BOA_ADD_OFFSET name="B04">-1000</BOA_ADD_OFFSET>
                <BOA_ADD_OFFSET name="B08">-1000</BOA_ADD_OFFSET>
            </Product_Image_Characteristics>
        </General_Info>
    </n1:Level-2A_User_Product>
    """
    
    # 2. Create tiny GeoTIFF files to store in the ZIP representing bands (renamed as jp2)
    # We will write a tiny 5x5 TIFF and save it inside the zip
    band_data = np.arange(25, dtype=np.uint16).reshape((5, 5))
    
    profile = {
        "driver": "GTiff",
        "dtype": "uint16",
        "count": 1,
        "width": 5,
        "height": 5,
        "crs": "EPSG:32632",
        "transform": rasterio.transform.from_origin(600000, 5000000, 10, 10),
        "nodata": 0
    }
    
    temp_tif_b02 = tmp_path / "B02.tif"
    with rasterio.open(temp_tif_b02, "w", **profile) as dst:
        dst.write(band_data, 1)
        
    temp_tif_b03 = tmp_path / "B03.tif"
    # Create B03 with half size (representing a 20m band to test resampling/scaling!)
    profile_20m = profile.copy()
    profile_20m.update({
        "width": 3,
        "height": 3,
        "transform": rasterio.transform.from_origin(600000, 5000000, 20, 20)
    })
    with rasterio.open(temp_tif_b03, "w", **profile_20m) as dst:
        dst.write(np.ones((3, 3), dtype=np.uint16) * 100, 1)
        
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("S2A_MSIL2A_DUMMY.SAFE/MTD_MSIL2A.xml", xml_content)
        # Store B02
        zf.write(temp_tif_b02, "S2A_MSIL2A_DUMMY.SAFE/GRANULE/L2A_T32TNS_A000000/IMG_DATA/R10m/T32TNS_20231010T102021_B02_10m.jp2")
        # Store B03 (representing another band path)
        zf.write(temp_tif_b03, "S2A_MSIL2A_DUMMY.SAFE/GRANULE/L2A_T32TNS_A000000/IMG_DATA/R20m/T32TNS_20231010T102021_B03_20m.jp2")
        
    return zip_path

def test_find_band_members(dummy_sentinel2_zip):
    band_map = find_band_members(dummy_sentinel2_zip, ["B02", "B03"])
    assert "B02" in band_map
    assert "B03" in band_map
    assert "B02_10m.jp2" in band_map["B02"]
    assert "B03_20m.jp2" in band_map["B03"]

def test_find_band_members_missing(dummy_sentinel2_zip):
    with pytest.raises(ValueError):
        find_band_members(dummy_sentinel2_zip, ["B02", "B04"]) # B04 not in zip

def test_parse_metadata_from_zip(dummy_sentinel2_zip):
    quant, offsets = parse_metadata_from_zip(dummy_sentinel2_zip)
    assert quant == 10000.0
    assert offsets["B02"] == -1000.0
    assert offsets["B03"] == -1000.0

def test_get_reference_grid(dummy_sentinel2_zip):
    band_map = find_band_members(dummy_sentinel2_zip, ["B02"])
    grid = get_reference_grid(dummy_sentinel2_zip, band_map["B02"])
    
    assert grid["width"] == 5
    assert grid["height"] == 5
    assert grid["crs"].to_epsg() == 32632

def test_read_and_resample_band(dummy_sentinel2_zip):
    band_map = find_band_members(dummy_sentinel2_zip, ["B02", "B03"])
    ref_grid = get_reference_grid(dummy_sentinel2_zip, band_map["B02"])
    
    # Resample B03 (originally 3x3 at 20m) to match B02's grid (5x5 at 10m)
    resampled, nodata = read_and_resample_band(
        zip_path=dummy_sentinel2_zip,
        member_name=band_map["B03"],
        target_crs=ref_grid["crs"],
        target_transform=ref_grid["transform"],
        target_width=ref_grid["width"],
        target_height=ref_grid["height"],
        resampling_method="bilinear"
    )
    
    assert resampled.shape == (5, 5)
    assert np.all(resampled == 100) # Since it's all 100 in input, output is also 100
    assert nodata == 0
