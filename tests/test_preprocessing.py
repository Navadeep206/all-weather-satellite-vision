import pytest
import zipfile
import json
import numpy as np
import rasterio
from pathlib import Path
from src.data.scene_catalog import SceneCatalog
from src.data.preprocessing import process_scene

@pytest.fixture
def dummy_scene_setup(tmp_path):
    # Setup folders
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    metadata_dir = tmp_path / "metadata"
    
    # Init catalog
    catalog = SceneCatalog(metadata_dir=metadata_dir)
    
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
    
    # Write tiny tifs
    band_data = np.ones((5, 5), dtype=np.uint16) * 2000
    
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
    
    temp_tif = tmp_path / "temp_band.tif"
    with rasterio.open(temp_tif, "w", **profile) as dst:
        dst.write(band_data, 1)
        
    scene_id = "S2_L2A_TEST_SCENE"
    product_name = "S2A_MSIL2A_TEST_SCENE.SAFE"
    product_id = "test-product-uuid-9999"
    
    zip_dir = raw_dir / scene_id / "product"
    zip_dir.mkdir(parents=True, exist_ok=True)
    zip_path = zip_dir / f"{product_name}.zip"
    
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(f"{product_name}/MTD_MSIL2A.xml", xml_content)
        for band in ["B02", "B03", "B04", "B08"]:
            zf.write(temp_tif, f"{product_name}/GRANULE/L2A_T32TNS_A000000/IMG_DATA/R10m/T32TNS_20231010T102021_{band}_10m.jp2")
            
    # Add scene to catalog as validated
    scene_record = {
        "scene_id": scene_id,
        "product_id": product_id,
        "product_name": product_name,
        "processing_level": "L2A",
        "platform": "Sentinel-2A",
        "sensing_datetime": "2023-10-10T10:20:21.000Z",
        "tile_id": "32TNS",
        "cloud_cover": 0.0,
        "source": "CDSE",
        "status": "validated"
    }
    catalog.add_scene(scene_record)
    
    config = {
        "bands": ["B02", "B03", "B04", "B08"],
        "search": {
            "resampling_method": "bilinear"
        }
    }
    
    return scene_id, product_id, raw_dir, processed_dir, catalog, config

def test_process_scene_success(dummy_scene_setup):
    scene_id, product_id, raw_dir, processed_dir, catalog, config = dummy_scene_setup
    
    # Process the scene
    success, msg = process_scene(
        scene_id=scene_id,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        catalog=catalog,
        config=config
    )
    
    assert success, msg
    assert catalog.get_scene(product_id)["status"] == "processed"
    
    # Check outputs generated
    scene_proc_dir = processed_dir / scene_id
    ms_path = scene_proc_dir / "multispectral.tif"
    rgb_path = scene_proc_dir / "rgb.tif"
    meta_path = scene_proc_dir / "metadata.json"
    qual_path = scene_proc_dir / "quality.json"
    
    assert ms_path.exists()
    assert rgb_path.exists()
    assert meta_path.exists()
    assert qual_path.exists()
    
    # Reopen GeoTIFF and check values
    # DN was 2000, offset B02..B08 was -1000, quantification 10000
    # Expected reflectance: (2000 - 1000) / 10000 = 0.1
    with rasterio.open(ms_path) as src:
        assert src.count == 4
        assert src.dtypes[0] == "float32"
        data = src.read(1)
        assert np.allclose(data, 0.1)
        
    with open(meta_path, "r") as f:
        meta = json.load(f)
    assert meta["scene_id"] == scene_id
    assert meta["bands"] == ["B02", "B03", "B04", "B08"]
    
    with open(qual_path, "r") as f:
        qual = json.load(f)
    assert qual["processing_status"] == "passed"
    assert qual["valid_pixel_percentage"] == 100.0
    
    # Clean up generated preview file
    from src.utils.config import get_project_root
    preview_file = get_project_root() / "data" / "samples" / "previews" / f"{scene_id}_preview.png"
    if preview_file.exists():
        preview_file.unlink()


def test_process_scene_dry_run(dummy_scene_setup):
    scene_id, product_id, raw_dir, processed_dir, catalog, config = dummy_scene_setup
    
    success, msg = process_scene(
        scene_id=scene_id,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        catalog=catalog,
        config=config,
        dry_run=True
    )
    
    assert success
    assert "Dry-run checked" in msg
    
    # Check NO files were written
    scene_proc_dir = processed_dir / scene_id
    assert not scene_proc_dir.exists()

def test_process_scene_idempotency(dummy_scene_setup):
    scene_id, product_id, raw_dir, processed_dir, catalog, config = dummy_scene_setup
    
    # First processing
    success1, _ = process_scene(scene_id, raw_dir, processed_dir, catalog, config)
    assert success1
    
    # Second processing without force should be skipped (return success True immediately)
    success2, msg2 = process_scene(scene_id, raw_dir, processed_dir, catalog, config)
    assert success2
    assert "already exist" in msg2
