import pytest
import zipfile
import json
from pathlib import Path
from src.data.scene_catalog import SceneCatalog
from src.data.validator import SceneValidator

@pytest.fixture
def temp_catalog_and_dirs(tmp_path):
    catalog_dir = tmp_path / "metadata"
    output_dir = tmp_path / "raw"
    catalog = SceneCatalog(metadata_dir=catalog_dir)
    validator = SceneValidator(catalog=catalog, output_dir=output_dir)
    return catalog, validator, output_dir

def create_dummy_zip(zip_path: Path, filenames: list[str]) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w") as zf:
        for name in filenames:
            zf.writestr(name, "dummy_data")

def test_validate_valid_scene(temp_catalog_and_dirs):
    catalog, validator, output_dir = temp_catalog_and_dirs
    
    scene = {
        "scene_id": "S2_L2A_VALID",
        "product_id": "valid-uuid-1111",
        "product_name": "S2A_MSIL2A_VALID.SAFE",
        "sensing_datetime": "2023-10-10T10:20:21.000Z",
        "cloud_cover": 4.5,
        "platform": "Sentinel-2A",
        "tile_id": "32TNS",
        "source": "CDSE"
    }
    
    catalog.add_scene(scene)
    
    # Create valid dummy zip structure
    zip_path = output_dir / "S2_L2A_VALID" / "product" / "S2A_MSIL2A_VALID.SAFE.zip"
    filenames = [
        "S2A_MSIL2A_VALID.SAFE/MTD_MSIL2A.xml",
        "S2A_MSIL2A_VALID.SAFE/GRANULE/L2A_T32TNS_A000000_20231010T102021/IMG_DATA/R10m/T32TNS_20231010T102021_B02_10m.jp2",
        "S2A_MSIL2A_VALID.SAFE/GRANULE/L2A_T32TNS_A000000_20231010T102021/IMG_DATA/R10m/T32TNS_20231010T102021_B03_10m.jp2",
        "S2A_MSIL2A_VALID.SAFE/GRANULE/L2A_T32TNS_A000000_20231010T102021/IMG_DATA/R10m/T32TNS_20231010T102021_B04_10m.jp2",
        "S2A_MSIL2A_VALID.SAFE/GRANULE/L2A_T32TNS_A000000_20231010T102021/IMG_DATA/R10m/T32TNS_20231010T102021_B08_10m.jp2",
    ]
    create_dummy_zip(zip_path, filenames)
    
    success, msg = validator.validate_scene(scene)
    assert success, msg
    assert catalog.get_scene("valid-uuid-1111")["status"] == "validated"
    
    # Check that metadata file was preserved
    meta_file = output_dir / "S2_L2A_VALID" / "metadata" / "scene.json"
    assert meta_file.exists()
    with open(meta_file, "r") as f:
        meta_content = json.load(f)
    assert meta_content["scene_id"] == "S2_L2A_VALID"
    assert meta_content["validation_status"] == "validated"

def test_validate_missing_product(temp_catalog_and_dirs):
    catalog, validator, output_dir = temp_catalog_and_dirs
    scene = {
        "scene_id": "S2_L2A_MISSING_PROD",
        "product_id": "missing-prod-uuid",
        "product_name": "S2A_MSIL2A_MISSING.SAFE"
    }
    catalog.add_scene(scene)
    
    # Create product directory but not the ZIP file
    (output_dir / "S2_L2A_MISSING_PROD" / "product").mkdir(parents=True, exist_ok=True)
    success, msg = validator.validate_scene(scene)
    assert not success
    assert "ZIP not found" in msg
    assert catalog.get_scene("missing-prod-uuid")["status"] == "failed"

def test_validate_malformed_zip(temp_catalog_and_dirs):
    catalog, validator, output_dir = temp_catalog_and_dirs
    scene = {
        "scene_id": "S2_L2A_MALFORMED",
        "product_id": "malformed-uuid",
        "product_name": "S2A_MSIL2A_MALFORMED.SAFE"
    }
    catalog.add_scene(scene)
    
    zip_path = output_dir / "S2_L2A_MALFORMED" / "product" / "S2A_MSIL2A_MALFORMED.SAFE.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    # Write a plain text file instead of a real zip
    with open(zip_path, "w") as f:
        f.write("this is not a zip file")
        
    success, msg = validator.validate_scene(scene)
    assert not success
    assert "not a valid ZIP file" in msg
    assert catalog.get_scene("malformed-uuid")["status"] == "failed"

def test_validate_missing_b02(temp_catalog_and_dirs):
    catalog, validator, output_dir = temp_catalog_and_dirs
    scene = {
        "scene_id": "S2_L2A_MISSING_B02",
        "product_id": "missing-b02-uuid",
        "product_name": "S2A_MSIL2A_MISSING_B02.SAFE"
    }
    catalog.add_scene(scene)
    
    zip_path = output_dir / "S2_L2A_MISSING_B02" / "product" / "S2A_MSIL2A_MISSING_B02.SAFE.zip"
    # Missing B02 band
    filenames = [
        "S2A_MSIL2A_MISSING_B02.SAFE/MTD_MSIL2A.xml",
        "S2A_MSIL2A_MISSING_B02.SAFE/GRANULE/L2A_T32TNS_A000000_20231010T102021/IMG_DATA/R10m/T32TNS_20231010T102021_B03_10m.jp2",
        "S2A_MSIL2A_MISSING_B02.SAFE/GRANULE/L2A_T32TNS_A000000_20231010T102021/IMG_DATA/R10m/T32TNS_20231010T102021_B04_10m.jp2",
        "S2A_MSIL2A_MISSING_B02.SAFE/GRANULE/L2A_T32TNS_A000000_20231010T102021/IMG_DATA/R10m/T32TNS_20231010T102021_B08_10m.jp2",
    ]
    create_dummy_zip(zip_path, filenames)
    
    success, msg = validator.validate_scene(scene)
    assert not success
    assert "Missing required bands" in msg
    assert "B02" in msg
    assert catalog.get_scene("missing-b02-uuid")["status"] == "failed"

def test_validate_missing_b03(temp_catalog_and_dirs):
    catalog, validator, output_dir = temp_catalog_and_dirs
    scene = {
        "scene_id": "S2_L2A_MISSING_B03",
        "product_id": "missing-b03-uuid",
        "product_name": "S2A_MSIL2A_MISSING_B03.SAFE"
    }
    catalog.add_scene(scene)
    
    zip_path = output_dir / "S2_L2A_MISSING_B03" / "product" / "S2A_MSIL2A_MISSING_B03.SAFE.zip"
    filenames = [
        "S2A_MSIL2A_MISSING_B03.SAFE/MTD_MSIL2A.xml",
        "S2A_MSIL2A_MISSING_B03.SAFE/GRANULE/L2A_T32TNS_A000000_20231010T102021/IMG_DATA/R10m/T32TNS_20231010T102021_B02_10m.jp2",
        "S2A_MSIL2A_MISSING_B03.SAFE/GRANULE/L2A_T32TNS_A000000_20231010T102021/IMG_DATA/R10m/T32TNS_20231010T102021_B04_10m.jp2",
        "S2A_MSIL2A_MISSING_B03.SAFE/GRANULE/L2A_T32TNS_A000000_20231010T102021/IMG_DATA/R10m/T32TNS_20231010T102021_B08_10m.jp2",
    ]
    create_dummy_zip(zip_path, filenames)
    
    success, msg = validator.validate_scene(scene)
    assert not success
    assert "Missing required bands" in msg
    assert "B03" in msg

def test_validate_missing_b04(temp_catalog_and_dirs):
    catalog, validator, output_dir = temp_catalog_and_dirs
    scene = {
        "scene_id": "S2_L2A_MISSING_B04",
        "product_id": "missing-b04-uuid",
        "product_name": "S2A_MSIL2A_MISSING_B04.SAFE"
    }
    catalog.add_scene(scene)
    
    zip_path = output_dir / "S2_L2A_MISSING_B04" / "product" / "S2A_MSIL2A_MISSING_B04.SAFE.zip"
    filenames = [
        "S2A_MSIL2A_MISSING_B04.SAFE/MTD_MSIL2A.xml",
        "S2A_MSIL2A_MISSING_B04.SAFE/GRANULE/L2A_T32TNS_A000000_20231010T102021/IMG_DATA/R10m/T32TNS_20231010T102021_B02_10m.jp2",
        "S2A_MSIL2A_MISSING_B04.SAFE/GRANULE/L2A_T32TNS_A000000_20231010T102021/IMG_DATA/R10m/T32TNS_20231010T102021_B03_10m.jp2",
        "S2A_MSIL2A_MISSING_B04.SAFE/GRANULE/L2A_T32TNS_A000000_20231010T102021/IMG_DATA/R10m/T32TNS_20231010T102021_B08_10m.jp2",
    ]
    create_dummy_zip(zip_path, filenames)
    
    success, msg = validator.validate_scene(scene)
    assert not success
    assert "Missing required bands" in msg
    assert "B04" in msg

def test_validate_missing_b08(temp_catalog_and_dirs):
    catalog, validator, output_dir = temp_catalog_and_dirs
    scene = {
        "scene_id": "S2_L2A_MISSING_B08",
        "product_id": "missing-b08-uuid",
        "product_name": "S2A_MSIL2A_MISSING_B08.SAFE"
    }
    catalog.add_scene(scene)
    
    zip_path = output_dir / "S2_L2A_MISSING_B08" / "product" / "S2A_MSIL2A_MISSING_B08.SAFE.zip"
    filenames = [
        "S2A_MSIL2A_MISSING_B08.SAFE/MTD_MSIL2A.xml",
        "S2A_MSIL2A_MISSING_B08.SAFE/GRANULE/L2A_T32TNS_A000000_20231010T102021/IMG_DATA/R10m/T32TNS_20231010T102021_B02_10m.jp2",
        "S2A_MSIL2A_MISSING_B08.SAFE/GRANULE/L2A_T32TNS_A000000_20231010T102021/IMG_DATA/R10m/T32TNS_20231010T102021_B03_10m.jp2",
        "S2A_MSIL2A_MISSING_B08.SAFE/GRANULE/L2A_T32TNS_A000000_20231010T102021/IMG_DATA/R10m/T32TNS_20231010T102021_B04_10m.jp2",
    ]
    create_dummy_zip(zip_path, filenames)
    
    success, msg = validator.validate_scene(scene)
    assert not success
    assert "Missing required bands" in msg
    assert "B08" in msg

def test_validate_non_l2a_product(temp_catalog_and_dirs):
    catalog, validator, output_dir = temp_catalog_and_dirs
    scene = {
        "scene_id": "S2_L1C_PRODUCT",
        "product_id": "l1c-product-uuid",
        "product_name": "S2A_MSIL1C_PRODUCT.SAFE"
    }
    catalog.add_scene(scene)
    
    zip_path = output_dir / "S2_L1C_PRODUCT" / "product" / "S2A_MSIL1C_PRODUCT.SAFE.zip"
    filenames = [
        "S2A_MSIL1C_PRODUCT.SAFE/MTD_MSIL1C.xml",
        "S2A_MSIL1C_PRODUCT.SAFE/GRANULE/L1C_T32TNS_A000000_20231010T102021/IMG_DATA/T32TNS_20231010T102021_B02.jp2",
    ]
    create_dummy_zip(zip_path, filenames)
    
    success, msg = validator.validate_scene(scene)
    assert not success
    assert "not appear to be a Sentinel-2 Level-2A product" in msg
