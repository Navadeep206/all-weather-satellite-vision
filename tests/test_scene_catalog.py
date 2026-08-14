import pytest
from pathlib import Path
from src.data.scene_catalog import SceneCatalog

@pytest.fixture
def temp_catalog_dir(tmp_path):
    return tmp_path / "metadata"

def test_add_scene(temp_catalog_dir):
    catalog = SceneCatalog(metadata_dir=temp_catalog_dir)
    
    scene = {
        "scene_id": "S2_L2A_20231010_T32TNS_abcdefgh",
        "product_id": "abcdefgh-1234-5678-abcd-efghijklmnop",
        "product_name": "S2B_MSIL2A_20231010T102021_N0509_R065_T32TNS_20231010T121021.SAFE",
        "processing_level": "L2A",
        "platform": "Sentinel-2B",
        "sensing_datetime": "2023-10-10T10:20:21.000Z",
        "tile_id": "32TNS",
        "cloud_cover": 4.5,
        "source": "CDSE",
        "status": "discovered"
    }
    
    catalog.add_scene(scene)
    assert catalog.has_scene(scene["product_id"])
    
    loaded_scene = catalog.get_scene(scene["product_id"])
    assert loaded_scene["scene_id"] == scene["scene_id"]
    assert loaded_scene["status"] == "discovered"

def test_duplicate_scene_prevention(temp_catalog_dir):
    catalog = SceneCatalog(metadata_dir=temp_catalog_dir)
    
    product_id = "abcdefgh-1234-5678-abcd-efghijklmnop"
    scene1 = {
        "scene_id": "S2_L2A_20231010_T32TNS_abcdefgh",
        "product_id": product_id,
        "product_name": "S2B_MSIL2A_20231010T102021_N0509_R065_T32TNS_20231010T121021.SAFE",
        "status": "discovered",
        "cloud_cover": 4.5
    }
    
    scene2 = {
        "scene_id": "S2_L2A_20231010_T32TNS_abcdefgh",
        "product_id": product_id,
        "product_name": "S2B_MSIL2A_20231010T102021_N0509_R065_T32TNS_20231010T121021.SAFE",
        "status": "downloaded", # status changed
        "cloud_cover": 4.5
    }
    
    catalog.add_scene(scene1)
    catalog.add_scene(scene2)
    
    # Ensure there is only 1 entry in the catalog (prevented duplicates)
    assert len(catalog.list_scenes()) == 1
    # Ensure properties got updated/merged correctly
    assert catalog.get_scene(product_id)["status"] == "downloaded"

def test_status_update(temp_catalog_dir):
    catalog = SceneCatalog(metadata_dir=temp_catalog_dir)
    product_id = "abcdefgh-1234-5678-abcd-efghijklmnop"
    
    scene = {
        "scene_id": "S2_L2A_20231010_T32TNS_abcdefgh",
        "product_id": product_id,
        "product_name": "S2B_MSIL2A_20231010T102021_N0509_R065_T32TNS_20231010T121021.SAFE",
        "status": "discovered"
    }
    
    catalog.add_scene(scene)
    catalog.update_status(product_id, "validated")
    
    assert catalog.get_scene(product_id)["status"] == "validated"

def test_catalog_persistence(temp_catalog_dir):
    catalog = SceneCatalog(metadata_dir=temp_catalog_dir)
    product_id = "abcdefgh-1234-5678-abcd-efghijklmnop"
    
    scene = {
        "scene_id": "S2_L2A_20231010_T32TNS_abcdefgh",
        "product_id": product_id,
        "product_name": "S2B_MSIL2A_20231010T102021_N0509_R065_T32TNS_20231010T121021.SAFE",
        "status": "discovered",
        "cloud_cover": 4.5
    }
    
    catalog.add_scene(scene)
    
    # Re-initialize catalog pointing to same folder
    new_catalog = SceneCatalog(metadata_dir=temp_catalog_dir)
    assert new_catalog.has_scene(product_id)
    assert new_catalog.get_scene(product_id)["scene_id"] == "S2_L2A_20231010_T32TNS_abcdefgh"
    assert new_catalog.get_scene(product_id)["cloud_cover"] == 4.5
