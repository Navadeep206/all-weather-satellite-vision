import pytest
from src.utils.config import load_config

def test_load_acquisition_config():
    config = load_config("configs/acquisition.yaml")
    
    # Check key sections exist
    assert "dataset" in config
    assert "bands" in config
    assert "search" in config
    assert "download" in config
    assert "limits" in config
    
    # Check specific default values
    assert config["dataset"]["collection"] == "Sentinel-2"
    assert config["dataset"]["processing_level"] == "L2A"
    
    assert "B02" in config["bands"]
    assert "B03" in config["bands"]
    assert "B04" in config["bands"]
    assert "B08" in config["bands"]
    
    assert config["search"]["max_cloud_cover"] == 10
    assert config["download"]["output_directory"] == "data/raw/sentinel2"
    assert config["limits"]["smoke_test_scenes"] == 3
