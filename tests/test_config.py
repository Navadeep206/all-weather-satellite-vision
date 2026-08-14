import pytest
from pathlib import Path
from src.utils.config import load_config, get_project_root

def test_get_project_root():
    root = get_project_root()
    assert root.exists()
    assert (root / "pyproject.toml").exists()

def test_load_project_yaml():
    config = load_config("configs/project.yaml")
    assert config["project_name"] == "all-weather-satellite-vision"
    assert config["framework"] == "pytorch"
    assert config["seed"] == 42

def test_load_data_yaml():
    config = load_config("configs/data.yaml")
    assert config["input_channels"] == 4
    assert config["target_channels"] == 3
    assert len(config["input_bands"]) == 4

def test_load_missing_file_raises_error():
    with pytest.raises(FileNotFoundError):
        load_config("configs/non_existent_file.yaml")

def test_load_invalid_yaml_raises_error(tmp_path):
    invalid_yaml_file = tmp_path / "invalid.yaml"
    with open(invalid_yaml_file, "w") as f:
        # Write invalid YAML syntax
        f.write("invalid_key: : value")
        
    with pytest.raises(ValueError) as excinfo:
        load_config(invalid_yaml_file)
    assert "Failed to parse YAML" in str(excinfo.value)
