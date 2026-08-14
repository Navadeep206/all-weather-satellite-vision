import yaml
from pathlib import Path
from typing import Any, Dict

def get_project_root() -> Path:
    """Finds and returns the absolute path to the project root directory.
    
    Traverses upwards from the location of this file until a root-indicative
    file (like pyproject.toml or requirements.txt) is found, with a fallback
    to the direct parent folder offset.
    
    Returns:
        Path: The absolute path to the project root directory.
    """
    current_path = Path(__file__).resolve()
    # Check parents upwards
    for parent in [current_path] + list(current_path.parents):
        if (parent / "pyproject.toml").exists() or (parent / "requirements.txt").exists() or (parent / ".git").exists():
            return parent
    
    # Fallback to the 2nd parent of current file's directory: src/utils/config.py -> parents[2] is root
    return current_path.parents[2]

def load_config(path: str | Path) -> Dict[str, Any]:
    """Loads and validates a YAML configuration file.
    
    If a relative path is passed, it is resolved relative to the project root.
    
    Args:
        path (str or Path): Path to the YAML configuration file.
        
    Returns:
        dict: The loaded configuration dictionary.
        
    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If the file is not valid YAML.
    """
    path_obj = Path(path)
    if not path_obj.is_absolute():
        path_obj = get_project_root() / path_obj
        
    if not path_obj.exists():
        raise FileNotFoundError(f"Configuration file not found at: {path_obj}")
        
    try:
        with open(path_obj, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            if config is None:
                return {}
            if not isinstance(config, dict):
                raise ValueError(f"Configuration at {path_obj} must be a key-value mapping.")
            return config
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse YAML configuration file at {path_obj}: {e}")
