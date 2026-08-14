import torch
from pathlib import Path
from typing import Any, Dict, Optional, Union
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

def save_checkpoint(
    state: Dict[str, Any],
    filepath: Union[str, Path]
) -> Path:
    """Saves a training checkpoint containing model, optimizer, scheduler state and metadata.
    
    Args:
        state (dict): Dictionary containing keys like 'model_state_dict', 
                      'optimizer_state_dict', 'scheduler_state_dict', 'epoch',
                      'best_metric', 'config', 'seed', etc.
        filepath (str or Path): Path where the checkpoint will be saved.
        
    Returns:
        Path: The absolute path to the saved checkpoint file.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save checkpoint using PyTorch standard serialization
    torch.save(state, path)
    logger.info(f"Checkpoint successfully saved to {path}")
    return path

def load_checkpoint(
    filepath: Union[str, Path],
    map_location: Optional[Union[str, torch.device]] = None
) -> Dict[str, Any]:
    """Loads a training checkpoint from the specified path.
    
    Args:
        filepath (str or Path): Path to the checkpoint file.
        map_location (str or torch.device, optional): Device to map the loaded tensors to.
                      If None, maps to the current default device.
                      
    Returns:
        dict: The loaded state dictionary.
        
    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
    """
    path = Path(filepath)
    if not path.exists():
        logger.error(f"Checkpoint file not found at: {path}")
        raise FileNotFoundError(f"Checkpoint file not found at: {path}")
        
    logger.info(f"Loading checkpoint from {path} (map_location={map_location})")
    state = torch.load(path, map_location=map_location)
    return state
