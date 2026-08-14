import torch
from typing import Any, Dict

def get_device() -> torch.device:
    """Returns the best available torch.device.
    
    Priority:
    1. CUDA
    2. MPS (Apple Silicon GPU acceleration)
    3. CPU fallback
    
    Returns:
        torch.device: The resolved device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def get_device_info() -> Dict[str, Any]:
    """Retrieves metadata about PyTorch and execution hardware.
    
    Returns:
        dict: Device information metrics, including torch version, CUDA, and MPS status.
    """
    device = get_device()
    cuda_available = torch.cuda.is_available()
    mps_available = torch.backends.mps.is_available()
    
    info = {
        "device": str(device),
        "pytorch_version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda if cuda_available else None,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "mps_available": mps_available,
    }
    return info
