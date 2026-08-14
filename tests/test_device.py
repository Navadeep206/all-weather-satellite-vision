import torch
from unittest.mock import patch
from src.utils.device import get_device, get_device_info

def test_get_device_returns_torch_device():
    device = get_device()
    assert isinstance(device, torch.device)

def test_get_device_no_crash():
    try:
        get_device_info()
    except Exception as e:
        assert False, f"get_device_info() crashed with error: {e}"

def test_device_info_structure():
    info = get_device_info()
    assert "device" in info
    assert "pytorch_version" in info
    assert "cuda_available" in info
    assert "cuda_version" in info
    assert "gpu_name" in info
    assert "mps_available" in info

def test_cpu_fallback():
    # Mock cuda and mps availability to be False
    with patch("torch.cuda.is_available", return_value=False), \
         patch("torch.backends.mps.is_available", return_value=False):
        device = get_device()
        assert device.type == "cpu"
