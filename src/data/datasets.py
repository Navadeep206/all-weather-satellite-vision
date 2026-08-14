import torch
import rasterio
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, Any, Tuple

from src.data.manifest import read_manifest
from src.utils.config import get_project_root

class SatelliteStage1Dataset(Dataset):
    """PyTorch Dataset loading Stage 1 samples (degraded multispectral -> clean multispectral)."""
    
    def __init__(self, manifest_csv: Path, patch_size: int = 256):
        self.manifest_csv = Path(manifest_csv)
        self.patch_size = patch_size
        self.rows = read_manifest(self.manifest_csv)
        self.project_root = get_project_root()
        
    def __len__(self) -> int:
        return len(self.rows)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        row = self.rows[idx]
        
        y = int(row["patch_y"])
        x = int(row["patch_x"])
        window = rasterio.windows.Window(col_off=x, row_off=y, width=self.patch_size, height=self.patch_size)
        
        clean_path = self.project_root / row["clean_path"]
        deg_path = self.project_root / row["degraded_path"]
        
        # Read windows
        with rasterio.open(clean_path) as src_c, rasterio.open(deg_path) as src_d:
            clean_patch = src_c.read(window=window)
            deg_patch = src_d.read(window=window)
            
        # Replace NaNs with 0.0 (or keep them config-driven; standard ML datasets map NaNs to 0)
        clean_patch = np.nan_to_num(clean_patch, nan=0.0)
        deg_patch = np.nan_to_num(deg_patch, nan=0.0)
        
        # Convert to torch tensor float32
        x_tensor = torch.from_numpy(deg_patch).to(torch.float32)
        y_tensor = torch.from_numpy(clean_patch).to(torch.float32)
        
        metadata = {
            "sample_id": row["sample_id"],
            "scene_id": row["scene_id"],
            "split": row["split"],
            "patch_y": y,
            "patch_x": x,
            "haze_severity": row.get("haze_severity", "")
        }
        
        return x_tensor, y_tensor, metadata

class SatelliteStage2Dataset(Dataset):
    """PyTorch Dataset loading Stage 2 samples (degraded + mask -> clean)."""
    
    def __init__(self, manifest_csv: Path, patch_size: int = 256):
        self.manifest_csv = Path(manifest_csv)
        self.patch_size = patch_size
        self.rows = read_manifest(self.manifest_csv)
        self.project_root = get_project_root()
        
    def __len__(self) -> int:
        return len(self.rows)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        row = self.rows[idx]
        
        y = int(row["patch_y"])
        x = int(row["patch_x"])
        window = rasterio.windows.Window(col_off=x, row_off=y, width=self.patch_size, height=self.patch_size)
        
        clean_path = self.project_root / row["clean_path"]
        deg_path = self.project_root / row["degraded_path"]
        mask_path = self.project_root / row["mask_path"]
        
        # Read windows
        with rasterio.open(clean_path) as src_c, rasterio.open(deg_path) as src_d, rasterio.open(mask_path) as src_m:
            clean_patch = src_c.read(window=window)
            deg_patch = src_d.read(window=window)
            mask_patch = src_m.read(1, window=window)  # shape (H, W)
            
        # Add channel dim to mask -> (1, H, W)
        mask_patch = np.expand_dims(mask_patch, axis=0)
        
        # NaNs to 0
        clean_patch = np.nan_to_num(clean_patch, nan=0.0)
        deg_patch = np.nan_to_num(deg_patch, nan=0.0)
        
        x_tensor = torch.from_numpy(deg_patch).to(torch.float32)
        mask_tensor = torch.from_numpy(mask_patch).to(torch.float32)
        y_tensor = torch.from_numpy(clean_patch).to(torch.float32)
        
        metadata = {
            "sample_id": row["sample_id"],
            "scene_id": row["scene_id"],
            "split": row["split"],
            "patch_y": y,
            "patch_x": x,
            "mask_type": row.get("mask_type", ""),
            "occlusion_severity": row.get("occlusion_severity", "")
        }
        
        return x_tensor, mask_tensor, y_tensor, metadata

class SatelliteStage3Dataset(Dataset):
    """PyTorch Dataset loading Stage 3 samples (clean multispectral -> RGB)."""
    
    def __init__(self, manifest_csv: Path, patch_size: int = 256):
        self.manifest_csv = Path(manifest_csv)
        self.patch_size = patch_size
        self.rows = read_manifest(self.manifest_csv)
        self.project_root = get_project_root()
        
    def __len__(self) -> int:
        return len(self.rows)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        row = self.rows[idx]
        
        y = int(row["patch_y"])
        x = int(row["patch_x"])
        window = rasterio.windows.Window(col_off=x, row_off=y, width=self.patch_size, height=self.patch_size)
        
        clean_path = self.project_root / row["clean_path"]
        rgb_path = self.project_root / row["rgb_path"]
        
        with rasterio.open(clean_path) as src_c, rasterio.open(rgb_path) as src_rgb:
            clean_patch = src_c.read(window=window)
            rgb_patch = src_rgb.read(window=window)
            
        clean_patch = np.nan_to_num(clean_patch, nan=0.0)
        rgb_patch = np.nan_to_num(rgb_patch, nan=0.0)
        
        x_tensor = torch.from_numpy(clean_patch).to(torch.float32)
        y_tensor = torch.from_numpy(rgb_patch).to(torch.float32)
        
        metadata = {
            "sample_id": row["sample_id"],
            "scene_id": row["scene_id"],
            "split": row["split"],
            "patch_y": y,
            "patch_x": x
        }
        
        return x_tensor, y_tensor, metadata
