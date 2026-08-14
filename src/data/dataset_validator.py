import rasterio
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
from src.data.manifest import read_manifest
from src.utils.config import get_project_root
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

def check_data_leakage(dataset_dir: Path) -> Tuple[bool, str]:
    """Verifies that there is no geographic scene or sample-level leakage across splits.
    
    Args:
        dataset_dir (Path): Base directory containing splits and manifests.
        
    Returns:
        Tuple[bool, str]: (is_leakage_free, detailed_report)
    """
    manifests_dir = dataset_dir / "manifests"
    
    # Read all manifests
    splits = ["train", "val", "test"]
    stages = ["stage1", "stage2", "stage3"]
    
    scene_splits = {}
    sample_splits = {}
    
    leakage_detected = False
    details = []
    
    for split in splits:
        for stage in stages:
            csv_path = manifests_dir / f"{stage}_{split}.csv"
            if not csv_path.exists():
                continue
                
            rows = read_manifest(csv_path)
            for r in rows:
                sid = r["scene_id"]
                samp_id = r["sample_id"]
                
                # Check scene leakage
                if sid in scene_splits and scene_splits[sid] != split:
                    leakage_detected = True
                    details.append(f"SCENE LEAKAGE: Scene {sid} is assigned to split {scene_splits[sid]} and split {split}.")
                else:
                    scene_splits[sid] = split
                    
                # Check sample leakage
                if samp_id in sample_splits and sample_splits[samp_id] != split:
                    leakage_detected = True
                    details.append(f"SAMPLE LEAKAGE: Sample {samp_id} is assigned to split {sample_splits[samp_id]} and split {split}.")
                else:
                    sample_splits[samp_id] = split
                    
    # Also verify scene split text files match manifests
    splits_dir = dataset_dir / "splits"
    for split in splits:
        split_file = splits_dir / f"{split}.txt"
        if split_file.exists():
            with open(split_file, "r") as f:
                scene_list = [line.strip() for line in f if line.strip()]
            for sid in scene_list:
                if sid in scene_splits and scene_splits[sid] != split:
                    leakage_detected = True
                    details.append(f"SPLIT FILE LEAKAGE: Scene {sid} in {split}.txt is mapped to {scene_splits[sid]} in manifests.")
                    
    if leakage_detected:
        report = "DATA LEAKAGE: FAIL\n" + "\n".join(details)
        return False, report
    else:
        return True, "DATA LEAKAGE: PASS (No overlapping scene or sample assignments found)."

def validate_dataset_files(
    dataset_dir: Path,
    patch_size: int = 256
) -> Tuple[bool, List[str]]:
    """Checks that every file referenced in the manifests exists, is readable, and aligns correctly.
    
    Args:
        dataset_dir (Path): Base directory containing splits and manifests.
        patch_size (int): Expected height/width of training patches (e.g. 256).
        
    Returns:
        Tuple[bool, List[str]]: (is_valid, list_of_errors)
    """
    manifests_dir = dataset_dir / "manifests"
    errors = []
    
    splits = ["train", "val", "test"]
    stages = ["stage1", "stage2", "stage3"]
    
    project_root = get_project_root()
    
    for split in splits:
        for stage in stages:
            csv_path = manifests_dir / f"{stage}_{split}.csv"
            if not csv_path.exists():
                continue
                
            rows = read_manifest(csv_path)
            logger.info(f"Validating file integrity for {csv_path.name} ({len(rows)} samples)...")
            
            for idx, r in enumerate(rows):
                samp_id = r["sample_id"]
                y = r.get("patch_y")
                x = r.get("patch_x")
                
                # Check common relative clean file path
                clean_path = project_root / r["clean_path"]
                if not clean_path.exists():
                    errors.append(f"{samp_id}: Clean source {clean_path} does not exist.")
                    continue
                    
                # 1. Validate Stage 1 (degraded multispectral, clean multispectral)
                if stage == "stage1":
                    deg_path = project_root / r["degraded_path"]
                    if not deg_path.exists():
                        errors.append(f"{samp_id}: Degraded file {deg_path} does not exist.")
                        continue
                        
                    # Verify bands & shapes
                    try:
                        with rasterio.open(clean_path) as cln, rasterio.open(deg_path) as deg:
                            if cln.count != 4 or deg.count != 4:
                                errors.append(f"{samp_id}: Expected 4 bands for multispectral rasters; got clean={cln.count}, degraded={deg.count}")
                            if cln.width != deg.width or cln.height != deg.height:
                                errors.append(f"{samp_id}: Dimension mismatch between clean and degraded rasters.")
                            # Check window fits inside dimensions
                            if y + patch_size > cln.height or x + patch_size > cln.width:
                                errors.append(f"{samp_id}: Patch window offsets ({y}, {x}) exceed raster dimensions ({cln.height}, {cln.width})")
                    except Exception as e:
                        errors.append(f"{samp_id}: Failed to open/verify rasters: {e}")
                        
                # 2. Validate Stage 2 (degraded multispectral, clean multispectral, mask)
                elif stage == "stage2":
                    deg_path = project_root / r["degraded_path"]
                    mask_path = project_root / r["mask_path"]
                    if not deg_path.exists():
                        errors.append(f"{samp_id}: Degraded file {deg_path} does not exist.")
                        continue
                    if not mask_path.exists():
                        errors.append(f"{samp_id}: Mask file {mask_path} does not exist.")
                        continue
                        
                    try:
                        with rasterio.open(clean_path) as cln, rasterio.open(deg_path) as deg, rasterio.open(mask_path) as msk:
                            if cln.count != 4 or deg.count != 4 or msk.count != 1:
                                errors.append(f"{samp_id}: Band mismatch; expected clean=4, deg=4, mask=1; got {cln.count}/{deg.count}/{msk.count}")
                            if cln.width != msk.width or cln.height != msk.height:
                                errors.append(f"{samp_id}: Dimension mismatch between clean and mask rasters.")
                                
                            # Read window of mask to verify binary nature
                            win = rasterio.windows.Window(col_off=x, row_off=y, width=patch_size, height=patch_size)
                            mask_patch = msk.read(1, window=win)
                            if not np.all((mask_patch == 0) | (mask_patch == 1)):
                                errors.append(f"{samp_id}: Mask contains non-binary values.")
                    except Exception as e:
                        errors.append(f"{samp_id}: Failed to verify Stage 2 files: {e}")
                        
                # 3. Validate Stage 3 (clean multispectral, RGB)
                elif stage == "stage3":
                    rgb_path = project_root / r["rgb_path"]
                    if not rgb_path.exists():
                        errors.append(f"{samp_id}: RGB file {rgb_path} does not exist.")
                        continue
                        
                    try:
                        with rasterio.open(clean_path) as cln, rasterio.open(rgb_path) as rgb:
                            if cln.count != 4 or rgb.count != 3:
                                errors.append(f"{samp_id}: Band mismatch; expected clean=4, rgb=3; got {cln.count}/{rgb.count}")
                            # Spatial coordinate check
                            if not cln.crs == rgb.crs:
                                errors.append(f"{samp_id}: CRS mismatch between clean multispectral and RGB target.")
                            # Transform check
                            if not np.allclose(cln.transform, rgb.transform):
                                errors.append(f"{samp_id}: Geotransform alignment mismatch between clean and RGB target.")
                    except Exception as e:
                        errors.append(f"{samp_id}: Failed to verify Stage 3 files: {e}")
                        
    return (len(errors) == 0, errors)
