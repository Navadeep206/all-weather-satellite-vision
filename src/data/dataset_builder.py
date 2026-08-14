import json
import time
import numpy as np
import rasterio
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

from src.data.scene_catalog import SceneCatalog
from src.data.dataset_split import create_scene_split, write_split_files, save_split_metadata
from src.data.manifest import write_manifest
from src.data.degradation import generate_sample
from src.data.degradation_cli import save_degraded_sample
from src.utils.config import load_config, get_project_root
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

def generate_patch_indices(
    clean_path: Path,
    patch_size: int = 256,
    overlap: int = 0,
    min_valid_fraction: float = 0.90
) -> List[Tuple[int, int]]:
    """Scans the raster and determines grid offsets for valid 256x256 patches.
    
    Discards boundary patches smaller than patch_size, and filters patches 
    with too many nodata (NaN) values.
    
    Args:
        clean_path (Path): Path to the processed clean GeoTIFF.
        patch_size (int): Dimensions of the patch.
        overlap (int): Overlap pixels between adjacent patches.
        min_valid_fraction (float): Minimum proportion of non-NaN values required.
        
    Returns:
        List[Tuple[int, int]]: List of (y_offset, x_offset) coordinate pairs.
    """
    if not clean_path.exists():
        logger.warning(f"File not found for patch indexing: {clean_path}")
        return []
        
    patch_coords = []
    step = patch_size - overlap
    
    with rasterio.open(clean_path) as src:
        h, w = src.height, src.width
        
        for y in range(0, h, step):
            for x in range(0, w, step):
                # Discard patches extending past boundaries
                if y + patch_size > h or x + patch_size > w:
                    continue
                    
                # Read window to check valid pixels
                window = rasterio.windows.Window(col_off=x, row_off=y, width=patch_size, height=patch_size)
                # Read band 1 (or check all, B02 is band 1)
                band = src.read(1, window=window)
                
                # Check valid fraction (nodata is np.nan)
                valid_count = np.sum(~np.isnan(band))
                valid_fraction = valid_count / band.size
                
                if valid_fraction >= min_valid_fraction:
                    patch_coords.append((y, x))
                    
    return patch_coords

def build_datasets(
    processed_dir: Path,
    dataset_dir: Path,
    catalog: SceneCatalog,
    config: Dict[str, Any],
    degradation_config: Dict[str, Any],
    max_scenes: Optional[int] = None,
    force: bool = False,
    dry_run: bool = False
) -> Dict[str, Any]:
    """Coordinates splitting, synthetic generation, patch indexing, and manifest creation.
    
    Args:
        processed_dir (Path): Source processed scene directory.
        dataset_dir (Path): Target dataset directory.
        catalog (SceneCatalog): Local catalogue tracker.
        config (dict): Dataset configuration dictionary.
        degradation_config (dict): Degradation config.
        max_scenes (int, optional): Limit on input scenes.
        force (bool): Re-run even if already built.
        dry_run (bool): Dry-run only.
        
    Returns:
        Dict[str, Any]: Generation summary metrics.
    """
    logger.info("Initializing Dataset Builder...")
    
    # 1. Discover processed scenes
    scenes = []
    for s in catalog.list_scenes():
        # Check if the scene is marked as processed and directories exist
        scene_id = s["scene_id"]
        ms_path = processed_dir / scene_id / "multispectral.tif"
        rgb_path = processed_dir / scene_id / "rgb.tif"
        if s.get("status") == "processed" and ms_path.exists() and rgb_path.exists():
            scenes.append(scene_id)
            
    logger.info(f"Discovered {len(scenes)} eligible processed scenes.")
    
    if max_scenes:
        scenes = sorted(scenes)[:max_scenes]
        logger.info(f"Capping input scenes to {len(scenes)} by max_scenes parameter.")
        
    if not scenes:
        logger.warning("No eligible processed scenes found to build dataset. Aborting.")
        return {}

    # 2. Partition scenes deterministically
    seed = config["dataset"]["seed"]
    split_cfg = config["split"]
    train_scenes, val_scenes, test_scenes = create_scene_split(
        scenes=scenes,
        train_ratio=split_cfg["train_ratio"],
        val_ratio=split_cfg["val_ratio"],
        test_ratio=split_cfg["test_ratio"],
        seed=seed
    )
    
    split_map = {}
    for s in train_scenes: split_map[s] = "train"
    for s in val_scenes: split_map[s] = "val"
    for s in test_scenes: split_map[s] = "test"
    
    if dry_run:
        logger.info("[DRY-RUN] Dataset build simulation complete.")
        return {
            "train_scenes": train_scenes,
            "val_scenes": val_scenes,
            "test_scenes": test_scenes,
            "expected_scenes_count": len(scenes)
        }
        
    # Write split lists and split metadata
    splits_dir = dataset_dir / "splits"
    write_split_files(splits_dir, train_scenes, val_scenes, test_scenes)
    save_split_metadata(dataset_dir / "metadata", train_scenes, val_scenes, test_scenes, seed, config["dataset"]["version"])

    # 3. Setup folders
    degraded_dir = get_project_root() / degradation_config["degradation"]["degraded_directory"]
    degraded_dir.mkdir(parents=True, exist_ok=True)
    
    patch_size = config["patch"]["size"]
    overlap = config["patch"]["overlap"]
    min_valid_frac = config["patch"]["minimum_valid_fraction"]
    
    stage1_variants = config["generation"]["stage1_variants_per_scene"]
    stage2_variants = config["generation"]["stage2_variants_per_scene"]
    
    # Manifest row storage
    stage1_rows = []
    stage2_rows = []
    stage3_rows = []
    
    # Setup NumPy random generator for variant parameter choices
    rng = np.random.default_rng(seed)
    
    # 4. Generate variants & build manifest indexes
    for scene_id in scenes:
        split = split_map[scene_id]
        logger.info(f"Processing scene {scene_id} ({split})...")
        
        # Load clean multispectral raster to get profile & image array
        scene_proc_dir = processed_dir / scene_id
        ms_path = scene_proc_dir / "multispectral.tif"
        rgb_path = scene_proc_dir / "rgb.tif"
        
        with rasterio.open(ms_path) as src:
            clean_arr = src.read()
            profile = src.profile.copy()
            
        # Determine valid patch coordinates (the spatial index)
        patch_coords = generate_patch_indices(
            ms_path,
            patch_size=patch_size,
            overlap=overlap,
            min_valid_fraction=min_valid_frac
        )
        logger.info(f"Found {len(patch_coords)} valid patch(es) in clean scene {scene_id}")
        
        # Paths relative to project root
        clean_rel = str(ms_path.relative_to(get_project_root()))
        rgb_rel = str(rgb_path.relative_to(get_project_root()))
        
        # --- Stage 3: Clean Multispectral -> RGB (No degradations needed) ---
        for y, x in patch_coords:
            sample_id = f"SMP_{scene_id}_stage3_p{y}_{x}"
            stage3_rows.append({
                "sample_id": sample_id,
                "scene_id": scene_id,
                "split": split,
                "clean_path": clean_rel,
                "rgb_path": rgb_rel,
                "patch_y": y,
                "patch_x": x,
                "degradation_type": "none"
            })
            
        # --- Stage 1: Degraded (Haze) -> Clean ---
        haze_severities = ["low", "medium", "high", "extreme"]
        for v_idx in range(stage1_variants):
            v_seed = seed + hash(scene_id) % 10000 + v_idx * 10
            haze_rng = np.random.default_rng(v_seed)
            severity = haze_rng.choice(haze_severities)
            
            sample_name_id = f"SMP_{scene_id}_haze_{severity}_{v_seed}"
            output_sample_dir = degraded_dir / "haze" / sample_name_id
            degraded_rel = str((output_sample_dir / "degraded.tif").relative_to(get_project_root()))
            
            # Generate the degraded tif file on disk (using Phase 3 solver)
            if not (output_sample_dir / "degraded.tif").exists() or force:
                degraded_arr, _, metadata = generate_sample(
                    clean_image=clean_arr,
                    config=degradation_config,
                    seed=v_seed,
                    haze_severity=severity,
                    occlusion_severity=None
                )
                save_degraded_sample(
                    sample_id=sample_name_id,
                    output_dir=output_sample_dir,
                    clean_arr=clean_arr,
                    degraded_arr=degraded_arr,
                    mask_arr=None,
                    metadata=metadata,
                    profile=profile
                )
                
            # Add patch rows referencing this degraded file
            for y, x in patch_coords:
                patch_id = f"SMP_{scene_id}_stage1_v{v_idx}_p{y}_{x}"
                stage1_rows.append({
                    "sample_id": patch_id,
                    "scene_id": scene_id,
                    "split": split,
                    "clean_path": clean_rel,
                    "degraded_path": degraded_rel,
                    "patch_y": y,
                    "patch_x": x,
                    "seed": v_seed,
                    "degradation_type": "haze",
                    "haze_severity": severity
                })
                
        # --- Stage 2: Degraded (Occlusion/Haze) -> Clean ---
        # Stage 2 requires a mask. We can apply combined or occlusion-only
        occlusion_severities = ["low", "medium", "high", "extreme"]
        mask_types = ["cloud_like", "irregular", "rectangular"]
        
        for v_idx in range(stage2_variants):
            v_seed = seed + hash(scene_id) % 10000 + v_idx * 10 + 500
            stage2_rng = np.random.default_rng(v_seed)
            
            h_severity = stage2_rng.choice(haze_severities + ["none"])
            o_severity = stage2_rng.choice(occlusion_severities)
            m_type = stage2_rng.choice(mask_types)
            
            deg_type = "combined" if h_severity != "none" else "occlusion"
            sample_name_id = f"SMP_{scene_id}_{deg_type}_{o_severity}_{v_seed}"
            output_sample_dir = degraded_dir / deg_type / sample_name_id
            
            degraded_rel = str((output_sample_dir / "degraded.tif").relative_to(get_project_root()))
            mask_rel = str((output_sample_dir / "mask.tif").relative_to(get_project_root()))
            
            # Generate files
            if not (output_sample_dir / "degraded.tif").exists() or not (output_sample_dir / "mask.tif").exists() or force:
                degraded_arr, mask_arr, metadata = generate_sample(
                    clean_image=clean_arr,
                    config=degradation_config,
                    seed=v_seed,
                    haze_severity=h_severity,
                    occlusion_severity=o_severity,
                    mask_type=m_type
                )
                save_degraded_sample(
                    sample_id=sample_name_id,
                    output_dir=output_sample_dir,
                    clean_arr=clean_arr,
                    degraded_arr=degraded_arr,
                    mask_arr=mask_arr,
                    metadata=metadata,
                    profile=profile
                )
                
            # Add patch rows
            for y, x in patch_coords:
                patch_id = f"SMP_{scene_id}_stage2_v{v_idx}_p{y}_{x}"
                stage2_rows.append({
                    "sample_id": patch_id,
                    "scene_id": scene_id,
                    "split": split,
                    "clean_path": clean_rel,
                    "degraded_path": degraded_rel,
                    "mask_path": mask_rel,
                    "patch_y": y,
                    "patch_x": x,
                    "seed": v_seed,
                    "degradation_type": deg_type,
                    "haze_severity": h_severity,
                    "occlusion_severity": o_severity,
                    "mask_type": m_type
                })

    # 5. Write CSV manifests split-wise (train, val, test)
    manifests_dir = dataset_dir / "manifests"
    
    for split_name in ["train", "val", "test"]:
        # Stage 1
        s1_split = [r for r in stage1_rows if r["split"] == split_name]
        write_manifest(manifests_dir / f"stage1_{split_name}.csv", s1_split)
        
        # Stage 2
        s2_split = [r for r in stage2_rows if r["split"] == split_name]
        write_manifest(manifests_dir / f"stage2_{split_name}.csv", s2_split)
        
        # Stage 3
        s3_split = [r for r in stage3_rows if r["split"] == split_name]
        write_manifest(manifests_dir / f"stage3_{split_name}.csv", s3_split)

    # 6. Save metadata logs
    meta_dir = dataset_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    
    dataset_metadata = {
        "dataset_version": config["dataset"]["version"],
        "source_data_version": "v1",
        "split_seed": seed,
        "channel_configuration": ["B02", "B03", "B04", "B08"],
        "patch_size": patch_size,
        "overlap": overlap,
        "minimum_valid_fraction": min_valid_frac,
        "ratios": split_cfg,
        "scene_counts": {
            "total": len(scenes),
            "train": len(train_scenes),
            "val": len(val_scenes),
            "test": len(test_scenes)
        },
        "sample_counts": {
            "stage1": len(stage1_rows),
            "stage2": len(stage2_rows),
            "stage3": len(stage3_rows)
        },
        "generation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    
    with open(meta_dir / "dataset_metadata.json", "w", encoding="utf-8") as f:
        json.dump(dataset_metadata, f, indent=2)
        
    # Generate balanced summary
    stage1_haze_stats = {}
    for r in stage1_rows:
        sev = r["haze_severity"]
        stage1_haze_stats[sev] = stage1_haze_stats.get(sev, 0) + 1
        
    stage2_type_stats = {}
    stage2_sev_stats = {}
    for r in stage2_rows:
        mtype = r["mask_type"]
        osev = r["occlusion_severity"]
        stage2_type_stats[mtype] = stage2_type_stats.get(mtype, 0) + 1
        stage2_sev_stats[osev] = stage2_sev_stats.get(osev, 0) + 1
        
    generation_summary = {
        "stage1_haze_distribution": stage1_haze_stats,
        "stage2_mask_distribution": stage2_type_stats,
        "stage2_occlusion_distribution": stage2_sev_stats,
        "date_generated": time.strftime("%Y-%m-%d", time.gmtime())
    }
    
    with open(meta_dir / "generation_summary.json", "w", encoding="utf-8") as f:
        json.dump(generation_summary, f, indent=2)
        
    logger.info("Dataset construction complete.")
    
    return dataset_metadata
