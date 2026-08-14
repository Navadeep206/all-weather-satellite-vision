import sys
import argparse
import json
import time
import numpy as np
import rasterio
from pathlib import Path
from typing import Dict, Any, Optional

from src.data.scene_catalog import SceneCatalog
from src.data.degradation import generate_sample
from src.utils.config import load_config, get_project_root
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

def save_degraded_sample(
    sample_id: str,
    output_dir: Path,
    clean_arr: np.ndarray,
    degraded_arr: np.ndarray,
    mask_arr: Optional[np.ndarray],
    metadata: Dict[str, Any],
    profile: Dict[str, Any]
) -> None:
    """Writes clean, degraded, mask rasters and metadata JSON to disk, preserving geospatial data.
    
    Args:
        sample_id (str): The unique identifier for this sample.
        output_dir (Path): Base output directory (e.g. data/degraded/combined/sample_id/).
        clean_arr (np.ndarray): Clean image array (C, H, W).
        degraded_arr (np.ndarray): Degraded image array (C, H, W).
        mask_arr (np.ndarray, optional): Binary mask array (H, W).
        metadata (dict): Processing metadata dictionary.
        profile (dict): Geotiff writer profile from source.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    clean_path = output_dir / "clean.tif"
    degraded_path = output_dir / "degraded.tif"
    mask_path = output_dir / "mask.tif"
    meta_path = output_dir / "metadata.json"
    
    # 1. Save clean.tif (multispectral)
    ms_profile = profile.copy()
    ms_profile.update({
        "dtype": "float32",
        "count": clean_arr.shape[0],
        "driver": "GTiff",
        "nodata": np.nan
    })
    
    with rasterio.open(clean_path, "w", **ms_profile) as dst:
        for i in range(clean_arr.shape[0]):
            dst.write(clean_arr[i], i + 1)
            
    # 2. Save degraded.tif (multispectral)
    with rasterio.open(degraded_path, "w", **ms_profile) as dst:
        for i in range(degraded_arr.shape[0]):
            dst.write(degraded_arr[i], i + 1)
            
    # 3. Save mask.tif (1-band uint8) if occlusion occurred
    if mask_arr is not None:
        mask_profile = profile.copy()
        mask_profile.update({
            "dtype": "uint8",
            "count": 1,
            "driver": "GTiff",
            "nodata": 0
        })
        with rasterio.open(mask_path, "w", **mask_profile) as dst:
            dst.write(mask_arr, 1)
            
    # 4. Save metadata.json
    metadata["sample_id"] = sample_id
    metadata["processed_timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    metadata["processing_version"] = "phase3_v1"
    
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        
    logger.info(f"Sample {sample_id} saved to {output_dir}")
    
    # 5. Save visual preview under data/samples/phase3/
    deg_type = "combined" if len(metadata.get("degradations", [])) > 1 else (metadata.get("degradations", ["haze"])[0] if metadata.get("degradations") else "haze")
    save_degradation_visualization(sample_id, clean_arr, degraded_arr, mask_arr, deg_type)

def save_degradation_visualization(
    sample_id: str,
    clean_arr: np.ndarray,
    degraded_arr: np.ndarray,
    mask_arr: Optional[np.ndarray],
    deg_type: str
) -> None:
    """Creates and saves a visual debugging figure under data/samples/phase3/."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib is not available. Skipping visual preview generation.")
        return
        
    try:
        samples_dir = get_project_root() / "data" / "samples" / "phase3"
        samples_dir.mkdir(parents=True, exist_ok=True)
        out_path = samples_dir / f"{sample_id}_preview.png"
        
        # Extract RGB (B04, B03, B02) - index 2, 1, 0
        def to_rgb(arr):
            rgb = arr[[2, 1, 0]]
            img_clean = np.nan_to_num(rgb, nan=0.0)
            p2, p98 = np.percentile(img_clean, (2, 98), axis=(1, 2), keepdims=True)
            diff = p98 - p2
            diff[diff == 0] = 1.0
            stretched = np.clip((img_clean - p2) / diff, 0.0, 1.0)
            return np.moveaxis(stretched, 0, -1)
            
        clean_rgb = to_rgb(clean_arr)
        degraded_rgb = to_rgb(degraded_arr)
        
        if deg_type == "combined":
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(clean_rgb)
            axes[0].set_title("1. Clean RGB")
            axes[0].axis("off")
            
            axes[1].imshow(degraded_rgb)
            axes[1].set_title("2. Degraded RGB (Haze & Occluded)")
            axes[1].axis("off")
            
            if mask_arr is not None:
                axes[2].imshow(mask_arr, cmap="gray")
                axes[2].set_title("3. Spatial Mask (1=Valid)")
                axes[2].axis("off")
                
        elif deg_type == "haze":
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(clean_rgb)
            axes[0].set_title("Clean RGB")
            axes[0].axis("off")
            
            axes[1].imshow(degraded_rgb)
            axes[1].set_title("Hazy RGB")
            axes[1].axis("off")
            
        else:  # occlusion
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(clean_rgb)
            axes[0].set_title("Clean RGB")
            axes[0].axis("off")
            
            if mask_arr is not None:
                axes[1].imshow(mask_arr, cmap="gray")
                axes[1].set_title("Mask (1=Valid)")
                axes[1].axis("off")
                
            axes[2].imshow(degraded_rgb)
            axes[2].set_title("Occluded RGB")
            axes[2].axis("off")
            
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved visual debugging preview to {out_path}")
    except Exception as e:
        logger.warning(f"Failed to generate visualization preview: {e}")


def main():
    parser = argparse.ArgumentParser(description="Synthetic Degradation Engine CLI for Sentinel-2 data")
    parser.add_argument("--input-scene", required=True, help="Scene ID to load from data/processed/")
    parser.add_argument("--severity", default="medium", choices=["low", "medium", "high", "extreme"],
                        help="Degradation severity level (default: medium)")
    parser.add_argument("--mask-type", default="cloud_like", choices=["cloud_like", "irregular", "rectangular"],
                        help="Mask shape type (default: cloud_like)")
    parser.add_argument("--seed", type=int, default=42, help="Global seeding coefficient (default: 42)")
    
    # Mode arguments
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--haze-only", action="store_true", help="Generate haze atmospheric degradation only")
    mode_group.add_argument("--occlusion-only", action="store_true", help="Generate spatial occlusion masks only")
    mode_group.add_argument("--combined", action="store_true", help="Generate combined haze and occlusion degradation")
    
    parser.add_argument("--dry-run", action="store_true", help="Inspect and validate without writing files")
    parser.add_argument("--max-samples", type=int, default=1, help="Max synthetic variants to generate (default: 1)")
    
    args = parser.parse_args()
    
    # 1. Load config
    try:
        config = load_config("configs/degradation.yaml")
    except Exception as e:
        logger.error(f"Failed to load degradation config: {e}")
        sys.exit(1)
        
    processed_dir = get_project_root() / "data" / "processed" / "sentinel2"
    degraded_dir = Path(config["degradation"]["degraded_directory"])
    if not degraded_dir.is_absolute():
        degraded_dir = get_project_root() / degraded_dir
        
    scene_path = processed_dir / args.input_scene / "multispectral.tif"
    
    if not scene_path.exists():
        logger.error(f"Processed multispectral source raster not found at {scene_path}")
        sys.exit(1)
        
    # 2. Read source data
    logger.info(f"Loading source scene {args.input_scene} from {scene_path}...")
    with rasterio.open(scene_path) as src:
        clean_arr = src.read()
        profile = src.profile.copy()
        
    # Ensure correct band count
    if clean_arr.shape[0] != 4:
        logger.error(f"Invalid clean image band count {clean_arr.shape[0]}; expected 4 channels.")
        sys.exit(1)
        
    # 3. Determine degradations to apply
    h_sev = args.severity if (args.haze_only or args.combined) else None
    o_sev = args.severity if (args.occlusion_only or args.combined) else None
    m_type = args.mask_type if (args.occlusion_only or args.combined) else None
    
    deg_type_str = "haze" if args.haze_only else ("occlusion" if args.occlusion_only else "combined")
    
    print(f"\nSource Scene ID: {args.input_scene}")
    print(f"Degradation Type: {deg_type_str.upper()}")
    print(f"Severity:         {args.severity.upper()}")
    print(f"Mask Shape:       {args.mask_type.upper()}")
    print(f"Global Seed:      {args.seed}")
    
    # 4. Generate variants
    for idx in range(args.max_samples):
        # Sample-specific seed: global seed + sample index
        sample_seed = args.seed + idx
        sample_id = f"SMP_{args.input_scene}_{deg_type_str}_{args.severity}_{sample_seed}"
        
        if args.dry_run:
            logger.info(f"[DRY-RUN] Would generate sample {sample_id}")
            continue
            
        try:
            # Generate sample arrays
            degraded_arr, mask_arr, metadata = generate_sample(
                clean_image=clean_arr,
                config=config,
                seed=sample_seed,
                haze_severity=h_sev,
                occlusion_severity=o_sev,
                mask_type=m_type
            )
            
            # Post-generation checks
            assert degraded_arr.shape == clean_arr.shape, "Degraded array dimensions do not match clean array."
            assert not np.any(np.isnan(degraded_arr[:, mask_arr != 0] if mask_arr is not None else degraded_arr)), "Unexpected NaNs found in degraded array valid pixels."
            assert not np.any(np.isinf(degraded_arr)), "Unexpected Inf values found in degraded array."
            
            # Save files
            sample_out_dir = degraded_dir / deg_type_str / sample_id
            save_degraded_sample(
                sample_id=sample_id,
                output_dir=sample_out_dir,
                clean_arr=clean_arr,
                degraded_arr=degraded_arr,
                mask_arr=mask_arr,
                metadata=metadata,
                profile=profile
            )
            
        except Exception as e:
            logger.error(f"Failed to generate variant {idx + 1}: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
            
    print(f"\nSuccessfully generated {args.max_samples} sample(s).\n")

if __name__ == "__main__":
    main()
