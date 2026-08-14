import sys
import argparse
import json
import time
import numpy as np
import rasterio
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from src.data.scene_catalog import SceneCatalog
from src.data.band_utils import (
    find_band_members,
    parse_metadata_from_zip,
    get_reference_grid,
    read_and_resample_band
)
from src.data.quality import compute_raster_statistics, verify_processed_scene
from src.utils.config import load_config, get_project_root
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

def process_scene(
    scene_id: str,
    raw_dir: Path,
    processed_dir: Path,
    catalog: SceneCatalog,
    config: Dict[str, Any],
    force: bool = False,
    dry_run: bool = False
) -> Tuple[bool, str]:
    """Preprocesses a single Sentinel-2 scene and aligns B02, B03, B04, B08 bands to a common target grid.
    
    Args:
        scene_id (str): The project scene identifier.
        raw_dir (Path): Base directory containing raw scene files.
        processed_dir (Path): Base directory to write processed scene files.
        catalog (SceneCatalog): Local catalog database to track statuses.
        config (dict): Configuration dictionary containing bands and limits.
        force (bool): If True, re-processes even if files already exist.
        dry_run (bool): If True, performs checks without writing file outputs.
        
    Returns:
        Tuple[bool, str]: (success, status_message)
    """
    logger.info(f"Starting Phase 2 Preprocessing for scene: {scene_id}")
    
    # 1. Resolve paths
    scene_raw_dir = raw_dir / scene_id
    scene_proc_dir = processed_dir / scene_id
    
    product_dir = scene_raw_dir / "product"
    
    # Check if catalog has this product and if it's validated
    # Find matching product_id in catalog
    matching_scene = None
    for s in catalog.list_scenes():
        if s.get("scene_id") == scene_id:
            matching_scene = s
            break
            
    if not matching_scene:
        return False, f"Scene {scene_id} not found in local catalog metadata."
        
    product_id = matching_scene["product_id"]
    product_name = matching_scene["product_name"]
    
    # Only process validated scenes
    if matching_scene.get("status") not in ["validated", "processed", "processing_failed"]:
        return False, f"Scene {scene_id} has status '{matching_scene.get('status')}'. Bypassing processing."
        
    zip_filename = f"{product_name}.zip" if not product_name.endswith(".zip") else product_name
    zip_path = product_dir / zip_filename
    
    if not zip_path.exists():
        catalog.update_status(product_id, "processing_failed")
        return False, f"Raw product archive not found at {zip_path}"
        
    # Idempotency check
    ms_out_path = scene_proc_dir / "multispectral.tif"
    rgb_out_path = scene_proc_dir / "rgb.tif"
    proc_meta_path = scene_proc_dir / "metadata.json"
    proc_qual_path = scene_proc_dir / "quality.json"
    
    if (ms_out_path.exists() and rgb_out_path.exists() and 
        proc_meta_path.exists() and proc_qual_path.exists() and not force):
        logger.info(f"Processed files already exist for {scene_id}. Skipping processing.")
        catalog.update_status(product_id, "processed")
        return True, "Files already exist (skipped)."
        
    if dry_run:
        logger.info(f"[DRY-RUN] Preprocessing scene {scene_id} from {zip_path}")
        return True, "Dry-run checked successfully."

    catalog.update_status(product_id, "processing")
    scene_proc_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 2. Map required bands inside ZIP
        bands_to_extract = config.get("bands", ["B02", "B03", "B04", "B08"])
        band_members = find_band_members(zip_path, bands_to_extract)
        
        # 3. Inspect metadata scaling & offsets
        quantification, offsets = parse_metadata_from_zip(zip_path)
        logger.info(f"Scale factor (Quantification): {quantification}, Offsets parsed: {offsets}")
        
        # 4. Define reference grid using B02 (Blue band)
        ref_member = band_members["B02"]
        ref_grid = get_reference_grid(zip_path, ref_member)
        logger.info(f"Reference grid (B02): CRS={ref_grid['crs']}, Shape={ref_grid['width']}x{ref_grid['height']}, Res={ref_grid['res']}")
        
        # Output arrays setup
        target_shape = (ref_grid["height"], ref_grid["width"])
        multispectral_data = []
        
        # Resampling configuration
        resampling_method = config.get("search", {}).get("resampling_method", "bilinear")
        
        # 5. Extract, Resample and Scale each band to target grid
        for band in bands_to_extract:
            member = band_members[band]
            logger.info(f"Processing band {band} from member {member[:60]}...")
            
            # Read & warp band raw pixels
            resampled_dn, source_nodata = read_and_resample_band(
                zip_path=zip_path,
                member_name=member,
                target_crs=ref_grid["crs"],
                target_transform=ref_grid["transform"],
                target_width=ref_grid["width"],
                target_height=ref_grid["height"],
                resampling_method=resampling_method
            )
            
            # 6. Apply offset and scale for valid pixels (Bottom Of Atmosphere reflectance)
            # Default offset is 0 if not present in baseline metadata
            offset = offsets.get(band, 0.0)
            
            band_float = np.full(target_shape, np.nan, dtype=np.float32)
            
            # Define mask of valid source pixels
            # 0 is the standard nodata value for Sentinel-2 JP2 bands
            nodata_val = source_nodata if source_nodata is not None else 0
            valid_mask = (resampled_dn != nodata_val) & (resampled_dn >= 0) & (resampled_dn < 65535)
            
            # Scale valid pixels: (DN + Offset) / Scale
            band_float[valid_mask] = (resampled_dn[valid_mask].astype(np.float32) + offset) / quantification
            
            multispectral_data.append(band_float)
            
        # 7. Write multispectral.tif
        ms_meta = {
            "driver": "GTiff",
            "dtype": "float32",
            "count": len(bands_to_extract),
            "width": ref_grid["width"],
            "height": ref_grid["height"],
            "crs": ref_grid["crs"],
            "transform": ref_grid["transform"],
            "nodata": np.nan
        }
        
        with rasterio.open(ms_out_path, "w", **ms_meta) as dst:
            for i, data in enumerate(multispectral_data):
                dst.write(data, i + 1)
                
        # 8. Create RGB band order: B04 (Red), B03 (Green), B02 (Blue)
        # Note: bands_to_extract order is [B02, B03, B04, B08]
        # index 0: B02, index 1: B03, index 2: B04
        rgb_meta = ms_meta.copy()
        rgb_meta.update({"count": 3})
        
        with rasterio.open(rgb_out_path, "w", **rgb_meta) as dst:
            dst.write(multispectral_data[2], 1) # Red (B04)
            dst.write(multispectral_data[1], 2) # Green (B03)
            dst.write(multispectral_data[0], 3) # Blue (B02)
            
        logger.info("Wrote multispectral and RGB GeoTIFFs successfully.")
        
        # 9. Perform structural and georeference verification checks
        passed, val_msg = verify_processed_scene(ms_out_path, rgb_out_path, len(bands_to_extract))
        if not passed:
            logger.error(f"Post-processing alignment check failed: {val_msg}")
            # Delete corrupted outputs
            if ms_out_path.exists(): ms_out_path.unlink()
            if rgb_out_path.exists(): rgb_out_path.unlink()
            catalog.update_status(product_id, "processing_failed")
            return False, f"Alignment validation failed: {val_msg}"
            
        # 10. Compute numeric quality statistics
        # Compute quality statistics based on the multispectral composite
        combined_ms = np.stack(multispectral_data, axis=0)
        stats = compute_raster_statistics(combined_ms)
        
        # 11. Write quality.json
        quality_record = {
            "scene_id": scene_id,
            "valid_pixel_percentage": stats["valid_pixel_percentage"],
            "invalid_pixel_percentage": stats["invalid_pixel_percentage"],
            "nan_count": stats["nan_count"],
            "inf_count": stats["inf_count"],
            "nodata_count": stats["nodata_count"],
            "alignment_check": True,
            "band_resolution_check": True,
            "crs_check": True,
            "transform_check": True,
            "processing_status": "passed"
        }
        
        with open(proc_qual_path, "w", encoding="utf-8") as f:
            json.dump(quality_record, f, indent=2)
            
        # 12. Write metadata.json
        metadata_record = {
            "scene_id": scene_id,
            "source_product": product_name,
            "processing_level": "L2A",
            "bands": bands_to_extract,
            "target_resolution_m": int(ref_grid["res"][0]) if ref_grid["res"] else 10,
            "crs": str(ref_grid["crs"]),
            "width": ref_grid["width"],
            "height": ref_grid["height"],
            "bounds": [ref_grid["bounds"].left, ref_grid["bounds"].bottom, ref_grid["bounds"].right, ref_grid["bounds"].top],
            "reflectance_representation": "float32 Bottom-of-Atmosphere (BOA) Reflectance",
            "resampling_method": resampling_method,
            "processing_version": "phase2_v1",
            "processed_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        
        with open(proc_meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata_record, f, indent=2)
            
        # 13. Create debug preview
        try:
            samples_dir = get_project_root() / "data" / "samples" / "previews"
            create_preview(scene_id, ms_out_path, rgb_out_path, samples_dir)
        except Exception as preview_err:
            logger.warning(f"Failed to generate visualization preview for {scene_id}: {preview_err}")

        catalog.update_status(product_id, "processed")
        logger.info(f"Scene {scene_id} processed successfully.")
        return True, "Scene preprocessed and aligned successfully."
        
    except Exception as e:
        logger.error(f"Error processing scene {scene_id}: {e}")
        # Clean up any partial files
        for p in [ms_out_path, rgb_out_path, proc_meta_path, proc_qual_path]:
            if p.exists():
                try: p.unlink()
                except Exception: pass
        catalog.update_status(product_id, "processing_failed")
        return False, f"Processing failed: {e}"

def create_preview(scene_id: str, ms_path: Path, rgb_path: Path, output_dir: Path) -> Optional[Path]:
    """Generates a side-by-side RGB composite and grayscale NIR preview for debugging.
    
    Saves the output to data/samples/previews/<scene_id>_preview.png.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib is not available. Skipping debug preview generation.")
        return None
        
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_img_path = output_dir / f"{scene_id}_preview.png"
        
        with rasterio.open(rgb_path) as rgb_src, rasterio.open(ms_path) as ms_src:
            # Read RGB bands [R, G, B]
            rgb = rgb_src.read()  # shape: (3, H, W)
            # Read NIR band (index 4 in MS)
            nir = ms_src.read(4)  # shape: (H, W)
            
        def stretch_image(img):
            img_clean = np.nan_to_num(img, nan=0.0)
            p2, p98 = np.percentile(img_clean, (2, 98))
            if p98 > p2:
                img_stretched = np.clip((img_clean - p2) / (p98 - p2), 0.0, 1.0)
            else:
                img_stretched = np.clip(img_clean, 0.0, 1.0)
            return img_stretched
            
        rgb_display = np.moveaxis(stretch_image(rgb), 0, -1)
        nir_display = stretch_image(nir)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(rgb_display)
        axes[0].set_title(f"RGB Composite: {scene_id}")
        axes[0].axis("off")
        
        axes[1].imshow(nir_display, cmap="gray")
        axes[1].set_title(f"NIR Band (B08): {scene_id}")
        axes[1].axis("off")
        
        plt.tight_layout()
        plt.savefig(out_img_path, dpi=150, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Generated debug preview at {out_img_path}")
        return out_img_path
    except Exception as e:
        logger.warning(f"Error drawing debug preview: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Preprocess and align raw Sentinel-2 L2A scenes")
    parser.add_argument("--scene-id", help="Scene ID to process")
    parser.add_argument("--all", action="store_true", help="Process all validated scenes in catalog")
    parser.add_argument("--force", action="store_true", help="Force reprocessing even if outputs exist")
    parser.add_argument("--dry-run", action="store_true", help="Inspect and validate without writing files")
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config("configs/acquisition.yaml")
    except Exception as e:
        logger.error(f"Failed to load config configs/acquisition.yaml: {e}")
        sys.exit(1)
        
    raw_dir = get_project_root() / "data" / "raw" / "sentinel2"
    processed_dir = get_project_root() / "data" / "processed" / "sentinel2"
    
    catalog = SceneCatalog()
    
    if args.scene_id:
        success, msg = process_scene(
            scene_id=args.scene_id,
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            catalog=catalog,
            config=config,
            force=args.force,
            dry_run=args.dry_run
        )
        print(f"Result: {'SUCCESS' if success else 'FAILED'} - {msg}")
        sys.exit(0 if success else 1)
        
    elif args.all:
        scenes = catalog.list_scenes()
        validated_scenes = [s for s in scenes if s.get("status") in ["validated", "processed", "processing_failed"]]
        
        if not validated_scenes:
            print("No validated scenes found in catalog to process.")
            sys.exit(0)
            
        print(f"Found {len(validated_scenes)} scene(s) to process.")
        success_count = 0
        
        for s in validated_scenes:
            scene_id = s.get("scene_id")
            success, msg = process_scene(
                scene_id=scene_id,
                raw_dir=raw_dir,
                processed_dir=processed_dir,
                catalog=catalog,
                config=config,
                force=args.force,
                dry_run=args.dry_run
            )
            print(f"Scene {scene_id}: {'SUCCESS' if success else 'FAILED'} - {msg}")
            if success:
                success_count += 1
                
        print(f"\nProcessing complete: {success_count}/{len(validated_scenes)} passed.")
        sys.exit(0 if success_count == len(validated_scenes) else 1)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
