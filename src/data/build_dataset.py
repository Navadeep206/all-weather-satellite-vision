import sys
import argparse
from pathlib import Path
import json

from src.data.scene_catalog import SceneCatalog
from src.data.dataset_builder import build_datasets
from src.data.dataset_validator import check_data_leakage, validate_dataset_files
from src.utils.config import load_config, get_project_root
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

def generate_visual_report(dataset_dir: Path) -> None:
    """Generates visual comparisons for Stage 1, 2, and 3 under data/samples/phase4/."""
    try:
        import matplotlib.pyplot as plt
        import rasterio
        import numpy as np
    except ImportError:
        logger.warning("matplotlib or rasterio not available. Skipping visual dataset report.")
        return
        
    try:
        from src.data.manifest import read_manifest
        manifests_dir = dataset_dir / "manifests"
        out_samples_dir = get_project_root() / "data" / "samples" / "phase4"
        out_samples_dir.mkdir(parents=True, exist_ok=True)
        
        project_root = get_project_root()
        patch_size = 256
        
        def stretch(img):
            img_clean = np.nan_to_num(img, nan=0.0)
            p2, p98 = np.percentile(img_clean, (2, 98), axis=(1, 2), keepdims=True)
            diff = p98 - p2
            diff[diff == 0] = 1.0
            return np.clip((img_clean - p2) / diff, 0.0, 1.0)
            
        # --- 1. Stage 1 Visual preview ---
        s1_csv = manifests_dir / "stage1_train.csv"
        if s1_csv.exists():
            rows = read_manifest(s1_csv)
            if rows:
                r = rows[0]
                y, x = r["patch_y"], r["patch_x"]
                win = rasterio.windows.Window(col_off=x, row_off=y, width=patch_size, height=patch_size)
                
                with rasterio.open(project_root / r["clean_path"]) as cln, rasterio.open(project_root / r["degraded_path"]) as deg:
                    c_patch = cln.read(window=win)
                    d_patch = deg.read(window=win)
                    
                c_rgb = np.moveaxis(stretch(c_patch[[2, 1, 0]]), 0, -1)
                d_rgb = np.moveaxis(stretch(d_patch[[2, 1, 0]]), 0, -1)
                
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                axes[0].imshow(d_rgb)
                axes[0].set_title("Stage 1 Input (Degraded)")
                axes[0].axis("off")
                axes[1].imshow(c_rgb)
                axes[1].set_title("Stage 1 Target (Clean)")
                axes[1].axis("off")
                
                plt.tight_layout()
                plt.savefig(out_samples_dir / "stage1_sample.png", dpi=150)
                plt.close()
                logger.info(f"Saved Stage 1 visual sample to {out_samples_dir / 'stage1_sample.png'}")
                
        # --- 2. Stage 2 Visual preview ---
        s2_csv = manifests_dir / "stage2_train.csv"
        if s2_csv.exists():
            rows = read_manifest(s2_csv)
            if rows:
                r = rows[0]
                y, x = r["patch_y"], r["patch_x"]
                win = rasterio.windows.Window(col_off=x, row_off=y, width=patch_size, height=patch_size)
                
                with rasterio.open(project_root / r["clean_path"]) as cln, rasterio.open(project_root / r["degraded_path"]) as deg, rasterio.open(project_root / r["mask_path"]) as msk:
                    c_patch = cln.read(window=win)
                    d_patch = deg.read(window=win)
                    m_patch = msk.read(1, window=win)
                    
                c_rgb = np.moveaxis(stretch(c_patch[[2, 1, 0]]), 0, -1)
                d_rgb = np.moveaxis(stretch(d_patch[[2, 1, 0]]), 0, -1)
                
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(d_rgb)
                axes[0].set_title("Stage 2 Input (Degraded)")
                axes[0].axis("off")
                axes[1].imshow(m_patch, cmap="gray")
                axes[1].set_title("Stage 2 Input (Mask)")
                axes[1].axis("off")
                axes[2].imshow(c_rgb)
                axes[2].set_title("Stage 2 Target (Clean)")
                axes[2].axis("off")
                
                plt.tight_layout()
                plt.savefig(out_samples_dir / "stage2_sample.png", dpi=150)
                plt.close()
                logger.info(f"Saved Stage 2 visual sample to {out_samples_dir / 'stage2_sample.png'}")
                
        # --- 3. Stage 3 Visual preview ---
        s3_csv = manifests_dir / "stage3_train.csv"
        if s3_csv.exists():
            rows = read_manifest(s3_csv)
            if rows:
                r = rows[0]
                y, x = r["patch_y"], r["patch_x"]
                win = rasterio.windows.Window(col_off=x, row_off=y, width=patch_size, height=patch_size)
                
                with rasterio.open(project_root / r["clean_path"]) as cln, rasterio.open(project_root / r["rgb_path"]) as rgb:
                    c_patch = cln.read(window=win)
                    rgb_patch = rgb.read(window=win)
                    
                c_rgb = np.moveaxis(stretch(c_patch[[2, 1, 0]]), 0, -1)
                rgb_target = np.moveaxis(stretch(rgb_patch), 0, -1)
                
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                axes[0].imshow(c_rgb)
                axes[0].set_title("Stage 3 Input (Clean Multispectral RGB)")
                axes[0].axis("off")
                axes[1].imshow(rgb_target)
                axes[1].set_title("Stage 3 Target (RGB)")
                axes[1].axis("off")
                
                plt.tight_layout()
                plt.savefig(out_samples_dir / "stage3_sample.png", dpi=150)
                plt.close()
                logger.info(f"Saved Stage 3 visual sample to {out_samples_dir / 'stage3_sample.png'}")
                
    except Exception as e:
        logger.warning(f"Error drawing visual dataset report: {e}")

def main():
    parser = argparse.ArgumentParser(description="Dataset Construction CLI")
    parser.add_argument("--config", default="configs/dataset.yaml", help="Path to dataset config (default: configs/dataset.yaml)")
    parser.add_argument("--max-scenes", type=int, help="Limit number of processed scenes")
    parser.add_argument("--validate", action="store_true", help="Trigger dataset consistency validation checks")
    parser.add_argument("--dry-run", action="store_true", help="Inspect and output plan without writing files")
    parser.add_argument("--force", action="store_true", help="Force recreate synthetic samples")
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
        degradation_config = load_config("configs/degradation.yaml")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)
        
    project_root = get_project_root()
    processed_dir = project_root / "data" / "processed" / "sentinel2"
    dataset_dir = project_root / "data" / "dataset"
    
    # Initialize scene catalog
    catalog = SceneCatalog()
    
    max_scenes = args.max_scenes if args.max_scenes is not None else config["generation"]["max_scenes"]
    
    # Execute build
    summary = build_datasets(
        processed_dir=processed_dir,
        dataset_dir=dataset_dir,
        catalog=catalog,
        config=config,
        degradation_config=degradation_config,
        max_scenes=max_scenes,
        force=args.force,
        dry_run=args.dry_run
    )
    
    if args.dry_run:
        print("\n=== DRY-RUN PLAN ===")
        print(f"Train scenes:      {summary.get('train_scenes', [])}")
        print(f"Validation scenes: {summary.get('val_scenes', [])}")
        print(f"Test scenes:       {summary.get('test_scenes', [])}")
        print(f"Total scenes:      {summary.get('expected_scenes_count', 0)}")
        print("====================\n")
        return
        
    if not summary:
        print("\nNo dataset generated. Verify processed scenes are cataloged.\n")
        sys.exit(1)
        
    # Print build summary to stdout
    print("\n========================================")
    print("DATASET CONSTRUCTION COMPLETE")
    print("=============================")
    print(f"Dataset Version: {summary['dataset_version']}")
    print(f"Total scenes:    {summary['scene_counts']['total']}")
    print(f"Train scenes:    {summary['scene_counts']['train']}")
    print(f"Val scenes:      {summary['scene_counts']['val']}")
    print(f"Test scenes:     {summary['scene_counts']['test']}")
    print("-----------------------------")
    print(f"Stage 1 samples: {summary['sample_counts']['stage1']}")
    print(f"Stage 2 samples: {summary['sample_counts']['stage2']}")
    print(f"Stage 3 samples: {summary['sample_counts']['stage3']}")
    print("========================================\n")
    
    # Generate visual debugging reports
    generate_visual_report(dataset_dir)
    
    # 5. Perform validation checks if requested
    if args.validate:
        leakage_ok, leakage_report = check_data_leakage(dataset_dir)
        print(leakage_report)
        
        files_ok, file_errors = validate_dataset_files(dataset_dir, patch_size=config["patch"]["size"])
        if not files_ok:
            print("\nFILE VALIDATION FAILURE:")
            for err in file_errors[:10]:
                print(f" - {err}")
            if len(file_errors) > 10:
                print(f" ... and {len(file_errors) - 10} more errors.")
        else:
            print("FILE INTEGRITY CHECK: PASS")
            
        # Write reports/dataset_summary.md
        reports_dir = project_root / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        summary_md_path = reports_dir / "dataset_summary.md"
        
        with open(dataset_dir / "metadata" / "generation_summary.json", "r") as f:
            gen_sum = json.load(f)
            
        with open(summary_md_path, "w", encoding="utf-8") as f:
            f.write("# Dataset Construction Summary\n\n")
            f.write(f"* **Dataset Version**: {summary['dataset_version']}\n")
            f.write(f"* **Split Seed**: {summary['split_seed']}\n")
            f.write(f"* **Patch Dimensions**: {summary['patch_size']} x {summary['patch_size']} (Overlap: {summary['overlap']}px)\n")
            f.write(f"* **Date Generated**: {gen_sum.get('date_generated')}\n\n")
            
            f.write("## Geographic Split Distribution\n")
            f.write(f"* **Train scenes**: {summary['scene_counts']['train']}\n")
            f.write(f"* **Val scenes**: {summary['scene_counts']['val']}\n")
            f.write(f"* **Test scenes**: {summary['scene_counts']['test']}\n\n")
            
            f.write("## Sample Counts\n")
            f.write(f"* **Stage 1 (Haze)**: {summary['sample_counts']['stage1']} patch samples\n")
            f.write(f"* **Stage 2 (Occlusion)**: {summary['sample_counts']['stage2']} patch samples\n")
            f.write(f"* **Stage 3 (RGB translation)**: {summary['sample_counts']['stage3']} patch samples\n\n")
            
            f.write("## Integrity Verification\n")
            f.write(f"* **Data Leakage Status**: {'PASS' if leakage_ok else 'FAIL'}\n")
            f.write(f"* **File Integrity Status**: {'PASS' if files_ok else 'FAIL'}\n")
            
        logger.info(f"Human-readable summary report written to {summary_md_path}")
        
        if not leakage_ok or not files_ok:
            sys.exit(1)

if __name__ == "__main__":
    main()
