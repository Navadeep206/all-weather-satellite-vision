import json
import zipfile
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from src.data.scene_catalog import SceneCatalog
from src.utils.logging_utils import get_logger
from src.utils.config import get_project_root

logger = get_logger(__name__)

class SceneValidator:
    """Validates downloaded Sentinel-2 Level-2A product ZIP archives and saves metadata."""

    def __init__(self, catalog: SceneCatalog, output_dir: Optional[Path] = None):
        """Initializes the SceneValidator.
        
        Args:
            catalog (SceneCatalog): Local catalog database to track statuses.
            output_dir (Path, optional): Directory containing downloaded scenes.
                Defaults to data/raw/sentinel2/ relative to the project root.
        """
        self.catalog = catalog
        if output_dir is None:
            self.output_dir = get_project_root() / "data" / "raw" / "sentinel2"
        else:
            self.output_dir = Path(output_dir)

    def validate_scene(self, scene: Dict[str, Any]) -> Tuple[bool, str]:
        """Validates the downloaded ZIP product and updates the catalog.
        
        Checks ZIP integrity, confirms the Level-2A identity, verifies the presence
        of required bands (B02, B03, B04, B08), and preserves metadata.
        
        Args:
            scene (dict): The normalized scene metadata dictionary.
            
        Returns:
            Tuple[bool, str]: (is_valid, validation_message)
        """
        scene_id = scene.get("scene_id")
        product_id = scene.get("product_id")
        product_name = scene.get("product_name")
        
        if not scene_id or not product_id or not product_name:
            return False, "Invalid scene metadata. Missing keys."
            
        scene_dir = self.output_dir / scene_id
        product_dir = scene_dir / "product"
        metadata_dir = scene_dir / "metadata"
        
        # 1. Verify expected scene directories exist
        if not scene_dir.exists() or not product_dir.exists():
            error_msg = f"Scene directories do not exist for scene_id {scene_id}."
            logger.error(error_msg)
            self.catalog.update_status(product_id, "failed")
            return False, error_msg
            
        # Find zip file in product directory
        zip_filename = f"{product_name}.zip" if not product_name.endswith(".zip") else product_name
        zip_path = product_dir / zip_filename
        
        # 2. Verify downloaded product exists
        if not zip_path.exists():
            error_msg = f"Downloaded product ZIP not found at {zip_path}."
            logger.error(error_msg)
            self.catalog.update_status(product_id, "failed")
            return False, error_msg
            
        # 3. Check product is not obviously incomplete / is a valid zip
        if not zipfile.is_zipfile(zip_path):
            error_msg = f"File at {zip_path} is not a valid ZIP file."
            logger.error(error_msg)
            self.catalog.update_status(product_id, "failed")
            return False, error_msg
            
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                # 4. Check for obvious corruption by testing zip files
                corrupted = zf.testzip()
                if corrupted is not None:
                    error_msg = f"ZIP file {zip_path} contains corrupted file: {corrupted}."
                    logger.error(error_msg)
                    self.catalog.update_status(product_id, "failed")
                    return False, error_msg
                    
                namelist = zf.namelist()
                
                # 5. Check expected Level-2A identity (MSIL2A is present in paths/filenames)
                l2a_identity = any("MSIL2A" in name for name in namelist) or "MSIL2A" in product_name
                if not l2a_identity:
                    error_msg = f"Product {product_name} does not appear to be a Sentinel-2 Level-2A product."
                    logger.error(error_msg)
                    self.catalog.update_status(product_id, "failed")
                    return False, error_msg
                    
                # 6. Verify required 10m bands exist (B02, B03, B04, B08)
                required_bands = ["B02", "B03", "B04", "B08"]
                missing_bands = []
                for band in required_bands:
                    # S2 L2A 10m bands are named ending with _B02_10m.jp2 or similar inside the GRANULE folder
                    band_found = any(
                        name.endswith(f"_{band}_10m.jp2") or 
                        f"/R10m/" in name and f"_{band}_" in name 
                        for name in namelist
                    )
                    if not band_found:
                        missing_bands.append(band)
                        
                if missing_bands:
                    error_msg = f"Missing required bands: {', '.join(missing_bands)} in {product_name}."
                    logger.error(error_msg)
                    self.catalog.update_status(product_id, "failed")
                    return False, error_msg
                    
        except Exception as e:
            error_msg = f"Exception occurred while inspecting ZIP archive: {e}."
            logger.error(error_msg)
            self.catalog.update_status(product_id, "failed")
            return False, error_msg

        # 7. Preserve machine-readable metadata file
        metadata_dir.mkdir(parents=True, exist_ok=True)
        meta_json_path = metadata_dir / "scene.json"
        
        metadata_record = {
            "scene_id": scene_id,
            "product_id": product_id,
            "product_name": product_name,
            "sensing_time": scene.get("sensing_datetime"),
            "cloud_cover": scene.get("cloud_cover"),
            "platform": scene.get("platform"),
            "tile": scene.get("tile_id"),
            "source": scene.get("source"),
            "download_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "local_storage_path": str(zip_path.relative_to(get_project_root())) if zip_path.is_relative_to(get_project_root()) else str(zip_path),
            "validation_status": "validated"
        }
        
        try:
            with open(meta_json_path, "w", encoding="utf-8") as f:
                json.dump(metadata_record, f, indent=2)
            logger.info(f"Metadata preserved for scene {scene_id} at {meta_json_path}")
        except Exception as e:
            logger.error(f"Failed to preserve metadata scene.json: {e}")
            # Do not fail validation just because we couldn't write the local metadata file, 
            # but it is good to flag.
            
        # Update catalog status to validated
        self.catalog.update_status(product_id, "validated")
        logger.info(f"Scene {scene_id} validated successfully.")
        return True, "Validation passed successfully."
