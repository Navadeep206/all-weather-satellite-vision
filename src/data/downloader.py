from pathlib import Path
from typing import Dict, Any, Optional
from src.data.cdse_client import CDSEClient
from src.data.scene_catalog import SceneCatalog
from src.utils.logging_utils import get_logger
from src.utils.config import get_project_root

logger = get_logger(__name__)

class SceneDownloader:
    """Manages downloading Sentinel-2 Level-2A products from CDSE."""

    def __init__(
        self,
        client: CDSEClient,
        catalog: SceneCatalog,
        output_dir: Optional[Path] = None
    ):
        """Initializes the SceneDownloader.
        
        Args:
            client (CDSEClient): Authenticated CDSE API client.
            catalog (SceneCatalog): Local catalog database to track statuses.
            output_dir (Path, optional): Directory to save raw downloads.
                Defaults to data/raw/sentinel2/ relative to the project root.
        """
        self.client = client
        self.catalog = catalog
        if output_dir is None:
            self.output_dir = get_project_root() / "data" / "raw" / "sentinel2"
        else:
            self.output_dir = Path(output_dir)

    def download_scene(self, scene: Dict[str, Any], dry_run: bool = False) -> bool:
        """Downloads a single scene and updates the catalog.
        
        Args:
            scene (dict): The normalized scene metadata dictionary.
            dry_run (bool): If True, previews the download without calling the CDSE API.
            
        Returns:
            bool: True if downloaded (or dry-run skipped), False if failed.
        """
        scene_id = scene.get("scene_id")
        product_id = scene.get("product_id")
        product_name = scene.get("product_name")
        
        if not scene_id or not product_id or not product_name:
            logger.error("Invalid scene metadata. Missing scene_id, product_id, or product_name.")
            return False
            
        # Standardize ZIP output filename
        zip_filename = f"{product_name}.zip" if not product_name.endswith(".zip") else product_name
        dest_path = self.output_dir / scene_id / "product" / zip_filename
        
        # Idempotency check: Skip if it already exists
        if dest_path.exists():
            logger.info(f"Product {product_name} already exists at {dest_path}. Skipping download.")
            self.catalog.update_status(product_id, "downloaded")
            return True
            
        if dry_run:
            logger.info(f"[DRY-RUN] Would download scene {scene_id} ({product_name}) to {dest_path}")
            return True

        if not self.client.has_credentials():
            logger.error("CDSE Credentials are not set. Cannot start download.")
            self.catalog.update_status(product_id, "failed")
            return False

        try:
            # Update status to started
            self.catalog.update_status(product_id, "download_started")
            
            # Execute download using CDSE client
            self.client.download_product(product_id, dest_path)
            
            # Update status to downloaded
            self.catalog.update_status(product_id, "downloaded")
            return True
        except Exception as e:
            logger.error(f"Failed to download scene {scene_id}: {e}")
            self.catalog.update_status(product_id, "failed")
            return False
