import csv
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from src.utils.logging_utils import get_logger
from src.utils.config import get_project_root

logger = get_logger(__name__)

class SceneCatalog:
    """Manages the local metadata catalog of Sentinel-2 scenes.
    
    Prevents duplicates, maintains file outputs in CSV and JSON formats, and
    tracks product statuses (discovered, download_started, downloaded, validated, failed).
    """
    
    COLUMNS = [
        "scene_id",
        "product_id",
        "product_name",
        "processing_level",
        "platform",
        "sensing_datetime",
        "tile_id",
        "cloud_cover",
        "source",
        "status"
    ]

    def __init__(self, metadata_dir: Optional[Path] = None):
        """Initializes the SceneCatalog.
        
        Args:
            metadata_dir (Path, optional): Directory to save catalog files.
                Defaults to data/metadata/ relative to the project root.
        """
        if metadata_dir is None:
            self.metadata_dir = get_project_root() / "data" / "metadata"
        else:
            self.metadata_dir = Path(metadata_dir)
            
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.metadata_dir / "scenes.csv"
        self.json_path = self.metadata_dir / "scenes.json"
        
        # In-memory dictionary mapped by product_id
        self.scenes: Dict[str, Dict[str, Any]] = {}
        self.load()

    def load(self) -> None:
        """Loads catalog records from the CSV file if it exists, falling back to JSON."""
        self.scenes = {}
        
        # Load from CSV if exists
        if self.csv_path.exists():
            try:
                with open(self.csv_path, "r", encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        pid = row.get("product_id")
                        if pid:
                            # Parse cloud cover to float
                            if "cloud_cover" in row:
                                try:
                                    row["cloud_cover"] = float(row["cloud_cover"])
                                except ValueError:
                                    row["cloud_cover"] = 100.0
                            self.scenes[pid] = dict(row)
                logger.info(f"Loaded {len(self.scenes)} scene(s) from CSV catalog.")
                return
            except Exception as e:
                logger.warning(f"Failed to read CSV catalog at {self.csv_path}: {e}. Trying JSON...")

        # Fallback to JSON if CSV load failed/missing
        if self.json_path.exists():
            try:
                with open(self.json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data:
                        pid = item.get("product_id")
                        if pid:
                            self.scenes[pid] = item
                logger.info(f"Loaded {len(self.scenes)} scene(s) from JSON catalog.")
            except Exception as e:
                logger.error(f"Failed to read JSON catalog at {self.json_path}: {e}")

    def save(self) -> None:
        """Saves current memory catalog state to both CSV and JSON formats."""
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Write CSV
        try:
            with open(self.csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.COLUMNS)
                writer.writeheader()
                for scene in self.scenes.values():
                    # Format float and slice items to COLUMNS keys
                    row = {col: scene.get(col, "") for col in self.COLUMNS}
                    writer.writerow(row)
        except Exception as e:
            logger.error(f"Failed to save CSV catalog: {e}")
            
        # 2. Write JSON
        try:
            with open(self.json_path, "w", encoding="utf-8") as f:
                json.dump(list(self.scenes.values()), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save JSON catalog: {e}")

    def add_scene(self, scene: Dict[str, Any]) -> None:
        """Adds a new scene to the catalog, updating it if it already exists (preventing duplicates).
        
        Args:
            scene (dict): Normalized scene metadata dictionary.
        """
        pid = scene.get("product_id")
        if not pid:
            logger.error("Cannot add scene to catalog without a valid product_id.")
            return
            
        if pid in self.scenes:
            # Update existing, keeping status if new is only 'discovered'
            existing = self.scenes[pid]
            new_status = scene.get("status", "discovered")
            if new_status == "discovered" and existing.get("status") != "discovered":
                scene["status"] = existing["status"]
            existing.update(scene)
            logger.debug(f"Updated scene {pid[:8]} in catalog.")
        else:
            self.scenes[pid] = dict(scene)
            logger.info(f"Added new scene {pid[:8]} to catalog.")
            
        self.save()

    def update_status(self, product_id: str, status: str) -> None:
        """Updates the status of an existing scene in the catalog.
        
        Args:
            product_id (str): The product ID to search.
            status (str): The new status to apply.
        """
        if product_id in self.scenes:
            self.scenes[product_id]["status"] = status
            logger.info(f"Updated status of product {product_id[:8]} to '{status}'.")
            self.save()
        else:
            logger.warning(f"Product {product_id[:8]} not found in catalog; status update to '{status}' skipped.")

    def has_scene(self, product_id: str) -> bool:
        """Checks if a product is already registered in the catalog."""
        return product_id in self.scenes

    def get_scene(self, product_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves catalog metadata for a specific scene."""
        return self.scenes.get(product_id)

    def list_scenes(self) -> List[Dict[str, Any]]:
        """Returns all scenes currently in the catalog."""
        return list(self.scenes.values())
