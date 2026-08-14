import csv
from pathlib import Path
from typing import List, Dict, Any, Optional
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

MANIFEST_COLUMNS = [
    "sample_id",
    "scene_id",
    "split",
    "clean_path",
    "degraded_path",
    "mask_path",
    "rgb_path",
    "patch_y",
    "patch_x",
    "seed",
    "degradation_type",
    "haze_severity",
    "occlusion_severity",
    "mask_type"
]

def write_manifest(filepath: Path, rows: List[Dict[str, Any]]) -> None:
    """Writes a list of manifest rows to a CSV file.
    
    Args:
        filepath (Path): Destination CSV file path.
        rows (list of dict): List of rows matching MANIFEST_COLUMNS schema.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Check for duplicate sample_ids
    seen_ids = set()
    cleaned_rows = []
    for r in rows:
        sid = r.get("sample_id")
        if sid in seen_ids:
            logger.warning(f"Duplicate sample_id found during writing: {sid}. Skipping duplicate.")
            continue
        seen_ids.add(sid)
        cleaned_rows.append(r)
        
    try:
        with open(filepath, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
            writer.writeheader()
            for r in cleaned_rows:
                # Filter to only contain columns in schema
                row_dict = {col: r.get(col, "") for col in MANIFEST_COLUMNS}
                writer.writerow(row_dict)
        logger.info(f"Successfully wrote {len(cleaned_rows)} rows to manifest {filepath.name}")
    except Exception as e:
        logger.error(f"Failed to write manifest CSV {filepath}: {e}")
        raise

def read_manifest(filepath: Path) -> List[Dict[str, Any]]:
    """Reads and parses a manifest CSV file.
    
    Args:
        filepath (Path): CSV file path to load.
        
    Returns:
        list of dict: Parsed manifest records.
    """
    if not filepath.exists():
        logger.warning(f"Manifest file not found: {filepath}")
        return []
        
    rows = []
    try:
        with open(filepath, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Parse numeric patch offsets if present
                if "patch_y" in row and row["patch_y"]:
                    row["patch_y"] = int(row["patch_y"])
                if "patch_x" in row and row["patch_x"]:
                    row["patch_x"] = int(row["patch_x"])
                if "seed" in row and row["seed"]:
                    row["seed"] = int(row["seed"])
                rows.append(dict(row))
    except Exception as e:
        logger.error(f"Failed to read manifest CSV {filepath}: {e}")
        raise
        
    return rows

def validate_manifest_schema(rows: List[Dict[str, Any]]) -> Tuple[bool, str]:
    """Validates that a manifest list conforms to the schema rules and contains no duplicate sample_ids.
    
    Args:
        rows (list of dict): Manifest rows to validate.
        
    Returns:
        Tuple[bool, str]: (is_valid, reason)
    """
    seen_ids = set()
    for idx, r in enumerate(rows):
        sid = r.get("sample_id")
        if not sid:
            return False, f"Row {idx} is missing required field 'sample_id'."
        if sid in seen_ids:
            return False, f"Duplicate sample_id '{sid}' detected at row {idx}."
        seen_ids.add(sid)
        
        # Verify columns exist
        for col in ["scene_id", "split", "clean_path"]:
            if col not in r or not r[col]:
                return False, f"Sample '{sid}' at row {idx} is missing required column '{col}'."
                
    return True, "Schema is valid and contain no duplicates."
