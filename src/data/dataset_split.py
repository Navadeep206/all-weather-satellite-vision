import json
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

def create_scene_split(
    scenes: List[str],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """Splits a list of scene IDs deterministically into train, validation, and test sets.
    
    Args:
        scenes (list of str): List of scene identifiers to split.
        train_ratio (float): Fraction of scenes for training.
        val_ratio (float): Fraction of scenes for validation.
        test_ratio (float): Fraction of scenes for testing.
        seed (int): Random seed for reproducibility.
        
    Returns:
        Tuple[List[str], List[str], List[str]]: (train_scenes, val_scenes, test_scenes)
    """
    if not scenes:
        return [], [], []
        
    # Sort to ensure order is deterministic before shuffling
    sorted_scenes = sorted(scenes)
    
    rng = np.random.default_rng(seed)
    shuffled_scenes = list(sorted_scenes)
    rng.shuffle(shuffled_scenes)
    
    n_scenes = len(shuffled_scenes)
    
    # Handle small datasets (e.g. 1, 2, or 3 scenes) by allocating at least 1 scene
    # to each split where possible, or falling back.
    if n_scenes == 1:
        # Single scene must go to train
        return shuffled_scenes, [], []
    elif n_scenes == 2:
        # Two scenes: 1 train, 1 val
        return shuffled_scenes[:1], shuffled_scenes[1:], []
    elif n_scenes == 3:
        # Three scenes: 1 train, 1 val, 1 test
        return [shuffled_scenes[0]], [shuffled_scenes[1]], [shuffled_scenes[2]]
        
    # Standard splitting calculations for larger numbers of scenes
    n_train = max(1, int(round(train_ratio * n_scenes)))
    n_val = max(1, int(round(val_ratio * n_scenes)))
    n_test = n_scenes - n_train - n_val
    
    # Check if calculation led to out of bounds/empty splits
    if n_test < 0:
        n_val = max(1, n_val - abs(n_test))
        n_test = n_scenes - n_train - n_val
        
    train_scenes = shuffled_scenes[:n_train]
    val_scenes = shuffled_scenes[n_train:n_train + n_val]
    test_scenes = shuffled_scenes[n_train + n_val:]
    
    # Final sanity checks
    assert len(train_scenes) + len(val_scenes) + len(test_scenes) == n_scenes, "Scene count mismatch after split!"
    assert len(set(train_scenes).intersection(val_scenes)) == 0, "Data leakage between train and val!"
    assert len(set(train_scenes).intersection(test_scenes)) == 0, "Data leakage between train and test!"
    assert len(set(val_scenes).intersection(test_scenes)) == 0, "Data leakage between val and test!"
    
    return train_scenes, val_scenes, test_scenes

def write_split_files(
    splits_dir: Path,
    train_scenes: List[str],
    val_scenes: List[str],
    test_scenes: List[str]
) -> None:
    """Writes scene split IDs to txt files on disk."""
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    for split_name, scenes in [("train", train_scenes), ("val", val_scenes), ("test", test_scenes)]:
        file_path = splits_dir / f"{split_name}.txt"
        with open(file_path, "w", encoding="utf-8") as f:
            for s in scenes:
                f.write(f"{s}\n")
        logger.info(f"Wrote {len(scenes)} scene IDs to {file_path}")

def save_split_metadata(
    metadata_dir: Path,
    train_scenes: List[str],
    val_scenes: List[str],
    test_scenes: List[str],
    seed: int,
    version: str = "v1"
) -> None:
    """Saves machine-readable split metadata configuration JSON to disk."""
    metadata_dir.mkdir(parents=True, exist_ok=True)
    meta_path = metadata_dir / "split_metadata.json"
    
    total = len(train_scenes) + len(val_scenes) + len(test_scenes)
    
    record = {
        "split_seed": seed,
        "split_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_version": version,
        "source_scenes_count": total,
        "split_counts": {
            "train": len(train_scenes),
            "val": len(val_scenes),
            "test": len(test_scenes)
        },
        "actual_ratios": {
            "train": len(train_scenes) / total if total > 0 else 0,
            "val": len(val_scenes) / total if total > 0 else 0,
            "test": len(test_scenes) / total if total > 0 else 0
        },
        "train_scenes": train_scenes,
        "val_scenes": val_scenes,
        "test_scenes": test_scenes
    }
    
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    logger.info(f"Split metadata configuration written to {meta_path}")
