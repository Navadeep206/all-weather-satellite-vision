import pytest
import tempfile
from pathlib import Path
from src.data.manifest import write_manifest, read_manifest, validate_manifest_schema

def test_manifest_write_and_read(tmp_path):
    csv_path = tmp_path / "test_manifest.csv"
    
    rows = [
        {
            "sample_id": "SMP_001",
            "scene_id": "SCENE_A",
            "split": "train",
            "clean_path": "data/processed/clean_A.tif",
            "degraded_path": "data/degraded/haze/clean_A_haze.tif",
            "patch_y": 0,
            "patch_x": 0,
            "seed": 42,
            "degradation_type": "haze",
            "haze_severity": "medium"
        },
        {
            "sample_id": "SMP_002",
            "scene_id": "SCENE_A",
            "split": "train",
            "clean_path": "data/processed/clean_A.tif",
            "degraded_path": "data/degraded/haze/clean_A_haze.tif",
            "patch_y": 0,
            "patch_x": 256,
            "seed": 42,
            "degradation_type": "haze",
            "haze_severity": "medium"
        }
    ]
    
    # Write and reload
    write_manifest(csv_path, rows)
    loaded = read_manifest(csv_path)
    
    assert len(loaded) == 2
    assert loaded[0]["sample_id"] == "SMP_001"
    assert loaded[0]["patch_y"] == 0
    assert loaded[0]["seed"] == 42
    assert loaded[1]["patch_x"] == 256

def test_manifest_duplicate_detection(tmp_path):
    csv_path = tmp_path / "dup_manifest.csv"
    
    # Rows with duplicate sample_ids
    rows = [
        {"sample_id": "SMP_001", "scene_id": "SCENE_A", "split": "train", "clean_path": "p1"},
        {"sample_id": "SMP_001", "scene_id": "SCENE_B", "split": "train", "clean_path": "p2"}
    ]
    
    # write_manifest handles/filters duplicates internally with a warning
    write_manifest(csv_path, rows)
    loaded = read_manifest(csv_path)
    assert len(loaded) == 1  # Second duplicate skipped

def test_validate_manifest_schema():
    # Valid
    valid_rows = [
        {"sample_id": "SMP_001", "scene_id": "SCENE_A", "split": "train", "clean_path": "p1"},
        {"sample_id": "SMP_002", "scene_id": "SCENE_B", "split": "train", "clean_path": "p2"}
    ]
    ok, err = validate_manifest_schema(valid_rows)
    assert ok, err
    
    # Invalid missing scene_id
    invalid_rows = [
        {"sample_id": "SMP_001", "split": "train", "clean_path": "p1"}
    ]
    ok, err = validate_manifest_schema(invalid_rows)
    assert not ok
    assert "missing required column" in err
    
    # Invalid duplicate sample_ids
    dup_rows = [
        {"sample_id": "SMP_001", "scene_id": "SCENE_A", "split": "train", "clean_path": "p1"},
        {"sample_id": "SMP_001", "scene_id": "SCENE_A", "split": "train", "clean_path": "p2"}
    ]
    ok, err = validate_manifest_schema(dup_rows)
    assert not ok
    assert "Duplicate sample_id" in err
