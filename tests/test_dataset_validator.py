import pytest
import csv
from pathlib import Path
from src.data.dataset_validator import check_data_leakage, validate_dataset_files
from src.data.manifest import write_manifest

def test_check_data_leakage_success(tmp_path):
    manifests_dir = tmp_path / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    
    # Write train and val manifests
    train_rows = [
        {"sample_id": "SMP_01", "scene_id": "SCENE_A", "split": "train", "clean_path": "c1"},
        {"sample_id": "SMP_02", "scene_id": "SCENE_A", "split": "train", "clean_path": "c2"}
    ]
    val_rows = [
        {"sample_id": "SMP_03", "scene_id": "SCENE_B", "split": "val", "clean_path": "c3"}
    ]
    
    write_manifest(manifests_dir / "stage1_train.csv", train_rows)
    write_manifest(manifests_dir / "stage1_val.csv", val_rows)
    
    ok, report = check_data_leakage(tmp_path)
    assert ok, report
    assert "PASS" in report

def test_check_data_leakage_failure(tmp_path):
    manifests_dir = tmp_path / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    
    # Scene_A appears in both train and val splits (leakage!)
    train_rows = [
        {"sample_id": "SMP_01", "scene_id": "SCENE_A", "split": "train", "clean_path": "c1"}
    ]
    val_rows = [
        {"sample_id": "SMP_02", "scene_id": "SCENE_A", "split": "val", "clean_path": "c2"}
    ]
    
    write_manifest(manifests_dir / "stage1_train.csv", train_rows)
    write_manifest(manifests_dir / "stage1_val.csv", val_rows)
    
    ok, report = check_data_leakage(tmp_path)
    assert not ok
    assert "FAIL" in report
    assert "SCENE LEAKAGE" in report
