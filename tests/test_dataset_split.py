import pytest
from src.data.dataset_split import create_scene_split

def test_create_scene_split_deterministic():
    scenes = [f"SCENE_{i}" for i in range(20)]
    
    # Run 1
    t1, v1, te1 = create_scene_split(scenes, 0.70, 0.15, 0.15, seed=42)
    # Run 2
    t2, v2, te2 = create_scene_split(scenes, 0.70, 0.15, 0.15, seed=42)
    
    assert t1 == t2
    assert v1 == v2
    assert te1 == te2

def test_create_scene_split_no_overlap():
    scenes = [f"SCENE_{i}" for i in range(15)]
    train, val, test = create_scene_split(scenes, 0.60, 0.20, 0.20, seed=10)
    
    # Verify subsets are mutually exclusive
    assert len(set(train).intersection(val)) == 0
    assert len(set(train).intersection(test)) == 0
    assert len(set(val).intersection(test)) == 0
    
    # Verify all input scenes accounted for
    assert len(train) + len(val) + len(test) == len(scenes)

def test_create_scene_split_small_datasets():
    # 1 Scene
    train, val, test = create_scene_split(["S1"], seed=42)
    assert train == ["S1"]
    assert val == []
    assert test == []
    
    # 2 Scenes
    train, val, test = create_scene_split(["S1", "S2"], seed=42)
    assert len(train) == 1
    assert len(val) == 1
    assert len(test) == 0
    
    # 3 Scenes
    train, val, test = create_scene_split(["S1", "S2", "S3"], seed=42)
    assert len(train) == 1
    assert len(val) == 1
    assert len(test) == 1
