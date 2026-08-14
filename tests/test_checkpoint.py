import pytest
import torch
import torch.nn as nn
from src.utils.checkpoint import save_checkpoint, load_checkpoint

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
        
    def forward(self, x):
        return self.fc(x)

def test_checkpoint_save_and_load(tmp_path):
    model = DummyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Save original weights to compare later
    original_weights = model.fc.weight.clone()
    
    checkpoint_path = tmp_path / "checkpoint.pth"
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": 5,
        "best_metric": 0.85,
        "config": {"model_name": "dummy"},
        "seed": 42
    }
    
    # Save checkpoint
    saved_path = save_checkpoint(state, checkpoint_path)
    assert saved_path.exists()
    
    # Create new model instance (with different weights) and load checkpoint
    new_model = DummyModel()
    new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.01)
    
    # Check that new model weights are different from original model weights (almost certainly)
    # loading state_dict overrides them.
    loaded_state = load_checkpoint(saved_path)
    
    new_model.load_state_dict(loaded_state["model_state_dict"])
    new_optimizer.load_state_dict(loaded_state["optimizer_state_dict"])
    
    # Assert model weights are restored
    assert torch.equal(new_model.fc.weight, original_weights)
    
    # Assert metadata is restored
    assert loaded_state["epoch"] == 5
    assert loaded_state["best_metric"] == 0.85
    assert loaded_state["config"]["model_name"] == "dummy"
    assert loaded_state["seed"] == 42

def test_load_checkpoint_missing_file():
    with pytest.raises(FileNotFoundError):
        load_checkpoint("non_existent_checkpoint.pth")
