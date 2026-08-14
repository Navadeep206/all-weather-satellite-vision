import sys
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from src.utils.device import get_device, get_device_info
from src.utils.seed import set_seed
from src.utils.config import load_config
from src.utils.checkpoint import save_checkpoint, load_checkpoint

class TinyCNN(nn.Module):
    def __init__(self, in_channels: int = 4, out_channels: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

def run_smoke_test() -> bool:
    try:
        # 1. Load project configuration
        project_cfg = load_config("configs/project.yaml")
        data_cfg = load_config("configs/data.yaml")
        config_ok = (project_cfg.get("project_name") == "all-weather-satellite-vision" and
                     data_cfg.get("input_channels") == 4)
        
        # 2. Set random seed
        seed = project_cfg.get("seed", 42)
        set_seed(seed)
        
        # 3. Detect device
        device = get_device()
        device_info = get_device_info()
        
        # Seed reproducibility verification
        set_seed(seed)
        t1 = torch.rand(2, 2)
        set_seed(seed)
        t2 = torch.rand(2, 2)
        seed_ok = torch.equal(t1, t2)
        
        # 4. Create a tiny CNN
        in_ch = data_cfg.get("input_channels", 4)
        out_ch = data_cfg.get("target_channels", 3)
        model = TinyCNN(in_channels=in_ch, out_channels=out_ch).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        
        # 5. Create a random input tensor
        patch_size = data_cfg.get("patch_size", 256)
        x = torch.rand(1, in_ch, patch_size, patch_size).to(device)
        y_target = torch.rand(1, out_ch, patch_size, patch_size).to(device)
        
        # 6. Run forward pass
        y_pred = model(x)
        forward_ok = y_pred.shape == (1, out_ch, patch_size, patch_size)
        
        # 7. Calculate L1 loss
        loss_fn = nn.L1Loss()
        loss = loss_fn(y_pred, y_target)
        
        # 8. Run backward pass
        loss.backward()
        backward_ok = True
        
        # 9. Run optimizer step
        optimizer.step()
        optimizer_ok = True
        
        # 10. Save checkpoint
        checkpoint_dir = Path("checkpoints")
        checkpoint_path = checkpoint_dir / "smoke_test_checkpoint.pth"
        state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": 0,
            "best_metric": loss.item(),
            "config": project_cfg,
            "seed": seed
        }
        saved_path = save_checkpoint(state, checkpoint_path)
        save_ok = saved_path.exists()
        
        # 11. Load checkpoint
        loaded_state = load_checkpoint(saved_path, map_location=device)
        load_ok = (loaded_state["epoch"] == 0 and 
                   "model_state_dict" in loaded_state)
        
        # Clean up smoke test checkpoint
        if saved_path.exists():
            saved_path.unlink()
            
        # 12. Print success summary
        print("=" * 40)
        print("ALL-WEATHER SATELLITE VISION")
        print("PHASE 0 SMOKE TEST")
        print("==================")
        print(f"\nProject: {project_cfg.get('project_name')}")
        print(f"PyTorch: {device_info['pytorch_version']}")
        print(f"Device: {device_info['device']}")
        print(f"\nForward pass: {'PASS' if forward_ok else 'FAIL'}")
        print(f"Backward pass: {'PASS' if backward_ok else 'FAIL'}")
        print(f"Optimizer step: {'PASS' if optimizer_ok else 'FAIL'}")
        print(f"Checkpoint save: {'PASS' if save_ok else 'FAIL'}")
        print(f"Checkpoint load: {'PASS' if load_ok else 'FAIL'}")
        print(f"Configuration load: {'PASS' if config_ok else 'FAIL'}")
        print(f"Seed reproducibility: {'PASS' if seed_ok else 'FAIL'}")
        print("\n" + "=" * 40)
        
        all_passed = all([forward_ok, backward_ok, optimizer_ok, save_ok, load_ok, config_ok, seed_ok])
        if all_passed:
            print("PHASE 0 STATUS: PASS")
            print("=" * 20)
            return True
        else:
            print("PHASE 0 STATUS: FAIL")
            print("=" * 20)
            return False
            
    except Exception as e:
        print("\n" + "=" * 40)
        print(f"PHASE 0 SMOKE TEST CRASHED: {e}")
        print("=" * 40)
        print("PHASE 0 STATUS: FAIL")
        print("=" * 20)
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
