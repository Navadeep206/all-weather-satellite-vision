import random
import numpy as np
import torch

def set_seed(seed: int = 42) -> None:
    """Sets random seeds for reproducibility across random, numpy, and torch.
    
    Args:
        seed (int): Seed value to set. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Keep deterministic behavior where practical without hurting performance too much:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif torch.backends.mps.is_available():
        # MPS doesn't have separate manual_seed_all or backend settings as of now,
        # but setting torch.manual_seed handles it.
        pass
