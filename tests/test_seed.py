import random
import numpy as np
import torch
from src.utils.seed import set_seed

def test_set_seed_python_random():
    set_seed(42)
    val1 = random.random()
    val2 = random.randint(0, 100)
    
    set_seed(42)
    val3 = random.random()
    val4 = random.randint(0, 100)
    
    assert val1 == val3
    assert val2 == val4

def test_set_seed_numpy():
    set_seed(42)
    arr1 = np.random.rand(5)
    
    set_seed(42)
    arr2 = np.random.rand(5)
    
    assert np.array_equal(arr1, arr2)

def test_set_seed_pytorch():
    set_seed(42)
    tensor1 = torch.rand(5)
    
    set_seed(42)
    tensor2 = torch.rand(5)
    
    assert torch.equal(tensor1, tensor2)
