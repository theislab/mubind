import numpy as np
import pandas as pd
import torch
from mubind.tl.graph import compute_contributions
import pytest

def test_compute_contributions_hadamard():
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    G = torch.tensor([[1.0, 0.5], [0.5, 1.0]])
    D = torch.tensor([[0.5, 1.0], [1.0, 0.5]])

    indices, contributions, max_singular_value = compute_contributions(A, G, D, use_hadamard=True)
    
    assert indices.shape == contributions.shape
    assert len(contributions) == A.shape[0]
    assert max_singular_value >= 0

def test_compute_contributions_no_hadamard():
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    G = torch.tensor([[1.0, 0.5], [0.5, 1.0]])
    D = torch.tensor([[0.5, 1.0], [1.0, 0.5]])

    indices, contributions, max_singular_value = compute_contributions(A, G, D, use_hadamard=False)
    
    assert indices.shape == contributions.shape
    assert len(contributions) == A.shape[0]
    assert max_singular_value >= 0

def test_compute_contributions_empty():
    A = torch.tensor([[]])
    G = torch.tensor([[]])
    D = torch.tensor([[]])

    with pytest.raises(ValueError):
        compute_contributions(A, G, D)

def test_compute_contributions_single_value():
    A = torch.tensor([[1.0]])
    G = torch.tensor([[2.0]])
    D = torch.tensor([[3.0]])

    indices, contributions, max_singular_value = compute_contributions(A, G, D, use_hadamard=True)
    
    assert indices.shape == contributions.shape
    assert len(contributions) == A.shape[0]
    assert max_singular_value >= 0

def test_compute_contributions_large_matrix():
    A = torch.rand(100, 100)
    G = torch.rand(100, 100)
    D = torch.rand(100, 100)

    indices, contributions, max_singular_value = compute_contributions(A, G, D, use_hadamard=False)
    
    assert indices.shape == contributions.shape
    assert len(contributions) == A.shape[0]
    assert max_singular_value >= 0

def test_compute_contributions_different_dimensions_hadamard():
    A = torch.rand(2, 3)
    G = torch.rand(3, 2)
    D = torch.rand(2, 3)

    with pytest.raises(ValueError):
        compute_contributions(A, G, D, use_hadamard=True)

def test_compute_contributions_different_dimensions_no_hadamard():
    A = torch.rand(2, 4)
    G = torch.rand(3, 3)
    D = torch.rand(3, 3)

    with pytest.raises(ValueError):
        compute_contributions(A, G, D, use_hadamard=False)