import torch
import pandas as pd
import numpy as np

def compute_contributions(A, G, D, use_hadamard=True):
    """Computes contribution scores for activities linked to a filter.

    Arguments:
    ---------
    A: : `torch.Tensor`
        Activities matrix.
    G: `torch.Tensor`
        Graph matrix.
    D: `torch.Tensor`
        Graph scaling matrix.
    use_hadamard: `bool` (default: `True`)
        Use hadamard product instead of matrix multiplication.

    Returns:
    -------
    indices: `torch.Tensor`
        Indices of the contributions sorted by their absolute values descendingly
    contributions: `torch.Tensor`
        Contribution scores for each column of matrix A
    max_singular_value: `float`
        Maximum singular value of the matrix H.T with H = G * D or H = G @ D
    """

    # check if input matrices are torch tensors
    if not isinstance(A, torch.Tensor) or not isinstance(G, torch.Tensor) or not isinstance(D, torch.Tensor):
        raise TypeError("A, G, and D must be torch tensors")
    # check if input matrices are not empty
    if A.numel() == 0 or G.numel() == 0 or D.numel() == 0:
        raise ValueError("A, G, and D must not be empty")
    # check if dimensions are correct
    if A.shape[1] != G.shape[0]:
        raise ValueError("Dimension mismatch: number of columns of A must match the number of rows of G")
    if not use_hadamard and G.shape[1] != D.shape[0]:
        raise ValueError("Dimension mismatch: number of columns of G must match the number of rows of D")
    if use_hadamard and G.shape != D.shape:
        raise ValueError("Dimension mismatch: G and D must have the same shape when use_hadamard is True")
    
    H = G * D if use_hadamard else G @ D
    contributions = torch.norm(A @ H, dim=1) / torch.norm(A, dim=1)

    ''' 
    this implementation for contributions is not efficient, but easier to interpret
    contributions = torch.zeros(A.T.shape[1]) # number of columns of A.T
    i = 0
    for column in torch.unbind(A.T, dim=1):
        contributions[i] = (torch.norm((H.T @ column)) / torch.norm(column)).item()
        i += 1
    '''

    max_singular_value = np.linalg.svd(H.T.to_dense().detach().numpy(), compute_uv=False)[0]
    # sort the contributions by their absolute values descendingly
    _, indices = torch.sort(contributions, descending=True)
    return indices, contributions, max_singular_value


def metric_scramble_comparison(C,
                               D,
                               metric,
                               scramble_type,
                               n_scrambles=1000,
                               verbose=True):
    """Comparing metric scores of the original matrix with metric scores of scrambled matrices

    Arguments:
    ---------
    C: : `torch.Tensor`
        Graph matrix.
    D: `torch.Tensor`
        Graph scaling matrix.
    scramble_type: `str`
        Type of scrambling: 'flat', 'row', or 'column'
    n_scrambles: `int` (default: `1000`)
        Number of scrambled matrices to compare
    verbose: `bool` (default: `True`)
        Print summary statistics of the scores of scrambled matrices and the score of the original matrix

    Returns:
    -------
    scores_scrambled_df: `pandas.DataFrame`
        Results of the metric scores of the scrambled matrices
    """
    if C.is_sparse:
        C = C.to_dense()
    if D.is_sparse:
        D = D.to_dense()
    score_D = metric(C, D)
    scores_scrambled = []
    if scramble_type == "flat":
        # scrambling flattened D - this way we get (n^2)! different possible matrices instead of (n!)^2
        D_flat = D.flatten()
        scores_scrambled = [metric(C, D_flat[torch.randperm(496**2)].reshape(496, 496)) for _ in range(n_scrambles)]
    elif scramble_type == "row":
        scores_scrambled = [metric(C, D[torch.randperm(496), :]) for _ in range(n_scrambles)]
    elif scramble_type == "column":
        scores_scrambled = [metric(C, D[:, torch.randperm(496)]) for _ in range(n_scrambles)]
    else:
        raise ValueError("scramble_type must be one of 'flat', 'row', 'column'")
    scores_scrambled_df = pd.DataFrame(scores_scrambled, columns=['score'])
    if verbose:
        print(f"Summary statistics of the scores of scrambled matrices: \n{scores_scrambled_df.describe()} \n \n")
        print(f"This is the score of the original matrix: {score_D}")
    
    return scores_scrambled_df