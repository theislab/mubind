import torch
import pandas as pd
import numpy as np

'''
We want to understand, how each filter contributes to the overall result together with C @ D := H.

Since the i-th row of (A @ H), now denoted as (A @ H)_{i,:}, is nothing but (H.T @ A.T_{:,i}).T, (X_{:,i} is the i-th column of X) we want to compute the matrix vector product between H.T and the i-th column of A.T and find out how much it scales.
We'll later normalize this with the maximum singular value of H.
'''

def compute_contributions(A, G, D, use_hadamard=True):
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


# comparing the metric scores of the original matrix with metric scores of scrambled matrices
def metric_scramble_comparison(C,
                               D,
                               metric,
                               scramble_type,
                               n_scrambles=1000,
                               verbose=True):
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