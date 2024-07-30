import torch
import pandas as pd
import numpy as np

'''
We want to understand, how each filter contributes to the overall result together with C @ D := H.

Since the i-th row of (A @ H), now denoted as (A @ H)_{i,:}, is nothing but (H.T @ A.T_{:,i}).T, (X_{:,i} is the i-th column of X) we want to compute the matrix vector product between H.T and the i-th column of A.T and find out how much it scales.
We'll later normalize this with the maximum singular value of H.
'''

def compute_contributions(A, G, D, use_hadamard=True):
    H = G * D if use_hadamard else G @ D
    """
    efficient implementation:
    contributions = torch.norm(A @ H, dim=1) / torch.norm(A, dim=1)
    """
    # this implementation for contributions is not efficient, but easier to interpret
    contributions = torch.zeros(A.T.shape[1]) # number of columns of A.T
    i = 0
    for column in torch.unbind(A.T, dim=1):
        contributions[i] = (torch.norm((H.T @ column)) / torch.norm(column)).item()
        i += 1
    max_singular_value = np.linalg.svd(H.to_dense().detach().numpy(), compute_uv=False)[0]# torch.max(torch.abs(torch.linalg.eigvals(H.to_dense()))).item()
    # sort the contributions by their absolute values descendingly
    _, indices = torch.sort(contributions, descending=True)
    return indices, contributions, max_singular_value


# evaluate_metric function, that compares the metric of the original matrix with the metric of the scrambled matrices
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


def normalized_alignment_score(C, D):
    # Compute the element-wise product and then sum all elements
    numerator = torch.sum(C * D)
    # Compute the Frobenius norms of C and D
    norm_C = torch.norm(C, p='fro')
    norm_D = torch.norm(D, p='fro')
    # Compute the normalized alignment score
    score = numerator / (norm_C * norm_D)
    return score.item()