import numpy as np
import pandas as pd
import torch
import torch.optim as topti
import torch.utils.data as tdata


def test_model_aggregation():
    import warnings

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import mubind as mb

    # Use a GPU if available, as it should be faster.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: " + str(device))

    combined_model = mb.tl.aggregation.combine_models('tests')


