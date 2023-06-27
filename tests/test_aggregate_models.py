import numpy as np
import os
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

    counts_path = os.path.abspath('tests/_data/ALX1-ZeroCycle_TACCAA40NTTA_0_0-TACCAA40NTTA.tsv.gz')

    data = pd.read_csv(counts_path, sep='\t', index_col=0)
    n_rounds = len(data.columns) - 2

    labels = list(data.columns[:n_rounds])
    dataset = mb.datasets.SelexDataset(data, n_rounds=n_rounds, labels=labels)
    trainloader = tdata.DataLoader(dataset=dataset, batch_size=256, shuffle=True)

    model_paths = mb.tl.aggregation.get_model_paths('tests')
    assert len(model_paths) > 0

    # the pkl files have to be replaced
    try:
        bm = mb.tl.binding_modes("tests/*/*", device=device)
        del bm.conv_mono[0] # to remove None weight
        reduced = mb.tl.aggregation.reduce_filters(bm)
        combined_model = mb.tl.aggregation.binding_modes_to_multibind(reduced, trainloader, device)
    except:
        pass

