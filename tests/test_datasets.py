import numpy as np
import pandas as pd
import torch
import torch.optim as topti
import torch.utils.data as tdata


def test_dataset_index_int():
    import warnings

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import mubind as mb

    # Use a GPU if available, as it should be faster.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: " + str(device))

    motif = "AGGAACCTA"
    x2, y2 = mb.datasets.simulate_xy(motif, n_trials=1000, seqlen=30, max_mismatches=2)
    y2 = ((y2 - y2.min()) / (np.max(y2) - np.min(y2))).astype(np.float32)
    # data = pd.DataFrame({'seq': x1, 'enr_approx': y1})
    data = pd.DataFrame(
        {
            0: np.where(y2 == 0, 1, 0).astype(float),
            1: np.where(y2 == 1, 1, 0).astype(float),
        }
    )
    data.index = x2
    n_rounds = data.shape[1]

    train_data = mb.datasets.SelexDataset(data, single_encoding_step=False, n_rounds=n_rounds, index_type=int)
    train_loader = tdata.DataLoader(dataset=train_data, batch_size=256, shuffle=True)

    return train_loader


def test_seq_conversion():
    import warnings

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import mubind as mb

    # Use a GPU if available, as it should be faster.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: " + str(device))

    motif = "AGGAACCTA"
    x2, _ = mb.datasets.simulate_xy(motif, n_trials=1000, seqlen=30, max_mismatches=2)

    ints = list(map(mb.tl.encoding.string2bin, x2))
    strs = list(map(mb.tl.encoding.bin2string, ints))

    assert x2 == strs

