import numpy as np
import pandas as pd
import torch
import torch.optim as topti
import torch.utils.data as tdata
import mubind as mb

def test_dataset_index_int():
    import warnings

    warnings.filterwarnings("ignore", category=DeprecationWarning)

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

    return None


def test_seq_conversion():
    import warnings

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    # Use a GPU if available, as it should be faster.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: " + str(device))

    motif = "AGGAACCTA"
    x2, _ = mb.datasets.simulate_xy(motif, n_trials=1000, seqlen=30, max_mismatches=2)

    ints = list(map(mb.tl.encoding.string2bin, x2))
    strs = list(map(mb.tl.encoding.bin2string, ints))

    assert (x2 == strs).all()

def test_download_and_load_dataset():
    import warnings
    ad = mb.datasets.pancreas_rna_pytest()
    return None

    
def test_dataset_memory_increase():
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from sys import getsizeof
    import scipy
    # TODO
    def get_size_mb(a):
        if isinstance(a, pd.DataFrame):
            return round(a.memory_usage(deep=True).sum() / 1024 / 1024, 2)
        elif isinstance(a, scipy.sparse.csr.csr_matrix):
            size = a.data.nbytes + a.indptr.nbytes + a.indices.nbytes
            round(size / 1024 / 1024, 2)
        return round(getsizeof(a) / 1024 / 1024, 2)

    # Use a GPU if available, as it should be faster.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: " + str(device))

    motif = "AGGAACCTA"

    for index_type in str, int:
        for use_sparse in [0, 1]:
            for n_trials in [1000, 10000]:
                x2, y2 = mb.datasets.simulate_xy(motif, n_trials=n_trials,
                                                 seqlen=18, max_mismatches=2)
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

                train_data = mb.datasets.SelexDataset(data, single_encoding_step=False, n_rounds=n_rounds,
                                                      index_type=index_type, use_sparse=use_sparse)
                import pickle

                # in bytes
                size_seq = get_size_mb(train_data.seq) * 1024 * 1024
                size_rounds = get_size_mb(train_data.rounds) * 1024 * 1024

                # train = len(pickle.dumps(train_data.seq))
                # print(index_type, use_sparse, n_trials, size_seq, size_rounds)
                ## let o be the object whose size you want to measure
                train_loader = tdata.DataLoader(dataset=train_data, batch_size=256, shuffle=True)
    return None
