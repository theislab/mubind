import pytest

import mubind as mb
import numpy as np
import pandas as pd
import torch
import torch.nn as tnn
import torch.optim as topti


@pytest.mark.skip(reason="This decorator should be removed when test passes.")
@pytest.mark.filterwarnings("ignore:DeprecationWarning")
def test_simdata1():
    # Use a GPU if available, as it should be faster.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: " + str(device))

    motif = "AGGAACCTA"
    x2, y2 = mb.datasets.simulate_xy(motif, n_trials=1000, seqlen=30, max_mismatches=2)
    y2 = ((y2 - y2.min()) / (np.max(y2) - np.min(y2))).astype(np.float32)
    # data = pd.DataFrame({'seq': x1, 'enr_approx': y1})
    data = pd.DataFrame({"seq": x2, 0: np.where(y2 == 0, 1, 0).astype(float), 1: np.where(y2 == 1, 1, 0).astype(float)})

    # divide in train and test data -- copied from above, organize differently!
    train_dataframe = data.copy()
    n_sample = data.shape[0]
    train_dataframe = train_dataframe  # .sample(n=n_sample)
    test_dataframe.index = range(len(test_dataframe))
    train_dataframe.index = range(len(train_dataframe))
    train_dataframe.index = range(n_sample)

    # create datasets and dataloaders
    train_data = mb.datasets.SelexDataset(train_dataframe)
    train_loader = tdata.DataLoader(dataset=train_data, batch_size=256, shuffle=True)
    model = mb.models.DinucSelex(kernels=[0, 12]).to(device)
    optimiser = topti.Adam(net.parameters(), lr=0.01, weight_decay=0.01)
    criterion = mb.tl.PoissonLoss()

    # optimiser = topti.Adam(net.parameters(), lr=0.001)
    l2 = mb.tl.train_network(model, train_100_loader, device, optimiser, criterion, num_epochs=100, log_each=10)
    mb.pl.conv_mono(model)
