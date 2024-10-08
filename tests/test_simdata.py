import numpy as np
import pandas as pd
import torch
import torch.optim as topti
import torch.utils.data as tdata
import mubind as mb


def test_simdata_train():
    import warnings

    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # Use a GPU if available, as it should be faster.
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
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
    # divide in train and test data -- copied from above, organize differently!
    train_dataframe = data.copy()
    # data.shape[0]
    train_dataframe = train_dataframe  # .sample(n=n_sample)
    # create datasets and dataloaders
    n_rounds = train_dataframe.shape[1]
    train_data = mb.datasets.SelexDataset(train_dataframe, single_encoding_step=False, n_rounds=n_rounds)
    train_loader = tdata.DataLoader(dataset=train_data, batch_size=256, shuffle=True)
    model = mb.models.Mubind('selex', n_rounds=n_rounds, kernels=[0, 12]).to(device)
    optimiser = topti.Adam(model.parameters(), lr=0.01, weight_decay=0.01)

    criterion = mb.tl.PoissonLoss()
    model = mb.models.Mubind.make_model(train_loader, 4, criterion, device=device, use_dinuc=False) # .cuda()

    optimiser = topti.Adam(model.parameters(), lr=0.01, weight_decay=0.01)
    l2 = model.optimize_simple(train_loader, optimiser, num_epochs=10, log_each=1)

    # mb.pl.conv_mono(model)
