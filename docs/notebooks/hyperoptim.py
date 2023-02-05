from __future__ import print_function

import mubind as mb
import numpy as np
import pandas as pd
import torch
import bindome as bd
bd.constants.ANNOTATIONS_DIRECTORY = 'annotations'
# mb.models.MultiBind
import torch.optim as topti
import torch.utils.data as tdata
import matplotlib.pyplot as plt
import logomaker

# Use a GPU if available, as it should be faster.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device: " + str(device))

import warnings
warnings.filterwarnings("ignore")

def get_data():
    n_rounds = 1
    df = mb.bindome.datasets.ProBound.ctcf()
    df = df.sort_values(1, ascending=False).reset_index(drop=True)
    # data = data.sample(n=1000)
    # data.index = range(len(data))
    # dataset = mb.datasets.SelexDataset(data, n_rounds=n_rounds)
    # train = tdata.DataLoader(dataset=dataset, batch_size=256, shuffle=True)

    # data = df.head(1000)
    data = df.sample(n=1000)
    # data = df.copy()

    set(data[0])

    n_rounds = 1

    from matplotlib import rcParams
    rcParams['figure.figsize'] = 5, 1

    print('loading object (# entries)', data.shape[0])
    dataset = mb.datasets.SelexDataset(data, n_rounds=n_rounds)
    train = tdata.DataLoader(dataset=dataset, batch_size=256, shuffle=True)
    return train


import numpy as np
from hyperas import optim
from hyperas.distributions import choice
from hyperopt import Trials, STATUS_OK, tpe


def create_model(x_train):
    print('create model...')
    model_by_k, res_next = mb.tl.train_iterative(x_train, device, min_w=14, show_logo=True, optimize_motif_shift=True,
                                                 # criterion=mb.tl.ProboundLoss(),
                                                 dirichlet_regularization=dirichlet_regularization, # 10 ** dirichlet_regularization_log,
                                                 lr=[0.01, 0.01, 0.01],
                                                 weight_decay=[0.01, 0.001, 0.001], ignore_kernel=ignore_kernel,
                                                 num_epochs=1000, early_stopping=50, use_dinuc=False, # optimiser=torch.optim.LBFGS,
                                                 max_w=15, n_kernels=3, log_each=50, stop_at_kernel=None) #  seed=seed) # seeds.index[0]) #

    best_loss = res_next[0][-1]
    print('Best Loss of model:', best_loss)
    return {'loss': best_loss, 'status': STATUS_OK, 'model': model_by_k}





if __name__ == '__main__':
    print('here...')
    best_run, best_model = optim.minimize(model=create_model,
                                          data=get_data,
                                          algo=tpe.suggest,
                                          max_evals=100,
                                          trials=Trials())

    x_train = get_data()
    # print("Evalutation of best performing model:")
    # print(best_model.evaluate([x_train.X]))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
