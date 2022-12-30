import numpy as np
import pandas as pd
import itertools
import sklearn.metrics
import scipy
import torch
import torch.optim as topti
import torch.utils.data as tdata
import pickle
import os
from pathlib import Path
import mubind as mb

def load_model(path):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model


# currently assumes we have trained models using a single set of hyperparams
    # and not the grid approach (i.e. single model per setup)
# pickle file name does not matter as long as it is somewhere under output_dir

def _get_models(output_dir, extension='.pkl', device=None, stop_at=None):
    """
    output_dir (str): some parent dir of the output directory
    extension (str): of model files (.h5 or .pkl)
    """
    models = []
    tfs = os.listdir(output_dir)
    path = Path(output_dir)
    # matches all files with `extension` under `output_dir`
    for p in path.rglob("*"):
        if extension in p.name:
            with open(p, 'rb') as f:
                models.append(pickle.load(f))
                print(f'Loaded model {p.name}')

                if stop_at is not None and len(models) >= stop_at:
                    break

    print('# of models', len(models))
    return models


def combine_models(output_dir, extension='.pkl', device=None):
    models = _get_models(output_dir, extension=extension, device=device)
    combined_model = mb.models.Multibind(n_rounds=1, datatype='selex').to(device)

    del combined_model.binding_modes.conv_mono[0:]
    del combined_model.binding_modes.conv_di[0:] # to remove Nones from module lists
    for i, model in enumerate(models):
        print(i, model)
        combined_model.binding_modes.conv_mono.extend(model.binding_modes.conv_mono[1:])
        combined_model.binding_modes.conv_di.extend(model.binding_modes.conv_di[1:])
        combined_model.activities.log_activities += model.activities.log_activities

        print(combined_model.selex_module._parameters['log_etas'].shape)
        print(model.selex_module._parameters['log_etas'].shape)
        combined_model.selex_module._parameters['log_etas'] = combined_model.selex_module._parameters['log_etas'].add(model.selex_module._parameters['log_etas'])

    return combined_model


def binding_modes(output_dir, extension='.pkl', device=None, **kwargs):
    """
    output_dir (str): some parent dir of the output directory
    extension (str): of model files (.h5 or .pkl)
    """
    models = _get_models(output_dir, extension=extension, device=device, **kwargs)
    combined_model = mb.models.Multibind(n_rounds=1, datatype='selex').to(device)
    del combined_model.binding_modes.conv_mono[0:]
    del combined_model.binding_modes.conv_di[0:] # to remove Nones from module lists
    for i, model in enumerate(models):
        # print(i, model)
        combined_model.binding_modes.conv_mono.extend(model.binding_modes.conv_mono[1:])
        combined_model.binding_modes.conv_di.extend(model.binding_modes.conv_di[1:])

    return combined_model.binding_modes


def submatrix(m, start, length, flip, filter_neg_weights=True):
    sub_m = m[:, start: start + length]
    if flip:
        sub_m = np.flip(sub_m, [1])
        sub_m = np.flip(sub_m, [0])

    if filter_neg_weights:
        sub_m[sub_m < 0] = 0

    return sub_m


def distances_dataframe(a, b, min_w_sum=3):
    d = []
    min_w = min(a.shape[-1], b.shape[-1])
    lowest_d = -1, -1
    for k in range(5, min_w):
        # print(k)
        for i in range(0, a.shape[-1] - k + 1):
            ai = submatrix(a, i, k, 0)
            ai_sum = ai.sum()
            if ai_sum < min_w_sum:
                continue

            if ai_sum == 0:
                continue
            for j in range(0, b.shape[-1] - k + 1):

                bi = submatrix(b, j, k, 0)
                bi_sum = bi.sum()
                if bi_sum < min_w_sum:
                    continue

                # print(type(ai), type(bi), ai.shape, bi.shape)
                d.append([i, j, k, ai.shape[-1], bi.shape[-1],
                          ai.sum(), bi.sum(), 0, ((bi - ai) ** 2).sum() / bi.shape[-1]])
                if lowest_d[-1] == -1 or d[-1] < lowest_d[-1] or d[-2] < lowest_d[-1]:
                    lowest_d = i, 0, d[-1]

                bi_rev = submatrix(b, j, k, 1)
                # flipped version
                d.append([i, j, k, ai.shape[-1], bi.shape[-1],
                          ai.sum(), bi.sum(), 1, ((bi_rev - ai) ** 2).sum() / bi.shape[-1]])
                if lowest_d[-1] == -1 or d[-1] < lowest_d[-1] or d[-2] < lowest_d[-1]:
                    lowest_d = i, 1, d[-1]

    res = pd.DataFrame(d, columns=['a_start', 'b_start', 'k', 'a_shape', 'b_shape',
                                   'a_sum', 'b_sum', 'b_flip', 'distance']).sort_values('distance')
    return res

def min_distance(w1, w2):
    df = distances_dataframe(w1, w2)
    if df.shape[0] == 0:
        return np.nan
    return min(df['distance'])

def distances(w1, w2):
    return distances_dataframe(w1, w2)

