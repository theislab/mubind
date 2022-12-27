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
def combine_models(output_dir, extension='.pkl'):
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

    combined_model = mb.models.Multibind(n_rounds=1, datatype='selex')
    del combined_model.binding_modes.conv_mono[0:]
    del combined_model.binding_modes.conv_di[0:] # to remove Nones from module lists 
    for model in models:
        combined_model.binding_modes.conv_mono.extend(model.binding_modes.conv_mono[1:])
        combined_model.binding_modes.conv_di.extend(model.binding_modes.conv_di[1:])
        combined_model.activities.log_activities += model.activities.log_activities
        combined_model.selex_module._parameters['log_etas'] = combined_model.selex_module._parameters['log_etas'].add(model.selex_module._parameters['log_etas'])

    return combined_model

        

