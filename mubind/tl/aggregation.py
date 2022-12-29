import itertools
import torch
import pickle
from pathlib import Path

import mubind as mb


def get_model_paths(output_path, extension='.pkl'):
    """
    Given a path to a directory containing model pickle files, return a list of paths to those files.
    output_path (str): path to a directory containing model pickle files
    extension (str): of model files (.h5 or .pkl)
    """
    paths = []
    path = Path(output_path)
    # matches all files with `extension` under `output_path`
    for p in path.rglob("*"):
        if extension in p.name:
            paths.append(str(p))
    return paths


def get_binding_modes(items):
    """
    Given a list of model objects or paths to model pickle files (or mixed), return a list of binding modes.
    items (list): list of model objects or paths to model pickle files
    """
    binding_modes, activations, etas = [], [], []
    for item in items:
        if type(item) == str:
            with open(item, 'rb') as f:
                model = pickle.load(f)
        else:
            model = item
        binding_modes.append(model.binding_modes)
        activations.append(model.activities.log_activities)
        etas.append(model.selex_module._parameters['log_etas'])
        etas 
    return binding_modes, activations, etas


def concatanate(binding_modes, activations, etas):
    """
    Combine given binding modes, activations, and etas into a single model.
    """
    model = mb.models.Multibind(n_rounds=1, datatype='selex')
    del model.binding_modes.conv_mono[0:]
    del model.binding_modes.conv_di[0:] # to remove Nones from module lists
    for bm, a, e in zip(binding_modes, activations, etas):
        model.binding_modes.conv_mono.extend(bm.conv_mono[1:])
        model.binding_modes.conv_di.extend(bm.conv_di[1:])
        model.activities.log_activities += a
        model.selex_module._parameters['log_etas'] = model.selex_module._parameters['log_etas'].add(e)
    return model


def combine_models(output_path):
    """
    Given a path to a directory containing model pickle files, return a single model combining all of them.
    output_path (str): path to a directory containing model pickle files
    """
    paths = get_model_paths(output_path)
    binding_modes, activations, etas = get_binding_modes(paths)
    model = concatanate(binding_modes, activations, etas)
    return model

        
def weight_distances(mono, min_k=5):
    d = []
    for a, b in itertools.combinations(mono, r=2): # r-length combinations of elements in the iterable
        a, b = a.weight, b.weight
        a_width, b_width = a.shape[-1], b.shape[-1]
        min_w = min(a_width, b_width)

        for k in range(5, min_w): 
            for i in range(0, a_width - k + 1): 
                ai = a[:, :, :, i : i + k]
                for j in range(0, b_width - k + 1):
                    bi = b[:, :, :, j : j + k]
                    bi_rev = torch.flip(bi, [3])[:, :, [3, 2, 1, 0], :]
                    d.append(((ai-bi)**2).sum() / ai.shape[-1])
                    d.append(((ai-bi_rev)**2).sum() / ai.shape[-1])

    return min(d)