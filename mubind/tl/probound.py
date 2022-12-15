import json

import numpy as np
import torch
import torch.nn as tnn

import mubind as mb


def load_probound(json_path):
    with open(json_path) as json_reader:
        json_obj = json.load(json_reader)
        json_reader.close()
    n_rounds = len(json_obj["modelSettings"]["countTable"][0]["modeledColumns"]) - 1
    kernels = []
    for bm in json_obj["modelSettings"]["bindingModes"]:
        kernels.append(bm["size"])
    n_batches = 1
    model = mb.models.DinucSelex(n_rounds=n_rounds, n_batches=n_batches, kernels=kernels)
    model.log_activities = tnn.ParameterList()
    for i, bm in enumerate(json_obj["coefficients"]["bindingModes"]):
        model.log_activities.append(tnn.Parameter(torch.tensor(bm["activity"])))
        if kernels[i] > 0:
            weights = np.array(bm["mononucleotide"])
            weights = np.reshape(weights, (1, 1, 4, -1), order="F")
            model.conv_mono[i].weight = tnn.Parameter(torch.tensor(weights, dtype=torch.float32))
    log_etas = np.reshape(np.array(json_obj["coefficients"]["countTable"][0]["h"]), (1, -1))
    model.log_etas = tnn.Parameter(torch.tensor(log_etas, dtype=torch.float32))
    return model
