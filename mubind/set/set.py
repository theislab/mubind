import torch
import torch.nn as tnn
import numpy as np

def conv_mono(model, df, position, device):
    v = np.reshape(df.values, (1, 1, 4, -1), order="F")
    # v.values.reshape(1, 1, 4, 12).shape
    model.binding_modes.conv_mono[position].weight = tnn.Parameter(torch.tensor(v, dtype=torch.float32, device=device))

