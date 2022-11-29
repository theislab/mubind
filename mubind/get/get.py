
import numpy as np
import pandas as pd


def conv_mono_df(model):
    df = []
    for conv_mono in model.binding_modes.conv_mono:
        if conv_mono is None:
            continue
        weights = conv_mono.weight
        weights = weights.squeeze().cpu().detach().numpy()
        weights = pd.DataFrame(weights)
        weights.index = "A", "C", "G", "T"
        df.append(weights)
    return df

