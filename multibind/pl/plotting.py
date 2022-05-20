import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import logomaker
import seaborn as sns
import numpy as np
import torch


def scatter(x, y):
    plt.scatter(x, y)


def create_heatmap(net):
    weights = net.conv_mono.weight
    weights = weights.squeeze().cpu().detach().numpy()
    weights = pd.DataFrame(weights)
    weights.index = 'A', 'C', 'G', 'T'
    sns.heatmap(weights, cmap='Reds', vmin=0)


def create_logo(net):
    weights = net.conv_mono.weight
    weights = weights.squeeze().cpu().detach().numpy()
    weights = pd.DataFrame(weights)
    weights.index = 'A', 'C', 'G', 'T'
    crp_logo = logomaker.Logo(weights.T, shade_below=.5, fade_below=.5)
    
def conv_mono(model):
    n_cols = len(model.conv_mono)
    plt.figure(figsize=(min(6 * n_cols, 20), 5))
    for i, m in enumerate(model.conv_mono):
        # print(i, m)
        ax = plt.subplot(1, n_cols, i + 1)
        if m is None:
            continue
        weights = m.weight
        weights = weights.squeeze().cpu().detach().numpy()
        weights = pd.DataFrame(weights)
        weights.index = 'A', 'C', 'G', 'T'
        crp_logo = logomaker.Logo(weights.T, shade_below=.5, fade_below=.5, ax=ax)
    plt.show()

def conv_di(model):
    n_cols = len(model.conv_mono)
    plt.figure(figsize=(min(6 * n_cols, 20), 5))
    for i, m in enumerate(model.conv_di):
        # print(i, m)
        ax = plt.subplot(1, n_cols, i + 1)
        if m is None:
            continue
        weights = m.weight
        weights = weights.squeeze().cpu().detach().numpy()
        weights = pd.DataFrame(weights)
        weights.index = 'AA', 'AC', 'AG', 'AT', 'CA', 'CC', 'CG', 'CT', 'GA', 'GC', 'GG', 'GT', 'TA', 'TC', 'TG', 'TT'
        sns.heatmap(weights, cmap='coolwarm', center=0, ax=ax)
    plt.show()

def plot_activities(model):
    # shape of activities: [n_libraries, len(kernels), n_rounds+1]
    activities = np.exp(torch.stack(list(model.log_activities), dim=1).cpu().detach().numpy())
    n_cols = activities.shape[0]
    plt.figure(figsize=(min(6*n_cols, 20), 5))
    for i in range(n_cols):
        ax = plt.subplot(1, n_cols, i + 1)
        rel_activity = activities[i, :, :] / np.sum(activities[i, :, :])
        sns.heatmap(rel_activity.T, cmap='Reds', ax=ax)
        plt.title('rel contrib. to lib.' + str(i))
        plt.ylabel('selection round')
        plt.xlabel('binding mode rel activity')
    plt.show()
