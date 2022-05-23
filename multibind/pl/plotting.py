import logomaker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch


def scatter(x, y):
    plt.scatter(x, y)


def create_heatmap(net):
    weights = net.conv_mono.weight
    weights = weights.squeeze().cpu().detach().numpy()
    weights = pd.DataFrame(weights)
    weights.index = "A", "C", "G", "T"
    sns.heatmap(weights, cmap="Reds", vmin=0)


def create_logo(net):
    weights = net.conv_mono.weight
    weights = weights.squeeze().cpu().detach().numpy()
    weights = pd.DataFrame(weights)
    weights.index = "A", "C", "G", "T"
    crp_logo = logomaker.Logo(weights.T, shade_below=0.5, fade_below=0.5)


def conv_mono(model, figsize=None):
    n_cols = len(model.conv_mono)
    if figsize is not None:
        plt.figure(figsize=figsize)
    for i, m in enumerate(model.conv_mono):
        # print(i, m)
        ax = plt.subplot(1, n_cols, i + 1)
        if m is None:
            continue
        weights = m.weight
        weights = weights.squeeze().cpu().detach().numpy()
        weights = pd.DataFrame(weights)
        weights.index = "A", "C", "G", "T"
        crp_logo = logomaker.Logo(weights.T, shade_below=0.5, fade_below=0.5, ax=ax)
    plt.show()


def conv_di(model, figsize=None):
    n_cols = len(model.conv_mono)
    if figsize is not None:
        plt.figure(figsize=figsize)
    for i, m in enumerate(model.conv_di):
        # print(i, m)
        ax = plt.subplot(1, n_cols, i + 1)
        if m is None:
            continue
        weights = m.weight
        weights = weights.squeeze().cpu().detach().numpy()
        weights = pd.DataFrame(weights)
        weights.index = (
            "AA",
            "AC",
            "AG",
            "AT",
            "CA",
            "CC",
            "CG",
            "CT",
            "GA",
            "GC",
            "GG",
            "GT",
            "TA",
            "TC",
            "TG",
            "TT",
        )
        sns.heatmap(weights, cmap="coolwarm", center=0, ax=ax)
    plt.show()


def plot_activities(model, dataloader, figsize=None):
    # shape of activities: [n_libraries, len(kernels), n_rounds+1]
    activities = np.exp(torch.stack(list(model.log_activities), dim=1).cpu().detach().numpy())
    n_cols = activities.shape[0]
    batch_names = dataloader.dataset.batch_names
    if figsize is not None:
        plt.figure(figsize=figsize)
    for i in range(n_cols):
        ax = plt.subplot(1, n_cols, i + 1)
        rel_activity = activities[i, :, :] / np.sum(activities[i, :, :])
        sns.heatmap(rel_activity.T, cmap="Reds", ax=ax)
        plt.title("rel contrib. to batch " + str(batch_names[i]))
        plt.ylabel("selection round")
        plt.xlabel("binding mode rel activity")
    plt.show()
