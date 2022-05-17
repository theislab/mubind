import pandas as pd
import matplotlib.pyplot as plt
import logomaker
import seaborn as sns


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
