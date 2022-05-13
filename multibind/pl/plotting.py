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
    for i, m in enumerate(model.conv_mono):
        print(i, m)
        if m is None:
            continue
        weights = m.weight
        weights = weights.squeeze().cpu().detach().numpy()
        weights = pd.DataFrame(weights)
        weights.index = 'A', 'C', 'G', 'T'
        crp_logo = logomaker.Logo(weights.T, shade_below=.5, fade_below=.5)
        plt.show()