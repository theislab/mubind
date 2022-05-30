import logomaker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import multibind as mb
from sklearn.metrics import r2_score


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
    print('log_etas')
    print(model.log_etas)
    n_cols = len(model.conv_mono)
    if figsize is not None:
        plt.figure(figsize=figsize)
    print('activities')
    activities = np.exp(torch.stack(list(model.log_activities), dim=1).cpu().detach().numpy())
    print(activities)
    for i, m in enumerate(model.conv_mono):
        # print(i, m)

        if m is None:
            continue
        ax = plt.subplot(1, n_cols - 1, i)
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

def plot_loss(model):
    h, c = model.loss_history, model.loss_color
    for i in range(len(h) - 2):
        plt.plot([i, i + 1], h[i: i + 2], c=c[i])
    plt.xlabel('# epochs')
    plt.ylabel('loss')
    plt.show()


# enr_round=-1 means that the last round is used
def kmer_enrichment(model, train, k=8, base_round=0, enr_round=-1):
    # getting the targets and predictions from the model
    seqs, targets, pred = mb.tl.test_network(model, train, next(model.parameters()).device)

    target_kmers = mb.tl.seqs2kmers(seqs, k=k, counts=targets)
    target_labels = ['t'+str(i) for i in range(train.dataset.n_rounds + 1)]
    target_kmers[target_labels] = np.stack(target_kmers['counts'].to_numpy())

    pred_kmers = mb.tl.seqs2kmers(seqs, k=k, counts=pred)
    pred_labels = ['p'+str(i) for i in range(train.dataset.n_rounds + 1)]
    pred_kmers[pred_labels] = np.stack(pred_kmers['counts'].to_numpy())

    counts = target_kmers[target_labels].merge(pred_kmers[pred_labels], left_index=True, right_index=True, how='outer').fillna(0)
    if enr_round == -1:
        enr_round = train.dataset.n_rounds
    counts['enr_pred'] = (1 + counts[pred_labels[enr_round]]) / (1 + counts[pred_labels[base_round]])
    counts['enr_obs'] = (1 + counts[target_labels[enr_round]]) / (1 + counts[target_labels[base_round]])
    counts['f_pred'] = (1 / (enr_round - base_round)) * np.log10(counts['enr_pred'])
    counts['f_obs'] = (1 / (enr_round - base_round)) * np.log10(counts['enr_obs'])

    # plotting
    p = sns.displot(counts, x='enr_pred', y='enr_obs', cbar=True)
    p.set(xscale='log', yscale='log')
    plt.plot([0.1, 10], [0.1, 10], linewidth=2)

    print('R^2:', r2_score(counts['f_obs'], counts['f_pred']))
