import logomaker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import r2_score

import scipy
import scipy.cluster.hierarchy as hc
import multibind as mb


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


def conv_mono(model, figsize=None, flip=False, log=True):

    activities = np.exp(model.get_log_activities().cpu().detach().numpy())

    if log:
        print("\n#activities")
        print(activities)
        print("\n#log_etas")
        print(model.get_log_etas())
    n_cols = len(model.binding_modes)
    if figsize is not None:
        plt.figure(figsize=figsize)
    for i in range(n_cols):
        weights = model.get_kernel_weights(i)
        if weights is None:
            continue
        ax = plt.subplot(1, n_cols - 1, i)
        weights = weights.squeeze().cpu().detach().numpy()
        weights = pd.DataFrame(weights)
        weights.index = "A", "C", "G", "T"

        if flip:
            weights = weights.loc[::-1, ::-1].copy()
            weights.columns = range(weights.shape[1])
            weights.index = "A", "C", "G", "T"
        # print(weights)

        crp_logo = logomaker.Logo(weights.T, shade_below=0.5, fade_below=0.5, ax=ax)
        plt.title(i)
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
    activities = np.exp(model.get_log_activities().cpu().detach().numpy())
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
        plt.plot([i, i + 1], h[i : i + 2], c=c[i])
    plt.xlabel("# epochs")
    plt.ylabel("loss")
    plt.show()


# enr_round=-1 means that the last round is used
def kmer_enrichment(model, train, k=8, base_round=0, enr_round=-1, show=True):
    # getting the targets and predictions from the model
    seqs, targets, pred = mb.tl.test_network(model, train, next(model.parameters()).device)

    target_kmers = mb.tl.seqs2kmers(seqs, k=k, counts=targets)
    target_labels = ["t" + str(i) for i in range(train.dataset.n_rounds + 1)]
    target_kmers[target_labels] = np.stack(target_kmers["counts"].to_numpy())

    pred_kmers = mb.tl.seqs2kmers(seqs, k=k, counts=pred)
    pred_labels = ["p" + str(i) for i in range(train.dataset.n_rounds + 1)]
    pred_kmers[pred_labels] = np.stack(pred_kmers["counts"].to_numpy())

    counts = (
        target_kmers[target_labels]
        .merge(pred_kmers[pred_labels], left_index=True, right_index=True, how="outer")
        .fillna(0)
    )

    if model.datatype == 'selex':
        if enr_round == -1:
            enr_round = train.dataset.n_rounds
        counts["enr_pred"] = (1 + counts[pred_labels[enr_round]]) / (1 + counts[pred_labels[base_round]])
        counts["enr_obs"] = (1 + counts[target_labels[enr_round]]) / (1 + counts[target_labels[base_round]])
        counts["f_pred"] = (1 / (enr_round - base_round)) * np.log10(counts["enr_pred"])
        counts["f_obs"] = (1 / (enr_round - base_round)) * np.log10(counts["enr_obs"])
    elif model.datatype == 'pbm':  # assuming only one column of numbers to be modeled
        counts["enr_pred"] = counts['p0']
        counts["enr_obs"] = counts['t0']
        counts["f_pred"] = counts['p0']
        counts["f_obs"] = counts['t0']
    else:
        assert False

    r2 = r2_score(counts["f_obs"], counts["f_pred"])
    if show:
        # plotting
        p = sns.displot(counts, x="enr_pred", y="enr_obs", cbar=True)
        p.set(xscale="log", yscale="log")
        plt.title('R^2: %.2f' % r2)
        # plt.plot([0.1, 10], [0.1, 10], linewidth=2)
        plt.show()
    return r2


def get_dism(words):
    entries = []
    for i, a in enumerate(words):
        for j, b in enumerate(words):
            if j < i:
                continue
            d = dist(a, b)
            entries.append([i, j, d])
            if i != j:
                entries.append([j, i, d])

    df_dism = pd.DataFrame(entries, columns=['i', 'j', 'd'])
    df_dism = df_dism.pivot('i', 'j', 'd')
    return df_dism


def dist(a, b):
    assert len(a) == len(b)
    return len(a) - sum(ai == bi for ai, bi in zip(a, b))


def alignment_protein(seqs, out_basename=None, cluster=False, figsize=[10, 5], n_min_annot=50):
    linkage = None
    dism = get_dism(seqs)
    linkage = hc.linkage(scipy.spatial.distance.squareform(dism), method='average')

    df = pd.DataFrame([[letter for letter in w] for w in seqs])
    df.index = ['w.%i' % i for i in range(df.shape[0])]
    df.columns = ['c.%i' % i for i in range(df.shape[1])]
    import numpy as np
    df_mask = pd.DataFrame(index=df.index)
    # df_mask['challenge'] = 'white'
    df_colors = df.copy()
    letters = pd.DataFrame(index=list(mb.tl.get_protein_aa_index()))
    # print(letters)
    cmap = {k: idx for idx, k in enumerate(letters.sample(letters.shape[0], random_state=500).index)}
    for c in df_colors:
        df_colors[c] = df[c].map(cmap).astype(int)
    df_dism = pd.DataFrame([[i, j, dist(a, b)] for i, a in enumerate(seqs) for j, b in enumerate(seqs)],
                           columns=['i', 'j', 'd'])
    df_dism = df_dism.pivot('i', 'j', 'd')
    linkage = hc.linkage(scipy.spatial.distance.squareform(df_dism), method='average') if cluster else None
    g = sns.clustermap(df_colors, row_linkage=linkage if cluster else None, col_cluster=False,
                       row_cluster=linkage is not None, annot=df if df.shape[0] < n_min_annot else None,
                       fmt='', cmap=sns.color_palette("Paired"), mask=df == '-',  # row_colors=df_mask,
                       figsize=figsize)  # xticklabels=None, yticklabels=None)
    g.cax.set_visible(False)
    g.ax_heatmap.tick_params(left=False, bottom=False)


def R2_per_protein(model, dataloader, device, show_plot=True):
    target_signal = dataloader.dataset.signal
    pred_signal = pd.DataFrame(data=np.zeros(target_signal.shape), index=dataloader.dataset.seq)
    store_rev = dataloader.dataset.store_rev
    with torch.no_grad():  # we don't need gradients in the testing phase
        for i, batch in enumerate(dataloader):
            # Get a batch and potentially send it to GPU memory.
            mononuc = batch["mononuc"].to(device)
            b = batch["batch"].to(device) if "batch" in batch else None
            countsum = batch["countsum"].to(device) if "countsum" in batch else None
            seq = batch["seq"] if "seq" in batch else None
            residues = batch["residues"].to(device) if "residues" in batch else None
            y = batch["protein_id"] if "protein_id" in batch else None
            if residues is not None and store_rev:
                mononuc_rev = batch["mononuc_rev"].to(device)
                inputs = {"mono": mononuc, "mono_rev": mononuc_rev, "batch": b, "countsum": countsum,
                          "residues": residues}
            elif residues is not None:
                inputs = {"mono": mononuc, "batch": b, "countsum": countsum, "residues": residues}
            else:
                inputs = {"mono": mononuc, "batch": b, "countsum": countsum, "residues": residues, "protein_id": y}

            output = model(**inputs)
            output = output.cpu().detach().numpy()
            y = y.cpu().detach().numpy()

            for i in range(len(output)):
                pred_signal.loc[seq[i], y[i]] = output[i]
    pred_signal = np.array(pred_signal)
    R2_values = np.zeros(target_signal.shape[1])
    for i in range(len(R2_values)):
        R2_values[i] = r2_score(target_signal[:, i], pred_signal[:, i])
    if show_plot:
        plt.hist(R2_values)
        plt.show()
    return R2_values


def R2_calculation(model, train):
    if isinstance(train.dataset, mb.datasets.SelexDataset):
        return [kmer_enrichment(model, train, show=False)]
    elif isinstance(train.dataset, mb.datasets.PBMDataset):
        return R2_per_protein(model, train, next(model.parameters()).device, show_plot=False)
