import logomaker
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import r2_score

import scipy
import scipy.cluster.hierarchy as hc
import mubind as mb


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


def conv_mono(model=None, weights_list=None, n_cols=None, n_rows=None,
              figsize=None, flip=False, log=False, show=True):

    if log:
        activities = np.exp(model.get_log_activities().cpu().detach().numpy())
        print("\n#activities")
        print(activities)
        print("\n#log_etas")
        print(model.get_log_etas())

    if n_cols is None:
        n_cols = len(model.binding_modes) if model is not None else len(weights_list)
    if n_rows is None:
        n_rows = 1

    if figsize is not None:
        plt.figure(figsize=figsize)
    for i in range(max(n_cols, n_rows)):
        weights = None
        if model is not None:
            weights = model.get_kernel_weights(i)
            if weights is None:
                continue
            weights = weights.squeeze().cpu().detach().numpy()
        else:
            weights = weights_list[i]
            if weights is None:
                continue
            weights = weights.squeeze()

        weights = pd.DataFrame(weights)
        weights.index = "A", "C", "G", "T"

        ax = plt.subplot2grid((n_rows, n_cols),
                              (i if n_rows != 1 else 0, i if n_cols != 1 else 0), frame_on=False)

        if flip:
            weights = weights.loc[::-1, ::-1].copy()
            weights.columns = range(weights.shape[1])
            weights.index = "A", "C", "G", "T"
        # print(weights)

        crp_logo = logomaker.Logo(weights.T, shade_below=0.5, fade_below=0.5, ax=ax)
        plt.title(i)

    if show:
        plt.show()


def conv_di(model, figsize=None, mode='complex', show=True, ax=None): # modes include simple/complex/triangle
    n_cols = len(model.binding_modes)
    if figsize is not None:
        plt.figure(figsize=figsize)

    for i, m in enumerate(model.binding_modes.conv_di):
        # print(i, m)
        weights = model.get_kernel_weights(i, dinucleotide=True)
        if weights is None:
            continue
        ax = plt.subplot(1, n_cols - 1, i, frame_on=False)
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

        if mode == 'complex':
            df = []
            for c in weights:
                a = weights.index.str[0]
                b = weights.index.str[1]
                df2 = pd.DataFrame()
                df2['a'] = a
                df2['b'] = b
                df2['weights'] = weights[c].values
                df2['pos'] = c
                # df.append(df2)
                df.append(df2.pivot('a', 'b', 'weights'))

            m = pd.concat(df, axis=1)

            sns.heatmap(m, xticklabels=True, cmap="coolwarm", center=0)
            plt.yticks(rotation=0, fontsize=5);
            plt.xticks(rotation=45, ha='center', fontsize=5);

        elif mode == 'triangle':
            df = []
            for c in weights:
                a = weights.index.str[0]
                b = weights.index.str[1]
                df2 = pd.DataFrame()
                df2['a'] = a + str(c)
                df2['b'] = b + str(c)
                df2['weights'] = weights[c].values
                df2['pos'] = c
                df.append(df2.pivot('a', 'b', 'weights'))

            C = pd.concat(df).fillna(0)
            C = np.array(C)
            C = np.tril(C)
            C = np.ma.masked_array(C, C == 0)
            # print(C)
            _heatmap_triangle_horizontal(C, ax)
            # sns.heatmap(pd.concat(df), cmap='coolwarm', center=0)
        else:
            sns.heatmap(weights, cmap="coolwarm", center=0, ax=ax)

    if show:
        plt.show()


def conv(model, figsize=None, flip=False, log=False, mode='triangle', show=True, **kwargs):
    if log:
        activities = np.exp(model.get_log_activities().cpu().detach().numpy())
        print("\n#activities")
        print(activities)
        print("\n#log_etas")
        print(model.get_log_etas())

    is_multibind = isinstance(model, mb.models.Multibind)
    binding_modes = model.binding_modes if is_multibind else model # model can be a list of binding modes

    n_cols = len(binding_modes)
    if figsize is not None:
        plt.figure(figsize=figsize)

    print(n_cols)
    # mono
    for i, m in enumerate(binding_modes.conv_mono):
        # weights = model.get_kernel_weights(i)
        if m is None:
            continue

        weights = m.weight
        if weights is None:
            continue
        weights = weights.squeeze().cpu().detach().numpy()
        weights = pd.DataFrame(weights)
        weights.index = "A", "C", "G", "T"

        ax = plt.subplot2grid((2, (n_cols)), (0, i), frame_on=False)
        if flip:
            weights = weights.loc[::-1, ::-1].copy()
            weights.columns = range(weights.shape[1])
            weights.index = "A", "C", "G", "T"
        # print(weights.shape)
        crp_logo = logomaker.Logo(weights.T, shade_below=0.5, fade_below=0.5, ax=ax)
        # print(type(weights.T.shape[1]))
        xticks = [i for i in list(range(0, weights.T.shape[0], 5))]
        # print(xticks)
        plt.xticks(xticks)
        plt.title(i)

    # dinuc
    for i, m in enumerate(binding_modes.conv_di):
        if m is None:
            continue

        # print(i, m)
        # weights = model.get_kernel_weights(i, dinucleotide=True)
        weights = m.weight
        if weights is None:
            continue
        ax = plt.subplot2grid((2, (n_cols)), (1, i), frame_on=False)
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

        if mode == 'complex':
            df = []
            for c in weights:
                a = weights.index.str[0]
                b = weights.index.str[1]
                df2 = pd.DataFrame()
                df2['a'] = a
                df2['b'] = b
                df2['weights'] = weights[c].values
                df2['pos'] = c
                # df.append(df2)
                df.append(df2.pivot('a', 'b', 'weights'))

            m = pd.concat(df, axis=1)

            sns.heatmap(m, xticklabels=True, cmap="coolwarm", center=0)
            plt.yticks(rotation=0, fontsize=5);
            plt.xticks(rotation=45, ha='center', fontsize=5);

        elif mode == 'triangle':
            df = []

            a = weights.index.str[0]
            b = weights.index.str[1]
            for c in weights:
                # real values
                df2 = pd.DataFrame()
                df2['a'] = a + str(c)
                df2['b'] = b + str(c + 1)
                df2['weights'] = weights[c].values
                df2['pos'] = c
                df.append(df2.pivot('a', 'b', 'weights'))

            C = pd.concat(df).fillna(np.nan)

            # add border columns
            df2 = pd.DataFrame(index=C.index)
            for nt in ['A', 'C', 'G', 'T']:
                df2[nt + str(0)] = np.nan
            df3 = pd.DataFrame(index=C.index)
            for nt in ['A', 'C', 'G', 'T']:
                df3[nt + str(weights.shape[1] + 1)] = np.nan
            C = pd.concat([df2, C, df3], axis=1)

            C = np.array(C)
            a = np.empty((4, C.shape[1]))
            a[:] = np.nan
            C = np.concatenate([a, C, a])

            C = np.tril(C, k=8)
            C = np.ma.masked_array(C, C == 0)
            _heatmap_triangle_horizontal(C, ax)
            # sns.heatmap(pd.concat(df), cmap='coolwarm', center=0)
        else:
            sns.heatmap(weights, cmap="coolwarm", center=0, ax=ax)

    if show:
        plt.show()

def _heatmap_triangle_horizontal(C, axes):
    N = len(C)
    # Transformation matrix for rotating the heatmap.
    A = np.array([(y, x) for x in range(N, -1, -1) for y in range(N + 1)])
    a = .5
    b = 1
    # t = np.array([[a, b], [a, -b]])
    t = np.array([[.5, -.25], [-.5, -.25]])
    A = np.dot(A, t)
    # -1.0 correlation is blue, 0.0 is white, 1.0 is red.
    cmap = plt.cm.coolwarm
    axes.set_xticks([])
    axes.set_yticks([])
    # Plot the correlation heatmap triangle.
    X = A[:, 1].reshape(N + 1, N + 1)
    Y = A[:, 0].reshape(N + 1, N + 1)
    caxes = plt.pcolormesh(X, Y, np.flipud(C), axes=axes, cmap=cmap) #  norm=norm)
    # Remove the ticks and reset the x limit.
    axes.set_xlim(right=0)
    # Add a colorbar below the heatmap triangle.
    cb = plt.colorbar(caxes, ax=axes, orientation='horizontal', shrink=0.9825,
                     fraction=0.05, pad=-0.035, ticks=np.linspace(-1, 1, 5),
                     use_gridspec=True)
    cb.set_label("weight")
    return caxes


def activities(model, n_rows=None, n_cols=None, batch_i=0, batch_names=None, figsize=None):
    # shape of activities: [n_libraries, len(kernels), n_rounds+1]
    activities = np.exp(model.get_log_activities().cpu().detach().numpy())
    if n_cols is None:
        n_cols = activities.shape[0]
    if figsize is not None:
        plt.figure(figsize=figsize)
    for i in range(activities.shape[0]):
        print(i)
        ax = plt.subplot(n_cols, n_cols, i + 1)
        rel_activity = activities[i, :, :] / np.sum(activities[i, :, :])
        sns.heatmap(rel_activity.T, cmap="Reds", ax=ax)
        plt.title("rel contrib. to batch " + str(batch_names[i] if batch_names is not None else i))
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
def kmer_enrichment(model, train, k=None, base_round=0, enr_round=-1, show=True, hue='batch',
                    log_scale=True, style='distplot', xlab='enr_pred', ylab='enr_obs'):
    # getting the targets and predictions from the model
    counts = mb.tl.kmer_enrichment(model, train, k, base_round, enr_round)
    scores = mb.tl.scores(model, train)

    r2_counts = scores['r2_counts']
    r2_fc = scores['r2_fc']
    pearson_fc = scores['pearson_foldchange']

    hue = hue if hue in counts else None
    # plotting

    p = None
    if style == 'distplot':
        p = sns.displot(counts, x=xlab, y=ylab, cbar=True, hue=hue)
    elif style == 'scatter':
        p = sns.scatterplot(counts, x=xlab, y=ylab, hue=hue)
    if log_scale:
        p.set(xscale='log', yscale='log')
    plt.title('k-mer length = %i, n=%i\n' % (k if k is not None else -1, counts.shape[0]) +
              r'$R^2 (counts)$ = %.2f' % (r2_counts) +
              r', $R^2 (fc)$ = %.2f' % r2_fc + ', Pearson\'s R(fc) = %.2f' % pearson_fc)
    # plt.plot([0.1, 10], [0.1, 10], linewidth=2)
    if show:
        plt.show()

    return scores


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


def R2_calculation(model, train, show=True):
    if isinstance(train.dataset, mb.datasets.SelexDataset):
        return [kmer_enrichment(model, train, show=show)]
    elif isinstance(train.dataset, mb.datasets.PBMDataset):
        return R2_per_protein(model, train, next(model.parameters()).device, show_plot=show)
    elif isinstance(train.dataset, mb.datasets.GenomicsDataset):
        return R2_per_protein(model, train, next(model.parameters()).device, show_plot=show)
