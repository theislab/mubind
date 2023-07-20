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

def set_rcParams(parms):
    for k in parms:
        matplotlib.rcParams[k] = parms[k]

def logo_mono(model=None, weights_list=None, n_cols=None, n_rows=None, xticks=True, yticks=True,
              figsize=None, flip=False, log=False, show=True, title=True):
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

    fig, axs = plt.subplots(n_rows, n_cols)

    print('logo mono')
    for i, mi in enumerate(range(n_rows * n_cols) if subset is None else subset):
        ax = axs.flatten()[i]
        ax.set_frame_on(False)

        weights = None

        # print('weights')
        # print(weights)
        if model is not None:
            weights = model.get_kernel_weights(mi)
            print(weights)
            if weights is None:
                fig.delaxes(ax)
                continue
            weights = weights.squeeze().cpu().detach().numpy()
        else:
            if i >= len(weights_list):
                break
            weights = weights_list[i]
            if weights is None:
                fig.delaxes(ax)
                continue
            weights = weights.squeeze()

        weights = pd.DataFrame(weights)
        weights.index = "A", "C", "G", "T"

        # print(i, mi, weights.shape)

        if flip:
            weights = weights.loc[::-1, ::-1].copy()
            weights.columns = range(weights.shape[1])
            weights.index = "A", "C", "G", "T"
        # print(weights)

        # info content

        crp_logo = logomaker.Logo(weights.T, shade_below=0.5, fade_below=0.5, ax=ax)
        if title:
            ax.set_title(mi)
        if not xticks:
            ax.set_xticks([])

    if show:
        plt.show()


def logo_di(model, figsize=None, mode='complex', show=True, ax=None): # modes include simple/complex/triangle
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


def logo(model, figsize=None, flip=False, log=False, mode='triangle',
         show=True, rowspan_mono = 1, rowspan_dinuc = 1, **kwargs):
    import matplotlib.pyplot as plt
    import numpy as np
    if log:
        activities = np.exp(model.get_log_activities().cpu().detach().numpy())
        print("\n#activities")
        print(activities)
        print("\n#log_etas")
        print(model.get_log_etas())

    is_multibind = not isinstance(model, list) # mb.models.Multibind)
    binding_modes = model.binding_modes if is_multibind else model # model can be a list of binding modes

    print(is_multibind)

    n_cols = kwargs.get('n_cols', None)
    if n_cols is None:
        n_cols = len(binding_modes) if not is_multibind else len(model.binding_modes.conv_mono)
    if figsize is not None:
        plt.figure(figsize=figsize)

    print(n_cols)
    # mono
    ci = 0

    n_rows = kwargs.get('n_rows', None)
    if n_rows is None:
        n_rows = rowspan_mono + rowspan_dinuc

    order = kwargs.get('order')

    print('order', order)
    for i, m in enumerate(binding_modes.conv_mono) if order is None else enumerate(order):
        if kwargs.get('log') is True:
            print(i, m)
        if isinstance(m, int):
            m = binding_modes.conv_mono[m]

        # print(i)
        # print(m)
        # weights = model.get_kernel_weights(i)
        if kwargs.get('stop_at') is not None and i >= kwargs.get('stop_at'):
            print('break')
            break

        if i % 10 == 0:
            print('%i out of %i...' % (i, len(binding_modes.conv_mono)))
        if kwargs.get('log'):
            print('mono', i, m)
        if m is None:
            continue

        shape = (n_rows, (n_cols))
        size = (int((i) / n_cols), ci)
        # print(n_rows, n_cols, ci, shape, size, rowspan_mono)
        ax = plt.subplot2grid(shape,
                              size,
                              rowspan=rowspan_mono,
                              frame_on=False)
        ci += 1
        ci = ci % n_cols

        weights = m.weight
        # print(weights.T)

        if weights is None:
            continue
        weights = weights.squeeze().cpu().detach().numpy()
        weights = pd.DataFrame(weights)
        weights.index = "A", "C", "G", "T"
        if flip:
            weights = weights.loc[::-1, ::-1].copy()
            weights.columns = range(weights.shape[1])
            weights.index = "A", "C", "G", "T"

        # print(weights.shape)
        # print(weights)
        if kwargs.get('zero_norm'):
            for c in weights:
                weights[c] = np.where(weights[c] < 0, 0, weights[c])
                weights[c] = np.log2(weights[c] / .25)
                weights[c] = np.where(weights[c] < 0, 0, weights[c])

        # print(weights.T)
        crp_logo = logomaker.Logo(weights.T, shade_below=0.5, fade_below=0.5, ax=ax)


        # print(type(weights.T.shape[1]))
        xticks = [i for i in list(range(0, weights.T.shape[0], 5))]
        # print(xticks)
        plt.xticks([])
        if kwargs.get('title') is not False:
            plt.title(i)

    print('done with mono')

    # dinuc
    ci = 0
    for i, m in enumerate(binding_modes.conv_di):
        if m is None:
            continue
        ax = plt.subplot2grid((n_rows, n_cols), (int((i - 1) / n_cols) + rowspan_mono, ci), rowspan=rowspan_dinuc, frame_on=False)
        ci += 1

        if mode == 'complex':
            # print(i, m)
            # weights = model.get_kernel_weights(i, dinucleotide=True)
            weights = m.weight
            if weights is None:
                continue
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
            df_final = []
            next = binding_modes.conv_di[i]
            rescale_y = False
            if isinstance(next, torch.nn.modules.conv.Conv2d):
                next = [next]
                rescale_y = True
                # print('this is a single array')
                # m = next.weight
                # print('initial weight', m.shape)
                # conv_di_next = []
                # k = binding_modes.conv_mono[i].weight.shape[-1]
                # for i in range(1, k - i):
                #     conv_di_next.append(torch.nn.Conv2d(1, 1, kernel_size=(4, 4)))
                #     m_reshaped = m[:, :, :, i].reshape(conv_di_next[-1].weight.shape)
                #     print(conv_di_next[-1].weight.shape, m_reshaped.shape)
                #     conv_di_next[-1].weight = torch.nn.Parameter(torch.tensor(m_reshaped, dtype=torch.float))
                # print(m, m.shape)
                # next = conv_di_next
            for ki, m in enumerate(next):
                if m is None:
                    continue
                # weights = model.get_kernel_weights(i, dinucleotide=True)
                weights = m.weight

                # print(ki, m)
                # print(weights)
                # print(weights.shape)

                if weights is None:
                    continue
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

                j = weights.columns.shape[0]
                # weights.columns = range(i, weights.columns.shape[0] + i)
                # if i > j:
                #     continue

                # print(weights.shape)
                a = weights.index.str[0]
                b = weights.index.str[1]
                # print(weights.shape)
                df = []
                for c in range(0, j):
                    # real values
                    df2 = pd.DataFrame()
                    df2['a'] = a + str(c)
                    df2['b'] = b + str(c + 1 + ki)
                    df2['weights'] = weights[c].values if c in weights.columns else np.nan
                    df2['pos'] = c
                    df.append(df2.pivot(index='b', columns='a', values='weights'))

                # print(ki, len(df))
                if len(df) != 0:
                    df_concat = pd.concat(df).fillna(np.nan)
                    # sns.heatmap(df_concat)
                    # plt.show()
                    df_final.append(df_concat)

                # if len(df_final) == 10:
                #     break

            df = df_final[0].copy()
            for df2 in df_final:
                df[~pd.isnull(df2)] = df2

            # print(df.shape)
            C = df.copy()
            C = np.array(C)


            import numpy as np
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            import matplotlib as mpl
            cmap = mpl.cm.get_cmap('RdBu_r')
            norm = mpl.colors.Normalize(vmin=np.nanmin(C), vmax=np.nanmax(C))
            size = 20
            delta = 10
            min_x, max_x = None, None
            min_y, max_y = None, None
            for i in range(C.shape[0]):
                for j in range(C.shape[1]):
                    if int(j / 4) > int(i / 4):
                        continue
                    # print(i, j, int(i / 4), int(j / 4))
                    color = cmap(norm(C[i, j]))
                    # print(C[i, j], norm(C[i, j]), color)
                    r2 = patches.Rectangle((-j * size, -i * size), size - delta, size - delta,
                                           color=color, edgecolor=None)
                    t2 = mpl.transforms.Affine2D().rotate_deg(-45) + ax.transData
                    r2.set_transform(t2)

                    # x
                    min_x = r2.get_x() if min_x is None else min(min_x, r2.get_x())
                    max_x = r2.get_x() if max_x is None else max(max_x, r2.get_x())
                    # y
                    min_y = r2.get_y() if min_y is None else min(min_y, r2.get_y())
                    max_y = r2.get_y() if max_y is None else max(max_y, r2.get_y())

                    # print(r2.xy)
                    ax.add_patch(r2)


            scale_x = 1.41421
            n_pos = C.shape[0]
            size_x = size * scale_x
            plt.xlim(-(n_pos + 1) * size_x, size_x * 2.5)
            plt.ylim(-n_pos * (size if not rescale_y else 1.5) * .70710, scale_x * size * 8 / 4)
            # ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            if not xticks:
                plt.xticks([])

            # ax.axis('off')
            # plt.show()

            # sns.heatmap(pd.concat(df), cmap='coolwarm', center=0)
        else:
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


def loss(model):
    h, c = model.loss_history, model.loss_color
    for i in range(len(h) - 2):
        plt.plot([i, i + 1], h[i : i + 2], c=c[i])
    plt.xlabel("# epochs")
    plt.ylabel("loss")
    plt.show()


# enr_round=-1 means that the last round is used
def kmer_enrichment(model, train, k=None, base_round=0, enr_round=-1, show=True, hue='batch',
                    log_scale=True, style='distplot', xlab='enr_pred', ylab='enr_obs', by=None):
    # getting the targets and predictions from the model
    counts = mb.tl.kmer_enrichment(model, train, k, base_round, enr_round)
    scores = mb.tl.scores(model, train, by=by)

    if by != 'batch':
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
    else:
        scores_by_batch = mb.tl.scores(model, train, by='batch')
        for batch in scores_by_batch:
            print(batch)
            scores = scores_by_batch[batch]
            r2_counts = scores['r2_counts']
            r2_fc = scores['r2_fc']
            pearson_fc = scores['pearson_foldchange']
            counts_batch = counts[counts['batch'] == batch]
            hue = hue if hue in counts_batch else None
            # plotting
            p = None
            if style == 'distplot':
                p = sns.displot(counts_batch, x=xlab, y=ylab, cbar=True, hue=hue)
            elif style == 'scatter':
                p = sns.scatterplot(counts_batch, x=xlab, y=ylab, hue=hue)
            if log_scale:
                p.set(xscale='log', yscale='log')
            plt.title('k-mer length = %i, n=%i\n' % (k if k is not None else -1, counts_batch.shape[0]) +
                      r'$R^2 (counts)$ = %.2f' % (r2_counts) +
                      r', $R^2 (fc)$ = %.2f' % r2_fc + ', Pearson\'s R(fc) = %.2f' % pearson_fc)
            # plt.plot([0.1, 10], [0.1, 10], linewidth=2)
            if show:
                plt.show()
        return scores_by_batch


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
