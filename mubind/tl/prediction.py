import copy
import time
import datetime

import numpy as np
import pandas as pd
import sklearn.metrics
import scipy
import torch
import torch.optim as topti
import torch.utils.data as tdata

import mubind as mb

from torch.profiler import profile, ProfilerActivity

def calculate_enrichment(data, approx=True, cols=[0, 1]):
    data["p0"] = data[cols[0]] / np.sum(data[cols[0]])
    data["p1"] = data[cols[1]] / np.sum(data[cols[1]])
    data["enr"] = data["p1"] / data["p0"]
    if approx:
        data["enr_approx"] = np.where(data["p0"] == 0, data["p1"] / (data["p0"] + 1e-06), data["enr"])
    return data


def create_datasets(data_file):
    # read data and calculate additional columns
    data = pd.read_csv(data_file, sep="\t", header=None)
    data.columns = ["seq", 0, 1]
    data = calculate_enrichment(data)
    # divide in train and test data
    test_dataframe = data.sample(frac=0.001)
    train_dataframe = data.drop(test_dataframe.index)
    test_dataframe.index = range(len(test_dataframe))
    train_dataframe.index = range(len(train_dataframe))
    # create datasets and dataloaders
    train_data = mb.datasets.SelexDataset(data_frame=train_dataframe)
    train_loader = tdata.DataLoader(dataset=train_data, batch_size=256, shuffle=True)
    test_data = mb.datasets.SelexDataset(data_frame=test_dataframe)
    test_loader = tdata.DataLoader(dataset=test_data, batch_size=1, shuffle=False)
    return train_loader, test_loader


def test_network(model, dataloader, device):
    all_seqs = []
    all_targets = np.zeros((len(dataloader.dataset), max(dataloader.dataset.n_rounds)), dtype=np.float32)
    all_preds = np.zeros((len(dataloader.dataset), max(dataloader.dataset.n_rounds)), dtype=np.float32)
    position = 0
    store_rev = dataloader.dataset.store_rev
    with torch.no_grad():  # we don't need gradients in the testing phase
        for i, batch in enumerate(dataloader):
            # Get a batch and potentially send it to GPU memory.
            mononuc = batch["mononuc"].to(device)
            b = batch["batch"].to(device) if "batch" in batch else None
            rounds = batch["rounds"].to(device) if "rounds" in batch else None
            countsum = batch["countsum"].to(device) if "countsum" in batch else None
            seq = batch["seq"] if "seq" in batch else None
            residues = batch["residues"].to(device) if "residues" in batch else None
            if residues is not None and store_rev:
                mononuc_rev = batch["mononuc_rev"].to(device)
                inputs = {"mono": mononuc, "mono_rev": mononuc_rev, "batch": b, "countsum": countsum,
                          "residues": residues}
            elif residues is not None:
                inputs = {"mono": mononuc, "batch": b, "countsum": countsum, "residues": residues}
            elif store_rev:
                mononuc_rev = batch["mononuc_rev"].to(device)
                inputs = {"mono": mononuc, "mono_rev": mononuc_rev, "batch": b, "countsum": countsum}
            else:
                inputs = {"mono": mononuc, "batch": b, "countsum": countsum}

            inputs['scale_countsum'] = model.datatype == 'selex'
            output = model(**inputs)

            output = output.cpu().detach().numpy()
            if len(output.shape) == 1:
                output = output.reshape(output.shape[0], 1)
            target = rounds.cpu().detach().numpy()
            if len(target.shape) == 1:
                target = target.reshape(output.shape[0], 1)

            all_preds[position:(position + len(seq)), :] = output
            all_targets[position:(position + len(seq)), :] = target
            all_seqs.extend(seq)
            position += len(seq)

    return all_seqs, all_targets, all_preds


def create_simulated_data(motif="GATA", n_batch=None, n_trials=20000, seqlen=100, batch_sizes=10):
    x2, y2 = mb.datasets.simulate_xy(motif, n_trials=n_trials, seqlen=seqlen, max_mismatches=-1, batch=1)

    # print('skip normalizing...')
    # y2 = ((y2 - y2.min()) / (np.max(y2) - np.min(y2))).astype(np.float32)

    # data = pd.DataFrame({'seq': x1, 'enr_approx': y1})
    data = pd.DataFrame({"seq": x2, "target": y2})
    batch = mb.tl.onehot_covar(np.random.random_integers(0, n_batch - 1, len(y2)))

    # assert n_batch == len(batch_sizes)
    batch_mult = np.array(batch_sizes)[(np.argmax(batch, axis=1))]
    # print(batch_mult)
    # df = pd.DataFrame()
    # df['y.before'] = np.array(data['target'])
    # print(np.argmax(batch, axis=1))
    # df['batch'] = np.argmax(batch, axis=1)
    # df['mult'] = batch_mult
    # df['y.after'] = np.array(data['target']) * np.array(batch_mult)
    # print('head...')
    # print(df.head(25))
    # print('tail')
    # print(df.tail(25))

    data["target"] = np.array(data["target"]) * np.array(batch_mult)
    data["is_count_data"] = np.repeat(1, data.shape[0])
    # assert False

    # divide in train and test data -- copied from above, organize differently!
    test_dataframe = data.sample(frac=0.01)
    train_dataframe = data.drop(test_dataframe.index)
    batch_test = np.argmax(batch[test_dataframe.index], axis=1)
    batch_train = np.argmax(batch[train_dataframe.index], axis=1)

    test_dataframe.index = range(len(test_dataframe))
    train_dataframe.index = range(len(train_dataframe))

    n_train = len(train_dataframe.index)
    n_test = len(test_dataframe.index)
    n_train + n_test

    # create datasets and dataloaders
    # print(train_dataframe)
    # print(batch, n_train)
    # print(batch[:n_train])
    train_data = mb.datasets.ChipSeqDataset(data_frame=train_dataframe)
    train_data.batch = batch_train
    train_data.seq = train_dataframe.seq

    train_loader = tdata.DataLoader(dataset=train_data, batch_size=256, shuffle=True)
    test_data = mb.datasets.ChipSeqDataset(data_frame=test_dataframe)
    test_data.batch = batch_test
    test_data.seq = test_dataframe.seq

    test_loader = tdata.DataLoader(dataset=test_data, batch_size=1, shuffle=False)
    return train_loader, test_loader


def create_multi_data(n_chip=100, n_selex=100, n_batch_selex=3):
    # chip seq
    x_chip, y_chip = mb.datasets.gata_remap(n_sample=n_chip) if n_chip > 0 else [[], []]
    df_chip = pd.DataFrame({"seq": x_chip, "target": y_chip})
    df_chip["is_count_data"] = np.repeat(0, df_chip.shape[0])
    df_chip["batch"] = "0_chip"
    # selex
    # seqlen = 10
    # x_selex, y_selex = mb.datasets.simulate_xy('GATA', n_trials=n_selex, seqlen=seqlen, max_mismatches=-1, batch=50)
    # df_selex = pd.DataFrame({'seq': x_selex, 'target': y_selex})
    # df_selex['is_count_data'] = np.repeat(1, df_selex.shape[0])
    # n_batch = n_batch_selex
    # batch sizes. Dataset 2 has many times more reads than Dataset 1
    batch_sizes = [int(5 * 10**i) for i in range(n_batch_selex)]
    train1, test1 = mb.tl.create_simulated_data(
        motif="GATA",
        n_batch=n_batch_selex,
        n_trials=n_selex,
        seqlen=10,
        batch_sizes=batch_sizes,
    )  # multiplier=100)
    # print(train1.dataset.seq, train1.shape)
    # assert False
    df_selex = pd.DataFrame({"seq": train1.dataset.seq, "target": train1.dataset.target})
    df_selex["batch"] = pd.Series(train1.dataset.batch).astype(str).values + "_selex"
    df_selex["is_count_data"] = np.repeat(1, df_selex.shape[0])
    datasets = [df_chip, df_selex]
    data = pd.concat(datasets)

    # pad nucleotides for leaking regions

    data["seqlen"] = data["seq"].str.len()
    # print(data.head())
    # print(data.tail())
    # assert False

    max_seqlen = data["seq"].str.len().max()
    print("max seqlen", max_seqlen)
    max_seqlen = int(max_seqlen)
    data["k"] = "_"
    padded_seq = data["seq"] + data["k"].str.repeat(max_seqlen - data["seqlen"].astype(int))
    data["seq"] = np.where(data["seqlen"] < max_seqlen, padded_seq, data["seq"])
    data["seqlen"] = data["seq"].str.len()
    assert len(set(data["seqlen"])) == 1

    # batches
    # n_batch = len(datasets)
    # batch = []
    # for i, df in enumerate(datasets):
    #     batch.append(np.repeat(i, df.shape[0]))
    # batch = np.array(np.concatenate(batch))
    # data['batch'] = batch
    # data['batch'] -= np.min(data['batch'])

    # print(data.batch.value_counts())
    # assert False

    #     print(data['batch'].value_counts())
    #     print(data.head())

    # print(data.columns)

    # divide in train and test data -- copied from above, organize differently!
    test_dataframe = data.sample(frac=0.01)
    train_dataframe = data.drop(test_dataframe.index)
    batch_test = test_dataframe["batch"]
    batch_train = train_dataframe["batch"]


    #     print(batch_train)
    #     print(batch_train.value_counts())

    test_dataframe.index = range(len(test_dataframe))
    train_dataframe.index = range(len(train_dataframe))
    n_train = len(train_dataframe.index)
    n_test = len(test_dataframe.index)
    n_train + n_test

    train_data = mb.datasets.MultiDataset(data_frame=train_dataframe)
    train_data.batch = np.array(batch_train)
    train_data.batch_one_hot = mb.tl.onehot_covar(train_data.batch)
    train_loader = tdata.DataLoader(dataset=train_data, batch_size=256, shuffle=True)
    # print(np.bincount(batch))

    test_data = mb.datasets.MultiDataset(data_frame=test_dataframe)
    test_data.batch = np.array(batch_test)
    test_data.batch_one_hot = mb.tl.onehot_covar(test_data.batch)
    test_loader = tdata.DataLoader(dataset=test_data, batch_size=1, shuffle=False)

    return train_loader, test_loader

def scores(model, train, by=None, **kwargs):
    counts = mb.tl.kmer_enrichment(model, train, **kwargs)

    if by is not None and by == 'batch':
        results_by_batch = {}
        for batch_id, grp in counts.groupby('batch'):
            counts_batch = counts[counts['batch'] == batch_id]
            targets = counts_batch[[c for c in counts_batch if c.startswith('t')]]
            pred = counts_batch[[c for c in counts_batch if c.startswith('p')]]
            mask = np.isnan(targets)

            r2_counts = sklearn.metrics.r2_score(targets.to_numpy()[~mask], pred.to_numpy()[~mask])
            # print(counts)
            r2_enr = sklearn.metrics.r2_score(counts_batch["enr_obs"], counts_batch["enr_pred"])
            try:
                r2_f = sklearn.metrics.r2_score(counts_batch["f_obs"], counts_batch["f_pred"])
            except:
                r2_f = np.nan
            try:
                r_f = scipy.stats.pearsonr(counts_batch["f_obs"], counts_batch["f_pred"])[0]
            except:
                r_f = np.nan
            result = {'r2_counts': r2_counts,
                    'r2_foldchange': r2_f, 'r2_enr': r2_enr,
                    'r2_fc': r_f ** 2,
                    'pearson_foldchange': r_f}
            results_by_batch[batch_id] = result
        return results_by_batch

    else:
        targets = counts[[c for c in counts if c.startswith('t')]]
        pred = counts[[c for c in counts if c.startswith('p')]]
        mask = np.isnan(targets)

        r2_counts = sklearn.metrics.r2_score(targets.to_numpy()[~mask], pred.to_numpy()[~mask])
        # print(counts)
        r2_enr = sklearn.metrics.r2_score(counts["enr_obs"], counts["enr_pred"])
        try:
            r2_f = sklearn.metrics.r2_score(counts["f_obs"], counts["f_pred"])
        except:
            r2_f = np.nan
        try:
            r_f = scipy.stats.pearsonr(counts["f_obs"], counts["f_pred"])[0]
        except:
            r_f = np.nan
        return {'r2_counts': r2_counts,
                'r2_foldchange': r2_f, 'r2_enr': r2_enr,
                'r2_fc': r_f ** 2,
                'pearson_foldchange': r_f}

def kmer_enrichment(model, train, k=None, base_round=0, enr_round=-1, pseudo_count=1):
    # getting the targets and predictions from the model
    seqs, targets, pred = mb.tl.test_network(model, train, next(model.parameters()).device)
    counts = None
    target_labels = ["t" + str(i) for i in range(max(train.dataset.n_rounds))]
    pred_labels = ["p" + str(i) for i in range(max(train.dataset.n_rounds))]

    if k is not None:
        target_kmers = mb.tl.seqs2kmers(seqs, k=k, counts=targets)
        # print(target_kmers.shape)
        # print(target_kmers.head())
        # print(np.stack(target_kmers["counts"].to_numpy()))

        target_kmers[target_labels] = np.stack(target_kmers["counts"].to_numpy())
        pred_kmers = mb.tl.seqs2kmers(seqs, k=k, counts=pred)
        pred_kmers[pred_labels] = np.stack(pred_kmers["counts"].to_numpy())

        counts = (
            target_kmers[target_labels]
            .merge(pred_kmers[pred_labels], left_index=True, right_index=True, how="outer")
            .fillna(0)
        )
    else:
        t = pd.DataFrame(targets, index = seqs,
                         columns=target_labels)
        p = pd.DataFrame(pred, index = seqs,
                         columns=pred_labels)
        for i in range(max(train.dataset.n_rounds)):
            # print(train.dataset.n_rounds - 1)
            p['p' + str(i)] = np.where(~(train.dataset.n_rounds - 1 < i), p['p' + str(i)], np.nan)
            t['t' + str(i)] = np.where(~(train.dataset.n_rounds - 1 < i), t['t' + str(i)], np.nan)
        counts = pd.concat([t, p], axis=1)
        # print(counts)
        # assert False

        if train.dataset.batch is not None:
            counts['batch'] = train.dataset.batch
        counts['n_rounds'] = train.dataset.n_rounds

    if model.datatype == 'selex':
        if enr_round == -1:
            enr_round = max(train.dataset.n_rounds) - 1

        # print(enr_round)
        # print((pseudo_count + counts[pred_labels[train.dataset.n_rounds]]))
        # assert False
        # print('iterative assignment of scores in last round...')
        # last_target = (~pd.isnull(counts[counts.columns[:9]])).sum(axis=1).values
        target_last_round = np.nan
        pred_last_round = np.nan
        for c in counts.columns:
            if c.startswith('p'):
                pred_last_round = np.where(~pd.isnull(counts[c]), counts[c], pred_last_round)
            if c.startswith('t'):
                target_last_round = np.where(~pd.isnull(counts[c]), counts[c], target_last_round)

        # label_last_round = ('p' + (counts['n_rounds'] - 1).astype(str))
        # pred_last_round = np.array([r[label_last_round[ri]] for ri, r in counts.iterrows()])
        # label_last_round = ('t' + (counts['n_rounds'] - 1).astype(str))
        # target_last_round = np.array([r[label_last_round[ri]] for ri, r in counts.iterrows()])

        counts["enr_pred"] = (pseudo_count + pred_last_round) / (pseudo_count + counts[pred_labels[base_round]])
        # counts["enr_pred"] = (pseudo_count + counts[pred_labels[enr_round]]) / (pseudo_count + counts[pred_labels[base_round]])
        counts["enr_obs"] = (pseudo_count + target_last_round) / (pseudo_count + counts[target_labels[base_round]])
        # counts["enr_obs"] = (pseudo_count + counts[target_labels[enr_round]]) / (pseudo_count + counts[target_labels[base_round]])

        counts["f_pred"] = (1 / (enr_round - base_round)) * np.log10(counts["enr_pred"])
        counts["f_obs"] = (1 / (enr_round - base_round)) * np.log10(counts["enr_obs"])
    elif model.datatype == 'pbm':  # assuming only one column of numbers to be modeled
        counts["enr_pred"] = counts['p0']
        counts["enr_obs"] = counts['t0']
        counts["f_pred"] = counts['p0']
        counts["f_obs"] = counts['t0']
    else:
        assert False

    return counts

def predict(model, train, show=True):
    counts = mb.tl.kmer_enrichment(model, train)
    return counts
