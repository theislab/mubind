import itertools
import random
from difflib import SequenceMatcher

import numpy as np
import torch.utils.data as tdata
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import multibind as mb

import pandas as pd


# Class for reading training/testing SELEX dataset files.
class SelexDataset(tdata.Dataset):
    def __init__(self, df, n_rounds=1, enr_series=True, single_encoding_step=False, store_rev=False,
                 labels=None):
        self.n_rounds = n_rounds
        self.enr_series = enr_series
        self.store_rev = store_rev
        self.length = len(df)

        # df = df.reset_index(drop=True)

        # this only works if the columns are equal to the round names (partly obsolete)
        # labels = [i for i in range(n_rounds + 1)]
        # self.rounds = np.array(df[labels])
        self.rounds = np.array(df) if labels is None else np.array(df[labels])
        # print(self.rounds.shape)

        delete_batch_col = False
        if "batch" not in df.columns:
            df["batch"] = np.repeat(0, df.shape[0])
            delete_batch_col = True
        self.batch_names = {}
        for i, name in enumerate(set(df["batch"])):
            self.batch_names[i] = name
            mask = df["batch"] == name
            df.loc[mask, "batch"] = i
        self.batch = np.array(df["batch"])
        self.n_batches = len(set(df["batch"]))
        if delete_batch_col:
            del df['batch']

        seq = df["seq"] if "seq" in df else df.index
        self.seq = np.array(seq)
        self.countsum = np.sum(self.rounds, axis=1).astype(np.float32)

        if single_encoding_step:
            assert len(set(seq.str.len())) == 1
            n_entries = df.shape[0]
            single_seq = "".join(seq.head(n_entries))
            df_single_entry = df.head(1).copy()
            df_single_entry["seq"] = [single_seq]
            le = LabelEncoder()
            oe = OneHotEncoder(sparse=False)
            # single encoding step
            self.mononuc = np.array(
                [mb.tl.onehot_mononuc(row["seq"], le, oe) for index, row in df_single_entry.iterrows()]
            )
            # splitting step
            self.mononuc = np.array(np.split(self.mononuc, n_entries, axis=2)).squeeze(1)
        else:
            max_length = max(set(seq.str.len()))
            self.mononuc = mb.tl.onehot_mononuc_multi(pd.Series(seq), max_length=max_length)

        if store_rev:
            self.mononuc_rev = mb.tl.revert_onehot_mononuc(self.mononuc)

    def __getitem__(self, index):
        # Return a single input/label pair from the dataset.
        sample = {
            "mononuc": self.mononuc[index],
            "batch": self.batch[index],
            "rounds": self.rounds[index],
            "seq": self.seq[index],
            "countsum": self.countsum[index],
        }
        if self.store_rev:
            sample["mononuc_rev"] = self.mononuc_rev[index]
        return sample

    def __len__(self):
        return self.length


# Class for reading training/testing PBM dataset files.
class PBMDataset(tdata.Dataset):
    def __init__(self, df, single_encoding_step=False, store_rev=False, labels=None):
        self.store_rev = store_rev
        self.signal = np.array(df).astype(np.float32) if labels is None else np.array(df[labels]).astype(np.float32)
        self.n_proteins = self.signal.shape[1]
        self.length = self.signal.shape[0] * self.signal.shape[1]
        seq = df["seq"] if "seq" in df else df.index
        self.seq = np.array(seq)

        if single_encoding_step:
            assert len(set(seq.str.len())) == 1
            n_entries = df.shape[0]
            single_seq = "".join(seq.head(n_entries))
            df_single_entry = df.head(1).copy()
            df_single_entry["seq"] = [single_seq]
            le = LabelEncoder()
            oe = OneHotEncoder(sparse=False)
            # single encoding step
            self.mononuc = np.array(
                [mb.tl.onehot_mononuc(row["seq"], le, oe) for index, row in df_single_entry.iterrows()]
            )
            # splitting step
            self.mononuc = np.array(np.split(self.mononuc, n_entries, axis=2)).squeeze(1)
        else:
            max_length = max(set(seq.str.len()))
            self.mononuc = mb.tl.onehot_mononuc_multi(pd.Series(seq), max_length=max_length)

        if store_rev:
            self.mononuc_rev = mb.tl.revert_onehot_mononuc(self.mononuc)

    def __getitem__(self, index):
        # split up the index to one position in the signal matrix
        x = index % self.signal.shape[0]
        y = int((index - x) / self.signal.shape[0])
        # Return a single input/label pair from the dataset.
        sample = {
            "mononuc": self.mononuc[x],
            "rounds": self.signal[x, y:(y + 1)],
            "seq": self.seq[x],
            "protein_id": y,
        }
        if self.store_rev:
            sample["mononuc_rev"] = self.mononuc_rev[index]
        return sample

    def __len__(self):
        return self.length


# Class for reading training/testing genomic datasets, e.g. ATAC data.
class GenomicsDataset(tdata.Dataset):
    def __init__(self, df, single_encoding_step=False, store_rev=False, labels=None):
        self.store_rev = store_rev
        self.signal = np.array(df).astype(np.float32) if labels is None else np.array(df[labels]).astype(np.float32)
        self.n_cells = self.signal.shape[1]
        self.length = self.signal.shape[0] * self.signal.shape[1]
        seq = df["seq"] if "seq" in df else df.index
        self.seq = np.array(seq)

        if single_encoding_step:
            assert len(set(seq.str.len())) == 1
            n_entries = df.shape[0]
            single_seq = "".join(seq.head(n_entries))
            df_single_entry = df.head(1).copy()
            df_single_entry["seq"] = [single_seq]
            le = LabelEncoder()
            oe = OneHotEncoder(sparse=False)
            # single encoding step
            self.mononuc = np.array(
                [mb.tl.onehot_mononuc(row["seq"], le, oe) for index, row in df_single_entry.iterrows()]
            )
            # splitting step
            self.mononuc = np.array(np.split(self.mononuc, n_entries, axis=2)).squeeze(1)
        else:
            max_length = max(set(seq.str.len()))
            self.mononuc = mb.tl.onehot_mononuc_multi(pd.Series(seq), max_length=max_length)

        if store_rev:
            self.mononuc_rev = mb.tl.revert_onehot_mononuc(self.mononuc)

    def __getitem__(self, index):
        # split up the index to one position in the signal matrix
        x = index % self.signal.shape[0]
        y = int((index - x) / self.signal.shape[0])
        # Return a single input/label pair from the dataset.
        sample = {
            "mononuc": self.mononuc[x],
            "rounds": self.signal[x, y:(y + 1)],
            "seq": self.seq[x],
            "protein_id": y,
        }
        if self.store_rev:
            sample["mononuc_rev"] = self.mononuc_rev[index]
        return sample

    def __len__(self):
        return self.length


# Class for reading training/testing PBM data with residue sequences.
class ResiduePBMDataset(tdata.Dataset):
    def __init__(self, df, msa_onehot, single_encoding_step=False, store_rev=False):
        self.n_rounds = 0
        self.store_rev = store_rev
        self.length = df.shape[0] * df.shape[1]
        self.signal = np.array(df).astype(np.float32)
        self.msa_onehot = np.stack(msa_onehot).transpose((0, 2, 1)).astype(np.float32)

        delete_batch_col = False
        if "batch" not in df.columns:
            df["batch"] = np.repeat(0, df.shape[0])
            delete_batch_col = True
        self.batch_names = {}
        for i, name in enumerate(set(df["batch"])):
            self.batch_names[i] = name
            mask = df["batch"] == name
            df.loc[mask, "batch"] = i
        self.batch = np.array(df["batch"])
        self.n_batches = len(set(df["batch"]))
        if delete_batch_col:
            del df['batch']

        seq = df["seq"] if "seq" in df else df.index
        self.seq = np.array(seq)

        if single_encoding_step:
            assert len(set(seq.str.len())) == 1
            n_entries = df.shape[0]
            single_seq = "".join(seq.head(n_entries))
            df_single_entry = df.head(1).copy()
            df_single_entry["seq"] = [single_seq]
            le = LabelEncoder()
            oe = OneHotEncoder(sparse=False)
            # single encoding step
            self.mononuc = np.array(
                [mb.tl.onehot_mononuc(row["seq"], le, oe) for index, row in df_single_entry.iterrows()]
            )
            # splitting step
            self.mononuc = np.array(np.split(self.mononuc, n_entries, axis=2)).squeeze(1)
        else:
            max_length = max(set(seq.str.len()))
            self.mononuc = mb.tl.onehot_mononuc_multi(pd.Series(seq), max_length=max_length)

        if store_rev:
            self.mononuc_rev = mb.tl.revert_onehot_mononuc(self.mononuc)

    def __getitem__(self, index):
        # split up the index to one position in the signal matrix
        x = index % self.signal.shape[0]
        y = int((index - x) / self.signal.shape[0])
        # Return a single input/label pair from the dataset.
        sample = {
            "mononuc": self.mononuc[x],
            "batch": self.batch[x],
            "rounds": self.signal[x, y:(y+1)],
            "seq": self.seq[x],
            "residues": self.msa_onehot[y],
            "protein_id": y,
        }
        if self.store_rev:
            sample["mononuc_rev"] = self.mononuc_rev[index]
        return sample

    def __len__(self):
        return self.length

    def get_max_residue_length(self):
        return self.msa_onehot.shape[1]


# Class for reading training/testing ChIPSeq dataset files.
class ChipSeqDataset(tdata.Dataset):
    def __init__(self, data_frame, use_dinuc=False, batch=None):
        self.batch = batch
        self.target = data_frame["target"].astype(np.float32)
        # self.rounds = self.data[[0, 1]].to_numpy()
        self.le = LabelEncoder()
        self.oe = OneHotEncoder(sparse=False)
        self.length = len(data_frame)
        self.mononuc = np.array(
            [mb.tl.onehot_mononuc(row["seq"], self.le, self.oe) for index, row in data_frame.iterrows()]
        )
        self.dinuc = np.array(
            [mb.tl.onehot_dinuc(row["seq"], self.le, self.oe) for index, row in data_frame.iterrows()]
        )
        self.is_count_data = np.array(data_frame.is_count_data)

    def __getitem__(self, index):
        # Return a single input/label pair from the dataset.
        mononuc_sample = self.mononuc[index]
        target_sample = self.target[index]

        # print(self.batch)
        # print(self.batch.shape)
        batch = self.batch[index]
        dinuc_sample = self.dinuc[index]
        is_count_data = self.is_count_data[index]
        sample = {
            "mononuc": mononuc_sample,
            "dinuc": dinuc_sample,
            "target": target_sample,
            "batch": batch,
            "is_count_data": is_count_data,
        }
        return sample

    def __len__(self):
        return self.length


# Class for curating multi-source data (chip/selex/PBM).
class MultiDataset(tdata.Dataset):
    def __init__(self, data_frame, use_dinuc=False, batch=None):
        self.batch = batch
        self.target = data_frame["target"].astype(np.float32)
        # self.rounds = self.data[[0, 1]].to_numpy()
        self.le = LabelEncoder()
        self.oe = OneHotEncoder(sparse=False)
        self.length = len(data_frame)

        # mononuc = []
        # for index, row in data_frame.iterrows():
        #     # print(row['seq'], self.le, self.oe)
        #     m = mb.tl.onehot_mononuc(row['seq'], self.le, self.oe)
        #     mononuc.append(m)
        # # assert False
        # self.mononuc = np.array(mononuc)
        # print('prepare mononuc feats...')
        self.mononuc = np.array(
            [mb.tl.onehot_mononuc_with_gaps(row["seq"], self.le, self.oe) for index, row in data_frame.iterrows()]
        )
        # print('prepare dinuc feats...')
        self.dinuc = np.array([mb.tl.onehot_dinuc_with_gaps(row["seq"]) for index, row in data_frame.iterrows()])
        self.is_count_data = data_frame["is_count_data"].astype(int)

    def __getitem__(self, index):
        # Return a single input/label pair from the dataset.
        mononuc_sample = self.mononuc[index]
        target_sample = self.target[index]

        # print(self.batch)l
        # print(self.batch.shape)
        batch = self.batch_one_hot[index]
        dinuc_sample = self.dinuc[index]
        is_count_data = self.is_count_data[index]
        sample = {
            "mononuc": mononuc_sample,
            "dinuc": dinuc_sample,
            "target": target_sample,
            "batch": batch,
            "is_count_data": is_count_data,
        }
        return sample

    def __len__(self):
        return self.length


def _get_random_sequence(seqlen=50, options="ACTG"):
    return "".join(random.choice(options) for _ in range(seqlen))


def _similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def simulate_data(motif, seqlen=50, n_trials=1000, random_seed=500):
    seqs = [_get_random_sequence(seqlen) for i in range(n_trials)]

    random.seed(random_seed)

    for i in range(len(seqs)):
        s = seqs[i]
        p = random.randint(0, len(s) - len(motif) + 1)
        s = s[:p] + motif + s[p + len(motif) :]
        seqs[i] = s

    return seqs


def _mismatch(word, letters, num_mismatches):
    for locs in itertools.combinations(range(len(word)), num_mismatches):
        this_word = [[char] for char in word]
        for loc in locs:
            orig_char = word[loc]
            this_word[loc] = [l for l in letters if l != orig_char]
        for poss in itertools.product(*this_word):
            yield "".join(poss)


# Generate a seq of sequences with a seuquence embedded
def simulate_xy(motif, batch=100, n_trials=500, seqlen=50, max_mismatches=-1, min_count=1):
    import numpy as np

    x = np.array([_get_random_sequence(seqlen) + "ACGT" for k in range(n_trials)])

    options = []
    mismatch_values = list(range(0, len(motif)))
    if max_mismatches > 0:
        mismatch_values = mismatch_values[:max_mismatches]

    for n_mismatches in mismatch_values:
        next_options = np.random.choice(
            list(_mismatch(motif, "ACGT", n_mismatches)),
            int(n_trials / len(mismatch_values)),
        )
        # print(n_mismatches, len(next_options), next_options)
        options += list(next_options)

    y = []
    for i, opt in zip(range(len(x)), options):
        p = np.random.choice(range(len(x[0]) - 4 - len(opt)))
        x[i] = x[i][:p] + opt + x[i][p + len(opt) :]
        y.append(int(_similar(motif, opt) * batch))

    y = np.array(y) + min_count
    x = np.array(x)

    return x, y


def gata_remap(n_sample=5000):
    from os.path import join

    import screg as scr

    # this is the directory where we are saving the annotations at the moment (hard to distribute for multiple genomes, so only ICB at the moment)
    mb.bindome.constants.ANNOTATIONS_DIRECTORY = "/mnt/znas/icb_zstore01/groups/ml01/datasets/annotations"

    # get data for gata1 using remap2020
    peaks = mb.bindome.datasets.REMAP2020.get_remap_peaks("GATA1")
    peaks_cd34 = peaks[peaks[3].str.contains("GATA1:CD34")].sample(n_sample)
    gen_path = join(mb.bindome.constants.ANNOTATIONS_DIRECTORY, "hg38/genome/hg38.fa")

    import tempfile

    fa_path = tempfile.mkstemp()[1]
    seqs = scr.gen.get_sequences_from_bed(
        peaks_cd34[["chr", "summit.start", "summit.end"]],
        fa_path,
        gen_path=gen_path,
        uppercase=True,
    )
    # only shuffling, no dinucleotide content control
    shuffled_seqs = [[h + "_rand", scr.gen.randomize_sequence(s)] for h, s in seqs]

    x = np.array([s[1] for s in seqs] + [s[1] for s in shuffled_seqs])
    y = np.array([int(i % 2 == 0) for i, s in enumerate([seqs, shuffled_seqs]) for yi in range(len(s))])
    return x, y
