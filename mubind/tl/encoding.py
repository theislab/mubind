import itertools

import numpy as np
import torch
from numba import jit
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def mono2revmono(x, fast=True):
    rev = torch.flip(x, [2])
    # rev = rev[:, [3, 2, 1, 0], :]
    rev = torch.flip(rev, [1]) # this 2nd flip only works if the indices are encoding AT in [0, 3] and/or CG in [1, 2]
    return rev


def mono2dinuc(mono):
    # this is a concatenation of columns (i : i - 1) and (i + 1 : i)
    n_mono = mono.shape[1]
    x = torch.cat([mono[:, :, :-1], mono[:, :, 1:]], dim=1)
    # print(x.shape)
    dinuc = torch.cat(
        [  # AX
            (x[:, 0, :] * x[:, 4, :]),
            (x[:, 0, :] * x[:, 5, :]),
            (x[:, 0, :] * x[:, 6, :]),
            (x[:, 0, :] * x[:, 7, :]),
            # CX
            (x[:, 1, :] * x[:, 4, :]),
            (x[:, 1, :] * x[:, 5, :]),
            (x[:, 1, :] * x[:, 6, :]),
            (x[:, 1, :] * x[:, 7, :]),
            # GX
            (x[:, 2, :] * x[:, 4, :]),
            (x[:, 2, :] * x[:, 5, :]),
            (x[:, 2, :] * x[:, 6, :]),
            (x[:, 2, :] * x[:, 7, :]),
            # TX
            (x[:, 3, :] * x[:, 4, :]),
            (x[:, 3, :] * x[:, 5, :]),
            (x[:, 3, :] * x[:, 6, :]),
            (x[:, 3, :] * x[:, 7, :]),
        ],
        dim=1,
    ).reshape(x.shape[0], n_mono**2, x.shape[2])
    return dinuc


dict_dna = "ACGT"
dict_prot = "ACDEFGHIKLMNPQRSTVWY"


def string2bin(s, mode="dna"):
    code = None
    if mode == "dna":
        code = dict_dna
    elif mode == "protein":
        code = dict_prot

    q = " %s" % code
    return sum(5**i * q.index(p) for i, p in enumerate(s))


def bin2string(n, mode="dna"):
    result = ""

    code = None
    if mode == "dna":
        code = dict_dna
    elif mode == "protein":
        code = dict_prot

    while n:
        n, p = divmod(n, 5)
        result += (" %s" % code)[p]
    return result


# models N as (0.25, 0.25, 0.25, 0.25)
@jit
def onehot_mononuc(seq, label_encoder=LabelEncoder(), onehot_encoder=OneHotEncoder(sparse=False)):
    seq_arr = np.array(list(seq + "ACGNT"))
    seq_int = label_encoder.fit_transform(seq_arr)
    pre_onehot = onehot_encoder.fit_transform(seq_int.reshape(-1, 1))
    return (
        pre_onehot.T[[0, 1, 2, 4], :-5] + 0.25 * np.repeat(np.reshape(pre_onehot.T[3, :-5], [1, -1]), 4, axis=0)
    ).astype(np.float32)

def get_protein_aa_index():
    keys_aa = dict_prot + '-'
    return keys_aa

@jit
def onehot_protein(seq, label_encoder=LabelEncoder(), onehot_encoder=OneHotEncoder(sparse=False)):
    keys_aa = get_protein_aa_index()
    # print(seq + keys_aa)
    seq_arr = np.array(list(seq + keys_aa))
    seq_int = label_encoder.fit_transform(seq_arr)
    pre_onehot = onehot_encoder.fit_transform(seq_int.reshape(-1, 1))
    return (pre_onehot.T[:, :len(seq)]).astype(np.float32)


def onehot_mononuc_multi(seqs, max_length):
    result = np.full([len(seqs), 4, max_length], 0.25, dtype=np.float32)
    for i, seq in seqs.items():
        shift = int((max_length - len(seq)) / 2)
        for j in range(len(seq)):
            base = seq[j]
            if base == "A":
                result[i, :, j + shift] = [1, 0, 0, 0]
            elif base == "C":
                result[i, :, j + shift] = [0, 1, 0, 0]
            elif base == "G":
                result[i, :, j + shift] = [0, 0, 1, 0]
            elif base == "T":
                result[i, :, j + shift] = [0, 0, 0, 1]
    return result


def revert_onehot_mononuc(mononuc):
    return np.flip(mononuc, (1, 2)).copy()


def onehot_covar(covar, label_encoder=LabelEncoder(), onehot_encoder=OneHotEncoder(sparse=False)):
    covar_arr = np.array(list(covar))
    covar_int = label_encoder.fit_transform(covar_arr)
    pre_onehot = onehot_encoder.fit_transform(covar_int.reshape(-1, 1))
    return pre_onehot.astype(np.int)


def onehot_dinuc(seq, label_encoder=LabelEncoder(), onehot_encoder=OneHotEncoder(sparse=False)):
    # The added string contains each possible dinucleotide feature once
    extended_seq = seq + "AACAGATCCGCTGGTTA"
    dinuc_arr = np.array([extended_seq[i : i + 2] for i in range(len(extended_seq) - 1)])
    seq_int = label_encoder.fit_transform(dinuc_arr)
    pre_onehot = onehot_encoder.fit_transform(seq_int.reshape(-1, 1))
    return pre_onehot.T[:, :-17].astype(np.float32)


# models _ as (0, 0, 0, 0)
def onehot_mononuc_with_gaps(seq, label_encoder=LabelEncoder(), onehot_encoder=OneHotEncoder(sparse=False)):
    seq_arr = np.array(list(seq + "ACGT_"))
    seq_int = label_encoder.fit_transform(seq_arr)
    pre_onehot = onehot_encoder.fit_transform(seq_int.reshape(-1, 1)).T
    return pre_onehot[:4, :-5].astype(np.float32)


def onehot_dinuc_with_gaps(seq):
    r = 2
    index = np.array(["".join(c) for c in itertools.product("ACTG", repeat=r)])
    m = np.zeros([len(index), len(seq) - r + 1])  # pd.DataFrame(index=index)
    # print(m.shape)
    for i in range(len(seq) - r + 1):
        di = seq[i : i + 2]
        m[:, i] = np.where(di == index, 1, 0)
    return np.array(m)


# works only for sequences which have the same length
def onehot_dinuc_fast(seqs):
    result = np.zeros([len(seqs), 16, len(seqs[0]) - 1], dtype=np.float32)
    for i, seq in seqs.iteritems():
        b2 = seq[0]
        for j in range(len(seq) - 1):
            b1 = b2
            b2 = seq[j + 1]
            if b1 == "A":
                if b2 == "A":
                    result[i, 0, j] = 1
                elif b2 == "C":
                    result[i, 1, j] = 1
                elif b2 == "G":
                    result[i, 2, j] = 1
                elif b2 == "T":
                    result[i, 3, j] = 1
                else:
                    result[i, 0:4, j] = [0.25, 0.25, 0.25, 0.25]
            elif b1 == "C":
                if b2 == "A":
                    result[i, 4, j] = 1
                elif b2 == "C":
                    result[i, 5, j] = 1
                elif b2 == "G":
                    result[i, 6, j] = 1
                elif b2 == "T":
                    result[i, 7, j] = 1
                else:
                    result[i, 4:8, j] = [0.25, 0.25, 0.25, 0.25]
            elif b1 == "G":
                if b2 == "A":
                    result[i, 8, j] = 1
                elif b2 == "C":
                    result[i, 9, j] = 1
                elif b2 == "G":
                    result[i, 10, j] = 1
                elif b2 == "T":
                    result[i, 11, j] = 1
                else:
                    result[i, 8:12, j] = [0.25, 0.25, 0.25, 0.25]
            elif b1 == "T":
                if b2 == "A":
                    result[i, 12, j] = 1
                elif b2 == "C":
                    result[i, 13, j] = 1
                elif b2 == "G":
                    result[i, 14, j] = 1
                elif b2 == "T":
                    result[i, 15, j] = 1
                else:
                    result[i, 12:16, j] = [0.25, 0.25, 0.25, 0.25]
            else:
                if b2 == "A":
                    result[i, [0, 4, 8, 12], j] = [0.25, 0.25, 0.25, 0.25]
                elif b2 == "C":
                    result[i, [1, 5, 9, 13], j] = [0.25, 0.25, 0.25, 0.25]
                elif b2 == "G":
                    result[i, [2, 6, 10, 14], j] = [0.25, 0.25, 0.25, 0.25]
                elif b2 == "T":
                    result[i, [3, 7, 11, 15], j] = [0.25, 0.25, 0.25, 0.25]
                else:
                    result[i, :, j] = [0.0625] * 16
    return result
