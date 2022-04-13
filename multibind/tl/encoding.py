import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import itertools

def onehot_mononuc(seq, label_encoder=LabelEncoder(), onehot_encoder=OneHotEncoder(sparse=False)):
    seq_arr = np.array(list(seq + 'ACGT_'))
    seq_int = label_encoder.fit_transform(seq_arr)
    pre_onehot = onehot_encoder.fit_transform(seq_int.reshape(-1, 1))
    return pre_onehot.T[:, :-4].astype(np.float32)

def onehot_covar(covar, label_encoder=LabelEncoder(), onehot_encoder=OneHotEncoder(sparse=False)):
    covar_arr = np.array(list(covar))
    covar_int = label_encoder.fit_transform(covar_arr)
    pre_onehot = onehot_encoder.fit_transform(covar_int.reshape(-1, 1))
    return pre_onehot.astype(np.int)

def onehot_dinuc(seq, label_encoder=LabelEncoder(), onehot_encoder=OneHotEncoder(sparse=False)):
    extended_seq = seq + 'AACAGATCCGCTGGTTA'  # The added string contains each possible dinucleotide feature once
    dinuc_arr = np.array([extended_seq[i:i+2] for i in range(len(extended_seq) - 1)])
    seq_int = label_encoder.fit_transform(dinuc_arr)
    pre_onehot = onehot_encoder.fit_transform(seq_int.reshape(-1, 1))
    return pre_onehot.T[:, :-17].astype(np.float32)

def onehot_mononuc_with_gaps(seq, label_encoder=LabelEncoder(), onehot_encoder=OneHotEncoder(sparse=False)):
    seq_arr = np.array(list(seq + 'ACGT_'))
    seq_int = label_encoder.fit_transform(seq_arr)
    pre_onehot = onehot_encoder.fit_transform(seq_int.reshape(-1, 1)).T
    return pre_onehot[:4, :-5].astype(np.float32)

def onehot_dinuc_with_gaps(seq):
    r = 2
    index = np.array([''.join(c) for c in itertools.product('ACTG', repeat=r)])
    m = np.zeros([len(index), len(seq) - r + 1]) # pd.DataFrame(index=index)
    # print(m.shape)
    for i in range(len(seq) - r + 1):
        di = seq[i: i + 2]
        m[:, i] = np.where(di == index, 1, 0)
    return np.array(m)