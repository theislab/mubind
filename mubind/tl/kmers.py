import gzip
import os

import numpy as np
import pandas as pd

base_for = "ACGT"
base_rev = "TGCA"
comp_tab = str.maketrans(base_for, base_rev)


def seqs2kmers(seqs, k, counts=None, kmc=False, log_each=None):
    if counts is None:
        counts = np.repeat(1, len(seqs))
    kmers = {}
    for i, s in enumerate(seqs):
        if log_each is not None and i % log_each == 0:
            print("%i out of %i" % (i, len(seqs)))

        for si in range(0, len(s) - k + 1):

            kmer_for = s[si : si + k]
            kmer_rev = kmer_for.translate(comp_tab)[::-1]

            if kmer_for < kmer_rev:
                kmer = kmer_for
            else:
                kmer = kmer_rev

            if not kmer in kmers:
                kmers[kmer] = 0
            kmers[kmer] = counts[i] + (kmers[kmer] if kmer in kmers else 0)

    df_kmers = pd.DataFrame()
    df_kmers.index = kmers.keys()
    df_kmers["counts"] = kmers.values()

    return df_kmers


def _is_tool(name):
    """Check whether `name` is on PATH and marked as executable."""
    # from whichcraft import which
    from shutil import which

    return which(name) is not None


def fastq2kmers(fastq_path, k, kmc_tmp="kmc_tmp", log=False):
    # for f in ['NA.kmc_pre', 'NA.kmc_suf', 'NA.txt']:
    #     if os.path.exists('NA.kmc_pre'):
    #         os.path.
    if not _is_tool("kmc"):
        print("kmc is not found in your system. Please install and add to path binaries")
        print("https://github.com/refresh-bio/KMC")
        assert _is_tool("kmc")

    if not os.path.exists(kmc_tmp):
        os.mkdir(kmc_tmp)

    # the maximum counter needs to be fixed
    n_lines = len(gzip.open(fastq_path).readlines())

    cmd = " ".join(
        [
            "kmc",
            "-cs%i" % n_lines,
            "-k%i" % k,
            "-ci1",
            "-m8",
            fastq_path,
            "NA",
            kmc_tmp,
            "1>",
            "in.txt",
            "2>",
            "err.txt",
        ]
    )
    if log:
        print(cmd)
    os.system(cmd)  # !kmc -k$k -m8 $fastq_path NA kmc_tmp 1> in.txt 2> err.txt
    cmd = " ".join(["kmc_tools", "transform", "NA", "dump", "NA.txt"])
    if log:
        print(cmd)
    os.system(cmd)  # !kmc_tools transform NA dump NA.txt
    df = pd.read_csv("NA.txt", sep="\t", index_col=0, header=None)
    df.columns = ["counts"]
    # df = df.sort_values('counts', ascending=False)
    return df


def log2fc_vs_zero(data, k):
    # get a vector that indicates the mean log2fc between non-zero cycles and the zero cycle
    kmers_by_k = {}
    for round_k in data.columns[1:]:
        print("checking kmers at round", round_k, data.shape)
        kmers_by_k[round_k] = seqs2kmers(data.seq, k, counts=data[round_k])
    ref = kmers_by_k[0]
    pos = pd.concat([kmers_by_k[i] for i in kmers_by_k if i != 0], axis=1)
    seeds = np.log2(pos.mean(axis=1) / (ref.mean(axis=1) + 1)).sort_values(ascending=False)
    return seeds


def levenshteinDistanceDP(token1, token2):
    distances = np.zeros((len(token1) + 1, len(token2) + 1))

    for t1 in range(len(token1) + 1):
        distances[t1][0] = t1

    for t2 in range(len(token2) + 1):
        distances[0][t2] = t2

    a = 0
    b = 0
    c = 0

    for t1 in range(1, len(token1) + 1):
        for t2 in range(1, len(token2) + 1):
            if token1[t1 - 1] == token2[t2 - 1]:
                distances[t1][t2] = distances[t1 - 1][t2 - 1]
            else:
                a = distances[t1][t2 - 1]
                b = distances[t1 - 1][t2]
                c = distances[t1 - 1][t2 - 1]

                if a <= b and a <= c:
                    distances[t1][t2] = a + 1
                elif b <= a and b <= c:
                    distances[t1][t2] = b + 1
                else:
                    distances[t1][t2] = c + 1

    # printDistances(distances, len(token1), len(token2))
    return distances[len(token1)][len(token2)]


def levenshteinDistanceDNA(seq1, seq2):
    return min(levenshteinDistanceDP(seq1, seq2), levenshteinDistanceDP(seq1, seq2.translate(comp_tab)[::-1]))


def printDistances(distances, token1Length, token2Length):
    for t1 in range(token1Length + 1):
        for t2 in range(token2Length + 1):
            print(int(distances[t1][t2]), end=" ")
        print()


# seeds = seeds.index
def get_seed(data, k, n=100):
    seeds = log2fc_vs_zero(data, k)
    seqs = seeds.index[:n]
    m = np.zeros((len(seqs), len(seqs)))
    for i in range(len(seqs)):
        for j in range(i, len(seqs)):
            seq1 = seqs[i]
            seq2 = seqs[j]
            m[i, j] = levenshteinDistanceDNA(seq1, seq2)
            m[j, i] = m[i, j]

    best_seed = np.argmin(m.mean(axis=1))
    seed = seqs[best_seed]
    return seed
