
import pandas as pd
import numpy as np
import os

base_for = "ACGT"
base_rev = "TGCA"
comp_tab = str.maketrans(base_for, base_rev)

def seqs2kmers(seqs, k, counts=None, kmc=False, log_each=None):
    if counts is None:
        counts = np.repeat(1, len(seqs))
    kmers = {}
    for i, s in enumerate(seqs):
        if log_each is not None and i % log_each == 0:
            print('%i out of %i' % (i, len(seqs)))

        for si in range(0, len(s) - k + 1):

            kmer_for = s[si: si + k]
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
    df_kmers['counts'] = kmers.values()

    return df_kmers

def _is_tool(name):
    """Check whether `name` is on PATH and marked as executable."""
    # from whichcraft import which
    from shutil import which
    return which(name) is not None

def fastq2kmers(fastq_path, k, kmc_tmp='kmc_tmp', log=False):
    # for f in ['NA.kmc_pre', 'NA.kmc_suf', 'NA.txt']:
    #     if os.path.exists('NA.kmc_pre'):
    #         os.path.
    if not _is_tool('kmc'):
        print('kmc is not found in your system. Please install and add to path binaries')
        print('https://github.com/refresh-bio/KMC')
        assert _is_tool('kmc')

    bin = 'kmc'

    if not os.path.exists(kmc_tmp):
        os.mkdir(kmc_tmp)

    cmd = ' '.join(['kmc', '-k%i' % k, '-m8', fastq_path, 'NA', kmc_tmp, '1>', 'in.txt', '2>', 'err.txt'])
    if log:
        print(cmd)
    os.system(cmd) # !kmc -k$k -m8 $fastq_path NA kmc_tmp 1> in.txt 2> err.txt
    cmd = ' '.join(['kmc_tools', 'transform', 'NA', 'dump', 'NA.txt'])
    if log:
        print(cmd)
    os.system(cmd) # !kmc_tools transform NA dump NA.txt
    df = pd.read_csv('NA.txt', sep='\t', index_col=0, header=None)
    df.columns = ['counts']
    # df = df.sort_values('counts', ascending=False)
    return df
