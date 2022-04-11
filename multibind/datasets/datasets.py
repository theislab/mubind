import multibind as mb
import random
import numpy as np
import itertools
from difflib import SequenceMatcher
import random

def _get_random_sequence(seqlen=50, options='ACTG'):
    return ''.join(random.choice(options) for _ in range(seqlen))


def _similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def simulate_data(motif, seqlen=50, n_trials=1000, random_seed=500):
    seqs = [_get_random_sequence(seqlen) for i in range(n_trials)]
    
    random.seed(random_seed)
    
    for i in range(len(seqs)):
        s = seqs[i]
        p = random.randint(0, len(s) - len(motif) + 1)
        s = s[:p] + motif + s[p + len(motif):]
        seqs[i] = s

    return seqs

def _mismatch(word, letters, num_mismatches):
    for locs in itertools.combinations(range(len(word)), num_mismatches):
        this_word = [[char] for char in word]
        for loc in locs:
            orig_char = word[loc]
            this_word[loc] = [l for l in letters if l != orig_char]
        for poss in itertools.product(*this_word):
            yield ''.join(poss)
            
# Generate a seq of sequences with a seuquence embedded
def simulate_xy(motif, batch=100, n_trials=500, seqlen=50, max_mismatches=-1, min_count=1):
    import numpy as np
    x = np.array([_get_random_sequence(seqlen) + 'ACGT' for k in range(n_trials)])
    
    options = []
    mismatch_values = list(range(0, len(motif)))
    if max_mismatches > 0:
        mismatch_values = mismatch_values[:max_mismatches]
        
    for n_mismatches in mismatch_values:
        next_options = np.random.choice(list(_mismatch(motif, 'ACGT', n_mismatches)), int(n_trials / len(mismatch_values)))
        # print(n_mismatches, len(next_options), next_options)
        options += list(next_options)
        
    y = []
    for i, opt in zip(range(len(x)), options):
        p = np.random.choice(range(len(x[0]) - 4 - len(opt)))
        x[i] = x[i][:p] + opt + x[i][p + len(opt):]
        y.append(int(_similar(motif, opt) * batch))
    
    y = np.array(y) + min_count
    x = np.array(x)
    
    return x, y
    
def gata_remap(n_sample=5000):
    import screg as scr
    from os.path import join
    # this is the directory where we are saving the annotations at the moment (hard to distribute for multiple genomes, so only ICB at the moment)
    mb.bindome.constants.ANNOTATIONS_DIRECTORY = '/mnt/znas/icb_zstore01/groups/ml01/datasets/annotations'

    # get data for gata1 using remap2020
    peaks = mb.bindome.datasets.REMAP2020.get_remap_peaks('GATA1')
    peaks_cd34 = peaks[peaks[3].str.contains('GATA1:CD34')].sample(n_sample)
    gen_path = join(mb.bindome.constants.ANNOTATIONS_DIRECTORY, 'hg38/genome/hg38.fa')

    import tempfile
    fa_path = tempfile.mkstemp()[1]
    seqs = scr.gen.get_sequences_from_bed(peaks_cd34[['chr', 'summit.start', 'summit.end']],
                                          fa_path, gen_path=gen_path, uppercase=True)
    # only shuffling, no dinucleotide content control
    shuffled_seqs = [[h + '_rand', scr.gen.randomize_sequence(s)] for h, s in seqs]

    x = np.array([s[1] for s in seqs] + [s[1] for s in shuffled_seqs])
    y = np.array([int(i % 2 == 0) for i, s in enumerate([seqs, shuffled_seqs]) for yi in range(len(s))])
    return x, y
