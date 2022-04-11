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
    
    
    