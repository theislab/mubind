from .encoding import (
    bin2string,
    mono2dinuc,
    mono2revmono,
    onehot_covar,
    onehot_dinuc,
    onehot_dinuc_fast,
    onehot_dinuc_with_gaps,
    onehot_mononuc,
    onehot_mononuc_multi,
    onehot_mononuc_with_gaps,
    revert_onehot_mononuc,
    onehot_protein,
    get_protein_aa_index,
    string2bin,
)
from .kmers import fastq2kmers, get_seed, log2fc_vs_zero, seqs2kmers
from .loss import MSELoss, MultiDatasetLoss, PoissonLoss, ProboundLoss
from .prediction import (  # SelexDataset,; ChipSeqDataset,; create_datasets,
    create_multi_data,
    create_simulated_data,
    test_network,
    kmer_enrichment,
    scores,
    predict,
    dynamic_score
)
from .aggregation import (
    calculate_distances,
    combine_models,
    binding_modes,
    distances,
    min_distance,
    submatrix,
    distances_dataframe,
    reduce_filters,
    binding_modes_to_multibind,
)
from .probound import load_probound

from .graph import compute_contributions, metric_scramble_comparison