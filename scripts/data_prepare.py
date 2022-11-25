#!/usr/bin/env python
# coding: utf-8
import mubind as mb
import numpy as np
import pandas as pd
import bindome as bd
import sys
from pathlib import Path

if __name__ == '__main__':
    """
    read fastq files, prepare input files for modeling
    """

    import argparse
    import os

    parser = argparse.ArgumentParser(
        description='Precompute diffusion connectivities for knn data integration methods.')

    parser.add_argument('--annot', help='annotations directory')
    parser.add_argument('-o', '--output', required=True, help='output directory for counts and queries metadata file')
    parser.add_argument('--tf_name', required=True)
    parser.add_argument('--n_sample', default=None, type=int)

    args = parser.parse_args()

    bd.constants.ANNOTATIONS_DIRECTORY = args.annot
    data = bd.bindome.datasets.SELEX.get_data()
    tf_query = args.tf_name
    tf_queries = {tf_query}
    model_by_k = {}

    queries_tsv_outpath = args.output
    queries_directory = Path(queries_tsv_outpath).parent.absolute()
    if not os.path.exists(queries_directory):
        print(queries_directory)
        os.makedirs(queries_directory)

    queries = []
    for tf in tf_queries: # set(data['tf.name']):
        if 'ZERO' in tf:
            continue
        print(tf)

        for library, grp in data.groupby('library'):
            data_sel_tf = grp[(grp['tf.name'] == tf)] #  & (grp['cycle'] == '1')]
            if data_sel_tf.shape[0] == 0:
                continue

            print('loading', tf, ':', library)
            reads_tf = mb.bindome.datasets.SELEX.load_read_counts(tf, data=data_sel_tf)
            data_sel_zero = grp[(grp['cycle'] == 0) & grp['library'].isin(set(grp[grp['tf.name'] == tf]['library']))]  # & grp['accession'].isin(set(grp[grp['tf.name'] == tf]['accession']))]
            reads_zero = mb.bindome.datasets.SELEX.load_read_counts(data=data_sel_zero, library=library)

            print('# zero files found', data_sel_zero.shape)
            print(reads_tf.keys())
            print(reads_zero.keys())

            for k_r0 in reads_zero:
                print(k_r0)

                k_model = tf + '-' + k_r0 + '-' + library

                n_rounds = 1

                next_data = reads_zero[k_r0].copy()
                new_cols = ['seq', k_r0]
                for k_tf in reads_tf:
                    next_data = next_data.merge(reads_tf[k_tf], on='seq', how='outer').fillna(0) # .astype(int)
                    new_cols.append(k_tf)
                # next_data = reads_zero[k_r0].merge(reads_tf[k_tf], on='seq', how='outer').fillna(0) # .astype(int)
                # new_cols = ['seq', k_r0, k_tf]
                next_data.columns = new_cols
                for i, k in enumerate(new_cols[1:]):
                    next_data[k] = next_data[k].astype(int)
                    # next_data[i] = next_data[k].astype(int)

                # next_data = next_data.head(10000)
                if args.n_sample is not None and args.n_sample != -1:
                    next_data = next_data.sample(n=args.n_sample)

                # print(next_data.shape)
                next_data = next_data.set_index('seq')
                # print(next_data.head())

                # not needed for the current model, because the enrichment is not predicted
                # next_data = mb.tl.calculate_enrichment(next_data, cols=next_data.columns[1:])

                # assign batch and data type
                next_data['batch'] = 1
                next_data['is_count_data'] = 1

                next_outpath = str(queries_directory) + '/' + k_model + '.tsv.gz'
                next_data.to_csv(next_outpath, sep='\t')

                queries.append([tf_query, k_r0, library, next_outpath, next_data.shape[0]])

    queries = pd.DataFrame(queries, columns=['tf_name', 'r0', 'library', 'counts_path', 'n_sample'])
    queries.to_csv(queries_tsv_outpath, sep='\t')
    sys.exit()
    #                 dataset = mb.datasets.SelexDataset(next_data) # n_rounds=n_rounds)
    #                 train = tdata.DataLoader(dataset=dataset, batch_size=256, shuffle=True)
    #                 train_test = tdata.DataLoader(dataset=dataset, batch_size=1, shuffle=False)

    #                 ### steps to train model
    #                 model = mb.models.DinucSelex(use_dinuc=False, kernels=[0, 14, 12]).to(device) #, n_rounds=n_rounds)
    #                 optimiser = topti.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    #                 criterion = mb.tl.PoissonLoss()
    #                 mb.tl.train_network(model, train, device, optimiser, criterion, num_epochs=200, early_stopping=5, log_each=11)

    #                 # probably here load the state of the best epoch and save
    #                 model.load_state_dict(model.best_model_state)

    #                 # store model parameters and fit for later visualization
    #                 model_by_k[k_model] = model

                    # assert False

                    # stop (debugging)
