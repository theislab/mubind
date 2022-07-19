import multibind as mb
import numpy as np
import pandas as pd
import torch
from os.path import exists
import bindome as bd
import torch.optim as topti
import torch.utils.data as tdata
import matplotlib.pyplot as plt
import logomaker

# Use a GPU if available, as it should be faster.
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print("Using device: " + str(device))
if __name__ == '__main__':
    """
    read fastq files, prepare input files for modeling
    """

    import argparse
    import os

    parser = argparse.ArgumentParser(
        description='Precompute diffusion connectivities for knn data integration methods.')

    parser.add_argument('-i', '--queries', help='path to queries.tsv with entries')
    parser.add_argument('--out_model', required=True, help='directory to save learned model')
    parser.add_argument('--out_tsv', required=True, help='output path for metrics')
    parser.add_argument('--n_epochs', default=50, help='# of epochs for training')
    parser.add_argument('--is_count_data', default=True)
    
    # Use a GPU if available, as it should be faster.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: " + str(device))

    args = parser.parse_args()
    
    queries = pd.read_csv(args.queries, sep='\t', index_col=0)
    
    out_model = args.out_model
    if not os.path.exists(out_model):
        os.mkdir(out_model)
        
    metrics = []
    for ri, r in queries.iterrows():
        
        counts_path = r['counts_path']
        
        model_path = out_model + '/' + os.path.basename(counts_path).replace('.tsv.gz', '.h5')
        
        data = pd.read_csv(counts_path, sep='\t', index_col=0)
        n_rounds = len(data.columns) - 3
        n_batches = len(set(data.batch))
        print('# rounds', n_rounds)
        print('# batches', n_batches)

        print(data.shape)
        # data = data.head(1000)
        # print(data.shape)

        labels = list(data.columns[:n_rounds + 1])
        print('labels', labels)
        dataset = mb.datasets.SelexDataset(data, n_rounds=n_rounds, labels=labels)
        train = tdata.DataLoader(dataset=dataset, batch_size=256, shuffle=True)
        train_test = tdata.DataLoader(dataset=dataset, batch_size=1, shuffle=False)                
        ### steps to train model

        model = mb.models.Multibind(n_rounds, n_batches, use_dinuc=False, kernels=[0, 14, 12]).to(device) #, n_rounds=n_rounds)
        print(exists(model_path), model_path)
        if not exists(model_path):
            optimiser = topti.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
            criterion = mb.tl.PoissonLoss()

            print('training starts...')

            mb.tl.train_network(model, train, device, optimiser, criterion, num_epochs=args.n_epochs, early_stopping=5, log_each=11)
            # probably here load the state of the best epoch and save 
            model.load_state_dict(model.best_model_state)
            torch.save(model.state_dict(), model_path)
            
        else:
            print('loading model from', model_path)
            model.load_state_dict(torch.load(model_path))
            
        
        r2 = mb.pl.kmer_enrichment(model, train, k=8, show=False)
        print("R^2:", r2)
        
        metrics.append(list(r.values[:-1]) + [args.n_epochs, r2])
        print(metrics[-1])
        
    metrics = pd.DataFrame(metrics, columns=list(queries.columns[:-1]] + ['n_epochs', 'r_2'])
    print(metrics)