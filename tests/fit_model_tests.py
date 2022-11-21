import os
import sys
import multibind as mb
import pandas as pd
import torch
import torch.utils.data as tdata

import unittest

class ModelTests(unittest.TestCase):
    N_EPOCHS = 50

    def setUp(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        early_stopping = 100
        counts_path = os.path.abspath('tests/_data/ALX1-ZeroCycle_TACCAA40NTTA_0_0-TACCAA40NTTA.tsv.gz')

        data = pd.read_csv(counts_path, sep='\t', index_col=0)
        n_rounds = len(data.columns) - 3

        labels = list(data.columns[:n_rounds + 1])
        dataset = mb.datasets.SelexDataset(data, n_rounds=n_rounds, labels=labels)
        self.train = tdata.DataLoader(dataset=dataset, batch_size=256, shuffle=True)

        self.model, _ = mb.tl.train_iterative(self.train, device, num_epochs=self.N_EPOCHS, show_logo=False,
                                                    early_stopping=early_stopping, log_each=50,
                                                    verbose=0)


    # just to formalize that the code above raises no errors
    #   if it does, this method won't be called anyways
    def test_no_errors(self):
        self.assertTrue(True)


    @unittest.skip
    def test_r2_positive(self):
        r2 = mb.pl.kmer_enrichment(self.model, self.train, k=8, show=False)
        self.assertTrue(r2 > 0)
    

if __name__ == '__main__':
    if len(sys.argv) > 1:
        ModelTests.N_EPOCHS = int(sys.argv.pop())
    unittest.main()