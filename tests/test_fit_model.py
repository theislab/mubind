import os
import sys
import pandas as pd
import torch
import torch.utils.data as tdata
import warnings

import unittest

class ModelTests(unittest.TestCase):
    N_EPOCHS = 2

    @classmethod
    def setUpClass(cls):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        import mubind as mb

        # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # device = 'cpu'

        early_stopping = 10 
        counts_path = os.path.abspath('tests/_data/ALX1-ZeroCycle_TACCAA40NTTA_0_0-TACCAA40NTTA.tsv.gz')

        data = pd.read_csv(counts_path, sep='\t', index_col=0)
        n_rounds = len(data.columns) - 2

        labels = list(data.columns[:n_rounds])
        dataset = mb.datasets.SelexDataset(data, n_rounds=n_rounds, labels=labels)
        cls.train = tdata.DataLoader(dataset=dataset, batch_size=256, shuffle=True)

        model = mb.models.Mubind.make_model(cls.train,
                                            4,
                                            mb.tl.PoissonLoss()) # .cuda()
        cls.model, _ = model.optimize_iterative(cls.train, n_epochs=cls.N_EPOCHS, show_logo=False,
                                                early_stopping=early_stopping, log_each=50,
                                                opt_kernel_shift=[0, 0, 1, 1], opt_kernel_length=[0, 0, 1, 1],
                                                dinuc_mode='local',
                                                verbose=0)

        # setting dinuc to false
        model = mb.models.Mubind.make_model(cls.train,
                                            4,
                                            mb.tl.PoissonLoss(), dinuc_mode='local') # .cuda()

        cls.model, _ = model.optimize_iterative(cls.train, n_epochs=cls.N_EPOCHS, show_logo=False,
                                                early_stopping=early_stopping, log_each=50,
                                                opt_kernel_shift=0, opt_kernel_length=0,
                                                verbose=0,
                                                use_dinuc=False)

        # using dinuc full
        model = mb.models.Mubind.make_model(cls.train,
                                            4,
                                            mb.tl.PoissonLoss(), dinuc_mode='full') # .cuda()
        cls.model, _ = model.optimize_iterative(cls.train, n_epochs=cls.N_EPOCHS, show_logo=False,
                                                early_stopping=early_stopping, log_each=50,
                                                opt_kernel_shift=0, opt_kernel_length=0,
                                                verbose=0,
                                                use_dinuc=True,)

    # just to formalize that the code above raises no errors
    #   if it does, this method won't be called anyways
    def test_no_errors(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        import mubind as mb

        self.assertTrue(True)


    @unittest.skip
    def test_r2_positive(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        import mubind as mb

        r2 = mb.pl.kmer_enrichment(ModelTests.model, ModelTests.train, k=8, show=False)
        self.assertTrue(r2 > 0)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        ModelTests.N_EPOCHS = int(sys.argv.pop())
    unittest.main()
