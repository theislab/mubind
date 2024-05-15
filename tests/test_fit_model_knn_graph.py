import os
import sys
import pandas as pd
import torch
import torch.utils.data as tdata
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
import unittest
import mubind as mb

class ModelTests(unittest.TestCase):
    N_EPOCHS = 10

    @classmethod
    def setUpClass(cls):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        import anndata

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        early_stopping = 10 
        ad_path = os.path.abspath('tests/_data/pancreas_multiome.h5ad')
        dataloader_path = os.path.abspath('tests/_data/pancreas_multiome.pth')

        ad = anndata.read_h5ad(ad_path)
        cls.train = torch.load(dataloader_path)

        print(ad)
        print(cls.train)

        optimize_log_dynamic = True
        w = [15]
        model = mb.models.Mubind.make_model(cls.train, 4, mb.tl.PoissonLoss(), kernels=[0, 2,] + w, # [0, 2] + w,
                                           # use_dinuc=True, dinuc_mode='full',
                                           optimize_sym_weight=False,
                                           optimize_exp_barrier=True,
                                           optimize_prob_act=True,
                                           optimize_log_dynamic=optimize_log_dynamic,
                                           use_dinuc=False,
                                           device=device,
                                           p_dropout=0.8,
                                           prepare_knn=optimize_log_dynamic,
                                           knn_free_weights=False,
                                           adata=None if not optimize_log_dynamic else ad,
                                           dinuc_mode=None).cuda()
        
        cls.model, _ = model.optimize_iterative(cls.train, n_epochs=cls.N_EPOCHS, show_logo=False,
                                                early_stopping=early_stopping, log_each=50,
                                                opt_kernel_shift=0, opt_kernel_length=0,
                                                verbose=1,
                                                use_dinuc=False,
                                                log_next_r2=False)
        

    # just to formalize that the code above raises no errors
    #   if it does, this method won't be called anyways
    def test_no_errors(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        import mubind as mb

        self.assertTrue(True)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        ModelTests.N_EPOCHS = int(sys.argv.pop())
    unittest.main()
