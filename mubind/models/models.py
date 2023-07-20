import itertools
import time
import datetime
import torch
import torch.nn as tnn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import torch.optim as topti
import pandas as pd
import copy

import mubind as mb


class Mubind(tnn.Module):
    """
    Implements the MUBIND model.

    Args:
        datatype (String): Type of the experimental data. "selex" and "pbm" are supported.

    Keyword Args:
        n_rounds (int): Necessary for SELEX data: Number of rounds to be predicted.
        init_random (bool): Use a random initialization for all parameters. Default: True
        padding_const (double): Value for padding DNA-seqs. Default: 0.25
        use_dinuc (bool): Use dinucleotide contributions (not fully implemented for all kind of models). Default: False
        enr_series (bool): Whether the data should be handled as enrichment series. Default: True
        n_batches (int): Number of batches that will occur in the data. Default: 1
        ignore_kernel (list[bool]): Whether a kernel should be ignored. Default: None.
        kernels (List[int]): Size of the binding modes (0 indicates non-specific binding). Default: [0, 15]
        n_kernels (int). Number of filters to be used (including non-specific binding, as a constant).
                         Default: 2 (ns-binding, and one filter)
        init_random (bool): Use a random initialization for all parameters. Default: True
        n_proteins (int): Number of proteins in the dataset. Either n_proteins or n_batches may be used. Default: 1

        bm_generator (torch.nn.Module): PyTorch module which has a weight matrix as output.
        add_intercept (bool): Whether an intercept is used in addition to the predicted binding modes. Default: True
    """

    def __init__(self, datatype, **kwargs):
        super().__init__()

        self.device = kwargs.get('device')
        if self.device is None:
            # Use a GPU if available, as it should be faster.
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print("Using device: " + str(self.device))


        self.optimize_exp_barrier = kwargs.get('optimize_exp_barrier', False)
        self.optimize_kernel_rel = kwargs.get('optimize_kernel_rel', False)
        self.optimize_sym_weight = kwargs.get('optimize_sym_weight', True)
        self.optimize_log_dynamic = kwargs.get('optimize_log_dynamic', False)
        self.optimize_prob_act = kwargs.get('optimize_prob_act', False)

        self.datatype = datatype.lower()
        assert self.datatype in ["selex", "pbm"]
        self.padding_const = kwargs.get("padding_const", 0.25)
        self.use_mono = True
        self.use_dinuc = kwargs.get("use_dinuc", True)
        if "kernels" not in kwargs and "n_kernels" not in kwargs:
            kwargs["kernels"] = [0, 15]
            kwargs["n_kernels"] = len(kwargs["kernels"])
        elif "n_kernels" not in kwargs:
            kwargs["n_kernels"] = len(kwargs["kernels"])
        elif "kernels" not in kwargs and "bm_generator" not in kwargs:
            kwargs["kernels"] = [0] + [15] * (kwargs["n_kernels"] - 1)
        elif "bm_generator" not in kwargs:
            assert len(kwargs["kernels"]) == kwargs["n_kernels"]
        self.kernels = kwargs.get("kernels")
        if self.datatype == "pbm":
            kwargs["target_dim"] = kwargs.get('target_dim', 1)
        elif self.datatype == "selex":
            if "n_rounds" in kwargs:
                kwargs["target_dim"] = kwargs["n_rounds"]
            elif "target_dim" in kwargs:
                kwargs["n_rounds"] = kwargs["target_dim"] - 1
            else:
                print("n_rounds must be provided.")
                assert False

        # assert not ("n_batches" in kwargs and "n_proteins" in kwargs)

        # only keep one padding equals to the length of the max kernel
        if self.kernels is None:
            self.padding = tnn.ConstantPad2d((12, 12, 0, 0), self.padding_const)
        else:
            self.padding = tnn.ConstantPad2d((max(self.kernels) - 1, max(self.kernels) - 1, 0, 0), self.padding_const)
        if "bm_generator" in kwargs and kwargs["bm_generator"] is not None:
            self.binding_modes = BindingModesPerProtein(**kwargs)
        else:
            self.binding_modes = BindingModesSimple(**kwargs)
        self.activities = ActivitiesLayer(**kwargs)
        if self.datatype == "selex":
            self.graph_module = GraphModule(**kwargs)

        self.kernel_rel = None
        if kwargs.get('kernel_sim') is not None:
            self.kernel_rel = torch.tensor(kwargs.get('kernel_sim'))

        self.n_kernels = kwargs['n_kernels']
        self.best_model_state = None
        self.best_loss = None
        self.loss_history = []
        self.loss_history_sym_weights = []
        self.loss_history_log_dynamic = []
        self.r2_history = []
        self.loss_color = []
        self.total_time = 0


    @staticmethod
    def make_model(train, n_kernels, criterion, init_random=True, **kwargs):
        # verbose print declaration
        if kwargs.get('verbose'):
            def vprint(*args, **kwargs):
                print(*args, **kwargs)
        else:
            vprint = lambda *a, **k: None  # do-nothing function

        if (isinstance(train, list) and 'SelexDataset' in str(type(train[0].dataset))) or\
            ('SelexDataset' in str(type(train.dataset))):
            if criterion is None:
                criterion = mb.tl.PoissonLoss()

            if not isinstance(train, list):
                n_rounds = train.dataset.n_rounds
                n_batches = train.dataset.n_batches
                enr_series = train.dataset.enr_series
            else:
                n_rounds = max([max(t.dataset.n_rounds) for t in train])
                n_batches = len(train)
                enr_series = True

            n_batches = kwargs.get('n_batches', n_batches)
            if 'n_batches' in kwargs:
                del kwargs['n_batches']
            n_rounds = kwargs.get('n_rounds', n_rounds)
            if 'n_rounds' in kwargs:
                del kwargs['n_rounds']

            use_mono = kwargs.get('use_mono', True)
            use_dinuc = kwargs.get('use_dinuc', True)

            dinuc_mode = kwargs.get('dinuc_mode', 'local')

            vprint("# rounds", set(n_rounds) if not isinstance(n_rounds, int) else n_rounds)
            vprint("# rounds", set(n_rounds) if not isinstance(n_rounds, int) else n_rounds)
            vprint('# use_mono', use_mono)
            vprint('# use_dinuc', use_dinuc)
            vprint('# dinuc_mode', dinuc_mode)
            vprint("# batches", n_batches)
            vprint("# kernels", n_kernels)
            vprint("# initial w", kwargs.get('w', 20))
            vprint("# enr_series", enr_series)
            vprint('# opt kernel shift', kwargs.get('opt_kernel_shift', False))
            vprint('# opt kernel length', kwargs.get('opt_kernel_length', False))
            vprint("# custom kernels", kwargs.get('kernels'))

            kwargs['kernels'] = kwargs.get('kernels', [0] + [kwargs.get('w', 20)] * (n_kernels - 1))

            model = mb.models.Mubind(
                datatype="selex",
                n_rounds=n_rounds,
                n_batches=n_batches,
                init_random=init_random,
                enr_series=enr_series,
                **kwargs,
            ) # .to(self.device)
        elif isinstance(train.dataset, mb.datasets.PBMDataset) or isinstance(train.dataset,
                                                                             mb.datasets.GenomicsDataset):
            if criterion is None:
                criterion = mb.tl.MSELoss()
            if isinstance(train.dataset, mb.datasets.PBMDataset):
                n_proteins = train.dataset.n_proteins
            else:
                n_proteins = kwargs.get('n_proteins', train.dataset.n_cells)
            vprint("# proteins", n_proteins)
            if kwargs.get('joint_learning', False) or n_proteins == 1:
                kwargs['kernels'] = kwargs.get('kernels', [0] + [w] * (n_kernels - 1))
                model = mb.models.Mubind(
                    datatype="pbm",
                    init_random=init_random,
                    n_batches=n_proteins,
                    **kwargs,
                ) # .to(self.device)
            else:
                bm_generator = mb.models.BMCollection(n_proteins=n_proteins, n_kernels=n_kernels,
                                                      init_random=init_random)
                model = mb.models.Mubind(
                    datatype="pbm",
                    init_random=init_random,
                    n_proteins=n_proteins,
                    bm_generator=bm_generator,
                    n_kernels=n_kernels,
                    **kwargs,
                ) # .to(self.device)
        elif isinstance(train.dataset, mb.datasets.ResiduePBMDataset):
            model = mb.models.Mubind(
                datatype="pbm",
                init_random=init_random,
                bm_generator=mb.models.BMPrediction(num_classes=1, input_size=21, hidden_size=2, num_layers=1,
                                                    seq_length=train.dataset.get_max_residue_length()),
                **kwargs,
            ) # .to(self.device)
        else:
            assert False  # not implemented yet

        # set criterion
        model.criterion = criterion

        if str(kwargs.get('device')) != 'cpu':
            return model.cuda()
        return model


    def forward(self, mono, **kwargs):
        # mono_rev=None, di=None, di_rev=None, batch=None, countsum=None, residues=None, protein_id=None):
        mono_rev = kwargs.get("mono_rev", None)
        di = kwargs.get("di", None)
        di_rev = kwargs.get("di_rev", None)
        mono = self.padding(mono)
        if mono_rev is None:
            mono_rev = mb.tl.mono2revmono(mono)
        else:
            mono_rev = self.padding(mono_rev)
            del kwargs['mono_rev']  # for later function calls, and to avoid duplicates

        # prepare the dinucleotide objects if we need them
        if self.use_dinuc:
            if di is None:
                di = mb.tl.mono2dinuc(mono)
            if di_rev is None:
                di_rev = mb.tl.mono2dinuc(mono_rev)
            di = torch.unsqueeze(di, 1)
            di_rev = torch.unsqueeze(di_rev, 1)
            kwargs["di"] = di
            kwargs["di_rev"] = di_rev

        # unsqueeze mono after preparing di and unsqueezing mono
        mono_rev = torch.unsqueeze(mono_rev, 1)
        mono = torch.unsqueeze(mono, 1)

        # binding_per_mode: matrix of size [batchsize, number of binding modes]
        binding_per_mode = self.binding_modes(mono=mono, mono_rev=mono_rev, **kwargs)
        binding_scores = self.activities(binding_per_mode, **kwargs)
        # print('mode')
        # print(binding_per_mode)
        # print('scores')
        # print(binding_scores)
        if self.datatype == "pbm":
            return binding_scores
        elif self.datatype == "selex":
            return self.graph_module(binding_scores, **kwargs)
        else:
            return None  # this line should never be called

    def set_seed(self, seed, index, max_value=0, min_value=-1):
        if isinstance(self.binding_modes, BindingModesSimple):
            self.binding_modes.set_seed(seed, index, max_value, min_value)
        else:
            print("Setting a seed is not possible for that kind of model.")
            assert False

    def modify_kernel(self, index=None, shift=0, expand_left=0, expand_right=0, device=None):
        self.binding_modes.modify_kernel(index, shift, expand_left, expand_right, device)

    def set_kernel_weights(self, weight, index):
        assert weight.shape == self.conv_mono[index].weight.shape
        self.conv_mono[index].weight = weight

    def update_grad(self, index, value):
        self.binding_modes.update_grad(index, value)
        self.activities.update_grad(index, value)

    def update_grad_activities(self, index, value):
        self.activities.update_grad(index, value)

    def update_grad_etas(self, value):
        self.graph_module.update_grad_etas(value)

    def set_ignore_kernel(self, ignore_kernel):
        self.activities.set_ignore_kernel(ignore_kernel)

    def get_ignore_kernel(self):
        return self.activities.get_ignore_kernel()

    def get_kernel_width(self, index):
        return self.binding_modes.get_kernel_width(index)

    def get_kernel_weights(self, index, **kwargs):
        return self.binding_modes.get_kernel_weights(index, **kwargs)

    def get_log_activities(self):
        return self.activities.get_log_activities()

    def get_log_etas(self):
        assert self.datatype == "selex"
        return self.graph_module.get_log_etas()

    def dirichlet_regularization(self):
        return self.binding_modes.dirichlet_regularization()

    def weight_distances_min_k(self, min_k=5, exp_delta=4):
        d = []
        for a, b in itertools.combinations(self.conv_mono[1:], r=2):
            a = a.weight
            b = b.weight
            min_w = min(a.shape[-1], b.shape[-1])

            # print(min_w)

            lowest_d = -1
            for k in range(5, min_w):
                # print(k)
                for i in range(0, a.shape[-1] - k + 1):
                    ai = a[:, :, :, i: i + k]
                    for j in range(0, b.shape[-1] - k + 1):
                        bi = b[:, :, :, j: j + k]
                        bi_rev = torch.flip(bi, [3])[:, :, [3, 2, 1, 0], :]
                        d.append(((bi - ai) ** 2).sum().cpu().detach() / bi.shape[-1])
                        d.append(((bi_rev - ai) ** 2).sum().cpu().detach() / bi.shape[-1])

                        if lowest_d == -1 or d[-1] < lowest_d or d[-2] < lowest_d:
                            next_d = min(d[-2], d[-1])
                            # print(i, i + k, j, j + k, d[-2], d[-1])
                            lowest_d = next_d

        if len(d) == 0:
            print(self.conv_mono)
            assert False

        return torch.exp(exp_delta - min(d))

    def loss_kernel_rel(self, log=False):
        """
        Return a loss associated to the similarity of weights that are assumed to be similar
        """
        loss = 0
        # relationship terms in the matrix
        if self.kernel_rel is not None and self.optimize_kernel_rel:
            # print('distances')
            # monos = [b.weight for b in self.binding_modes.conv_mono[1:]]
            # monos = [m.cpu().detach().numpy().squeeze() for m in monos]
            # # print(monos)
            # print('# of kernels', len(monos))
            # res = mb.tl.calculate_distances([m.copy() for m in monos], best=True, full=True,
            #                                 filter_neg_weights=False, min_w_sum=-10000)
            # distances_kernels = res[~pd.isnull(res['id'])]
            # d = distances_kernels.pivot('a', 'b', 'distance')
            #
            # print(d)
            # print(self.kernel_rel)
            # dist_loss = np.nansum(d.values[self.kernel_rel[d.index - 1, :-1] == 1])
            dist_loss = 0
            for ai, a in enumerate(self.binding_modes.conv_mono[1:]):
                for bi, b in enumerate(self.binding_modes.conv_mono[1:]):
                    if ai >= bi:
                        continue
                    if self.kernel_rel[ai, bi] == 1:
                        d = torch.norm(a.weight - b.weight)
                        if log:
                            print(ai, bi, d)
                        dist_loss += d

            # assert False
            # print(distances_kernels.head())
            # calculate the distances between kernels
            # print('here', dist_loss)
            # assert False

            loss += dist_loss
        return loss

    def print_weights(self):
        torch.set_printoptions(profile="default")  # reset
        torch.set_printoptions(linewidth=500)
        # torch.set_printoptions(threshold=10_000)
        print('\nmono')
        for b in self.binding_modes.conv_mono:
            if hasattr(b, 'weight'):
                print(b.weight)
        print('\ndinuc')
        for b in self.binding_modes.conv_di:
            if hasattr(b, 'weight'):
                print(b.weight)
        print('\nactivities')
        print(self.activities.get_log_activities())
        print('\netas')
        print(self.graph_module.log_etas)

    def loss_exp_barrier(self, exp_max):
        """
        We add an exponential negative term, to force weights to be more positive than negative
        """
        pos_weight_sum_abs_mono = [b.weight.sum(axis=2).abs() for b in self.binding_modes.conv_mono[1:]]
        mono = sum([torch.exp(p - exp_max).sum() for p in pos_weight_sum_abs_mono])

        di = None
        if self.use_dinuc and self.binding_modes.dinuc_mode == 'local':
            pos_weight_sum_abs_di = [b.weight.sum(axis=2).abs() for b in self.binding_modes.conv_di[1:]]
            di = sum([torch.exp(p - exp_max).sum() for p in pos_weight_sum_abs_di])
        elif self.use_dinuc and self.binding_modes.dinuc_mode == 'full':
            di = []
            for b in self.binding_modes.conv_di[1:]:
                for b2 in b:
                    di.append(b2.weight.sum(axis=2).abs().sum())
            di = sum(di)

            return mono + di

        return mono


    def loss_log_dynamic(self):

        if not hasattr(self.graph_module, 'conn_sparse'):
            return 0

        conn = self.graph_module.conn_sparse

        # log_dynamic = self.graph_module.D_tril # log_dynamic      
        # return 100

        # return torch.abs(torch.sparse.sum(self.graph_module.D_tril))

        # log_dynamic = self.graph_module.D_tril.coalesce().values() # self.graph_module.D_tril # log_dynamic
        log_dynamic = self.graph_module.log_dynamic


        idx = conn.indices()
        conn_vals = conn.values()
        pos = torch.arange(idx.size(1))
        
        # prepare combinations based on common indexes
        uniq_idx = idx.unique()
        all_combinations = []
        for u_idx in uniq_idx:
            # at least one common index has to be present in the position retrieved
            sub_pos = pos[(idx[0] == u_idx) | (idx[1] == u_idx)]
            c = torch.combinations(sub_pos, r=2)
            all_combinations.append(c)
        all_pos = torch.cat(all_combinations)
        pairs = idx[:, all_pos].reshape(all_pos.shape[0], 4)

        # pairs = idx[all_pos].reshape(all_pos.shape[0], 4)
        mask1 = (pairs[:, 0] == pairs[:, 2]) | (pairs[:, 1] == pairs[:, 3])
        mask2 = (pairs[:, 0] != pairs[:, 1]) & (pairs[:, 2] != pairs[:, 3])
        mask3 = ~((pairs[:, 0] == pairs[:, 2]) & (pairs[:, 1] != pairs[:, 3]))

        pairs = pairs[mask1 & mask2 & mask3]

        all_pos = all_pos[mask1 & mask2 & mask3]

        a = log_dynamic[all_pos[:, 0]]
        b = log_dynamic[all_pos[:, 1]]
        w_err = (a - b) ** 2
        conn_weight = conn_vals[all_pos[:, 0]] * conn_vals[all_pos[:, 1]]
        score = w_err * conn_weight

        w_err = score.sum() / idx.shape[0]
        # return sum(w_err + torch.rand(1, device=self.device))
        return w_err

    def exp_barrier(self, exp_max=40):
        out = 0
        for p in self.parameters():
            out += torch.sum(torch.exp(p - exp_max) + torch.exp(-p - exp_max))
        return out


    def loss_kernel_symmetrical_weights(self):
        """
        This loss calculates the squared sum of columns per position, and it is useful to detect
        strong positive/negative biases per position or in the whole object.
        """
        mono_sym_weight = sum([(b.weight.sum(axis=2) ** 2).sum() for b in self.binding_modes.conv_mono[1:]])

        di_sym_weight = None
        if self.use_dinuc and self.binding_modes.dinuc_mode == 'local':
            di_sym_weight = sum([(b.weight.sum(axis=2) ** 2).sum() for b in self.binding_modes.conv_di[1:]])
        elif self.use_dinuc and self.binding_modes.dinuc_mode == 'full':
            di_sym_weight = []
            for b in self.binding_modes.conv_di[1:]:
                for b2 in b:
                    di_sym_weight.append((b2.weight.sum(axis=2) ** 2).sum())
            di_sym_weight = sum(di_sym_weight)
            return mono_sym_weight + di_sym_weight

        return mono_sym_weight

    def loss_prob_act(self):
        prob = torch.cat((torch.ones(1, device=self.device), self.binding_modes.prob_act))
        prob = torch.sigmoid(torch.exp(prob))
        return torch.sum(prob)

    # if early_stopping is positive, training is stopped if over the length of early_stopping no improvement happened or
    # num_epochs is reached.
    def optimize_simple(self,
        dataloader,
        optimiser,
        # reconstruction_crit,
        num_epochs=15,
        early_stopping=-1,
        dirichlet_regularization=0,
        exp_max=40,  # if this value is negative, the exponential barrier will not be used.
        log_each=-1,
        verbose=0,
        r2_per_epoch=False,
        **kwargs,
    ):
        # global loss_history
        r2_history = []
        loss_history = []
        loss_history_sym_weights = []
        loss_history_log_dynamic = []

        best_loss = None
        best_epoch = -1
        if verbose != 0:
            print(
                "optimizer: ",
                str(type(optimiser)).split('.')[-1].split('\'>')[0],
                "\ncriterion:",
                str(type(self.criterion)).split('.')[-1].split('\'>')[0],
                "\n# epochs:",
                num_epochs,
                "\nearly_stopping:",
                early_stopping,
            )

        for f in ["lr", "weight_decay"]:
            if f in optimiser.param_groups[0]:
                if verbose != 0:
                    print("%s=" % f, optimiser.param_groups[0][f], end=", ")

        if verbose != 0:
            print("dir weight=", dirichlet_regularization)

        is_lbfgs = "LBFGS" in str(optimiser)

        store_rev = dataloader.dataset.store_rev if not isinstance(dataloader, list) else dataloader[
            0].dataset.store_rev

        t0 = time.time()
        n_batches = len(list(enumerate(dataloader)))

        # the total number of trials
        n_trials = None
        if isinstance(dataloader, list) and hasattr(dataloader[0].dasaset, 'signal'):
            n_trials = sum([d.dataset.signal.shape[0] for d in dataloader])
        elif isinstance(dataloader, list) and hasattr(dataloader[0].dataset, 'rounds'):
            n_trials = sum([d.dataset.rounds.shape[0] for d in dataloader])
        else:
            n_trials = sum(
                [d.dataset.rounds.shape[0] if hasattr(d.dataset, 'rounds') else d.dataset.signal.shape[0] for d in
                 [dataloader]])

        for epoch in range(num_epochs):
            running_loss = 0
            running_loss_sym_weights = 0
            running_loss_log_dynamic = 0

            running_rec = 0

            # if dataloader is a list of dataloaders, we have to iterate through those
            dataloader_queries = dataloader if isinstance(dataloader, list) else [dataloader]



            # print(len(dataloader_queries))
            for data_i, next_dataloader in enumerate(dataloader_queries):
                # print(data_i, next_dataloader, len(next_dataloader))
                for i, batch in enumerate(next_dataloader):
                    # print(i, 'batches out of', n_batches)
                    # Get a batch and potentially send it to GPU memory.
                    mononuc = batch["mononuc"].to(self.device)
                    b = batch["batch"].to(self.device) if "batch" in batch else None

                    rounds = batch["rounds"].to(self.device) if "rounds" in batch else None

                    # print(rounds.shape)
                    if next_dataloader.dataset.use_sparse:
                        rounds = rounds.squeeze(1)

                    n_rounds = batch["n_rounds"].to(self.device) if "n_rounds" in batch else None
                    countsum = batch["countsum"].to(self.device) if "countsum" in batch else None
                    residues = batch["residues"].to(self.device) if "residues" in batch else None
                    protein_id = batch["protein_id"].to(self.device) if "protein_id" in batch else None
                    inputs = {"mono": mononuc, "batch": b, "countsum": countsum}
                    if store_rev:
                        mononuc_rev = batch["mononuc_rev"].to(self.device)
                        inputs["mono_rev"] = mononuc_rev
                    if residues is not None:
                        inputs["residues"] = residues
                    if protein_id is not None:
                        inputs["protein_id"] = protein_id

                    # if not selex, do not scale by overall signal
                    inputs['scale_countsum'] = self.datatype == 'selex'
                    
                    loss = None
                    if is_lbfgs:
                        def closure():
                            optimiser.zero_grad()
                            # this statement here is mandatory to
                            outputs = self.forward(**inputs)

                            # weight_dist = model.weight_distances_min_k()
                            if dirichlet_regularization == 0:
                                dir_weight = 0
                            else:
                                dir_weight = dirichlet_regularization * self.dirichlet_regularization()

                            # loss = criterion(outputs, rounds) + weight_dist + dir_weight
                            loss = self.criterion(outputs, rounds) + dir_weight

                            # if exp_max >= 0:
                            #     loss += self.exp_barrier(exp_max)
                            loss.backward()  # retain_graph=True)
                            return loss

                        loss = optimiser.step(closure)  # Step to minimise the loss according to the gradient.
                    else:
                        # PyTorch calculates gradients by accumulating contributions to them (useful for
                        # RNNs).  Hence we must manully set them to zero before calculating them.
                        optimiser.zero_grad(set_to_none=None)

                        # outputs, reconstruction = model(inputs)  # Forward pass through the network.
                        outputs = self.forward(**inputs)  # Forward pass through the network.

                        # print(outputs.shape, rounds.shape)
                        # print(torch.cat([outputs, rounds], axis=1)[:3])
                        # assert False

                        # weight_dist = model.weight_distances_min_k()
                        if dirichlet_regularization == 0:
                            dir_weight = 0
                        else:
                            dir_weight = dirichlet_regularization * self.dirichlet_regularization()
                        # if the dataloader is a list, then we know the output shape directly by rounds
                        if isinstance(dataloader, list):
                            loss = self.criterion(outputs[:, :rounds.shape[1]], rounds)
                        else:
                            # define a mask to remove items on a rounds specific manner
                            if n_rounds is not None and len(set(dataloader.dataset.n_rounds)) != 1:
                                mask = torch.zeros((n_rounds.shape[0], outputs.shape[1]), dtype=torch.bool,
                                                   device=self.device)
                                for mi in range(mask.shape[1]):
                                    mask[:, mi] = ~(n_rounds - 1 < i)
                                loss = self.criterion(outputs[mask], rounds[mask])
                            else:
                                loss = self.criterion(outputs, rounds)

                        # print('pred/obs')
                        # print(outputs, outputs.shape)
                        # print(rounds, rounds.shape)
                        # print((rounds.sum(axis=1) == 0).any())
                        # assert False
                        # restart loss
                        # loss = 0.0

                        # skip loss
                        loss += dir_weight


                        # loss = criterion(outputs, rounds) + .01*reconstruct_crit(reconstruction, residues) + dir_w
                        # if exp_max >= 0:
                        #     loss += self.exp_barrier(exp_max)

                        loss_kernel_rel = self.loss_kernel_rel()
                        loss_neg_weights = self.loss_exp_barrier(exp_max=exp_max)
                        loss_sym_weights = self.loss_kernel_symmetrical_weights()
                        loss_log_dynamic = self.loss_log_dynamic()

                        # print(loss_log_dynamic)


                        # regularization of binding modes

                        # print(loss, loss_kernel_rel, loss_neg_weights, loss_sym_weights)
                        # assert False

                        if self.optimize_kernel_rel:
                            loss += loss_kernel_rel

                        if self.optimize_exp_barrier:
                            loss += loss_neg_weights
                        if self.optimize_sym_weight:
                            # print(loss_sym_weights)
                            loss += loss_sym_weights
                        if self.optimize_log_dynamic:
                            # print(loss_sym_weights)
                            # print(loss_log_dynamic)
                            loss += loss_log_dynamic
                        if self.optimize_prob_act:
                            loss_prob_act = self.loss_prob_act()
                            loss += loss_prob_act

                        # print(loss_kernel_rel)
                        # print(loss_neg_weights)
                        # print(loss_sym_weights)
                        # print('dynamic loss', loss_log_dynamic)
                        # print('sum activities', sum(p.sum() for p in self.activities.log_activities))
                        # print('# LOSS', loss)

                        loss.backward()  # Calculate gradients.
                        optimiser.step()

                    running_loss += loss.item()
                    running_loss_sym_weights += loss_sym_weights
                    running_loss_log_dynamic += loss_log_dynamic

                    # running_rec += reconstruction_crit(reconstruction, residues).item()

            loss_final = running_loss / len(dataloader)
            loss_final_sym_weights = running_loss_sym_weights / len(dataloader)
            loss_final_log_dynamic = running_loss_log_dynamic / len(dataloader)

            if log_each != -1 and epoch > 0 and (epoch % log_each == 0):
                # self.print_weights()
                if verbose != 0:
                    r2_epoch = None
                    if r2_per_epoch:
                        r2_epoch = mb.tl.scores(self, dataloader)['r2_counts']
                        # r2_history.append(mb.pl.kmer_enrichment(self, dataloader, k=8, show=False))

                    total_time = time.time() - t0
                    time_epoch_1k = (total_time / max(epoch, 1) / n_trials * 1e3)
                    print(
                        "Epoch: %2d, Loss: %.6f, %s" % (epoch + 1, loss_final,
                                                        'R2: %.2f, ' % r2_epoch if r2_epoch is not None else ''),
                        "best epoch: %i, " % best_epoch,
                        "secs per epoch: %.3f s, " % ((time.time() - t0) / max(epoch, 1)),
                        "secs epoch*1k trials: %.3fs" % time_epoch_1k,
                        "curr time:", datetime.datetime.now(),
                    )

                    if kwargs.get('print_weights', False):
                        self.print_weights()

            if best_loss is None or loss_final < best_loss:
                best_loss = loss_final
                best_epoch = epoch
                self.best_model_state = copy.deepcopy(self.state_dict())
                self.best_loss = best_loss

            # print("Epoch: %2d, Loss: %.3f" % (epoch + 1, running_loss / len(train_dataloader)))
            loss_history.append(float(loss_final))
            loss_history_sym_weights.append(float(loss_final_sym_weights))
            loss_history_log_dynamic.append(float(loss_final_log_dynamic))

            # model.crit_history.append(crit_final)
            # model.rec_history.append(rec_final)

            if early_stopping > 0 and epoch >= best_epoch + early_stopping:
                if verbose != 0:
                    r2_epoch = None
                    if r2_per_epoch:
                        r2_epoch = mb.tl.scores(self, dataloader)['r2_counts']
                        # r2_history.append(mb.pl.kmer_enrichment(self, dataloader, k=8, show=False))

                    total_time = time.time() - t0
                    time_epoch_1k = (total_time / max(epoch, 1) / n_trials * 1e3)
                    print(
                        "Epoch: %2d, Loss: %.6f, %s" % (epoch + 1, loss_final,
                                                        'R2: %.2f, ' % r2_epoch if r2_epoch is not None else ''),
                        "best epoch: %i, " % best_epoch,
                        "secs per epoch: %.3fs, " % ((time.time() - t0) / max(epoch, 1)),
                        "secs epoch*1k trials: %.3fs," % time_epoch_1k,
                        "curr time:", datetime.datetime.now(),
                    )
                if verbose != 0:
                    print("early stop!")
                break

        # Print if profiling included. Temporarily removed profiling to save memory.
        # print('Profiling epoch:')
        # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=25))
        # prof.export_chrome_trace(f'profile_{epoch}.json')
        total_time = time.time() - t0
        self.total_time += total_time
        if verbose:
            r2_epoch = None
            if r2_per_epoch:
                r2_epoch = mb.tl.scores(self, dataloader)['r2_counts']
                self.r2_final = r2_epoch
                # r2_history.append(mb.pl.kmer_enrichment(self, dataloader, k=8, show=False))

            print('Final loss: %.10f %s' % (loss_final, ', R2: %.2f' % r2_epoch if r2_epoch is not None else ''))
            print(f'Total time (model/function): (%.3fs / %.3fs)' % (self.total_time, total_time))
            print("Time per epoch (model/function): (%.3fs/ %.3fs)" %
                  ((self.total_time / max(epoch, 1)), (total_time / max(epoch, 1))))
            print('Time per epoch per 1k trials: %.3fs' % (total_time / max(epoch, 1) / n_trials * 1e3))
            print('Current time:', datetime.datetime.now())

        self.loss_history += loss_history
        self.loss_history_sym_weights += loss_history_sym_weights
        self.loss_history_log_dynamic += loss_history_log_dynamic

        self.r2_history += r2_history

    def optimize_iterative(self,
                           train,
                           # min_w=10,
                           max_w=20,
                           n_epochs=100, # int or list
                           early_stopping=15, # int or list
                           log_each=10,
                           opt_kernel_shift=True,
                           opt_kernel_length=True,
                           opt_one_step=False,
                           expand_length_max=3,
                           expand_length_step=1,
                           show_logo=False,
                           optimiser=None,
                           seed=None,
                           init_random=False,
                           joint_learning=False,
                           ignore_kernel=False,
                           lr=0.01,
                           weight_decay=0.001,
                           stop_at_kernel=None,
                           dirichlet_regularization=0,
                           verbose=2,
                           exp_max=-1,
                           shift_max=2,
                           shift_step=1,
                           r2_per_epoch=False,
                           skip_kernels=None,
                           log_next_r2=True,
                           **kwargs,
                           ):
        # color for visualization of history
        colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#ffff33", "#a65628"]

        # here he add a parameter to keep the r2 log by next parms
        self.best_r2_by_new_filter = []

        # verbose print declaration
        if verbose:
            def vprint(*args, **kwargs):
                print(*args, **kwargs)
        else:
            vprint = lambda *a, **k: None  # do-nothing function

        if not isinstance(opt_kernel_shift, list):
            opt_kernel_shift = [0] + [opt_kernel_shift] * (self.n_kernels - 1)
        if not isinstance(opt_kernel_length, list):
            opt_kernel_length = [0] + [opt_kernel_length] * (self.n_kernels - 1)

        # prepare model

        # this sets up the seed at the first position
        if seed is not None:
            # this sets up the seed at the first position
            for i, s, min_w, max_w in seed:
                if s is not None:
                    print(i, s)
                    self.set_seed(s, i, min=min_w, max=max_w)
            self = self.to(self.device)

        # step 1) freeze everything before the current binding mode
        for i in range(0, self.n_kernels):

            if skip_kernels is not None and i in skip_kernels:
                continue

            vprint('current kernels')
            # print(self.binding_modes)

            vprint("\Filter to optimize %i %s" % (i, '(intercept)' if i == 0 else ''))
            vprint("\nFREEZING KERNELS")

            for feat_i in ['mono', 'dinuc']:
                if i == 0 and feat_i == 'dinuc':
                    vprint('optimization of dinuc is not necessary for the intercepts (filter=0). Skip...')
                    continue

                vprint('optimizing feature type', feat_i)
                if i != 0:
                    if feat_i == 'dinuc' and not self.use_dinuc:
                        vprint('the optimization of dinucleotide features is skipped...')
                        continue
                    elif feat_i == 'mono' and not self.use_mono:
                        vprint('the optimization of mononucleotide features is skipped...')
                        continue

                # block kernels that we do not require to optimize
                for ki in range(self.n_kernels):
                    mask_mono = (ki == i) and (feat_i == 'mono')
                    mask_dinuc = (ki == i) and (feat_i == 'dinuc')

                    if opt_one_step: # skip freezing
                        if skip_kernels is None or (i in skip_kernels):
                            mask_mono = False
                            mask_dinuc = False

                    if verbose != 0:
                        if mask_mono or mask_dinuc:
                            vprint("setting grad status of kernel (mono, dinuc) at %i to (%i, %i)" % (
                            ki, mask_mono, mask_dinuc))

                    if hasattr(self.binding_modes, 'update_grad_mono'):
                        self.binding_modes.update_grad_mono(ki, mask_mono)
                        if self.use_dinuc:
                            self.binding_modes.update_grad_di(ki, mask_dinuc)

                    # activities are frozen during intercept optimization
                    self.update_grad_activities(ki, i != 0)
                    if opt_one_step:
                        self.update_grad_activities(ki, True)

                    # self.update_grad_etas(i != 0)

                if show_logo:
                    vprint("before filter optimization.")
                    mb.pl.plot_activities(self, train)
                    mb.pl.conv_mono(self)
                    # mb.pl.conv_mono(model, flip=False, log=False)
                    mb.pl.conv_di(self, mode='triangle')

                next_lr = lr if not isinstance(lr, list) else lr[i]
                next_weight_decay = weight_decay if not isinstance(weight_decay, list) else weight_decay[i]
                next_early_stopping = early_stopping if not isinstance(early_stopping, list) else early_stopping[i]

                next_optimiser = (
                    topti.Adam(self.parameters(), lr=next_lr, weight_decay=next_weight_decay)
                    if optimiser is None
                    else optimiser(self.parameters(), lr=next_lr)
                )

                # mask kernels to avoid using weights from further steps into early ones.
                if ignore_kernel:
                    self.set_ignore_kernel(np.array([0 for i in range(i + 1)] + [1 for i in range(i + 1, n_kernels)]))

                if verbose != 0:
                    print("filters mask", self.get_ignore_kernel())

                self.optimize_simple(
                    train,
                    next_optimiser,
                    num_epochs=n_epochs[i] if isinstance(n_epochs, list) else n_epochs,
                    early_stopping=next_early_stopping,
                    log_each=log_each,
                    dirichlet_regularization=dirichlet_regularization,
                    exp_max=exp_max,
                    verbose=verbose,
                    r2_per_epoch=r2_per_epoch,
                    **kwargs,
                )

                # vprint('grad')
                # vprint(model.binding_modes.conv_mono[1].weight.grad)
                # vprint(model.binding_modes.conv_di[1].weight.grad)
                # vprint('')

                self.loss_color += list(np.repeat(colors[i % len(colors)], len(self.loss_history) - len(self.loss_color)))
                # probably here load the state of the best epoch and save

                self.load_state_dict(self.best_model_state)

                # store model parameters and fit for later visualization
                # necessary?
                # self = copy.deepcopy(self)

                # optimizer for left / right flanks
                best_loss = self.best_loss

                if show_logo:
                    print("\n##After filter opt / before shift optim.")
                    mb.pl.plot_activities(self, train)
                    mb.pl.conv_mono(self)
                    # mb.pl.conv_mono(model, flip=True, log=False)
                    mb.pl.conv_di(self, mode='triangle')
                    mb.pl.plot_loss(self)

                # print(model_by_k[k_parms].loss_color)
                #######
                # optimize the flanks through +1/-1 shifts
                #######
                if (opt_kernel_shift[i] or opt_kernel_length[i]) and i != 0:
                    self = self.optimize_width_and_length(train,
                                                          expand_length_max,
                                                          expand_length_step,
                                                          shift_max,
                                                          shift_step,
                                                          i,
                                                          feat_i=feat_i,
                                                          colors=colors,
                                                          verbose=verbose,
                                                          lr=next_lr,
                                                          weight_decay=next_weight_decay,
                                                          optimiser=optimiser,
                                                          log_each=log_each,
                                                          exp_max=exp_max,
                                                          dirichlet_regularization=dirichlet_regularization,
                                                          early_stopping=next_early_stopping, criterion=self.criterion,
                                                          show_logo=show_logo,
                                                          n_kernels=self.n_kernels,
                                                          max_w=max_w,
                                                          r2_per_epoch=r2_per_epoch,
                                                          num_epochs=n_epochs[i] if isinstance(n_epochs, list) else n_epochs,
                                                          **kwargs)

                if show_logo:
                    vprint("after shift optimz model")
                    mb.pl.plot_activities(self, train)
                    mb.pl.conv_mono(self)
                    # mb.pl.conv_mono(model, log=False)
                    mb.pl.conv_di(self, mode='triangle')
                    mb.pl.plot_loss(self)
                    print("")

                # the first kernel does not require an additional fit.
                if i == 0:
                    continue

                vprint("\n\nfinal refinement step (after shift)...")
                vprint("\nunfreezing all layers for final refinement")

                for ki in range(self.n_kernels):
                    # vprint("kernel grad (%i) = %i \n" % (ki, True), sep=", ", end="")
                    self.update_grad(ki, ki == i)
                vprint("")

                # define the optimizer for final refinement of the model
                next_optimiser = (
                    topti.Adam(self.parameters(), lr=next_lr, weight_decay=next_weight_decay)
                    if optimiser is None
                    else optimiser(self.parameters(), lr=next_lr)
                )
                # mask kernels to avoid using weights from further steps into early ones.
                if ignore_kernel:
                    self.set_ignore_kernel(np.array([0 for i in range(i + 1)] +
                                                    [1 for i in range(i + 1, self.n_kernels)]))

                vprint("filters mask", self.get_ignore_kernel())
                vprint("filters mask", self.get_ignore_kernel())

                # final refinement of weights
                self.optimize_simple(
                    train,
                    next_optimiser,
                    num_epochs=n_epochs[i] if isinstance(n_epochs, list) else n_epochs,
                    early_stopping=next_early_stopping,
                    log_each=log_each,
                    dirichlet_regularization=dirichlet_regularization,
                    verbose=verbose,
                    r2_per_epoch=r2_per_epoch,
                )

                # load the best model after the final refinement
                self.loss_color += list(np.repeat(colors[i % len(colors)], len(self.loss_history) - len(self.loss_color)))
                self.load_state_dict(self.best_model_state)

                if stop_at_kernel is not None and stop_at_kernel == i:
                    break

                if show_logo:
                    vprint("\n##final motif signal (after final refinement)")
                    mb.pl.plot_activities(self, train)
                    mb.pl.conv_mono(self)
                    mb.pl.conv_di(self, mode='triangle')
                    # mb.pl.conv_mono(model, flip=True, log=False)

                vprint('best loss', self.best_loss)

                # calculate the current r2 and keep a log of it
                if log_next_r2:
                    next_r2 = mb.tl.scores(self, train)['r2_counts']
                    self.best_r2_by_new_filter.append(next_r2)
                    print('current r2 values by newly added filter')
                    print(self.best_r2_by_new_filter)


        vprint('\noptimization finished:')
        vprint(f'total time: {self.total_time}s')
        vprint("Time per epoch (total): %.3f s" %
               (self.total_time / max(n_epochs if not isinstance(n_epochs, list) else sum(n_epochs), 1)))


        return self, self.best_loss

    def optimize_width_and_length(self, train, expand_length_max, expand_length_step, shift_max, shift_step, i,
                                  colors=None, verbose=False, lr=0.01, weight_decay=0.001, optimiser=None, log_each=10,
                                  exp_max=40,
                                  num_epochs_shift_factor=3,
                                  dirichlet_regularization=0, early_stopping=15, criterion=None, show_logo=False,
                                  feat_i=None,
                                  n_kernels=4, w=15, max_w=20, num_epochs=100, loss_thr_pct=0.005, **kwargs, ):
        """
        A variation of the main optimization routine that attempts expanding the filter of the model at position i, and refines
        the weights and loss in order to find a better convergence.
        """

        # verbose print declaration
        if verbose:
            def vprint(*args, **kwargs):
                print(*args, **kwargs)
        else:
            vprint = lambda *a, **k: None  # do-nothing function

        n_attempts = 0  # to keep a log of overall attempts
        opt_expand_left = range(0, expand_length_max, expand_length_step)
        opt_expand_right = range(0, expand_length_max, expand_length_step)
        opt_shift = list(range(-shift_max, shift_max + 1, shift_step))
        for opt_option_text, opt_option_next in zip(
            ["WIDTH", "SHIFT"], [[opt_expand_left, opt_expand_right, [0]], [[0], [0], opt_shift]]
        ):
            next_loss = None
            loss_diff_pct = 0
            while next_loss is None or (next_loss < best_loss and loss_diff_pct > loss_thr_pct):
                n_attempts += 1

                vprint("\n%s OPTIMIZATION (%s)..." % (opt_option_text, "first" if next_loss is None else "again"),
                       end="")
                vprint("")
                curr_w = self.get_kernel_width(i)
                if curr_w >= max_w:
                    if opt_option_text == 'WIDTH':
                        print("Reached maximum w. Stop...")
                        break

                self = copy.deepcopy(self)
                best_loss = self.best_loss
                next_color = colors[-(1 if n_attempts % 2 == 0 else -2)]

                all_options = []

                options = [
                    [expand_left, expand_right, shift]
                    for expand_left in opt_option_next[0]
                    for expand_right in opt_option_next[1]
                    for shift in opt_option_next[2]
                ]

                if opt_option_text == 'SHIFT' and False:  # include shifts to center weights
                    m = torch.tensor(self.get_kernel_weights(i))
                    # print(m)
                    m[m < 0] = 0
                    m = m.reshape(m.shape[-2:])
                    col_pos_means = m.mean(axis=0).cpu()
                    w = int(m.shape[-1] / 2)
                    # print(w)
                    col_means = []
                    for j in range(m.shape[-1]):
                        a, b = max(j - w, 0), min(j + w, m.shape[-1])
                        ci = m[:, a:b].mean()
                        col_means.append(j)
                    pos_max = torch.argmax(torch.tensor(col_means))

                    # mb.pl.conv(model)
                    shift_center = (w - pos_max).cpu()
                    # print(shift)
                    # print('adding option', shift_center)
                    options = [[0, 0, -shift_center], [0, 0, shift_center]] + options
                    # assert False

                print('options to try', options)

                for expand_left, expand_right, shift in options:

                    # if abs(expand_left) + abs(expand_right) + abs(shift) == 0:
                    #     continue
                    # if abs(shift) > 0:  # skip shift for now.
                    #     continue
                    if curr_w + expand_left + expand_right > max_w:
                        continue

                    # print(expand_left, expand_right, shift)
                    # assert False

                    vprint("next expand left: %i, next expand right: %i, shift: %i"
                           % (expand_left, expand_right, shift))

                    model_shift = copy.deepcopy(self)
                    model_shift.loss_history = []
                    model_shift.r2_history = []
                    model_shift.loss_color = []

                    model_shift.optimize_modified_kernel(
                        train,
                        kernel_i=i,
                        shift=shift,
                        device=self.device,
                        expand_left=expand_left,
                        expand_right=expand_right,
                        num_epochs=num_epochs if opt_option_text == 'WIDTH' else num_epochs * num_epochs_shift_factor,
                        early_stopping=early_stopping,
                        log_each=log_each if opt_option_text == 'WIDTH' else log_each * num_epochs_shift_factor,
                        # log_each,
                        update_grad_i=i,
                        feat_i=feat_i,
                        lr=lr,
                        weight_decay=weight_decay,
                        optimiser=optimiser,
                        criterion=criterion,
                        dirichlet_regularization=dirichlet_regularization,
                        exp_max=exp_max,
                        verbose=verbose,
                        **kwargs,
                    )
                    vprint('')

                    model_shift.loss_color += list(np.repeat(next_color, len(model_shift.loss_history)))
                    # print('history left', len(model_left.loss_history))
                    weight_mono_i = model_shift.binding_modes.conv_mono[i].weight
                    pos_w_sum = float(weight_mono_i[weight_mono_i > 0].sum())

                    loss_diff_pct = (best_loss - model_shift.best_loss) / best_loss * 100

                    r2 = mb.tl.scores(model_shift, train)['r2_counts']

                    all_options.append([expand_left, expand_right, shift, model_shift,
                                        pos_w_sum, weight_mono_i.shape[-1], loss_diff_pct, model_shift.best_loss, r2])
                    # print('\n')
                    # vprint("after opt.")

                    if show_logo:
                        mb.pl.conv_mono(model_shift)
                        mb.pl.conv_di(model_shift, mode='triangle')

                # for shift, model_shift, loss in all_shifts:
                #     print('shift=%i' % shift, 'loss=%.4f' % loss)
                weight_ref_mono_i = model_shift.binding_modes.conv_mono[i].weight
                pos_w_ref_mono_i_sum = float(weight_ref_mono_i[weight_ref_mono_i > 0].sum())

                best_r2 = mb.tl.scores(self, train)['r2_counts']
                best = sorted(
                    all_options + [[0, 0, 0, self,
                                    pos_w_ref_mono_i_sum, weight_ref_mono_i.shape[-1], 0, self.best_loss, best_r2]],
                    key=lambda x: x[-1],
                )
                if verbose != 0:
                    print("sorted")
                best_df = pd.DataFrame(best, columns=["expand.left", "expand.right", "shift", "model",
                                                      'pos_w_sum', 'width', "loss_diff_pct", "loss", 'r2'],
                                       )
                best_df['last_loss'] = best_loss
                best_df = best_df.sort_values('loss')

                vprint(best_df[[c for c in best_df if c != 'model']])
                # print('\n history len')
                next_expand_left, next_expand_right, next_position, next_model, next_pos_w, w, \
                loss_diff_pct, next_loss, next_r2 = best_df.values[0][:-1]

                print(next_expand_left, next_expand_right, next_position, next_pos_w, w,
                      loss_diff_pct, next_loss, next_r2)

                if verbose != 0:
                    print("action (expand left, expand right, shift): (%i, %i, %i)\n" %
                          (next_expand_left, next_expand_right, next_position))

                if loss_diff_pct >= loss_thr_pct:
                    next_model.loss_history = self.loss_history + next_model.loss_history
                    next_model.r2_history = self.r2_history + next_model.r2_history
                    next_model.loss_color = self.loss_color + next_model.loss_color

                self = copy.deepcopy(next_model)

                if next_expand_left == 0 and next_expand_right == 0 and next_position == 0 and opt_option_text == 'SHIFT':
                    print('This was the last iteration. Done with filter shift optimization...')
                    break

        return self

    def optimize_modified_kernel(self,
        train,
        shift=0,
        expand_left=0,
        expand_right=0,
        device=None,
        num_epochs=500,
        early_stopping=15,
        log_each=-1,
        feat_i='mono',
        update_grad_i=None,
        use_dinuc=False,
        kernel_i=None,
        lr=0.01,
        weight_decay=0.001,
        optimiser=None,
        dirichlet_regularization=0,
        exp_max=40,
        verbose=0,
        r2_per_epoch=False,
        **kwargs,
    ):
        assert expand_left >= 0 and expand_right >= 0
        self.modify_kernel(kernel_i, shift, expand_left, expand_right, device)

        # requires grad update
        n_kernels = len(self.binding_modes)
        for ki in range(n_kernels):
            self.binding_modes.update_grad_mono(ki, (ki == update_grad_i) and (feat_i == 'mono'))

            if self.use_dinuc:
                self.binding_modes.update_grad_di(ki, (ki == update_grad_i) and (feat_i == 'dinuc'))

        # finally the optimiser has to be initialized again.
        optimiser = (
            topti.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
            if optimiser is None
            else optimiser(self.parameters(), lr=lr)
        )

        self.optimize_simple(
            train,
            optimiser,
            num_epochs=num_epochs,
            early_stopping=early_stopping,
            log_each=log_each,
            dirichlet_regularization=dirichlet_regularization,
            exp_max=exp_max,
            verbose=verbose,
            r2_per_epoch=r2_per_epoch,
        )

        return self


class BindingModesSimple(tnn.Module):
    """
    Implements binding modes (also non-specific binding) for one protein.

    Keyword Args:
        kernels (List[int]): Size of the binding modes (0 indicates non-specific binding). Default: [0, 15]
        init_random (bool): Use a random initialization for all parameters. Default: True
        use_dinuc (bool): Use dinucleotide contributions (not fully implemented for all kind of models). Default: False
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.kernels = kwargs.get("kernels", [0, 15])
        self.init_random = kwargs.get("init_random", True)
        self.use_dinuc = kwargs.get("use_dinuc", True)
        self.dinuc_mode = kwargs.get("dinuc_mode", 'local')  # local or full
        self.conv_mono = tnn.ModuleList()
        self.conv_di = tnn.ModuleList()
        self.ones = None  # aux ones tensor, for intercept init.


        for k in self.kernels:
            if k == 0:
                self.conv_mono.append(None)
                self.conv_di.append(None)
            else:
                next_mono = tnn.Conv2d(1, 1, kernel_size=(4, k), padding=(0, 0), bias=False)
                if not self.init_random:
                    next_mono.weight.data.uniform_(0, 0)
                else:
                    next_mono.weight.data.uniform_(.01, .05)  # problem with fitting  dinucleotides if (0, 0)

                self.conv_mono.append(next_mono)

                # create the conv_di layers. These are skipped during forward unless use_dinuc is True
                if self.dinuc_mode == 'local':
                    # the number of contiguous dinucleotides for k positions is k - 1
                    next_di = tnn.Conv2d(1, 1, kernel_size=(16, k - 1), padding=(0, 0), bias=False)
                    if not self.init_random:
                        next_di.weight.data.uniform_(-.01, .01)  # problem with fitting  dinucleotides if (0, 0)
                    self.conv_di.append(next_di)
                # a matrix of conv2d
                elif self.dinuc_mode == 'full':
                    conv_di_next = tnn.ModuleList()
                    for i in range(1, k):
                        conv_di_next.append(tnn.Conv2d(1, 1, kernel_size=(16, k - i)))
                    self.conv_di.append(conv_di_next)

        # regularization parameter
        self.prob_act = tnn.Parameter(torch.ones(len(self.conv_mono) - 1, dtype=torch.float32)) # minus the intercept
        # self.prob_thr = .1 # tnn.Parameter(torch.zeros(1, dtype=torch.float32))
        if kwargs.get('p_dropout', False):
            self.p_dropout = kwargs.get('p_dropout')
            self.dropout = tnn.Dropout(p=self.p_dropout)
        else:
            self.dropout = None

    def forward(self, mono, mono_rev, di=None, di_rev=None, **kwargs):
        bm_pred = []

        for i in range(len(self.kernels)):
            # print(i)
            if self.kernels[i] == 0:
                # intercept (will be scaled by the activity of the non-specific binding)
                # print(mono.shape[0])
                # temp = torch.tensor([1.0] * mono.shape[0], device=mono.device)
                if self.ones is None or self.ones.shape[0] < mono.shape[
                    0]:  # aux ones tensor, to avoid memory init delay
                    self.ones = torch.ones(mono.shape[0], device=mono.device)  # torch.ones is much faster

                temp = self.ones[:mono.shape[0]]  # subsetting of ones to fit batch
                bm_pred.append(temp)
            else:
                # check devices match
                if self.conv_mono[i].weight.device != mono.device:
                    self.conv_mono[i].weight.to(mono.device)

                # print('here...')
                # print(self.conv_di[i])

                if self.use_dinuc and self.dinuc_mode == 'local':
                    temp = torch.cat(
                        (
                            self.conv_mono[i](mono),
                            self.conv_mono[i](mono_rev),
                            self.conv_di[i](di),
                            self.conv_di[i](di_rev),
                        ),
                        dim=3,
                    )
                elif self.use_dinuc and self.dinuc_mode == 'full':
                    next_conv_di = self.conv_di[i]

                    k = self.conv_mono[i].weight.shape[-1]  # this is to infer the number of conv2d for dinuc
                    pi = -1  # the overall counter of the conv2d
                    # iterating in this way, we go from the largest axis i.e. diagonal to the corner
                    out_di = []
                    # print(next_conv_di)
                    # for pi, di_ij in enumerate(next_conv_di):
                    #     print(pi)
                    for ki in range(0, k):  # ki indicates the delta between positions i.e. the diagonal index
                        # print('\nI = %i' % ki)
                        p = mono[:, :, :, :mono.shape[-1] - ki]
                        q = mono[:, :, :, ki:]
                        assert p.shape[-1] == q.shape[-1]
                        p_max = torch.argmax(p, axis=2)
                        p_max = torch.mul(p_max, 4)
                        q_max = torch.argmax(q, axis=2)

                        mask = p_max + q_max
                        mask_flatten = mask.flatten()
                        one_hot = torch.nn.functional.one_hot(mask_flatten, num_classes=16)
                        m = one_hot.reshape(mono.shape[0], mono.shape[-1] - ki, 16)
                        m = m.float()
                        m = m.reshape(mono.shape[0], 16, mono.shape[-1] - ki)
                        m = torch.unsqueeze(m, 1)
                        di_ij = next_conv_di[pi]
                        next_out = di_ij(m)
                        # print(next_out.shape)
                        out_di.append(next_out)
                        # assert False

                    temp_mono = torch.cat(
                        (
                            self.conv_mono[i](mono),
                            self.conv_mono[i](mono_rev),
                        ),
                        dim=3,
                    )
                    temp_di = torch.cat(out_di, dim=3)

                    temp = torch.cat((temp_mono, temp_di), axis=3)
                    # print(temp.shape)
                    # print('here....')
                    # assert False
                else:
                    out_mono = self.conv_mono[i](mono)
                    out_mono_rev = self.conv_mono[i](mono_rev)
                    temp = torch.cat((out_mono, out_mono_rev), dim=3)

                # this particular step can generate out of bounds due to the exponential cost
                # print(temp_mono.type())
                # print(temp.shape, temp.type())
                # assert False

                temp = torch.exp(temp)
                temp = temp.view(temp.shape[0], -1)
                temp = torch.sum(temp, dim=1)
                bm_pred.append(temp)

        out = torch.stack(bm_pred).T

        # regularization step, using activation probability
        if self.dropout is not None:
            sparsity = False
            if not sparsity:
                prob = torch.ones(out.shape[-1] - 1, device=mono.device)
                prob = self.dropout(prob)
                prob = torch.cat((torch.ones(1, device=mono.device), prob))
                prob = prob > 0 # self.prob_thr
                return out * prob
            else:
                mask1 = self.prob_act > torch.ones(self.prob_act.shape[0], device=mono.device)
                out1 = torch.where(mask1, 1, self.prob_act)
                mask2 = out1 < torch.zeros(out1.shape[0], device=mono.device)
                z = torch.where(mask2, 0, out1)
                z = torch.cat((torch.ones(1, device=mono.device), z))
                return out * z
        else:
            return out


    def set_seed(self, seed, index, max, min):
        assert len(seed) <= self.conv_mono[index].kernel_size[1]
        shift = int((self.conv_mono[index].kernel_size[1] - len(seed)) / 2)
        seed_params = torch.full(self.conv_mono[index].kernel_size, max, dtype=torch.float32)
        for i in range(len(seed)):
            if seed[i] == "A":
                seed_params[:, i + shift] = torch.tensor([max, min, min, min])
            elif seed[i] == "C":
                seed_params[:, i + shift] = torch.tensor([min, max, min, min])
            elif seed[i] == "G":
                seed_params[:, i + shift] = torch.tensor([min, min, max, min])
            elif seed[i] == "T":
                seed_params[:, i + shift] = torch.tensor([min, min, min, max])
            else:
                seed_params[:, i + shift] = torch.tensor([0, 0, 0, 0])
        self.conv_mono[index].weight = tnn.Parameter(torch.unsqueeze(torch.unsqueeze(seed_params, 0), 0))

    def update_grad_mono(self, index, value):
        if self.conv_mono[index] is not None:
            self.conv_mono[index].weight.requires_grad = value
            if not value:
                self.conv_mono[index].weight.grad = None

    def update_grad_di(self, index, value):
        if len(self.conv_di) >= index and self.conv_di[index] is not None:
            if isinstance(self.conv_di[index], tnn.ModuleList):
                for conv_di in self.conv_di[index]:
                    conv_di.weight.requires_grad = value
                    if not value:
                        conv_di.weight.grad = None
            else:
                self.conv_di[index].weight.requires_grad = value
                if not value:
                    self.conv_di[index].weight.grad = None

    def update_grad(self, index, value):
        self.update_grad_mono(index, value)

        if self.use_dinuc:
            self.update_grad_di(index, value)

    def modify_kernel(self, index=None, shift=0, expand_left=0, expand_right=0, device=None):
        # shift mono

        shape_mono_before = None
        shape_di_before = None

        for i, m in enumerate(self.conv_mono):
            if index is not None and index != i:
                continue
            if m is None:
                continue
            before_w = m.weight.shape[-1]
            shape_mono_before = m.weight.shape

            # update the weight
            if shift >= 1:
                self.conv_mono[i].weight = torch.nn.Parameter(
                    torch.cat([m.weight[:, :, :, shift:], torch.zeros(1, 1, 4, shift, device=device)], dim=3)
                )
            elif shift <= -1:
                self.conv_mono[i].weight = torch.nn.Parameter(
                    torch.cat(
                        [
                            torch.zeros(1, 1, 4, -shift, device=device),
                            m.weight[:, :, :, :shift],
                        ],
                        dim=3,
                    )
                )

            # adding more positions left and right
            if expand_left > 0:
                self.conv_mono[i].weight = torch.nn.Parameter(
                    torch.cat([torch.zeros(1, 1, 4, expand_left, device=device), m.weight[:, :, :, :]], dim=3)
                )
            if expand_right > 0:
                self.conv_mono[i].weight = torch.nn.Parameter(
                    torch.cat([m.weight[:, :, :, :], torch.zeros(1, 1, 4, expand_right, device=device)], dim=3)
                )
            after_w = m.weight.shape[-1]
            if after_w != (before_w + expand_left + expand_right):
                assert after_w != (before_w + expand_left + expand_right)

        # shift di
        for i, m in enumerate(self.conv_di):
            if index is not None and index != i:
                continue
            if m is None:
                continue
            # update the weight
            if self.dinuc_mode == 'local':
                shape_di_before = m.weight.shape
                if shift >= 1:
                    m.weight = torch.nn.Parameter(
                        torch.cat([m.weight[:, :, :, shift:], torch.zeros(1, 1, 16, shift, device=device)], dim=3)
                    )
                elif shift <= -1:
                    m.weight = torch.nn.Parameter(
                        torch.cat([torch.zeros(1, 1, 16, -shift, device=device), m.weight[:, :, :, :shift]], dim=3)
                    )

                # adding more positions left and right
                if expand_left > 0:
                    self.conv_di[i].weight = torch.nn.Parameter(
                        torch.cat([torch.zeros(1, 1, 16, expand_left, device=device), m.weight[:, :, :, :]], dim=3)
                    )
                if expand_right > 0:
                    self.conv_di[i].weight = torch.nn.Parameter(
                        torch.cat([m.weight[:, :, :, :], torch.zeros(1, 1, 16, expand_right, device=device)], dim=3)
                    )

                # check that the differences between kernels are the same before and after updates
                diff_width_after = self.conv_di[i].weight.shape[-1] - self.conv_mono[i].weight.shape[-1]
                assert diff_width_after == (shape_di_before[-1] - shape_mono_before[-1])
            elif self.dinuc_mode == 'full':  # reset the dinuc weights to match the shape of the new conv2d for mononuc
                if shape_mono_before[-1] != self.conv_mono[i].weight.shape[-1]:
                    if False:
                        k = self.conv_mono[i].weight.shape[-1]
                        print('updating convdi triangle to have %i positions' % k)
                        conv_di_next = tnn.ModuleList()
                        for i in range(1, k + 1):
                            for j in range(k - i + 1):
                                # pi += 1
                                conv_di_next.append(tnn.Conv2d(1, 1, kernel_size=(16, i)))
                        self.conv_di[i] = conv_di_next
                        # print(len(self.conv_di))

    def dirichlet_regularization(self):
        out = 0
        for m in self.conv_mono:
            if m is None:
                continue
            elif m.weight.requires_grad:
                out -= torch.sum(m.weight - torch.logsumexp(m.weight, dim=2))
        if self.use_dinuc:
            for d in self.conv_di:
                if d is None:
                    continue
                elif d.weight.requires_grad:
                    out -= torch.sum(d.weight - torch.logsumexp(d.weight, dim=2))
        return out

    def get_kernel_width(self, index):
        return self.conv_mono[index].weight.shape[-1] if self.conv_mono[index] is not None else 0

    def get_kernel_weights(self, index, dinucleotide=False):
        values = self.conv_mono if not dinucleotide else self.conv_di
        return values[index].weight if len(values) >= (index + 1) and values[index] is not None else None

    def __len__(self):
        return len(self.conv_mono)


class BindingModesPerProtein(tnn.Module):
    """
    Implements binding modes (also non-specific binding) for multiple proteins in the same batch.

    Args:
        bm_generator (torch.nn.Module): PyTorch module which has a weight matrix as output

    Keyword Args:
        add_intercept (bool): Whether an intercept is used in addition to the predicted binding modes. Default: True
    """

    def __init__(self, bm_generator, **kwargs):
        super().__init__()
        self.generator = bm_generator
        self.use_intercept = kwargs.get("intercept", True)

    def forward(self, mono, mono_rev, di=None, di_rev=None, **kwargs):
        weights = self.generator(**kwargs)  # weights needs to be a list
        bm_pred = []
        if self.use_intercept:
            bm_pred.append(torch.Tensor([1.0] * mono.shape[0]).to(device=mono.device))
        for w in weights:
            w = torch.unsqueeze(w, 1)
            # Transposing batch dim and channels
            mono = torch.transpose(mono, 0, 1)
            mono_rev = torch.transpose(mono_rev, 0, 1)
            temp = torch.cat(
                (
                    F.conv2d(mono, w, groups=w.shape[0]),
                    F.conv2d(mono_rev, w, groups=w.shape[0]),
                ),
                dim=3,
            )
            # Transposing back
            mono = torch.transpose(mono, 0, 1)
            mono_rev = torch.transpose(mono_rev, 0, 1)
            temp = torch.transpose(temp, 0, 1)
            temp = torch.exp(temp)
            temp = temp.view(temp.shape[0], -1)
            temp = torch.sum(temp, dim=1)
            bm_pred.append(temp)
        return torch.stack(bm_pred).T

    def update_grad(self, index, value):
        self.generator.update_grad(index, value)

    def modify_kernel(self, index=None, shift=0, expand_left=0, expand_right=0, device=None):
        self.generator.modify_kernel(index, shift, expand_left, expand_right, device)

    def dirichlet_regularization(self):
        return 0  # could be implemented in the genaerators end then changed here

    def get_kernel_width(self, index):
        return self.generator.get_kernel_width(index)

    def get_kernel_weights(self, index):
        return self.generator.get_kernel_width(index)

    def __len__(self):
        return len(self.generator)


class ActivitiesLayer(tnn.Module):
    """
    Implements activities with batch effects.

    Args:
        target_dim: Second dimension of the output of forward

    Keyword Args:
        n_batches (int): Number of batches that will occur in the data. Default: 1
        n_proteins (int): Number of proteins in the dataset. Either n_proteins or n_batches may be used. Default: 1
        ignore_kernel (list[bool]): Whether a kernel should be ignored. Default: None.
    """

    def __init__(self, n_kernels, target_dim, **kwargs):
        super().__init__()
        self.n_kernels = n_kernels
        # due to having multiple batches in some rounds, the max n_rounds is stored
        self.target_dim = max(target_dim) if not isinstance(target_dim, int) else target_dim
        self.n_batches = kwargs.get("n_batches", 1) if "n_batches" in kwargs else kwargs.get("n_proteins", 1)
        self.ignore_kernel = kwargs.get("ignore_kernel", None)
        self.log_activities = tnn.ParameterList()
        for i in range(n_kernels):
            self.log_activities.append(
                tnn.Parameter(torch.zeros([self.n_batches, self.target_dim], dtype=torch.float32))
            )

    def forward(self, binding_per_mode, **kwargs):
        batch = kwargs.get("batch", None)
        if batch is None:
            batch = kwargs.get("protein_id", None)
        if batch is None:
            batch = torch.zeros([binding_per_mode.shape[0]], device=binding_per_mode.device)

        # print(scores.shape)
        # print(torch.stack(list(self.log_activities), dim=1).shape)

        # this is to compare old/new implementation of relevant low-level operations
        scores = None
        option = 1
        if option == 1:
            b = binding_per_mode.unsqueeze(1)
            a = torch.exp(torch.stack(list(self.log_activities), dim=1))
            # print(b.shape, a.shape, batch.shape)
            # print(b.type(), a.type(), batch.type())
            result = torch.matmul(b, a[batch, :, :])

            # print(a)
            # print('b')
            # print(b)

            scores = result.squeeze(1)
        else:
            scores = torch.zeros([binding_per_mode.shape[0], self.target_dim], device=binding_per_mode.device)
            for i in range(self.n_batches):
                a = torch.exp(torch.stack(list(self.log_activities), dim=1)[i, :, :])
                batch_mask = batch == i
                b = binding_per_mode[batch_mask]

                if self.ignore_kernel is not None:
                    mask = self.ignore_kernel != 1  # == False
                    scores[batch_mask] = torch.matmul(b[:, mask], a[mask, :])
                else:
                    scores[batch_mask] = torch.matmul(b, a)

        return scores

    def update_grad(self, index, value):
        self.log_activities[index].requires_grad = value
        if not value:
            self.log_activities[index].grad = None

    def set_ignore_kernel(self, ignore_kernel):
        self.ignore_kernel = ignore_kernel

    def get_ignore_kernel(self):
        return self.ignore_kernel

    def get_log_activities(self):
        return torch.stack(list(self.log_activities), dim=1)


class GraphModule(tnn.Module):
    """
    Implements the layer that calculates associations between samples and readouts

    Args:
        target_dim: Second dimension of the output of forward

    Keyword Args:
        enr_series (bool): Whether the data should be handled as enrichment series. Default: True
    """

    def __init__(self, n_rounds, **kwargs):
        super().__init__()
        self.n_rounds = max(n_rounds) if not isinstance(n_rounds, int) else n_rounds
        self.enr_series = kwargs.get("enr_series", True)
        self.n_batches = kwargs.get("n_batches", 1)
        self.log_etas = tnn.Parameter(torch.zeros([self.n_batches, self.n_rounds]))
        # log dynamic is a matrix with upper/lower triangle with opposite symbols
        # self.log_dynamic = tnn.Parameter(torch.zeros([int((self.n_rounds - 1) * (self.n_rounds) / 2)]))
        # self.log_dynamic = tnn.Parameter(torch.zeros([self.n_rounds, self.n_rounds]))
        # print(self.log_etas.shape, self.log_dynamic.shape)
        # print(self.log_etas.shape, self.log_dynamic.shape)
        if kwargs.get('prepare_knn'):
            self.prepare_knn(**kwargs)
            print('setting up log dynamic')
            # self.log_dynamic = tnn.Parameter(torch.rand(self.conn_sparse.indices().shape[1])).requires_grad_(True) # .cuda()

    import torch.nn as tnn
    def prepare_knn(self, **kwargs):
        '''
        This routine is in charge of the graph to be used during the assay-assay relatedness step
        '''
        adata = kwargs.get('adata')
        device = kwargs.get('device')

        # prepare the zero counts
        counts = adata.X.T
        next_data = pd.DataFrame(counts.A if type(counts) != np.ndarray else counts)  # sparse.from_spmatrix(counts.A)
        next_data['var'] = next_data.var(axis=1)
        del next_data['var']
        df = next_data.copy()  # sample
        zero_counts = df.sum(axis=1) == 0

        self.conn_sparse = torch.tensor(
            adata[:, ~zero_counts].obsp['connectivities'].A).to_sparse()
        
        if device != 'cpu':
            self.conn_sparse = self.conn_sparse.cuda()

        # do not activate the required grad of this function, otherwise, it does not optimize
        # if device == 'cpu':
        #     self.log_dynamic = tnn.Parameter(
        #         torch.rand(self.conn_sparse.indices().shape[1])) # .requires_grad_(True).cuda()
        
        # if device != 'cpu':
        #     self.log_dynamic = self.log_dynamic.cuda() # requires_grad_(True)
        # 
        #                
        # if self.log_dynamic.shape[0] == 0:
        #     print('Warning: Log dynamic is empty. This indicates an empty kNN representations.'
        #           'Please verify previous steps...')
        # else:
        
        # initialize log dynamic
        tspa = torch.sparse_coo_tensor
        t = torch.transpose
        C = self.conn_sparse # .cuda()
        a_ind = C.indices()

        # self.D = self.log_dynamic.cuda()

        # print(a_ind.device, self.D.device, C.device)
        # assert False

        # self.log_dynamic = tnn.Parameter(torch.rand(self.conn_sparse.indices().shape[1])).requires_grad_(True).cuda()
        
        # do not convert to cuda, otherwise, the optimization of these weights will not happen.
        self.log_dynamic = tnn.Parameter(torch.rand(self.conn_sparse.indices().shape[1])) # .cuda()

        # the opposite direction will always be rescaled into a negative sign, and this is the factor that controls the magnitude
        self.knn_free_weights = kwargs.get('knn_free_weights')
        if self.knn_free_weights:
            self.log_dynamic_scaling = tnn.Parameter(torch.rand(self.conn_sparse.indices().shape[1])) # .cuda()

        # self.D_tril = tspa(a_ind, torch.rand(self.conn_sparse.indices().shape[1]).cuda(), C.shape).requires_grad_(True).cuda()
        # self.D_triu = -self.D_tril # opposite sign


    def forward(self, binding_scores, countsum, **kwargs):
        batch = kwargs.get("batch", None)
        if batch is None:
            batch = torch.zeros([binding_scores.shape[0]], device=binding_scores.device)

        # assert hasattr(self, 'conn_sparse')

        out = None
        if self.enr_series:
            out = torch.cumprod(binding_scores, dim=1)  # cum product between rounds 0 and N
        elif hasattr(self, 'conn_sparse') and kwargs.get('use_conn', True): # in this particular step, we multiply by the dynamic or static scores.
            if True: # new solution:
                # operations
                tsum = torch.sum
                texp = torch.exp
                tspa = torch.sparse_coo_tensor
                tsmm = torch.sparse.mm
                t = torch.transpose

                # binding scores
                b = binding_scores
                b_T = torch.transpose(b, 0, 1)

                # connectivities
                C = self.conn_sparse
                a_ind = C.indices()
                # conn_spa_T = torch.transpose(C, 0, 1)

                # log dynamic weights
                D = self.log_dynamic
                D_tril = tspa(a_ind, D, C.shape)  # .requires_grad_(True).cuda()

                # scaling of weights yes/no
                if self.knn_free_weights:
                    D_triu = tspa(a_ind, -D * torch.exp(self.log_dynamic_scaling), C.shape)
                    # print('here...')
                else:
                    D_triu = tspa(a_ind, -D, C.shape)
                                
                D_all = D_tril + t(D_triu, 0, 1)

                # D_triu = -self.D_tril # opposite sign
                # D_all = self.D_tril + t(-self.D_tril, 0, 1) # opposite sign

                # print(self.D_tril)

                # update with pos and negs
                # D_all = self.D_tril + t(self.D_triu, 0, 1)

                # print(D.shape)
                # assert False
                # print('log dynamic1 after exp', torch.sum(D.to_dense()).cpu().detach().sum())
                # print('sum of dynamic mat1', tsum(D.to_dense()).cpu().detach().sum())

                # print(C.shape, b_T.shape)
                # assert False
                # print(conn_spa_T.shape, b.shape)
                # print(C.device, b_T.device, D_all.device)
                # assert False

                tmp = tsmm(C, b_T).T

                # print('')
                # print('sum of tmp 1', tmp.cpu().detach().sum())
                # print('shape tmp1', tmp.shape)
                tmp_T = t(tmp, 0, 1)
                # print('dynamic1 input', tsum(D.to_dense().cpu().detach().sum()),
                #       tsum(b.to_dense()).cpu().detach().sum())

                dynamic_out1 = tsmm(texp(t(D_all, 0, 1).to_dense()), tmp_T).T
                # print('dynamic out 1 sum', dynamic_out1.cpu().detach().sum())
                # print(torch.isnan(log_dynamic_spa).any(), torch.isnan(dyn1_T).any(), torch.isnan(dynamic_out).any())
                # print('static1 input', tsum(D.to_dense()).cpu().detach(), tsum(b.to_dense()).cpu().detach())
                static_out1 = tsmm(texp(D_all.to_dense()), b_T).T

                out = static_out1 + dynamic_out1

                # print(out)
                # print(binding_scores)
                # assert False
            else:
                # old solution
                # print(self.log_dynamic.shape)
                D = self.log_dynamic
                C_dense = C.to_dense()
                a_ind = C.indices()
                D = self.log_dynamic
                D_spa = tspa(a_ind, D, C.shape)  # .requires_grad_(True).cuda()
                D_mat = D_spa.to_dense()

                # print(log_dynamic_mat)
                D_mat = D_mat + torch.transpose(-D_mat, 0, 1)
                D2 = D_mat
                print('')
                print('sum of mat2', tsum(D_mat.to_dense()).cpu().detach().sum())
                print('static2 input', b.sum().cpu().detach().sum(), texp(-D_mat).cpu().detach().sum())

                tmp = (b @ C_dense)
                print('')
                print('sum of tmp 2', tmp.cpu().detach().sum())
                print('shape tmp1', tmp.shape)

                dynamic_out2 = tmp @ texp(D_mat)  # texp(D_mat)
                print('out 2 sum', dynamic_out2.cpu().detach().sum())
                static_out2 = b @ texp(-D_mat)  # texp(-D_mat)
                out2 = static_out2 + dynamic_out2
        else:
            out = binding_scores

        # print(hasattr(self, 'connectivities'))
        # assert False

        #
        # print('binding scores')
        # print(out[:5])
        #
        # print('log etas')
        # print(self.log_etas)

        # multiplication in one step
        etas = torch.exp(self.log_etas)
        #
        # print('etas exp')
        # print(etas)

        out = out * etas[batch, :]

        # print('out after eta scaling')
        # print(out[:5])

        # fluorescent data e.g. PBM, does not require scaling, to keep numbers beyond range [0 - 1]
        if not kwargs.get('scale_countsum', True):
            return out

        results = out.T / torch.sum(out, dim=1)

        # print('sums', torch.sum(out, dim=1))
        # print('results')
        # print(results)
        #
        # print('countsum')
        # print(countsum)
        # print('mat sum')
        # print(countsum.sum())

        return (results * countsum).T

    def get_log_etas(self):
        return self.log_etas

    def update_grad_etas(self, value):
        self.log_etas.requires_grad = value
        # if not value:
        #    self.log_etas.grad = None


def _weight_distances(mono, min_k=5):
    d = []
    for a, b in itertools.combinations(mono, r=2):
        a = a.weight
        b = b.weight
        min_w = min(a.shape[-1], b.shape[-1])
        # print(min_w)

        lowest_d = -1
        for k in range(5, min_w):
            # print(k)
            for i in range(0, a.shape[-1] - k + 1):
                ai = a[:, :, :, i: i + k]
                for j in range(0, b.shape[-1] - k + 1):
                    bi = b[:, :, :, j: j + k]
                    bi_rev = torch.flip(bi, [3])[:, :, [3, 2, 1, 0], :]
                    d.append(((bi - ai) ** 2).sum() / bi.shape[-1])
                    d.append(((bi_rev - ai) ** 2).sum() / bi.shape[-1])

                    if lowest_d == -1 or d[-1] < lowest_d or d[-2] < lowest_d:
                        next_d = min(d[-2], d[-1])
                        # print(i, i + k, j, j + k, d[-2], d[-1])
                        lowest_d = next_d
    return min(d)

class MubindFlexibleWeights(tnn.Module):
    def __init__(
        self,
        n_rounds,
        n_batches,
        use_dinuc=False,
        max_w=15,
        ignore_kernel=None,
        rho=1,
        gamma=0,
        init_random=True,
        enr_series=True,
        padding_const=0.25,
        datatype='selex',  # must be 'selex' ot 'pbm' (case-insensitive)
    ):
        super().__init__()
        self.datatype = datatype.lower()
        assert self.datatype in ['selex', 'pbm']
        self.use_dinuc = use_dinuc
        self.n_rounds = n_rounds
        self.n_batches = n_batches
        self.rho = rho
        self.gamma = gamma

        # self.padding = tnn.ModuleList()
        # only keep one padding equals to the length of the max kernel
        self.padding = tnn.ConstantPad2d((max_w - 1, max_w - 1, 0, 0), padding_const)

        self.log_activities = tnn.ParameterList()
        self.enr_series = enr_series
        # self.log_activities = tnn.Parameter(torch.zeros([n_batches, len(kernels), n_rounds+1]))
        self.log_etas = tnn.Parameter(torch.zeros([n_batches, n_rounds + 1]))
        self.ignore_kernel = ignore_kernel

        for _ in range(1, 3):
            self.log_activities.append(tnn.Parameter(torch.zeros([n_batches, n_rounds + 1], dtype=torch.float32)))

    def forward(self, x):
        # Create the forward pass through the network.
        mono, mono_rev, batch, countsum, weight = x
        # padding of sequences
        mono = self.padding(mono)
        mono_rev = self.padding(mono_rev)
        mono = torch.unsqueeze(mono, 1)
        mono_rev = torch.unsqueeze(mono_rev, 1)

        x_ = []
        temp = torch.Tensor([1.0] * mono.shape[0], device=mono.device)
        x_.append(temp)

        # Transposing batch dim and channels
        mono = torch.transpose(mono, 0, 1)
        mono_rev = torch.transpose(mono_rev, 0, 1)
        temp = torch.cat(
            (
                F.conv2d(mono, weight, groups=weight.shape[0]),
                F.conv2d(mono_rev, weight, groups=weight.shape[0]),
            ),
            dim=3,
        )
        temp = torch.transpose(temp, 0, 1)  # Transposing back
        temp = torch.exp(temp)
        temp = temp.view(temp.shape[0], -1)
        temp = torch.sum(temp, dim=1)
        x_.append(temp)
        x = torch.stack(x_).T

        scores = torch.zeros([x.shape[0], self.n_rounds + 1]).to(device=mono.device)
        for i in range(self.n_batches):
            # a = torch.exp(self.log_activities[i, :, :])
            a = torch.exp(torch.stack(list(self.log_activities), dim=1)[i, :, :])
            if self.ignore_kernel is not None:
                mask = self.ignore_kernel != 1  # == False
                # print(mask_kernel)
                # print(x.shape, a.shape, x[batch == i][:,mask_kernel], a[mask_kernel,:].shape)
                scores[batch == i] = torch.matmul(x[batch == i][:, mask], a[mask, :])
            else:
                scores[batch == i] = torch.matmul(x[batch == i], a)

        if self.datatype == "pbm":
            return scores
            # return torch.log(scores)

        # a = torch.reshape(a, [a.shape[0], a.shape[2]])
        # x = torch.matmul(x, a)
        # sequential enrichment or independent samples
        if self.enr_series:
            # print('using enrichment series')
            predictions_ = [scores[:, 0]]
            for i in range(1, self.n_rounds + 1):
                predictions_.append(predictions_[-1] * scores[:, i])
            out = torch.stack(predictions_).T
        else:
            out = scores

        for i in range(self.n_batches):
            eta = torch.exp(self.log_etas[i, :])
            out[batch == i] = out[batch == i] * eta

        results = out.T / torch.sum(out, dim=1)
        results = (results * countsum).T
        return results


# This class can be used to store binding modes for several proteins
class BMCollection(tnn.Module):
    """
    Implements binding modes for multiple proteins at once. Should be used as a generator in combination with
    BindingModesPerProtein.

    Keyword Args:
        kernels (List[int]): Size of the binding modes (0 indicates non-specific binding, and will be accomplished by
                setting add_intercept to True). Default: [0, 15]
        init_random (bool): Use a random initialization for all parameters. Default: True
    """

    def __init__(self, n_proteins, **kwargs):
        super().__init__()
        self.n_proteins = n_proteins
        if "kernels" not in kwargs and "n_kernels" not in kwargs:
            kwargs["kernels"] = [0, 15]
            kwargs["n_kernels"] = len(kwargs["kernels"])
        elif "n_kernels" not in kwargs:
            kwargs["n_kernels"] = len(kwargs["kernels"])
        elif "kernels" not in kwargs:
            kwargs["kernels"] = [0] + [15] * (kwargs["n_kernels"] - 1)
        else:
            assert len(kwargs["kernels"]) == kwargs["n_kernels"]
        self.kernels = kwargs.get("kernels")
        self.stored_indizes = {}
        self.init_random = kwargs.get("init_random", True)

        self.conv_mono_list = tnn.ModuleList()
        for k in self.kernels:
            if k != 0:
                conv_mono = tnn.ModuleList()
                for i in range(self.n_proteins):
                    next_mono = tnn.Conv2d(1, 1, kernel_size=(4, k), padding=(0, 0), bias=False)
                    if not self.init_random:
                        next_mono.weight.data.uniform_(0, 0)
                    conv_mono.append(next_mono)
                self.conv_mono_list.append(conv_mono)
            else:
                self.conv_mono_list.append(None)

    def forward(self, protein_id, **kwargs):
        output = []
        for l in range(len(self.kernels)):
            if self.conv_mono_list[l] is not None:
                kernel = torch.stack([self.conv_mono_list[l][i].weight for i in protein_id])
                output.append(torch.squeeze(torch.squeeze(kernel, 1), 1))
        return output

    def update_grad(self, index, value):
        if self.conv_mono_list[index] is not None:
            for i in range(self.n_proteins):
                self.conv_mono_list[index][i].weight.requires_grad = value
                if not value:
                    self.conv_mono_list[index][i].weight.grad = None

    def modify_kernel(self, index=None, shift=0, expand_left=0, expand_right=0, device=None):
        # shift mono
        for l in range(len(self.kernels)):
            if (index is None or index == l) and (self.conv_mono_list[l] is not None):
                if expand_left > 0:
                    self.kernels[l] += expand_left
                if expand_right > 0:
                    self.kernels[l] += expand_right
                for i, m in enumerate(self.conv_mono_list[l]):
                    before_w = m.weight.shape[-1]
                    # update the weight
                    if shift >= 1:
                        m.weight = torch.nn.Parameter(
                            torch.cat([m.weight[:, :, :, shift:], torch.zeros(1, 1, 4, shift, device=device)], dim=3)
                        )
                    elif shift <= -1:
                        m.weight = torch.nn.Parameter(
                            torch.cat(
                                [
                                    torch.zeros(1, 1, 4, -shift, device=device),
                                    m.weight[:, :, :, :shift],
                                ],
                                dim=3,
                            )
                        )
                    # adding more positions left and right
                    if expand_left > 0:
                        m.weight = torch.nn.Parameter(
                            torch.cat([torch.zeros(1, 1, 4, expand_left, device=device), m.weight[:, :, :, :]], dim=3)
                        )
                    if expand_right > 0:
                        m.weight = torch.nn.Parameter(
                            torch.cat([m.weight[:, :, :, :], torch.zeros(1, 1, 4, expand_right, device=device)], dim=3)
                        )
                    after_w = m.weight.shape[-1]
                    if after_w != (before_w + expand_left + expand_right):
                        assert after_w != (before_w + expand_left + expand_right)

    def get_kernel_width(self, index):
        return self.conv_mono_list[index][0].weight.shape[-1] if self.conv_mono_list[index] is not None else 0

    def get_kernel_weights(self, index):
        return self.conv_mono_list[index][0].weight if self.conv_mono_list[index] is not None else None

    def __len__(self):
        return len(self.kernels)


# This class could be used as a bm_generator
class BMPrediction(tnn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):  # state_size_buff=512):
        super().__init__()
        self.num_classes = num_classes  # number of classes
        self.num_layers = num_layers  # number of layers
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # hidden state
        self.seq_length = seq_length  # sequence length

        self.lstm = tnn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = tnn.Linear(hidden_size, 253)  # fully connected 1
        self.conv_mono = tnn.Linear(253, 60)  # fully connected last layer
        self.relu = tnn.ReLU()

        # self.state_size_buff = state_size_buff
        # self.h_0 = Variable(torch.zeros(self.num_layers, state_size_buff, self.hidden_size)) # .to(x.device)
        # self.c_0 = Variable(torch.zeros(self.num_layers, state_size_buff, self.hidden_size)) # .to(x.device)

    def forward(self, residues, **kwargs):
        # assert x.size(0) <= self.state_size_buff
        # h_0 = self.h_0[:,:x.size(0),:]
        # c_0 = self.c_0[:,:x.size(0),:]

        h_0 = Variable(
            torch.zeros(self.num_layers, residues.size(0), self.hidden_size, device=residues.device))  # hidden state
        c_0 = Variable(
            torch.zeros(self.num_layers, residues.size(0), self.hidden_size, device=residues.device))  # internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(residues, (h_0, c_0))  # lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size)  # reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.linear(out)  # first Dense
        out = self.relu(out)  # relu
        out = self.conv_mono(out)  # Final Output
        return [out.reshape(residues.shape[0], 4, 15)]


class Decoder(tnn.Module):
    def __init__(self, input_size=60, enc_size=21, seq_length=88, **kwargs):
        super().__init__()
        if "layers" in kwargs and kwargs["layers"] is not None:
            layers = kwargs["layers"]
        else:
            layers = [200, 500, 1000]
        self.input_size = input_size  # input size
        self.enc_size = enc_size
        self.seq_length = seq_length
        self.output_size = enc_size * seq_length  # output size
        modules = [tnn.Linear(input_size, layers[0])]
        for i in range(len(layers) - 1):
            modules.append(tnn.ReLU())
            modules.append(tnn.Linear(layers[i], layers[i + 1]))
        modules.append(tnn.ReLU())
        modules.append(tnn.Linear(layers[len(layers) - 1], self.output_size))
        self.decoder = tnn.Sequential(*modules)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.decoder(x)
        x = torch.reshape(x, (x.shape[0], self.enc_size, -1))
        return x
        # return tnn.functional.softmax(x, dim=1)


# This class should be deleted in the future
class ProteinDNABinding(tnn.Module):
    def __init__(self, n_rounds, n_batches, num_classes=1, input_size=21, hidden_size=2, num_layers=1, seq_length=88,
                 datatype="pbm", **kwargs):
        super().__init__()
        self.datatype = datatype

        self.bm_prediction = BMPrediction(num_classes, input_size, hidden_size, num_layers, seq_length)
        self.decoder = mb.models.Decoder(enc_size=input_size, seq_length=seq_length, **kwargs)
        self.mubind = MubindFlexibleWeights(n_rounds, n_batches, datatype=datatype)

        self.best_model_state = None
        self.best_loss = None
        self.loss_history = []
        self.r2_history = []
        self.crit_history = []
        self.rec_history = []
        self.loss_color = []
        self.total_time = 0

    def forward(self, x):
        if len(x) == 4:
            mono, batch, countsum, residues = x
            mono_rev = mb.tl.mono2revmono(mono)
        elif len(x) == 5:
            mono, mono_rev, batch, countsum, residues = x
        else:
            assert False

        weights = self.bm_prediction(residues)
        reconstruction = torch.transpose(self.decoder(weights), 1, 2)

        weights = tnn.Parameter(weights)
        weights = torch.unsqueeze(weights, 1)
        pred = self.mubind((mono, mono_rev, batch, countsum, weights))
        return pred.view(-1), reconstruction

    # expects msa as tensor with dims (n_seq, 21, n_residues)
    def get_predicted_bm(self, msa):
        msa = torch.transpose(msa, 1, 2)
        with torch.no_grad():
            weights = self.bm_prediction(msa)
        return weights


# Multiple datasets
class DinucMulti(tnn.Module):
    def __init__(self, use_dinuc=False, n_datasets=1, n_latent=1, w=8):
        super().__init__()
        self.use_dinuc = use_dinuc
        # Create and initialise weights and biases for the layers.
        self.conv_mono = tnn.Conv2d(1, 1, kernel_size=(4, w), bias=False)
        self.conv_di = tnn.Conv2d(1, 1, kernel_size=(16, w), bias=False)

        self.embedding = tnn.Embedding(n_datasets, n_latent)

        self.best_model_state = None
        # self.fc = tnn.Linear(193, 1, bias=True)
        # torch.nn.init.uniform_(self.fc.weight, 0.0, 2/193)
        # self.log_weight_1 = tnn.Parameter(torch.tensor(np.array([[0]]).astype(np.float32)))
        # self.log_weight_2 = tnn.Parameter(torch.tensor(np.array([[-5.3]]).astype(np.float32)))

    def forward(self, x):
        # Create the forward pass through the network.
        mono, di, batch = x[0], x[1], x[2]
        # print('input shape', mono.shape)

        mono = torch.unsqueeze(mono, 1)
        mono = mono.type(torch.float32)

        # print('mono type', mono.dtype)
        # mono = mono.type(torch.LongTensor)
        mono = self.conv_mono(mono)

        # print(di)
        di = torch.unsqueeze(di, 1)
        di = di.type(torch.float32)

        di = self.conv_di(di)

        # this is necessary but it needs to be rellocated
        di = di.type(torch.LongTensor)

        mono = torch.exp(mono)
        di = torch.exp(di)
        mono = mono.view(mono.shape[0], -1)  # Flatten tensor.
        di = di.view(di.shape[0], -1)

        if self.use_dinuc:
            x = torch.sum(mono, axis=1) + torch.sum(di, axis=1)
        else:
            x = torch.sum(mono, axis=1)

        x = x.view(-1)  # Flatten tensor.
        
        # print('x in shape', x.shape)
        emb = self.embedding
        # print('emb shape', emb.weight.shape)
        # print('batch shape', b.shape)

        # b = emb(batch).sum(axis=1)
        # b = b.view(-1)
        # print('b shape (after emb)', b.shape)
        b = emb.weight.T @ batch.T.type(torch.float32)

        # b = b.view(-1)
        x = torch.sum(x.reshape(x.shape[0], 1) * b.T, axis=1)
        # print('x after emb', b.shape)

        return x
