import itertools

import torch
import torch.nn as tnn
import torch.nn.functional as F

from torch.autograd import Variable

import mubind as mb


class Multibind(tnn.Module):
    """
    Implements the Multibind model as flexible as possible.

    Args:
        datatype (String): Type of the experimental data. "selex" and "pbm" are supported.

    Keyword Args:
        n_rounds (int): Necessary for selex data: Number of rounds to be predicted.
        init_random (bool): Use a random initialization for all parameters. Default: True
        padding_const (double): Value for padding DNA-seqs. Default: 0.25
        use_dinuc (bool): Use dinucleotide contributions (not fully implemented for all kind of models). Default: False
        enr_series (bool): Whether the data should be handled as enrichment series. Default: True
        n_batches (int): Number of batches that will occur in the data. Default: 1
        ignore_kernel (list[bool]): Whether a kernel should be ignored. Default: None.
        kernels (List[int]): Size of the binding modes (0 indicates non-specific binding). Default: [0, 15]
        n_kernels (int). Number of kernels to be used (including non-specific binding). Default: 2
        init_random (bool): Use a random initialization for all parameters. Default: True
        n_proteins (int): Number of proteins in the dataset. Either n_proteins or n_batches may be used. Default: 1

        bm_generator (torch.nn.Module): PyTorch module which has a weight matrix as output.
        add_intercept (bool): Whether an intercept is used in addition to the predicted binding modes. Default: True
    """
    def __init__(self, datatype, **kwargs):
        super().__init__()
        self.datatype = datatype.lower()
        assert self.datatype in ["selex", "pbm"]
        self.padding_const = kwargs.get("padding_const", 0.25)
        self.use_dinuc = kwargs.get("use_dinuc", False)
        if "kernels" not in kwargs and "n_kernels" not in kwargs:
            kwargs["kernels"] = [0, 15]
            kwargs["n_kernels"] = len(kwargs["kernels"])
        elif "n_kernels" not in kwargs:
            kwargs["n_kernels"] = len(kwargs["kernels"])
        elif "kernels" not in kwargs and "bm_generator" not in kwargs:
            kwargs["kernels"] = [0] + [15]*(kwargs["n_kernels"]-1)
        elif "bm_generator" not in kwargs:
            assert len(kwargs["kernels"]) == kwargs["n_kernels"]
        self.kernels = kwargs.get("kernels")
        if self.datatype == "pbm":
            kwargs["target_dim"] = 1
        elif self.datatype == "selex":
            if "n_rounds" in kwargs:
                kwargs["target_dim"] = kwargs["n_rounds"]
            elif "target_dim" in kwargs:
                kwargs["n_rounds"] = kwargs["target_dim"] - 1
            else:
                print("n_rounds must be provided.")
                assert False
        assert not ("n_batches" in kwargs and "n_proteins" in kwargs)

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
            self.selex_module = SelexModule(**kwargs)

        self.best_model_state = None
        self.best_loss = None
        self.loss_history = []
        self.r2_history = []
        self.loss_color = []
        self.total_time = 0

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
            return self.selex_module(binding_scores, **kwargs)
        else:
            return None  # this line should never be called

    def set_seed(self, seed, index, max=0, min=-1):
        if isinstance(self.binding_modes, BindingModesSimple):
            self.binding_modes.set_seed(seed, index, max, min)
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
        return self.selex_module.get_log_etas()

    def dirichlet_regularization(self):
        return self.binding_modes.dirichlet_regularization()

    def exp_barrier(self, exp_max=40):
        out = 0
        for p in self.parameters():
            out += torch.sum(torch.exp(p - exp_max) + torch.exp(-p - exp_max))
        return out

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
                    ai = a[:, :, :, i : i + k]
                    for j in range(0, b.shape[-1] - k + 1):
                        bi = b[:, :, :, j : j + k]
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
        self.use_dinuc = kwargs.get("use_dinuc", False)

        self.conv_mono = tnn.ModuleList()
        self.conv_di = tnn.ModuleList()
        for k in self.kernels:
            if k == 0:
                self.conv_mono.append(None)
                self.conv_di.append(None)
            else:
                next_mono = tnn.Conv2d(1, 1, kernel_size=(4, k), padding=(0, 0), bias=False)
                if not self.init_random:
                    next_mono.weight.data.uniform_(0, 0)
                self.conv_mono.append(next_mono)

                next_di = tnn.Conv2d(1, 1, kernel_size=(16, k), padding=(0, 0), bias=False)
                if not self.init_random:
                    next_di.weight.data.uniform_(-.5, .5) # problem with fitting  dinucleotides if (0, 0)
                self.conv_di.append(next_di)

    def forward(self, mono, mono_rev, di=None, di_rev=None, **kwargs):
        bm_pred = []
        for i in range(len(self.kernels)):
            # print(i)
            if self.kernels[i] == 0:
                # intercept (will be scaled by the activity of the non-specific binding)
                temp = torch.tensor([1.0] * mono.shape[0], device=mono.device)
                bm_pred.append(temp)
            else:
                # check devices match
                if self.conv_mono[i].weight.device != mono.device:
                    self.conv_mono[i].weight.to(mono.device)

                if self.use_dinuc:
                    temp = torch.cat(
                        (
                            self.conv_mono[i](mono),
                            self.conv_mono[i](mono_rev),
                            self.conv_di[i](di),
                            self.conv_di[i](di_rev),
                        ),
                        dim=3,
                    )
                else:
                    temp = torch.cat((self.conv_mono[i](mono), self.conv_mono[i](mono_rev)), dim=3)

                # this particular step can generate out of bounds due to the exponentail cost
                temp = torch.exp(temp)
                temp = temp.view(temp.shape[0], -1)
                temp = torch.sum(temp, dim=1)
                bm_pred.append(temp)


        return torch.stack(bm_pred).T

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
        if self.conv_di[index] is not None:
            self.conv_di[index].weight.requires_grad = value
            if not value:
                self.conv_di[index].weight.grad = None

    def update_grad(self, index, value):
        self.update_grad_mono(index, value)
        self.update_grad_di(index, value)

    def modify_kernel(self, index=None, shift=0, expand_left=0, expand_right=0, device=None):
        # shift mono
        for i, m in enumerate(self.conv_mono):
            if index is not None and index != i:
                continue
            if m is None:
                continue
            before_w = m.weight.shape[-1]
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
            if shift >= 1:
                m.weight = torch.nn.Parameter(
                    torch.cat([m.weight[:, :, :, shift:], torch.zeros(1, 1, 16, shift, device=device)], dim=3)
                )
            elif shift <= -1:
                m.weight = torch.nn.Parameter(
                    torch.cat([torch.zeros(1, 1, 16, -shift, device=device), m.weight[:, :, :, :-shift]], dim=3)
                )

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
        return values[index].weight if values[index] is not None else None

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
            result = torch.matmul(b, a[batch, :, :])
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


class SelexModule(tnn.Module):
    """
    Implements the final calculations for the prediction of selex data.

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

    def forward(self, binding_scores, countsum, **kwargs):
        batch = kwargs.get("batch", None)
        if batch is None:
            batch = torch.zeros([binding_scores.shape[0]], device=binding_scores.device)

        if self.enr_series:
            predictions_ = [binding_scores[:, 0]]
            for i in range(1, self.n_rounds):
                predictions_.append(predictions_[-1] * binding_scores[:, i])
            out = torch.stack(predictions_).T
        else:
            out = binding_scores

        # iterative multiplication
        # for i in range(self.n_batches):
        #     eta = torch.exp(self.log_etas[i, :])
        #     out[batch == i] = out[batch == i] * eta

        # multiplication in one step
        etas = torch.exp(self.log_etas)
        out = out * etas[batch, :]

        results = out.T / torch.sum(out, dim=1)
        return (results * countsum).T

    def get_log_etas(self):
        return self.log_etas


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
                ai = a[:, :, :, i : i + k]
                for j in range(0, b.shape[-1] - k + 1):
                    bi = b[:, :, :, j : j + k]
                    bi_rev = torch.flip(bi, [3])[:, :, [3, 2, 1, 0], :]
                    d.append(((bi - ai) ** 2).sum() / bi.shape[-1])
                    d.append(((bi_rev - ai) ** 2).sum() / bi.shape[-1])

                    if lowest_d == -1 or d[-1] < lowest_d or d[-2] < lowest_d:
                        next_d = min(d[-2], d[-1])
                        # print(i, i + k, j, j + k, d[-2], d[-1])
                        lowest_d = next_d
    return min(d)

# This class should be deleted in the future
class MultibindFlexibleWeights(tnn.Module):
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
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length): #  state_size_buff=512):
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
        #h_0 = self.h_0[:,:x.size(0),:]
        # c_0 = self.c_0[:,:x.size(0),:]

        h_0 = Variable(torch.zeros(self.num_layers, residues.size(0), self.hidden_size, device=residues.device))  # hidden state
        c_0 = Variable(torch.zeros(self.num_layers, residues.size(0), self.hidden_size, device=residues.device))  # internal state
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
        self.output_size = enc_size*seq_length  # output size
        modules = [tnn.Linear(input_size, layers[0])]
        for i in range(len(layers)-1):
            modules.append(tnn.ReLU())
            modules.append(tnn.Linear(layers[i], layers[i+1]))
        modules.append(tnn.ReLU())
        modules.append(tnn.Linear(layers[len(layers)-1], self.output_size))
        self.decoder = tnn.Sequential(*modules)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.decoder(x)
        x = torch.reshape(x, (x.shape[0], self.enc_size, -1))
        return x
        # return tnn.functional.softmax(x, dim=1)


# This class should be deleted in the future
class ProteinDNABinding(tnn.Module):
    def __init__(self, n_rounds, n_batches, num_classes=1, input_size=21, hidden_size=2, num_layers=1, seq_length=88, datatype="pbm", **kwargs):
        super().__init__()
        self.datatype = datatype

        self.bm_prediction = BMPrediction(num_classes, input_size, hidden_size, num_layers, seq_length)
        self.decoder = mb.models.Decoder(enc_size=input_size, seq_length=seq_length, **kwargs)
        self.mubind = MultibindFlexibleWeights(n_rounds, n_batches, datatype=datatype)

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
