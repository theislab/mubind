import torch
import torch.nn as tnn
import itertools


def _mono2revmono(x):
    return torch.flip(x, [1])[:, [3, 2, 1, 0], :]


def _mono2dinuc(mono):
    # this is a concatenation of columns (i : i - 1) and (i + 1 : i)
    n_mono = mono.shape[1]
    x = torch.cat([mono[:, :, :-1], mono[:, :, 1:]], dim=1)
    # print(x.shape)
    dinuc = torch.cat(
        [  # AX
            (x[:, 0, :] * x[:, 4, :]),
            (x[:, 0, :] * x[:, 5, :]),
            (x[:, 0, :] * x[:, 6, :]),
            (x[:, 0, :] * x[:, 7, :]),
            # CX
            (x[:, 1, :] * x[:, 4, :]),
            (x[:, 1, :] * x[:, 5, :]),
            (x[:, 1, :] * x[:, 6, :]),
            (x[:, 1, :] * x[:, 7, :]),
            # GX
            (x[:, 2, :] * x[:, 4, :]),
            (x[:, 2, :] * x[:, 5, :]),
            (x[:, 2, :] * x[:, 6, :]),
            (x[:, 2, :] * x[:, 7, :]),
            # TX
            (x[:, 3, :] * x[:, 4, :]),
            (x[:, 3, :] * x[:, 5, :]),
            (x[:, 3, :] * x[:, 6, :]),
            (x[:, 3, :] * x[:, 7, :]),
        ],
        dim=1,
    ).reshape(x.shape[0], n_mono**2, x.shape[2])
    return dinuc


# One selex datasets with multiple rounds of counts
class DinucSelex(tnn.Module):
    # n_rounds indicates the number of experimental rounds
    def __init__(
        self,
        n_rounds,
        n_batches,
        use_dinuc=False,
        kernels=[0, 14, 12],
        ignore_kernel=None,
        rho=1,
        gamma=0,
        init_random=True,
        enr_series=True,
        padding_const=0.25,
    ):
        super().__init__()
        self.use_dinuc = use_dinuc
        self.n_rounds = n_rounds
        self.n_batches = n_batches
        self.rho = rho
        self.gamma = gamma

        # self.padding = tnn.ModuleList()
        # only keep one padding equals to the length of the max kernel
        self.padding = tnn.ConstantPad2d((max(kernels) - 1, max(kernels) - 1, 0, 0), padding_const)

        self.conv_mono = tnn.ModuleList()
        self.conv_di = tnn.ModuleList()
        self.log_activities = tnn.ParameterList()
        self.kernels = kernels
        self.enr_series = enr_series
        # self.log_activities = tnn.Parameter(torch.zeros([n_batches, len(kernels), n_rounds+1]))
        self.log_etas = tnn.Parameter(torch.zeros([n_batches, n_rounds + 1]))
        self.ignore_kernel = ignore_kernel

        for k in self.kernels:
            if k == 0:
                # self.padding.append(None)
                self.conv_mono.append(None)
                self.conv_di.append(None)
            else:
                # self.padding.append(tnn.ConstantPad2d((k - 1, k - 1, 0, 0), 0.25))
                next_mono = tnn.Conv2d(1, 1, kernel_size=(4, k), padding=(0, 0), bias=False)
                if not init_random:
                    next_mono.weight.data.uniform_(0, 0)
                self.conv_mono.append(next_mono)

                next_di = tnn.Conv2d(1, 1, kernel_size=(16, k), padding=(0, 0), bias=False)
                if not init_random:
                    next_di.weight.data.uniform_(0, 0)
                self.conv_di.append(next_di)
            self.log_activities.append(tnn.Parameter(torch.zeros([n_batches, n_rounds + 1], dtype=torch.float32)))

        # self.log_activity = tnn.Embedding(len(kernels), n_rounds+1)
        # self.log_activity.weight.data.uniform_(0, 0)  # initialize log_activity as zeros.
        # self.log_eta = tnn.Embedding(n_rounds+1, 1)
        # self.log_eta.weight.data.uniform_(0, 0)
        self.best_model_state = None
        self.best_loss = None
        self.loss_history = []
        self.loss_color = []

    def forward(self, x, min_value=1e-15):
        # Create the forward pass through the network.
        mono, batch, countsum = x

        # convert mono to dinuc
        # print(mono.shape)

        # print(mono.shape, di.shape)
        # assert False

        # print(mono.shape)

        # padding of sequences
        mono = self.padding(mono)
        # print(mono.shape)
        # assert False

        # prepare the other three objects that we need
        mono_rev = _mono2revmono(mono)

        di = _mono2dinuc(mono)
        di_rev = _mono2dinuc(mono_rev)

        # unsqueeze mono after preparing di and unsqueezing mono
        mono_rev = torch.unsqueeze(mono_rev, 1)
        mono = torch.unsqueeze(mono, 1)
        di = torch.unsqueeze(di, 1)
        di_rev = torch.unsqueeze(di_rev, 1)

        # x = torch.zeros([mono.shape[0], len(self.kernels)], requires_grad=True)
        x_ = []
        # print(mono.device)
        # print(self.ignore_kernel)
        # assert False
        for i in range(len(self.kernels)):
            # print(i)
            # if self.ignore_kernel is not None and self.ignore_kernel[i]:
            #     temp = torch.Tensor([0.0] * mono.shape[0]).to(device=mono.device)
            #     x_.append(temp)
            if self.kernels[i] == 0:
                temp = torch.Tensor([1.0] * mono.shape[0]).to(device=mono.device)
                x_.append(temp)
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
                temp = torch.exp(temp)
                temp = temp.view(temp.shape[0], -1)
                temp = torch.sum(temp, axis=1)
                x_.append(temp)
                # print(temp.shape, x_.shape)


        x = torch.stack(x_).T

        scores = torch.zeros([x.shape[0], self.n_rounds + 1]).to(device=mono.device)  #  + min_value# conversion for gpu
        # print(scores)
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

        # print()
        # print(x)
        # print(a)

        # print('\n\nfinal scores')
        # print(scores)
        # assert False
        # a = torch.reshape(a, [a.shape[0], a.shape[2]])
        # x = torch.matmul(x, a)
        # sequential enrichment or independent samples
        if self.enr_series:
            # print('using enrichment series')
            predictions_ = [scores[:, 0]]
            for i in range(1, self.n_rounds + 1):
                predictions_.append(predictions_[-1] * scores[:, i])
            out = torch.stack(predictions_).T

            # print('here...')
            # print(out[:10])
        else:
            out = scores

        results = torch.zeros([mono.shape[0], self.n_rounds + 1]).to(
            device=mono.device
        )  #  + min_value # conversion for gpu

        for i in range(self.n_batches):
            eta = torch.exp(self.log_etas[i, :])
            out[batch == i] = out[batch == i] * eta
            results[batch == i] = (out[batch == i].T / torch.sum(out[batch == i], axis=1)).T
            results[batch == i] = (results[batch == i].T * countsum[batch == i].type(torch.float32)).T


        return results

    def set_seed(self, seed, index, max=0, min=-1):
        assert len(seed) <= self.conv_mono[index].kernel_size[1]
        shift = int((self.conv_mono[index].kernel_size[1] - len(seed)) / 2)
        seed_params = torch.full(self.conv_mono[index].kernel_size, max, dtype=torch.float32)
        for i in range(len(seed)):
            if seed[i] == 'A':
                seed_params[:, i+shift] = torch.tensor([max, min, min, min])
            elif seed[i] == 'C':
                seed_params[:, i + shift] = torch.tensor([min, max, min, min])
            elif seed[i] == 'G':
                seed_params[:, i + shift] = torch.tensor([min, min, max, min])
            elif seed[i] == 'T':
                seed_params[:, i + shift] = torch.tensor([min, min, min, max])
            else:
                seed_params[:, i + shift] = torch.tensor([0, 0, 0, 0])
        self.conv_mono[index].weight = tnn.Parameter(torch.unsqueeze(torch.unsqueeze(seed_params, 0), 0))

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
                    ai = a[:, :, :, i:i + k]
                    for j in range(0, b.shape[-1] - k + 1):
                        bi = b[:, :, :, j:j + k]
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
                ai = a[:, :, :, i:i + k]
                for j in range(0, b.shape[-1] - k + 1):
                    bi = b[:, :, :, j:j + k]
                    bi_rev = torch.flip(bi, [3])[:, :, [3, 2, 1, 0], :]
                    d.append(((bi - ai) ** 2).sum() / bi.shape[-1])
                    d.append(((bi_rev - ai) ** 2).sum() / bi.shape[-1])

                    if lowest_d == -1 or d[-1] < lowest_d or d[-2] < lowest_d:
                        next_d = min(d[-2], d[-1])
                        # print(i, i + k, j, j + k, d[-2], d[-1])
                        lowest_d = next_d
    return min(d)


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
