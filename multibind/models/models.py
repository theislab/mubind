import numpy as np
import torch
import torch.nn as tnn

# Class for creating the neural network.
class Mono(tnn.Module):
    def __init__(self):
        super().__init__()
        # Create and initialise weights and biases for the layers.
        self.conv_mono = tnn.Conv2d(1, 1, kernel_size=(4, 8), bias=False)
        # self.fc = tnn.Linear(193, 1, bias=True)
        # torch.nn.init.uniform_(self.fc.weight, 0.0, 2/193)
        self.log_weight = tnn.Parameter(torch.tensor(np.array([[-5.3]]).astype(np.float32)))

    def forward(self, x):
        # Create the forward pass through the network.
        x = torch.unsqueeze(x, 1)
        x = self.conv_mono(x)
        x = torch.exp(x)
        x = x.view(x.shape[0], -1)  # Flatten tensor.
        # x = self.fc(x)
        # x = tfunc.linear(x, weight=self.log_weight.exp(), bias=self.non_spec)
        x = torch.sum(x, axis=1)
        x = x*torch.exp(self.log_weight)
        x = x / (1 + x)
        x = x.view(-1)  # Flatten tensor.
        return x


# Class for creating the neural network.
class Dinuc(tnn.Module):
    def __init__(self, use_dinuc=False):
        super().__init__()
        self.use_dinuc = use_dinuc
        # Create and initialise weights and biases for the layers.
        self.conv_mono = tnn.Conv2d(1, 1, kernel_size=(4, 8), bias=False)
        self.conv_di = tnn.Conv2d(1, 1, kernel_size=(16, 8), bias=False)
        # self.fc = tnn.Linear(193, 1, bias=True)
        # torch.nn.init.uniform_(self.fc.weight, 0.0, 2/193)
        self.log_weight_1 = tnn.Parameter(torch.tensor(np.array([[0]]).astype(np.float32)))
        self.log_weight_2 = tnn.Parameter(torch.tensor(np.array([[-5.3]]).astype(np.float32)))

    def forward(self, x):
        # Create the forward pass through the network.
        if self.use_dinuc:
            mono, di = x[0], x[1]
            mono = torch.unsqueeze(mono, 1)
            mono = self.conv_mono(mono)
            di = torch.unsqueeze(di, 1)
            di = self.conv_di(di)
            mono = torch.exp(mono)
            di = torch.exp(di)
            mono = mono.view(mono.shape[0], -1)  # Flatten tensor.
            di = di.view(di.shape[0], -1)
            x = torch.exp(self.log_weight_1)*torch.sum(mono, axis=1) + torch.exp(self.log_weight_2)*torch.sum(di, axis=1)
        else:
            x = torch.unsqueeze(x, 1)
            x = self.conv_mono(x)
            x = torch.exp(x)
            x = x.view(x.shape[0], -1)  # Flatten tensor.
            # x = self.fc(x)
            # x = tfunc.linear(x, weight=self.log_weight.exp(), bias=self.non_spec)
            x = torch.sum(x, axis=1)
            x = x * torch.exp(self.log_weight_1)
        
        x = x / (1 + x)
        x = x.view(-1)  # Flatten tensor.
        return x


# One selex datasets with multiple rounds of counts
class DinucSelex(tnn.Module):
    # n_rounds indicates the number of experimental rounds
    def __init__(self, use_dinuc=False, kernels=[0, 14, 12], n_rounds=1, rho=1, gamma=0):
        super().__init__()
        if use_dinuc:
            print("Dinuc features not implemented yet. Using only mononuc features.")
        self.use_dinuc = use_dinuc
        self.n_rounds = n_rounds
        self.rho = rho
        self.gamma = gamma
        self.padding = tnn.ModuleList()
        self.conv_mono = tnn.ModuleList()
        self.conv_di = tnn.ModuleList()
        self.kernels = kernels
        for k in self.kernels:
            if k == 0:
                self.padding.append(tnn.ConstantPad2d((k - 1, k - 1, 0, 0), 0.25))
                self.conv_mono.append(None)
                self.conv_di.append(None)
            else:
                self.padding.append(tnn.ConstantPad2d((k-1, k-1, 0, 0), 0.25))
                self.conv_mono.append(tnn.Conv2d(1, 1, kernel_size=(4, k), padding=(0, 0), bias=False))
                self.conv_di.append(tnn.Conv2d(1, 1, kernel_size=(16, k), padding=(0, 0), bias=False))
        self.log_activity = tnn.Embedding(len(kernels), n_rounds+1)
        self.log_activity.weight.data.uniform_(0, 0)  # initialize log_activity as zeros.
        self.log_eta = tnn.Embedding(n_rounds+1, 1)
        self.log_eta.weight.data.uniform_(0, 0)
        self.best_model_state = None

    def forward(self, x):
        # Create the forward pass through the network.
        mono, mono_rev, di, di_rev, batch, seqlen, countsum = x
        mono = torch.unsqueeze(mono, 1)
        mono_rev = torch.unsqueeze(mono_rev, 1)
        # x = torch.zeros([mono.shape[0], len(self.kernels)], requires_grad=True)
        x_ = []
        for i in range(len(self.kernels)):
            if self.kernels[i] == 0:
                x_.append(torch.Tensor([1.0] * mono.shape[0]))
            else:
                temp = self.conv_mono[i](mono) + self.conv_mono[i](mono_rev)
                temp = torch.exp(temp)
                temp = temp.view(temp.shape[0], -1)
                temp = torch.sum(temp, axis=1)
                x_.append(temp)
        x = torch.stack(x_).T

        a = torch.exp(self.log_activity.weight)
        x = torch.matmul(x, a)

        eta = torch.exp(self.log_eta.weight.T)
        predictions_ = [x[:, 0]]
        for i in range(1, self.n_rounds+1):
            predictions_.append(predictions_[-1] * x[:, i])
        predictions = torch.stack(predictions_).T
        predictions = predictions * eta
        predictions = (predictions.T / torch.sum(predictions, axis=1))
        return (predictions * countsum).T


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