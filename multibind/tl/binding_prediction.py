import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import torch
import torch.utils.data as tdata
import torch.nn as tnn
import multibind as mb


def _onehot_mononuc(seq, label_encoder=LabelEncoder(), onehot_encoder=OneHotEncoder(sparse=False)):
    seq_arr = np.array(list(seq + 'ACGT'))
    seq_int = label_encoder.fit_transform(seq_arr)
    pre_onehot = onehot_encoder.fit_transform(seq_int.reshape(-1, 1))
    return pre_onehot.T[:, :-4].astype(np.float32)


def _onehot_dinuc(seq, label_encoder=LabelEncoder(), onehot_encoder=OneHotEncoder(sparse=False)):
    extended_seq = seq + 'AACAGATCCGCTGGTTA'  # The added string contains each possible dinucleotide feature once
    dinuc_arr = np.array([extended_seq[i:i+2] for i in range(len(extended_seq) - 1)])
    seq_int = label_encoder.fit_transform(dinuc_arr)
    pre_onehot = onehot_encoder.fit_transform(seq_int.reshape(-1, 1))
    return pre_onehot.T[:, :-17].astype(np.float32)


# Class for reading training/testing SELEX dataset files.
class SelexDataset(tdata.Dataset):
    def __init__(self, data_frame):
        self.target = data_frame['enr_approx']
        # self.rounds = self.data[[0, 1]].to_numpy()
        self.le = LabelEncoder()
        self.oe = OneHotEncoder(sparse=False)
        self.length = len(data_frame)
        self.inputs = np.array([_onehot_mononuc(row['seq'], self.le, self.oe) for index, row in data_frame.iterrows()])

    def __getitem__(self, index):
        # Return a single input/label pair from the dataset.
        input_sample = self.inputs[index]
        target_sample = self.target[index]
        sample = {"mononuc": input_sample, "target": target_sample}
        return sample

    def __len__(self):
        return self.length


# Class for reading training/testing ChIPSeq dataset files.
class ChipSeqDataset(tdata.Dataset):
    def __init__(self, data_frame, use_dinuc=False):
        self.use_dinuc = use_dinuc
        self.target = data_frame['target'].astype(np.float32)
        # self.rounds = self.data[[0, 1]].to_numpy()
        self.le = LabelEncoder()
        self.oe = OneHotEncoder(sparse=False)
        self.length = len(data_frame)
        self.mononuc = np.array([_onehot_mononuc(row['seq'], self.le, self.oe) for index, row in data_frame.iterrows()])
        if self.use_dinuc:
            self.dinuc = np.array([_onehot_dinuc(row['seq'], self.le, self.oe) for index, row in data_frame.iterrows()])

    def __getitem__(self, index):
        # Return a single input/label pair from the dataset.
        mononuc_sample = self.mononuc[index]
        target_sample = self.target[index]
        if self.use_dinuc:
            dinuc_sample = self.dinuc[index]
            sample = {"mononuc": mononuc_sample, "dinuc": dinuc_sample, "target": target_sample}
        else:
            sample = {"mononuc": mononuc_sample, "target": target_sample}
        return sample

    def __len__(self):
        return self.length


# (negative) Log-likelihood of the Poisson distribution
class PoissonLoss(tnn.Module):
    def __init__(self):
        super(PoissonLoss, self).__init__()

    def forward(self, inputs, targets):
        return torch.mean(inputs - targets*torch.log(inputs))


# Custom loss function
class CustomLoss(tnn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CustomLoss, self).__init__()

    def forward(self, inputs, rounds, avoid_zero=True):
        if avoid_zero:
            rounds = rounds + 0.1
        f = inputs*rounds[:, 0]
        return -torch.sum(rounds[:, 1]*torch.log(f+0.0001) - f)


def _calculate_enrichment(data, approx=True):
    data['p0'] = data[0] / np.sum(data[0])
    data['p1'] = data[1] / np.sum(data[1])
    data['enr'] = data['p1'] / data['p0']
    if approx:
        data['enr_approx'] = np.where(data['p0'] == 0, data['p1'] / (data['p0'] + 1e-06), data['enr'])
    return data


def create_datasets(data_file):
    # read data and calculate additional columns
    data = pd.read_csv(data_file, sep='\t', header=None)
    data.columns = ['seq', 0, 1]
    data = _calculate_enrichment(data)
    # divide in train and test data
    test_dataframe = data.sample(frac=0.001)
    train_dataframe = data.drop(test_dataframe.index)
    test_dataframe.index = range(len(test_dataframe))
    train_dataframe.index = range(len(train_dataframe))
    # create datasets and dataloaders
    train_data = SelexDataset(data_frame=train_dataframe)
    train_loader = tdata.DataLoader(dataset=train_data, batch_size=256, shuffle=True)
    test_data = SelexDataset(data_frame=test_dataframe)
    test_loader = tdata.DataLoader(dataset=test_data, batch_size=1, shuffle=False)
    return train_loader, test_loader


def test_network(net, test_dataloader, device):
    all_targets, all_outputs = [], []
    with torch.no_grad():  # we don't need gradients in the testing phase
        for i, batch in enumerate(test_dataloader):
            # Get a batch and potentially send it to GPU memory.
            # inputs, target = batch["input"].to(device), batch["target"].to(device)
            if "dinuc" in batch:
                mononuc, dinuc, target = batch["mononuc"].to(device), batch["dinuc"].to(device), batch["target"].to(device)
                inputs = (mononuc, dinuc)
            else:
                inputs, target = batch["mononuc"].to(device), batch["target"].to(device)
            output = net(inputs)
            all_outputs.append(output.squeeze().cpu().detach().numpy())
            all_targets.append(target)
    return np.array(all_targets), np.array(all_outputs)


def train_network(net, train_dataloader, device, optimiser, criterion, num_epochs=15):
    loss_history = []
    for epoch in range(num_epochs):
        running_loss = 0
        for i, batch in enumerate(train_dataloader):
            # Get a batch and potentially send it to GPU memory.
            if "dinuc" in batch:
                mononuc, dinuc, target = batch["mononuc"].to(device), batch["dinuc"].to(device), batch["target"].to(device)
                inputs = (mononuc, dinuc)
            else:
                inputs, target = batch["mononuc"].to(device), batch["target"].to(device)
            optimiser.zero_grad()  # PyTorch calculates gradients by accumulating contributions to them (useful for
            # RNNs).  Hence we must manully set them to zero before calculating them.
            outputs = net(inputs)  # Forward pass through the network.
            loss = criterion(outputs, target)
            loss.backward()  # Calculate gradients.
            optimiser.step()  # Step to minimise the loss according to the gradient.
            running_loss += loss.item()
        print("Epoch: %2d, Loss: %.3f" % (epoch + 1, running_loss / len(train_dataloader)))
        loss_history.append(running_loss / len(train_dataloader))
    return loss_history


def create_simulated_data():
    # motif = 'AGGAACCTA'
    motif = 'GATA'
    # SELEX
    x1, y1 = mb.datasets.simulate_xy(motif, n_trials=20000, seqlen=20, max_mismatches=5)
    # ChIP-seq
    x2, y2 = mb.datasets.simulate_xy(motif, n_trials=20000, seqlen=100, max_mismatches=-1)
    y2 = ((y2 - y2.min()) / (np.max(y2) - np.min(y2))).astype(np.float32)
    # data = pd.DataFrame({'seq': x1, 'enr_approx': y1})
    data = pd.DataFrame({'seq': x2, 'target': y2})
    # divide in train and test data -- copied from above, organize differently!
    test_dataframe = data.sample(frac=0.01)
    train_dataframe = data.drop(test_dataframe.index)
    test_dataframe.index = range(len(test_dataframe))
    train_dataframe.index = range(len(train_dataframe))
    # create datasets and dataloaders
    train_data = ChipSeqDataset(data_frame=train_dataframe)
    train_loader = tdata.DataLoader(dataset=train_data, batch_size=256, shuffle=True)
    test_data = ChipSeqDataset(data_frame=test_dataframe)
    test_loader = tdata.DataLoader(dataset=test_data, batch_size=1, shuffle=False)
    return train_loader, test_loader
