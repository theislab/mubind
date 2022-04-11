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

def _onehot_covar(covar, label_encoder=LabelEncoder(), onehot_encoder=OneHotEncoder(sparse=False)):
    covar_arr = np.array(list(covar))
    covar_int = label_encoder.fit_transform(covar_arr)
    pre_onehot = onehot_encoder.fit_transform(covar_int.reshape(-1, 1))
    return pre_onehot.astype(np.int)

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
    def __init__(self, data_frame, use_dinuc=False, batch=None):
        self.batch = batch
        self.target = data_frame['target'].astype(np.float32)
        # self.rounds = self.data[[0, 1]].to_numpy()
        self.le = LabelEncoder()
        self.oe = OneHotEncoder(sparse=False)
        self.length = len(data_frame)
        self.mononuc = np.array([_onehot_mononuc(row['seq'], self.le, self.oe) for index, row in data_frame.iterrows()])
        self.dinuc = np.array([_onehot_dinuc(row['seq'], self.le, self.oe) for index, row in data_frame.iterrows()])

    def __getitem__(self, index):
        # Return a single input/label pair from the dataset.
        mononuc_sample = self.mononuc[index]
        target_sample = self.target[index]
        
        # print(self.batch)
        # print(self.batch.shape)
        batch = self.batch[index]
        dinuc_sample = self.dinuc[index]
        sample = {"mononuc": mononuc_sample, "dinuc": dinuc_sample, "target": target_sample, "batch": batch}
        return sample

    def __len__(self):
        return self.length


# (negative) Log-likelihood of the Poisson distribution
class PoissonLoss(tnn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        return torch.mean(inputs - targets*torch.log(inputs))


# Custom loss function
class CustomLoss(tnn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

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
            mononuc = batch["mononuc"].to(device)
            dinuc = batch["dinuc"].to(device) if 'dinuc' in batch else None
            b = batch['batch'].to(device)
            target = batch["target"].to(device)
            inputs = (mononuc, dinuc, b, target)
            output = net(inputs)
            
            all_outputs.append(output.squeeze().cpu().detach().numpy())
            all_targets.append(target)
    return np.array(all_targets), np.array(all_outputs)


def train_network(net, train_dataloader, device, optimiser, criterion, num_epochs=15, log_each=None):
    loss_history = []
    for epoch in range(num_epochs):
        running_loss = 0
        for i, batch in enumerate(train_dataloader):
            # Get a batch and potentially send it to GPU memory.
            # print(batch.keys())
            mononuc = batch["mononuc"].to(device)
            dinuc = batch["dinuc"].to(device) if 'dinuc' in batch else None
            b = batch['batch'].to(device)
            target = batch["target"].to(device)
            inputs = (mononuc, dinuc, b, target)
            
            optimiser.zero_grad()  # PyTorch calculates gradients by accumulating contributions to them (useful for
            # RNNs).  Hence we must manully set them to zero before calculating them.
            outputs = net(inputs)  # Forward pass through the network.
            loss = criterion(outputs, target)
            loss.backward()  # Calculate gradients.
            optimiser.step()  # Step to minimise the loss according to the gradient.
            running_loss += loss.item()
        if log_each is None or (epoch % log_each == 0):
            print("Epoch: %2d, Loss: %.3f" % (epoch + 1, running_loss / len(train_dataloader)))

        # print("Epoch: %2d, Loss: %.3f" % (epoch + 1, running_loss / len(train_dataloader)))
        loss_history.append(running_loss / len(train_dataloader))
    
    return loss_history


def create_simulated_data(motif='GATA', n_batch=None, n_trials=20000, seqlen=100, batch_sizes=10):
    x2, y2 = mb.datasets.simulate_xy(motif, n_trials=n_trials, seqlen=seqlen, max_mismatches=-1, batch=1)
    
    # print('skip normalizing...')
    # y2 = ((y2 - y2.min()) / (np.max(y2) - np.min(y2))).astype(np.float32)
    
    # data = pd.DataFrame({'seq': x1, 'enr_approx': y1})
    data = pd.DataFrame({'seq': x2, 'target': y2})
    batch = _onehot_covar(np.random.random_integers(0, n_batch - 1, len(y2)))
    
    assert n_batch == len(batch_sizes)
    batch_mult = np.array(batch_sizes)[(np.argmax(batch, axis=1))]    
    # print(batch_mult)
    # df = pd.DataFrame()
    # df['y.before'] = np.array(data['target'])
    # print(np.argmax(batch, axis=1))
    # df['batch'] = np.argmax(batch, axis=1)
    # df['mult'] = batch_mult
    # df['y.after'] = np.array(data['target']) * np.array(batch_mult)
    # print('head...')
    # print(df.head(25))
    # print('tail')
    # print(df.tail(25))
    
    data['target'] = np.array(data['target']) * np.array(batch_mult)
    # assert False
    
    # divide in train and test data -- copied from above, organize differently!
    test_dataframe = data.sample(frac=0.01)
    train_dataframe = data.drop(test_dataframe.index)
    batch_test = np.argmax(batch[test_dataframe.index], axis=1)
    batch_train = np.argmax(batch[train_dataframe.index], axis=1)

    test_dataframe.index = range(len(test_dataframe))
    train_dataframe.index = range(len(train_dataframe))

    n_train = len(train_dataframe.index)
    n_test = len(test_dataframe.index)
    n_dim = n_train + n_test
    
    # create datasets and dataloaders
    # print(train_dataframe)
    # print(batch, n_train)
    # print(batch[:n_train])
    train_data = ChipSeqDataset(data_frame=train_dataframe)    
    train_data.batch = batch_train
    
    train_loader = tdata.DataLoader(dataset=train_data, batch_size=256, shuffle=True)
    test_data = ChipSeqDataset(data_frame=test_dataframe)
    test_data.batch = batch_test

    test_loader = tdata.DataLoader(dataset=test_data, batch_size=1, shuffle=False)
    return train_loader, test_loader