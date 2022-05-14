import numpy as np
import pandas as pd
import torch
import torch.utils.data as tdata
import torch.nn as tnn
import multibind as mb
import itertools
import copy

def calculate_enrichment(data, approx=True, cols=[0, 1]):
    data['p0'] = data[cols[0]] / np.sum(data[cols[0]])
    data['p1'] = data[cols[1]] / np.sum(data[cols[1]])
    data['enr'] = data['p1'] / data['p0']
    if approx:
        data['enr_approx'] = np.where(data['p0'] == 0, data['p1'] / (data['p0'] + 1e-06), data['enr'])
    return data

def create_datasets(data_file):
    # read data and calculate additional columns
    data = pd.read_csv(data_file, sep='\t', header=None)
    data.columns = ['seq', 0, 1]
    data = calculate_enrichment(data)
    # divide in train and test data
    test_dataframe = data.sample(frac=0.001)
    train_dataframe = data.drop(test_dataframe.index)
    test_dataframe.index = range(len(test_dataframe))
    train_dataframe.index = range(len(train_dataframe))
    # create datasets and dataloaders
    train_data = mb.datasets.SelexDataset(data_frame=train_dataframe)
    train_loader = tdata.DataLoader(dataset=train_data, batch_size=256, shuffle=True)
    test_data = mb.datasets.SelexDataset(data_frame=test_dataframe)
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


loss_history = []


# if early_stopping is positive, training is stopped if over the length of early_stopping no improvement happened or
# num_epochs is reached.
def train_network(net, train_dataloader, device, optimiser, criterion, num_epochs=15, early_stopping=-1, log_each=None):
    global loss_history
    loss_history = []
    best_loss = None
    best_epoch = -1
    for epoch in range(num_epochs):
        running_loss = 0
        for i, batch in enumerate(train_dataloader):
            # Get a batch and potentially send it to GPU memory.
            # print(batch.keys())
            # mononuc = batch["mononuc"].type(torch.LongTensor).to(device)
            mononuc = batch['mononuc'].to(device)
            mononuc_rev = batch['mononuc_rev'].to(device)
            dinuc = batch['dinuc'].type(torch.LongTensor).to(device) if 'dinuc' in batch else None
            dinuc_rev = batch['dinuc_rev'].to(device) if 'dinuc_rev' in batch else None
            b = batch['batch'].to(device) if 'batch' in batch else None
            target = batch['target'].to(device) if 'target' in batch else None
            rounds = batch['rounds'].to(device) if 'rounds' in batch else None
            is_count_data = batch['is_count_data'] if 'is_count_data' in batch else None
            seqlen = batch['seqlen'] if 'seqlen' in batch else None

            inputs = (mononuc, mononuc_rev, dinuc, dinuc_rev, b, seqlen, torch.sum(rounds, axis=1))
            optimiser.zero_grad()  # PyTorch calculates gradients by accumulating contributions to them (useful for
            # RNNs).  Hence we must manully set them to zero before calculating them.
            outputs = net(inputs)  # Forward pass through the network.
            loss = criterion(outputs, rounds)
            # loss = criterion(outputs/(1+outputs), target) #, is_count_data)
            # print('here...')
            # print(loss)
            # assert False
            loss.backward()  # Calculate gradients.
            optimiser.step()  # Step to minimise the loss according to the gradient.
            running_loss += loss.item()

        loss_final = running_loss / len(train_dataloader)
        if log_each is None or (epoch % log_each == 0):
            print("Epoch: %2d, Loss: %.3f" % (epoch + 1, loss_final))

        if best_loss is None or loss_final < best_loss:
            best_loss = loss_final
            best_epoch = epoch
            net.best_model_state = copy.deepcopy(net.state_dict())

        # print("Epoch: %2d, Loss: %.3f" % (epoch + 1, running_loss / len(train_dataloader)))
        loss_history.append(loss_final)

        if early_stopping > 0 and epoch >= best_epoch + early_stopping:
            break

    return loss_history


# returns the last loss value if it exists
def get_last_loss_value():
    global loss_history
    if len(loss_history) > 0:
        return loss_history[-1]
    return None


def create_simulated_data(motif='GATA', n_batch=None, n_trials=20000, seqlen=100, batch_sizes=10):
    x2, y2 = mb.datasets.simulate_xy(motif, n_trials=n_trials, seqlen=seqlen, max_mismatches=-1, batch=1)
    
    # print('skip normalizing...')
    # y2 = ((y2 - y2.min()) / (np.max(y2) - np.min(y2))).astype(np.float32)
    
    # data = pd.DataFrame({'seq': x1, 'enr_approx': y1})
    data = pd.DataFrame({'seq': x2, 'target': y2})
    batch = mb.tl.onehot_covar(np.random.random_integers(0, n_batch - 1, len(y2)))
    
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
    data['is_count_data'] = np.repeat(1, data.shape[0])
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
    train_data = mb.datasets.ChipSeqDataset(data_frame=train_dataframe)    
    train_data.batch = batch_train
    train_data.seq = train_dataframe.seq
    
    train_loader = tdata.DataLoader(dataset=train_data, batch_size=256, shuffle=True)
    test_data = mb.datasets.ChipSeqDataset(data_frame=test_dataframe)
    test_data.batch = batch_test
    test_data.seq = test_dataframe.seq

    test_loader = tdata.DataLoader(dataset=test_data, batch_size=1, shuffle=False)
    return train_loader, test_loader

def create_multi_data(n_chip=100, n_selex=100, n_batch_selex=3):
    # chip seq
    x_chip, y_chip = mb.datasets.gata_remap(n_sample=n_chip) if n_chip > 0 else [[], []]
    df_chip = pd.DataFrame({'seq': x_chip, 'target': y_chip})
    df_chip['is_count_data'] = np.repeat(0, df_chip.shape[0])
    df_chip['batch'] = '0_chip'
    # selex
    # seqlen = 10
    # x_selex, y_selex = mb.datasets.simulate_xy('GATA', n_trials=n_selex, seqlen=seqlen, max_mismatches=-1, batch=50)
    # df_selex = pd.DataFrame({'seq': x_selex, 'target': y_selex})
    # df_selex['is_count_data'] = np.repeat(1, df_selex.shape[0])
    # n_batch = n_batch_selex
    # batch sizes. Dataset 2 has many times more reads than Dataset 1
    batch_sizes = [int(5 * 10 ** i) for i in range(n_batch_selex)]
    train1, test1 = mb.tl.create_simulated_data(motif='GATA', n_batch=n_batch_selex, n_trials=n_selex,
                                                seqlen=10, batch_sizes=batch_sizes) # multiplier=100)
    # print(train1.dataset.seq, train1.shape)
    # assert False
    df_selex = pd.DataFrame({'seq': train1.dataset.seq, 'target': train1.dataset.target})
    df_selex['batch'] = pd.Series(train1.dataset.batch).astype(str).values + '_selex'
    df_selex['is_count_data'] = np.repeat(1, df_selex.shape[0])
    datasets = [df_chip, df_selex]
    data = pd.concat(datasets)
    
    # pad nucleotides for leaking regions
    
    data['seqlen'] = data['seq'].str.len()
    # print(data.head())
    # print(data.tail())
    # assert False
    
    max_seqlen = data['seq'].str.len().max()
    print('max seqlen', max_seqlen)
    max_seqlen = int(max_seqlen)
    data['k'] = "_"
    padded_seq = data['seq'] + data['k'].str.repeat(max_seqlen - data['seqlen'].astype(int))
    data['seq'] = np.where(data['seqlen'] < max_seqlen, padded_seq, data['seq'])    
    data['seqlen'] = data['seq'].str.len()
    assert len(set(data['seqlen'])) == 1
    
    # batches
    # n_batch = len(datasets)
    # batch = []
    # for i, df in enumerate(datasets):
    #     batch.append(np.repeat(i, df.shape[0]))    
    # batch = np.array(np.concatenate(batch))
    # data['batch'] = batch
    # data['batch'] -= np.min(data['batch'])
    
    # print(data.batch.value_counts())
    # assert False
    
#     print(data['batch'].value_counts())
#     print(data.head())
    
    # print(data.columns)
    
    # divide in train and test data -- copied from above, organize differently!
    test_dataframe = data.sample(frac=0.01)
    train_dataframe = data.drop(test_dataframe.index)
    batch_test = test_dataframe['batch']
    batch_train = train_dataframe['batch']
    
#     print('train batch labels')
#     print(batch_train)
#     print(batch_train.value_counts())
    
    test_dataframe.index = range(len(test_dataframe))
    train_dataframe.index = range(len(train_dataframe))
    n_train = len(train_dataframe.index)
    n_test = len(test_dataframe.index)
    n_dim = n_train + n_test

    train_data = mb.datasets.MultiDataset(data_frame=train_dataframe)
    train_data.batch = np.array(batch_train)
    train_data.batch_one_hot = mb.tl.onehot_covar(train_data.batch)
    train_loader = tdata.DataLoader(dataset=train_data, batch_size=256, shuffle=True)
    # print(np.bincount(batch))

    test_data = mb.datasets.MultiDataset(data_frame=test_dataframe)
    test_data.batch = np.array(batch_test)
    test_data.batch_one_hot = mb.tl.onehot_covar(test_data.batch)
    test_loader = tdata.DataLoader(dataset=test_data, batch_size=1, shuffle=False)

    return train_loader, test_loader
