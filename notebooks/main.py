import numpy as np

import torch
import torch.utils.data as tdata
import torch.nn as tnn
import torch.nn.functional as tfunc
import torch.optim as topti

# Class for reading training/testing dataset files.
class toyDataset(tdata.Dataset):
    def __init__(self, dataFile, labelFile):
        # Load data from files.
        self.inputs = np.loadtxt(dataFile, dtype = np.float32).reshape(-1, 4, 1000)
        self.labels = np.loadtxt(labelFile, dtype = np.float32)

        self.length = len(self.labels)

    def __getitem__(self, index):
        # Return a single input/label pair from the dataset.
        inputSample = self.inputs[index]
        labelSample = self.labels[index]
        sample = {"input": inputSample, "label": labelSample}

        return sample

    def __len__(self):

        return self.length

# Class for creating the neural network.
class network(tnn.Module):

    def __init__(self):
        super(network, self).__init__()

        # Create and initialise weights and biases for the layers.
        self.conv1 = tnn.Conv1d(4, 32, 4)
        self.conv2 = tnn.Conv1d(32, 64, 4)
        self.conv3 = tnn.Conv1d(64, 128, 4)

        self.fc1 = tnn.Linear(15616, 128)
        self.fc2 = tnn.Linear(128, 1)

    def forward(self, x):
        # Create the forward pass through the network.
        x = self.conv1(x)
        x = tfunc.leaky_relu(x, 0.1)
        x = tfunc.max_pool1d(x, 2)
        x = tfunc.dropout(x, 0.2)

        x = self.conv2(x)
        x = tfunc.leaky_relu(x, 0.1)
        x = tfunc.max_pool1d(x, 2)
        x = tfunc.dropout(x, 0.2)

        x = self.conv3(x)
        x = tfunc.leaky_relu(x, 0.1)
        x = tfunc.max_pool1d(x, 2)
        x = tfunc.dropout(x, 0.2)

        x = x.view(x.shape[0], -1) # Flatten tensor.

        x = self.fc1(x)
        x = tfunc.leaky_relu(x, 0.1)
        x = tfunc.dropout(x, 0.2)

        x = self.fc2(x)

        x = x.view(-1) # Flatten tensor.

        return x

def main():
    # Use a GPU if available, as it should be faster.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: " + str(device))

    # Load the training dataset, and create a data loader to generate a batch.
    trainDataset = toyDataset("toy_TrainData.csv", "toy_TrainLabel.csv")
    trainLoader = tdata.DataLoader(dataset = trainDataset, batch_size = 16, shuffle = True)

    net = network().to(device) # Create an instance of the network in memory (potentially GPU memory).
    criterion = tnn.BCEWithLogitsLoss() # Add a sigmoid activation function to the output.  Use a binary cross entropy
                                        # loss function.
    optimiser = topti.Adam(net.parameters(), lr = 0.001) # Minimise the loss using the Adam algorithm.

    for epoch in range(5):
        runningLoss = 0

        for i, batch in enumerate(trainLoader):
            # Get a batch and potentially send it to GPU memory.
            inputs, labels = batch["input"].to(device), batch["label"].to(device)

            optimiser.zero_grad() # PyTorch calculates gradients by accumulating contributions to them (useful for
                                  # RNNs).  Hence we must manully set them to zero before calculating them.

            outputs = net(inputs) # Forward pass through the network.
            loss = criterion(outputs, labels)
            loss.backward() # Calculate gradients.
            optimiser.step() # Step to minimise the loss according to the gradient.

            runningLoss += loss.item()

            if i % 32 == 31:
                print("Epoch: %2d, Batch: %4d, Loss: %.3f" % (epoch + 1, i + 1, runningLoss / 32))
                runningLoss = 0

    # Load the testing dataset, and create a data loader to generate a batch.
    testDataset = toyDataset("toy_TestData.csv", "toy_TestLabel.csv")
    testLoader = tdata.DataLoader(dataset = testDataset, batch_size = 16)

    truePos, trueNeg, falsePos, falseNeg = 0, 0, 0, 0

    # Evaluate network on the test dataset.  We aren't calculating gradients, so disable autograd to speed up
    # computations and reduce memory usage.
    with torch.no_grad():
        for batch in testLoader:
            # Get a batch and potentially send it to GPU memory.
            inputs, labels = batch["input"].to(device), batch["label"].to(device)

            outputs = torch.sigmoid(net(inputs))
            predicted = torch.round(outputs)

            truePos += torch.sum(labels * predicted).item()
            trueNeg += torch.sum((1 - labels) * (1 - predicted)).item()
            falsePos += torch.sum((1 - labels) * predicted).item()
            falseNeg += torch.sum(labels * (1 - predicted)).item()

    accuracy = 100 * (truePos + trueNeg) / len(testDataset)
    matthews = MCC(truePos, trueNeg, falsePos, falseNeg)

    print("Classification accuracy: %.2f%%\n"
          "Matthews Correlation Coefficient: %.2f" % (accuracy, matthews))

# Matthews Correlation Coefficient calculation.
def MCC(tp, tn, fp, fn):
    numerator = tp * tn - fp * fn
    denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5

    with np.errstate(divide = "ignore", invalid = "ignore"):
        return np.divide(numerator, denominator)

if __name__ == '__main__':
    main()