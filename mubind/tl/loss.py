import torch
import torch.nn as tnn


class PoissonLoss(tnn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        return torch.nanmean(y_pred - y_true * torch.log(y_pred))


class ProboundLoss(tnn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        return torch.mean(-y_true * torch.log((y_pred.T / torch.sum(y_pred, dim=1)).T))


class MSELoss(tnn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        # print(y_pred[:5], y_true[:5])
        # assert False
        return torch.mean((y_pred - y_true) ** 2)


# (negative) Log-likelihood of the Poisson distribution
class MultiDatasetLoss(tnn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true, is_count_data):

        a, b = y_pred[is_count_data == 1], y_pred[is_count_data == 0]
        # print(a.shape, b.shape)

        # add poisson loss
        poisson_loss = None
        if a.shape[0] > 0:
            loss = a - y_true[is_count_data == 1] * torch.log(a)
            # loss = torch.abs(loss)
            poisson_loss = torch.sum(loss)

        # m = torch.nn.Sigmoid()
        bce = torch.nn.BCELoss(reduction="sum")
        bce_loss = None
        if b.shape[0] > 0:
            # print(b)
            bce_loss = bce(b / (1 + b), y_true[is_count_data == 0])

        # print(poisson_loss, bce_loss)
        if a.shape[0] != 0 and b.shape[0] != 0:
            return (poisson_loss + bce_loss) / len(y_true)
        elif a.shape[0] != 0:
            return poisson_loss / len(y_true)
        elif b.shape[0] != 0:
            return bce_loss / len(y_true)
        assert False  # problem with the input data


# Custom loss function
class CustomLoss(tnn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, rounds, avoid_zero=True):
        if avoid_zero:
            rounds = rounds + 0.1
        f = inputs * rounds[:, 0]
        return -torch.sum(rounds[:, 1] * torch.log(f + 0.0001) - f)
