import torch
from sklearn import metrics
import torchmetrics.functional as MF

def accuracy(output, target):
    with torch.no_grad():
        if len(output.size()) > 1 and output.size(1) > 1:  # multi-class
            pred = torch.argmax(output, dim=1)
        else:
            pred = torch.as_tensor(
                (output - 0.5) > 0, dtype=torch.int32).squeeze()
        assert pred.shape[0] == len(target)
        # correct = 0
        # correct += torch.sum(pred == target).item()
    return MF.accuracy(pred, target)

    # return correct / len(target)

def recall(output, target):
    with torch.no_grad():
        if len(output.size()) > 1 and output.size(1) > 1:  # multi-class
            pred = torch.argmax(output, dim=1)
        else:
            pred = torch.as_tensor(
                (output - 0.5) > 0, dtype=torch.int32).squeeze()
        assert pred.shape[0] == len(target)

    # return metrics.recall_score(target.cpu().numpy(), pred.cpu().numpy())
    return MF.recall(pred, target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


if __name__ == '__main__':
    a = torch.Tensor([0, 0.5, 0.7, 0.9])
    b = torch.Tensor([0, 1, 0, 1])
    print(accuracy(a, b))
