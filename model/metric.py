import torch
from sklearn.metrics import precision_recall_fscore_support


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def precision(output, target):
    precision, _, _, _ = precision_recall_fscore_support(
        target, output, average='macro',
    )
    return precision


def recall(output, target):
    _, recall, _, _ = precision_recall_fscore_support(
        target, output, average='macro',
    )
    return recall


def f1(output, target):
    _, _, f1, _ = precision_recall_fscore_support(
        target, output, average='macro',
    )
    return f1
