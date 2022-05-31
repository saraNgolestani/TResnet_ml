from time import time
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score, precision_score


class Statistics(object):
    """Accumulator for loss statistics, inspired by ONMT.
    Keeps track of the following metrics:
    * F1
    * Precision
    * Recall
    * Accuracy
    """

    def __init__(self):
        self.loss_sum = 0
        self.examples = 0
        self._f1 = 0
        self._precision = 0
        self.start_time = time()

    def update(self, loss=0, precision=None, f1=0):
        if precision is None:
            precision = []
        examples = len(precision)
        self.loss_sum += loss * examples
        self.examples += examples
        self._precision += precision
        self._f1 += f1

    def loss(self):
        return self.loss_sum / self.examples

    def f1(self):
        return self._f1/ self.examples

    def precision(self):
        return self._precision.sum() / self.examples

    def examples_per_sec(self):
        return self.examples / (time() - self.start_time)


def calc_precision(preds, labels):
    best_th = 0.5
    pre = 0.0

    for th in np.arange(0.0, 1.0, 0.05):
        pred = []
        for item in preds:
            item_preds = [1 if p > th else 0 for p in item]
            pred.append(item_preds)
        new_pre = precision_score(labels, pred,  average='micro')
        if new_pre > pre:
            pre = new_pre
            best_th = th

    return pre, best_th


def compute_scores(output, target):

    precision = precision_score(target, output, average=None)
    f1 = f1_score(target, output, average='micro')

    return precision, f1


def average_precision(output, target):
    epsilon = 1e-8

    # sort examples
    indices = output.argsort()[::-1]
    # Computes prec@i
    total_count_ = np.cumsum(np.ones((len(output), 1)))

    target_ = target[indices]
    ind = target_ == 1
    pos_count_ = np.cumsum(ind)
    total = pos_count_[-1]
    pos_count_[np.logical_not(ind)] = 0
    pp = pos_count_ / total_count_
    precision_at_i_ = np.sum(pp)
    precision_at_i = precision_at_i_ / (total + epsilon)

    return precision_at_i


def mAP(targs, preds):
    """Returns the model's average precision for each class
    Return:
        ap (FloatTensor): 1xK tensor, with avg precision for each class k
    """

    if np.size(preds) == 0:
        return 0
    ap = np.zeros((preds.shape[1]))
    # compute average precision for each class
    for k in range(preds.shape[1]):
        # sort scores
        scores = preds[:, k]
        targets = targs[:, k]
        # compute average precision
        ap[k] = average_precision(scores, targets)
    return 100 * ap.mean()

