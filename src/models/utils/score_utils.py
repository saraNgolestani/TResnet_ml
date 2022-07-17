from time import time
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
        self.best_th = 0.45
        self._precision = 0
        self.start_time = time()

    def update(self, loss=0, precision=None, best_th=-1):
        if precision is None:
            precision = []
        examples = len(precision)
        self.loss_sum += loss * examples
        self.examples += examples
        try:
            self._precision += precision.sum()
        except:
            self._precision += sum(precision)

        if best_th != -1:
            self.best_th = best_th

    def loss(self):
        return self.loss_sum / self.examples

    def get_best_th(self):
        return self.best_th

    def precision(self):
        return self._precision / self.examples

    def examples_per_sec(self):
        return self.examples / (time() - self.start_time)


def compute_scores_and_th(preds, labels, th=None):
    if th is not None:
        output = (preds >= th)
        return precision_score(output, labels, average=None, zero_division=0), th

    best_th = 0.45
    best_pre_sum = 0.0
    pre = None
    for th in np.arange(0.15, 0.85, 0.05):
        pred = []
        for item in preds:
            item_preds = [1 if p > th else 0 for p in item]
            pred.append(item_preds)
        new_pre = precision_score(labels, pred,  average=None, zero_division=0)
        if new_pre.sum() > best_pre_sum:
            pre = new_pre
            best_pre_sum = new_pre.sum()
            best_th = th

    return pre, best_th


def compute_scores(output, target):

    precision = precision_score(target, output, average=None, zero_division=0)
    # f1 = f1_score(target, output, average='micro')

    return precision


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


def mAP(output, target, th):
    """Returns the model's average precision for each class
    Return:
        ap (FloatTensor): 1xK tensor, with avg precision for each class k
    """

    ap = np.zeros((output.shape[1]))
    #[[0, ...79],
    # [],
    # ...
    # [],
    # []]
    # compute average precision for each class
    for k in range(target.shape[1]):
        # sort scores
        pred_y = (output[:, k] > th)
        true_y = target[:, k]
        # compute average precision
        ap[k] = precision_score(true_y, pred_y, average=None, zero_division=0)
    return ap[k]

