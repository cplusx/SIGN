import torch
import numpy as np

def has_unseen_index(label, index_list):
    for idx in index_list:
        if idx in label:
            return True
    return False

def mean_iou_of_index(pred, label, index):
    pred_ = pred == index
    label_ = label == index
    intersection = (pred_ * label_)
    union = (pred_ + label_)
    assert intersection.dtype == 'bool' and union.dtype == 'bool'
    iou = np.sum(intersection).astype(np.float) / np.sum(union).astype(np.float)
    return iou

def acc_of_index(pred, label, index):
    pred_ = pred == index
    label_ = label == index
    intersection = (pred_ * label_)
    assert intersection.dtype == 'bool'
    acc = np.sum(intersection).astype(np.float) / np.sum(label_).astype(np.float)
    return acc

def harmonic_mean(seen, unseen):
    assert len(unseen) > 0
    unseen = torch.mean(torch.tensor(unseen))
    if len(seen) > 0:
        seen = torch.mean(torch.tensor(seen))
        return seen, unseen, 2 * seen * unseen / (seen + unseen)
    else:
        return torch.tensor(np.nan), unseen, unseen