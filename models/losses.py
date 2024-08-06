import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


def clf_loss(y_pred, y_true, mask=None, eps=1e-10,
             label_weight=None, pos_weight=None):
    if mask is not None:
        loss_fct = nn.BCEWithLogitsLoss(reduction='none')
        loss = loss_fct(y_pred, y_true)
        loss = torch.sum(loss * mask) / (torch.sum(mask) + eps)
    else:
        if label_weight is not None:
            label_weight = label_weight.to(y_true.device)
        if pos_weight is not None:
            pos_weight = pos_weight.to(y_true.device)
        loss_fct = nn.BCEWithLogitsLoss(label_weight, pos_weight=pos_weight)
        loss = loss_fct(y_pred, y_true)

    return loss

