# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch


def accuracy(pred, target, topk=1, thresh=None):
    """Calculate accuracy according to the prediction and target.

    Args:
        pred (torch.Tensor): The model prediction, shape (N, num_class, ...)
        target (torch.Tensor): The target of each prediction, shape (N, , ...)
        topk (int | tuple[int], optional): If the predictions in ``topk``
            matches the target, the predictions will be regarded as
            correct ones. Defaults to 1.
        thresh (float, optional): If not None, predictions with scores under
            this threshold are considered incorrect. Default to None.

    Returns:
        float | tuple[float]: If the input ``topk`` is a single integer,
            the function will return a single float as accuracy. If
            ``topk`` is a tuple containing multiple integers, the
            function will return a tuple containing accuracies of
            each ``topk`` number.
    """
    assert isinstance(topk, (int, tuple))
    if isinstance(topk, int):
        topk = (topk, )
        return_single = True
    else:
        return_single = False

    maxk = max(topk)
    if pred.size(0) == 0:
        accu = [pred.new_tensor(0.) for i in range(len(topk))]
        return accu[0] if return_single else accu
    assert pred.ndim == target.ndim + 1
    assert pred.size(0) == target.size(0)
    assert maxk <= pred.size(1), \
        f'maxk {maxk} exceeds pred dimension {pred.size(1)}'
    pred_value, pred_label = pred.topk(maxk, dim=1)
    # transpose to shape (maxk, N, ...)
    pred_label = pred_label.transpose(0, 1)
    correct = pred_label.eq(target.unsqueeze(0).expand_as(pred_label))
    if thresh is not None:
        # Only prediction values larger than thresh are counted as correct
        correct = correct & (pred_value > thresh).t()
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / target.numel()))
    return res[0] if return_single else res


def img_recall(hist, topk_label, mask=None):
    gt_label = (hist > 0).float()
    if mask is not None:
        gt_label[torch.logical_not(mask)] = 0
    recall = torch.gather(gt_label, 1, topk_label).sum(dim=1) / (hist > 0).float().sum(dim=1)
    if (hist > 0).sum() == 0:
        return recall.new_tensor(1)
    return recall[recall<float('inf')].mean()

def img_accuracy(hist, topk_label, mask=None):
    gt_label = (hist > 0).float()
    if mask is not None:
        gt_label[torch.logical_not(mask)] = 0
        recall = torch.gather(gt_label, 1, topk_label).sum(dim=1) / torch.gather(mask, 1, topk_label.repeat(mask.size(0), 1)).sum(dim=1)
    else:
        recall = torch.gather(gt_label, 1, topk_label).sum(dim=1) / topk_label.shape[1]
    return recall[recall<float('inf')].mean()

def pixel_recall(hist, topk_label, mask=None):
    if mask is not None:
        hist[torch.logical_not(mask)] = 0
    recall = torch.gather(hist, 1, topk_label).sum(dim=1) / hist.sum(dim=1)
    if (hist > 0).sum() == 0:
        return recall.new_tensor(1)
    return recall[recall<float('inf')].mean()


class Accuracy(nn.Module):
    """Accuracy calculation module."""

    def __init__(self, topk=(1, ), thresh=None):
        """Module to calculate the accuracy.

        Args:
            topk (tuple, optional): The criterion used to calculate the
                accuracy. Defaults to (1,).
            thresh (float, optional): If not None, predictions with scores
                under this threshold are considered incorrect. Default to None.
        """
        super().__init__()
        self.topk = topk
        self.thresh = thresh

    def forward(self, pred, target):
        """Forward function to calculate accuracy.

        Args:
            pred (torch.Tensor): Prediction of models.
            target (torch.Tensor): Target for each prediction.

        Returns:
            tuple[float]: The accuracies under different topk criterions.
        """
        return accuracy(pred, target, self.topk, self.thresh)

def get_average_precision(hist, topk_label):
    gt_label = (hist > 0)
    k_index = topk_label.new_tensor([[i for i in range(topk_label.shape[1])]] * topk_label.shape[0]).float() # B k
    inverted_label_index = float('inf') * topk_label.new_ones((hist.shape[0], hist.shape[1]))
    inverted_label_index.scatter_(1, topk_label, k_index) # B o, 'inf' for unpredicted classes, prediction rank for predicted classes
    ap_list = hist.new_zeros(hist.shape[0], dtype=torch.float) # calculate AP for each image in the batch
    for i in range(hist.shape[0]):
        V, _ = inverted_label_index[i, gt_label[i]].sort()  # for every groundtruth class, get its prediction rank (could be 'inf')
        I = V.new_tensor([i for i in range(1, len(V)+1)])   
        ap_list[i] = (I/(V+1)).sum()/gt_label[i].sum()
    return ap_list[ap_list<float('inf')].mean()