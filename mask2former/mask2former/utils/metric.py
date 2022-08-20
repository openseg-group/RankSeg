import torch
import torch.nn as nn


def img_recall(hist, topk_label, mask=None):
    gt_label = (hist > 0).float()
    if mask is not None:
        gt_label[torch.logical_not(mask)] = 0
    recall = torch.gather(gt_label, 1, topk_label).sum(dim=1) / (hist > 0).float().sum(dim=1)
    if (hist > 0).sum() == 0:
        return recall.new_tensor(1)
    return recall[recall<float('inf')].mean()

def img_precision(hist, topk_label, mask=None):
    gt_label = (hist > 0).float()
    if mask is not None:
        gt_label[torch.logical_not(mask)] = 0
        recall = torch.gather(gt_label, 1, topk_label).sum(dim=1) / torch.gather(mask, 1, topk_label.repeat(mask.size(0), 1)).sum(dim=1)
    else:
        recall = torch.gather(gt_label, 1, topk_label).sum(dim=1) / topk_label.shape[1]
    return recall[recall<float('inf')].mean()

def img_average_precision(hist, img_pred):
    topk_label =  torch.argsort(img_pred.squeeze(-1).squeeze(-1), dim=1, descending=True)
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