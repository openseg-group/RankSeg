import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.init import trunc_normal_

from ..builder import HEADS, build_loss
from .decode_head import BaseDecodeHead
from .query2label_head import Query2LabelHead as _Query2LabelHead
from mmcv.cnn import ConvModule
from mmcv.runner import force_fp32
from mmseg.ops import resize
from ..losses.accuracy import get_average_precision


@HEADS.register_module()
class Query2LabelHead(BaseDecodeHead):
    def __init__(
        self,
        n_cls,
        d_encoder,
        num_decoder_layers=2,
        loss_img=dict(type='AsymmetricLoss', gamma_neg=0, gamma_pos=0),
        **kwargs,
    ):
        # in_channels & channels are dummy arguments to satisfy signature of
        # parent's __init__
        super().__init__(
            in_channels=d_encoder,
            channels=1,
            num_classes=n_cls,
            **kwargs,
        )
        del self.conv_seg
        self.d_encoder = d_encoder
        self.n_cls = n_cls

        self.loss_img = build_loss(loss_img)

        self.img_cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((16, 16)),
            _Query2LabelHead(input_dim=d_encoder, hidden_dim=d_encoder, size=16, num_class=n_cls, num_encoder_layers=0, num_decoder_layers=num_decoder_layers)
            )
        # self.k = 0
        # self.x = torch.zeros(1955, 171)

    def init_weights(self):
        pass

    @staticmethod
    def _get_batch_hist_vector(target, nclass):
        # target is a 3D Variable BxHxW, output is 2D BxnClass
        batch, H, W = target.shape
        tvect = target.new_zeros((batch, nclass))
        for i in range(batch):
            hist = torch.histc(
                target[i].data.float(), bins=nclass, min=0, max=nclass - 1
            )
            tvect[i] = hist
        return tvect

    def forward(self, x):
        x = self._transform_inputs(x)
        B, C, H, W = x.size()

        img_pred = self.img_cls_head(x)
        # self.x[self.k] = img_pred.squeeze()
        # self.y[self.k] = topk_gt.squeeze()
        # self.k += 1
        # if self.k == 1955:
        #     torch.save(self.x, "tempo/img_pred_coco_asl_q2l_new.pt")
        #     torch.save(self.y, "tempo/groundtruth_coco_asl40_q2l.pt")

        if self.training:
            return img_pred

        else:
            full_masks = x.new_ones((B, self.n_cls, H, W))
            return full_masks
    
    
    @force_fp32(apply_to=('outputs', ))
    def losses(self, outputs, seg_label):
        """Compute segmentation loss."""
        hist = self._get_batch_hist_vector(seg_label.squeeze(1), self.num_classes)
        img_pred = outputs
        topk_index = torch.argsort(img_pred, dim=1, descending=True).squeeze(-1).squeeze(-1)
        loss = dict()
        loss['loss_img'] = self.loss_img(
            img_pred.squeeze(-1).squeeze(-1),
            (hist > 0).float())
        loss['acc_ap'] = get_average_precision(hist, topk_index)
        return loss

