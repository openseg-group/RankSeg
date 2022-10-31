# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
import numpy as np
from mmseg.ops import resize
from ..builder import HEADS, build_head, build_loss
from .decode_head import BaseDecodeHead
from mmcv.runner import force_fp32
from ..losses.accuracy import accuracy, img_recall, pixel_recall, img_accuracy


class ASPPModule(nn.ModuleList):
    """Atrous Spatial Pyramid Pooling (ASPP) Module.

    Args:
        dilations (tuple[int]): Dilation rate of each layer.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
    """

    def __init__(self, dilations, in_channels, channels, conv_cfg, norm_cfg,
                 act_cfg):
        super(ASPPModule, self).__init__()
        self.dilations = dilations
        self.in_channels = in_channels
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        for dilation in dilations:
            self.append(
                ConvModule(
                    self.in_channels,
                    self.channels,
                    1 if dilation == 1 else 3,
                    dilation=dilation,
                    padding=0 if dilation == 1 else dilation,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

    def forward(self, x):
        """Forward function."""
        aspp_outs = []
        for aspp_module in self:
            aspp_outs.append(aspp_module(x))

        return aspp_outs


@HEADS.register_module()
class ASPPMLSegHead(BaseDecodeHead):
    """Rethinking Atrous Convolution for Semantic Image Segmentation.

    This head is the implementation of `DeepLabV3
    <https://arxiv.org/abs/1706.05587>`_.

    Args:
        dilations (tuple[int]): Dilation rates for ASPP module.
            Default: (1, 6, 12, 18).
    """

    def __init__(self, mlseg, dilations=(1, 6, 12, 18), **kwargs):
        super(ASPPMLSegHead, self).__init__(**kwargs)

        # MLSeg
        self.img_cls_head = build_head(mlseg['head'])
        self.img_cls_loss = build_loss(mlseg['loss'])
        self.topk_cls = mlseg['topk_cls']
        self.img_loss_weight = mlseg['img_loss_weight']

        self.cls_emb = nn.Parameter(torch.randn(1, self.num_classes, mlseg['head']['hidden_dim']))
        self.cls_bias_emb = nn.Parameter(torch.randn(1, self.num_classes))

        from torch.nn.init import trunc_normal_
        trunc_normal_(self.cls_emb, std=0.02)
        trunc_normal_(self.cls_bias_emb, std=0.02)

        self.norm = nn.LayerNorm(self.topk_cls)
        del self.conv_seg

        assert isinstance(dilations, (list, tuple))
        self.dilations = dilations
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvModule(
                self.in_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        self.aspp_modules = ASPPModule(
            dilations,
            self.in_channels,
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.bottleneck = ConvModule(
            (len(dilations) + 1) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)

        aspp_outs = [
            resize(
                self.image_pool(x),
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        ]
        aspp_outs.extend(self.aspp_modules(x))
        aspp_outs = torch.cat(aspp_outs, dim=1)
        output = self.bottleneck(aspp_outs)
        
        # Multi-Label classification, using lowest-resolution feature
        inputs_ml = x
        if self.img_cls_head.share_embedding:
            img_pred = self.img_cls_head((inputs_ml, self.cls_emb.squeeze(0)))
        else:
            img_pred = self.img_cls_head(inputs_ml)

        topk_cls = self.topk_cls
        topk_index = torch.argsort(img_pred, dim=1, descending=True)[
            :, : topk_cls
        ].squeeze(-1).squeeze(-1)


        output = self.cls_seg(img_pred, topk_index, output)
            
        return output

    def cls_seg(self, img_pred, topk_index, feat):
        """Classify each pixel."""
        B, C, H, W = feat.shape

        if self.dropout is not None:
            feat = self.dropout(feat)

        # Select topK classes feature for every image
        cls_emb = self.cls_emb
        cls_emb = cls_emb.reshape(self.num_classes, -1)[topk_index.reshape(-1)].reshape(B, topk_index.shape[1], -1)
        cls_bias_emb = self.cls_bias_emb
        cls_bias_emb = cls_bias_emb.reshape(self.num_classes, -1)[topk_index.reshape(-1)].reshape(B, topk_index.shape[1])

        output = torch.einsum('bchw, bkc -> bkhw', feat, cls_emb) + cls_bias_emb.unsqueeze(-1).unsqueeze(-1)
        output = self.norm(output.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # return pixel prediction in topK classes during training
        # gt_labels are rearranged in self.losses() with topk_index
        if self.training:
            return img_pred, topk_index, output

        # return pixel prediction in all classes during inference
        # set the logits of classes not in topK classes to -100    
        else:
            topk_index_tmp = (
                topk_index.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H, W)
            )
            full_output = output.new_ones((B, self.num_classes, H, W)) * -100
            full_output.scatter_(1, topk_index_tmp, output)
            return full_output

    @force_fp32(apply_to=('seg_logit', ))
    def losses(self, seg_logit, seg_label):
        """Compute segmentation loss."""
        # generate histogram of gt_labels for each image
        hist = self._get_batch_hist_vector(seg_label.squeeze(1), self.num_classes)
        img_pred, topk_index, seg_logit = seg_logit

        loss = dict()
        # Image Multi-label Loss
        loss['loss_img'] = self.img_cls_loss(
            img_pred.squeeze(-1).squeeze(-1),
            (hist > 0).float()) * self.img_loss_weight
        
        # Seg Loss
        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        # rearrange gt_labels to topK index, pixel not predicted in topK classes will be set to ignore    
        topk_vector = (
            topk_index.unsqueeze(-1).unsqueeze(-1)
            == seg_label.repeat(1, self.topk_cls, 1, 1)
        ).long()
        topk_max_prob, topk_label = torch.max(topk_vector, dim=1)
        topk_label[topk_max_prob == 0] = self.ignore_index
        topk_label[seg_label.squeeze(1) == self.ignore_index] = self.ignore_index
        seg_label = topk_label.unsqueeze(1)
        
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)

        loss['loss_seg'] = self.loss_decode(
            seg_logit,
            seg_label,
            weight=seg_weight,
            ignore_index=self.ignore_index)
        loss['acc_pix'] = pixel_recall(hist, topk_index)
        loss['acc_img'] = img_recall(hist, topk_index)
        loss['acc_img_accuracy'] = img_accuracy(hist, topk_index)
        loss['acc_seg'] = accuracy(seg_logit, seg_label)
        return loss