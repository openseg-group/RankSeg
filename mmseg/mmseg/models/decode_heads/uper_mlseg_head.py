# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from .psp_head import PPM
from ..builder import build_head, build_loss
from mmcv.runner import force_fp32
from ..losses.accuracy import accuracy, img_recall, pixel_recall, img_accuracy

@HEADS.register_module()
class UPerMLSegHead(BaseDecodeHead):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """
    def __init__(self, mlseg, pool_scales=(1, 2, 3, 6), use_separable=False, **kwargs):
        super(UPerMLSegHead, self).__init__(
            input_transform='multiple_select', **kwargs)

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

        # PSP Module
        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1],
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)
        conv_builder = DepthwiseSeparableConvModule if use_separable else ConvModule
        self.bottleneck = conv_builder(
            self.in_channels[-1] + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = ConvModule(
                in_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            fpn_conv = conv_builder(
                self.channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = conv_builder(
            len(self.in_channels) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output

    def forward(self, inputs):
        """Forward function."""
        inputs = self._transform_inputs(inputs)

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.psp_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += resize(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)

        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        fpn_outs = torch.cat(fpn_outs, dim=1)
        output = self.fpn_bottleneck(fpn_outs)

        # Multi-Label classification, using lowest-resolution feature
        inputs_ml = inputs[-1]
        if self.img_cls_head.share_embedding:
            img_pred = self.img_cls_head((inputs_ml, self.cls_emb.squeeze(0)))
        else:
            img_pred = self.img_cls_head(inputs_ml)

        topk_cls = self.topk_cls
        topk_index = torch.argsort(img_pred, dim=1, descending=True)[
            :, : topk_cls
        ].squeeze(-1).squeeze(-1)

        output = self.cls_seg((img_pred, topk_index, output))
            
        return output

    def cls_seg(self, feat):
        """Classify each pixel."""
        img_pred, topk_index, feat = feat
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