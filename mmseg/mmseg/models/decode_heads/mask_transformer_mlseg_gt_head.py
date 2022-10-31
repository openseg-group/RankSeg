import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.init import trunc_normal_

from ..builder import HEADS, build_loss
from .decode_head import BaseDecodeHead
from mmcv.cnn import ConvModule
from mmcv.runner import force_fp32
from mmseg.ops import resize
from ..losses.accuracy import accuracy, img_recall, pixel_recall, img_accuracy


def init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


@HEADS.register_module()
class MaskTransformerMLSegGTHead(BaseDecodeHead):

    def __init__(
        self,
        n_cls,
        patch_size,
        d_encoder,
        n_layers,
        n_heads,
        d_model,
        d_ff,
        drop_path_rate,
        dropout,
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
        self.patch_size = patch_size
        self.n_layers = n_layers
        self.n_cls = n_cls
        self.d_model = d_model
        self.d_ff = d_ff
        self.scale = d_model**-0.5

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList([
            Block(d_model, n_heads, d_ff, dropout, dpr[i])
            for i in range(n_layers)
        ])

        self.cls_emb = nn.Parameter(torch.randn(1, n_cls, d_model))
        self.proj_dec = nn.Linear(d_encoder, d_model)

        self.proj_patch = nn.Parameter(self.scale *
                                       torch.randn(d_model, d_model))
        self.proj_classes = nn.Parameter(self.scale *
                                         torch.randn(d_model, d_model))

        self.decoder_norm = nn.LayerNorm(d_model)
        self.mask_norm = SelectedLayerNorm(n_cls)

    def init_weights(self):
        self.apply(init_weights)
        trunc_normal_(self.cls_emb, std=0.02)

    @staticmethod
    def _get_batch_hist_vector(target, nclass):
        # target is a 3D Variable BxHxW, output is 2D BxnClass
        batch, H, W = target.shape
        tvect = target.new_zeros((batch, nclass), dtype=torch.int64)
        for i in range(batch):
            hist = torch.histc(
                target[i].data.float(), bins=nclass, min=0, max=nclass - 1)
            tvect[i] = hist
        return tvect

    def forward(self, x):
        x, targets = x
        targets = targets.squeeze(1)
        x = self._transform_inputs(x)
        B, C, H, W = x.size()

        assert B == 1
        gt = self._get_batch_hist_vector(targets, self.num_classes) > 0
        gt_num = gt.sum(dim=1)
        gt_index = torch.nonzero(gt[0]).squeeze(-1).unsqueeze(0)

        if gt_num == 0:
            gt_index = gt_index.new_zeros((B, 1), dtype=torch.int64)
            gt_num = 1

        x = x.view(B, C, -1).permute(0, 2, 1)
        x = self.proj_dec(x)
        pixel_feat = x

        # Select gt classes feature for every image
        cls_topk_feat = self.cls_emb.view(self.n_cls,
                                          -1)[gt_index.view(-1)].view(
                                              B, gt_num, -1)
        patch_and_cls_feat = torch.cat((pixel_feat, cls_topk_feat), dim=1)

        # (nlayer) layers of self attention
        for blk in self.blocks:
            patch_and_cls_feat = blk(patch_and_cls_feat)
        patch_and_cls_feat = self.decoder_norm(patch_and_cls_feat)

        patches, cls_seg_feat = (
            patch_and_cls_feat[:, :-gt_num],
            patch_and_cls_feat[:, -gt_num:],
        )

        patches = patches @ self.proj_patch
        cls_seg_feat = cls_seg_feat @ self.proj_classes
        patches = patches / patches.norm(dim=-1, keepdim=True)
        cls_seg_feat = cls_seg_feat / cls_seg_feat.norm(dim=-1, keepdim=True)

        masks = patches @ cls_seg_feat.transpose(1, 2)
        masks = self.mask_norm(masks, gt_index)

        B, HW, N = masks.size()
        masks = masks.view(B, H, W, N).permute(0, 3, 1, 2)
        # return pixel prediction in topK classes during training
        # gt_labels are rearranged in self.losses() with topk_index
        if self.training:
            return gt_index, masks

        # return pixel prediction in all classes during inference
        # set the logits of classes not in topK classes to -100
        else:
            gt_index_tmp = gt_index.unsqueeze(-1).unsqueeze(-1).repeat(
                1, 1, H, W)
            full_masks = masks.new_ones((B, self.n_cls, H, W)) * -100
            full_masks.scatter_(1, gt_index_tmp, masks)
            return full_masks

    @staticmethod
    def _get_batch_hist_vector(target, nclass):
        # target is a 3D Variable BxHxW, output is 2D BxnClass
        batch, H, W = target.shape
        tvect = target.new_zeros((batch, nclass), dtype=torch.int64)
        for i in range(batch):
            hist = torch.histc(
                target[i].data.float(), bins=nclass, min=0, max=nclass - 1)
            tvect[i] = hist
        return tvect

    def losses(self, outputs, seg_label):
        """Compute segmentation loss."""
        # generate histogram of gt_labels for each image
        hist = self._get_batch_hist_vector(
            seg_label.squeeze(1), self.num_classes)
        topk_index, pixel_pred = outputs

        loss = dict()
        # Image Multi-label Loss

        # Seg Loss
        seg_logit = resize(
            input=pixel_pred,
            size=seg_label.shape[2:],
            mode="bilinear",
            align_corners=self.align_corners,
        )
        # rearrange gt_labels to topK index, pixel not predicted in topK classes will be set to ignore
        topk_vector = (
            topk_index.unsqueeze(-1).unsqueeze(-1) == seg_label.repeat(
                1, topk_index.shape[-1], 1, 1)).long()
        topk_max_prob, topk_label = torch.max(topk_vector, dim=1)
        topk_label[topk_max_prob == 0] = self.ignore_index
        topk_label[seg_label.squeeze(1) ==
                   self.ignore_index] = self.ignore_index

        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, topk_label)
        else:
            seg_weight = None

        topk_label = topk_label.squeeze(1)
        loss["loss_seg"] = self.loss_decode(
            seg_logit,
            topk_label,
            weight=seg_weight,
            ignore_index=self.ignore_index)

        loss["acc_seg"] = accuracy(seg_logit, topk_label)
        loss["acc_pix"] = pixel_recall(hist, topk_index)
        loss["acc_img"] = img_recall(hist, topk_index)
        loss["acc_img_accuracy"] = img_accuracy(hist, topk_index)
        return loss


class FeedForward(nn.Module):

    def __init__(self, dim, hidden_dim, dropout, out_dim=None):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        if out_dim is None:
            out_dim = dim
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(dropout)

    @property
    def unwrapped(self):
        return self

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):

    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads = heads
        head_dim = dim // heads
        self.scale = head_dim**-0.5
        self.attn = None

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    @property
    def unwrapped(self):
        return self

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = (
            self.qkv(x).reshape(B, N, 3, self.heads,
                                C // self.heads).permute(2, 0, 3, 1, 4))
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn


class Block(nn.Module):

    def __init__(self, dim, heads, mlp_dim, dropout, drop_path):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, mask=None, return_attention=False):
        y, attn = self.attn(self.norm1(x), mask)
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DecoderLinear(nn.Module):

    def __init__(self, n_cls, patch_size, d_encoder):
        super().__init__()

        self.d_encoder = d_encoder
        self.patch_size = patch_size
        self.n_cls = n_cls

        self.head = nn.Linear(self.d_encoder, n_cls)
        self.apply(init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def forward(self, x, im_size):
        H, W = im_size
        GS = H // self.patch_size
        x = self.head(x)
        B, HW, C = x.size()
        x = x.view(B, GS, HW // GS, C).permute(0, 3, 1, 2)
        # x = rearrange(x, "b (h w) c -> b c h w", h=GS)

        return x


class SelectedLayerNorm(nn.Module):
    """Reproduced LayerNorm
    https://github.com/pytorch/pytorch/blob/e2eb97dd7682d2810071ce78b76543acc1584a9c/torch/onnx/symbolic_opset9.py#L1311
    """

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x, index):
        B, k = index.shape
        assert B == 1
        index = index.new_tensor([i for i in range(k)])
        mean = x.mean(-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        gamma = self.gamma[index.view(-1)].view(B, k).unsqueeze(1)
        beta = self.beta[index.view(-1)].view(B, k).unsqueeze(1)
        if k == 1:
            return gamma * (x - mean) + beta.repeat(1, x.size(1), 1)
        else:
            std = (var + self.eps).sqrt()
            return gamma * (x - mean) / std + beta
