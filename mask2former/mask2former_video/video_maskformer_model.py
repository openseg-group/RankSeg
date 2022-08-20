# Copyright (c) Facebook, Inc. and its affiliates.
import pdb
import logging
import math
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks

from mask2former.modeling.multilabel_head import MultiLabelHead
from .modeling.criterion import VideoSetCriterion, AsymmetricLoss
from .modeling.matcher import VideoHungarianMatcher
from .utils.memory import retry_if_cuda_oom

logger = logging.getLogger(__name__)


@META_ARCH_REGISTRY.register()
class VideoMaskFormer(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """
    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # video
        num_frames,
        # mlseg parameters
        ml_use_gt: bool,
        ml_head: nn.Module,
        ml_loss: nn.Module,
        ml_loss_weight: float,
        ml_topk: int,
        ml_norm: nn.Module,
        semantic_on: bool,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
        """
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.semantic_on = semantic_on
        self.register_buffer("pixel_mean",
                             torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std",
                             torch.Tensor(pixel_std).view(-1, 1, 1), False)

        self.num_frames = num_frames
        # mlseg parameters
        self.ml_use_gt = ml_use_gt
        self.ml_head = ml_head
        self.ml_loss = ml_loss
        self.ml_loss_weight = ml_loss_weight
        self.ml_topk = ml_topk
        self.ml_norm = ml_norm

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

        # building criterion
        matcher = VideoHungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        weight_dict = {
            "loss_ce": class_weight,
            "loss_mask": mask_weight,
            "loss_dice": dice_weight,
        }

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update(
                    {k + f"_{i}": v
                     for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]

        if cfg.MODEL.RANKSEG.FLAG:
            ml_head = MultiLabelHead(
                sem_seg_head.num_classes, cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM,
                cfg.MODEL.RANKSEG.HIDDEN_DIM, cfg.MODEL.RANKSEG.HIDDEN_DIM // 64,
                cfg.MODEL.RANKSEG.HIDDEN_DIM * 4, 0.1,
                cfg.MODEL.RANKSEG.SHARE_EMBEDDING)
            
            ml_loss = AsymmetricLoss(2, 0)
            ml_loss_weight = cfg.MODEL.RANKSEG.LOSS_WEIGHT
            ml_norm = nn.LayerNorm(cfg.MODEL.RANKSEG.TOPK + 1)
        else:
            ml_head = None
            ml_loss = None
            ml_loss_weight = None
            ml_norm = None

        criterion = VideoSetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.
            IMPORTANCE_SAMPLE_RATIO,
        )

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold":
            cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": True,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # inference
            "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
            # video
            "num_frames": cfg.INPUT.SAMPLING_FRAME_NUM,
            "ml_use_gt": cfg.MODEL.RANKSEG.USE_GT,
            "ml_head": ml_head,
            "ml_loss": ml_loss,
            "ml_norm": ml_norm,
            "ml_loss_weight": ml_loss_weight,
            "ml_topk": cfg.MODEL.RANKSEG.TOPK,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        images = []
        for video in batched_inputs:
            for frame in video["image"]:
                images.append(frame.to(self.device))
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]

        images = ImageList.from_tensors(images, self.size_divisibility)
        features = self.backbone(images.tensor)

        outputs = self.sem_seg_head(features)


        B = len(batched_inputs)

        if self.ml_use_gt:
            targets = self.prepare_targets(batched_inputs, images)
            gt_classes = [torch.unique(target['labels']) for target in targets]

            not_gt_mask = torch.cat(
                (
                    images.tensor.new_ones(
                        (B, self.sem_seg_head.num_classes), dtype=torch.bool),
                    images.tensor.new_zeros((B, 1), dtype=torch.bool),
                ),
                dim=1,
            )
            for b in range(B):
                not_gt_mask[b].scatter_(0, gt_classes[b], 0)

        # predict multi-label classification scores
        if self.ml_head:
            image_patches_feature = outputs["query_feature"][2].permute(1, 2, 0)
            if (hasattr(self.ml_head, "share_embedding")
                    and self.ml_head.share_embedding):
                img_pred = self.ml_head((
                    image_patches_feature,
                    self.sem_seg_head.predictor.class_embed.weight[:-1],
                ))
            else:
                img_pred = self.ml_head(image_patches_feature)

            topk_index = (torch.argsort(
                img_pred, dim=1,
                descending=True)[:, :self.ml_topk].squeeze(-1).squeeze(-1))
            # Add the last "ignore" prediction
            topk_index = torch.cat(
                (topk_index,
                 topk_index.new_ones(B, 1) * self.sem_seg_head.num_classes),
                dim=1,
            )
        
        # Mask with GT
        if self.ml_use_gt:
            outputs["pred_logits"][not_gt_mask.unsqueeze(1).repeat(
                1, self.num_queries, 1)] = -100

        # Mask(Gather) with MLSeg
        if self.ml_head:
            outputs["pred_logits"] = torch.gather(
                outputs["pred_logits"],
                2,
                topk_index.unsqueeze(1).repeat(1, self.num_queries, 1),
            )
            for aux_output in outputs["aux_outputs"]:
                aux_output["pred_logits"] = torch.gather(
                    aux_output["pred_logits"],
                    2,
                    topk_index.unsqueeze(1).repeat(1, self.num_queries, 1),
                )

            # apply layernorm on the predictions
            outputs["pred_logits"] = self.ml_norm(outputs["pred_logits"])
            for aux_output in outputs["aux_outputs"]:
                aux_output["pred_logits"] = self.ml_norm(
                    aux_output["pred_logits"])

        if self.training:
            # mask classification target
            targets = self.prepare_targets(batched_inputs, images)
            
                    
            # bipartite matching-based loss
            if self.ml_head:
                losses = self.criterion(outputs, targets, topk_index)
            else:
                losses = self.criterion(outputs, targets)
                
            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)

            # Genarate GT Mask
            gt_classes = [torch.unique(target["labels"]) for target in targets]
            not_gt_mask = torch.cat(
                (
                    images.tensor.new_ones(
                        (B, self.sem_seg_head.num_classes), dtype=torch.bool),
                    images.tensor.new_zeros((B, 1), dtype=torch.bool),
                ),
                dim=1,
            )
            for b in range(B):
                not_gt_mask[b].scatter_(0, gt_classes[b], 0)

            if self.ml_head:
                losses["loss_ml"] = (self.ml_loss(
                    img_pred,
                    torch.logical_not(not_gt_mask[:, :-1]).float()) *
                                     self.ml_loss_weight)
            return losses
        else:
            if self.ml_head:
                outputs["pred_logits"] = (outputs["pred_logits"].new_ones(
                    (B, self.num_queries, self.sem_seg_head.num_classes + 1)) *
                                          -100).scatter(
                                              2,
                                              topk_index.unsqueeze(1).repeat(
                                                  1, self.num_queries, 1),
                                              outputs["pred_logits"],
                                          )

            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]

            mask_cls_result = mask_cls_results[0]
            # upsample masks
            mask_pred_result = retry_if_cuda_oom(F.interpolate)(
                mask_pred_results[0],
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

            del outputs

            input_per_image = batched_inputs[0]
            image_size = images.image_sizes[
                0]  # image size without padding after data augmentation

            height = input_per_image.get(
                "height",
                image_size[0])  # raw image size before data augmentation
            width = input_per_image.get("width", image_size[1])

            output = retry_if_cuda_oom(self.inference_video)(mask_cls_result,
                                                             mask_pred_result,
                                                             image_size,
                                                             height, width)

            if self.sem_seg_postprocess_before_inference:
                mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                    mask_pred_result, image_size, height, width)
                mask_cls_result = mask_cls_result.to(mask_pred_result)

            if self.semantic_on:
                r = retry_if_cuda_oom(self.semantic_inference)(
                    mask_cls_result, mask_pred_result)
                output['sem_seg'] = r

            return output

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        gt_instances = []
        for targets_per_video in targets:
            _num_instance = len(targets_per_video["instances"][0])
            mask_shape = [
                _num_instance,
                len(targets_per_video["instances"]), h_pad, w_pad
            ]
            gt_masks_per_video = torch.zeros(mask_shape,
                                             dtype=torch.bool,
                                             device=self.device)
            with_id = hasattr(targets_per_video["instances"], 'gt_ids')
            if with_id:
                gt_ids_per_video = []
            for f_i, targets_per_frame in enumerate(
                    targets_per_video["instances"]):
                targets_per_frame = targets_per_frame.to(self.device)
                h, w = targets_per_frame.image_size

                if with_id:
                    gt_ids_per_video.append(targets_per_frame.gt_ids[:, None])
                gt_masks_per_video[:, f_i, :h, :
                                   w] = targets_per_frame.gt_masks.tensor

            if with_id:
                gt_ids_per_video = torch.cat(gt_ids_per_video, dim=1)
                valid_idx = (gt_ids_per_video != -1).any(dim=-1)

                gt_classes_per_video = targets_per_frame.gt_classes[
                    valid_idx]  # N,
                gt_ids_per_video = gt_ids_per_video[valid_idx]  # N, num_frames
                gt_instances.append({
                    "labels": gt_classes_per_video,
                    "ids": gt_ids_per_video
                })
                gt_masks_per_video = gt_masks_per_video[valid_idx].float(
                )  # N, num_frames, H, W
            else:
                gt_classes_per_video = targets_per_frame.gt_classes
                gt_instances.append({
                    "labels": gt_classes_per_video,
                })
                gt_masks_per_video = gt_masks_per_video.float(
                )  # N, num_frames, H, W

            gt_instances[-1].update({"masks": gt_masks_per_video})

        return gt_instances

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qfhw->cfhw", mask_cls, mask_pred)
        return semseg

    def inference_video(self, pred_cls, pred_masks, img_size, output_height,
                        output_width):
        if len(pred_cls) > 0:
            scores = F.softmax(pred_cls, dim=-1)[:, :-1]
            labels = (torch.arange(self.sem_seg_head.num_classes,
                                   device=self.device).unsqueeze(0).repeat(
                                       self.num_queries, 1).flatten(0, 1))
            # keep top-10 predictions
            scores_per_image, topk_indices = scores.flatten(0, 1).topk(
                10, sorted=False)
            labels_per_image = labels[topk_indices]
            topk_indices = topk_indices // self.sem_seg_head.num_classes
            pred_masks = pred_masks[topk_indices]

            pred_masks = pred_masks[:, :, :img_size[0], :img_size[1]]
            pred_masks = F.interpolate(
                pred_masks,
                size=(output_height, output_width),
                mode="bilinear",
                align_corners=False,
            )
            masks = pred_masks > 0.0

            out_scores = scores_per_image.tolist()
            out_labels = labels_per_image.tolist()
            out_masks = [m for m in masks.cpu()]
        else:
            out_scores = []
            out_labels = []
            out_masks = []

        video_output = {
            "image_size": (output_height, output_width),
            "pred_scores": out_scores,
            "pred_labels": out_labels,
            "pred_masks": out_masks,
        }

        return video_output


def sem_seg_postprocess(result, img_size, output_height, output_width):
    """
    Return semantic segmentation predictions in the original resolution.

    The input images are often resized when entering semantic segmentor. Moreover, in same
    cases, they also padded inside segmentor to be divisible by maximum network stride.
    As a result, we often need the predictions of the segmentor in a different
    resolution from its inputs.

    Args:
        result (Tensor): semantic segmentation prediction logits. A tensor of shape (C, H, W),
            where C is the number of classes, and H, W are the height and width of the prediction.
        img_size (tuple): image size that segmentor is taking as input.
        output_height, output_width: the desired output resolution.

    Returns:
        semantic segmenation prediction (Tensor): A tensor of the shape
            (C, output_height, output_width) that contains per-pixel soft predictions.
    """
    result = result[..., :img_size[0], :img_size[1]].expand(-1, -1, -1, -1)
    result = F.interpolate(result,
                           size=(output_height, output_width),
                           mode="bilinear",
                           align_corners=False)
    return result
