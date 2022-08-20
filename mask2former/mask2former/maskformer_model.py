# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple

import pdb
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom

from .modeling.criterion import SetCriterion, AsymmetricLoss
from .modeling.matcher import HungarianMatcher
from .modeling.multilabel_head import (
    MultiLabelHead,
)


@META_ARCH_REGISTRY.register()
class MaskFormer(nn.Module):
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
        # inference
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
        # mlseg parameters
        ml_use_gt: bool,
        ml_head: nn.Module,
        ml_loss: nn.Module,
        ml_loss_weight: float,
        ml_topk: int,
        ml_norm: nn.Module,
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
            ml_head: a multi-label classification module
            ml_loss: the loss choice for multi-label classification
            ml_loss_weight: the loss weight of multi-label classification
            ml_topk: the number of kept categories with higher multi-label classification scores
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
        self.register_buffer("pixel_mean",
                             torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std",
                             torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image

        # mlseg parameters
        self.ml_use_gt = ml_use_gt
        self.ml_head = ml_head
        self.ml_loss = ml_loss
        self.ml_loss_weight = ml_loss_weight
        self.ml_topk = ml_topk
        self.ml_norm = ml_norm

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

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
        matcher = HungarianMatcher(
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
                sem_seg_head.num_classes,
                cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM,
                cfg.MODEL.RANKSEG.HIDDEN_DIM,
                cfg.MODEL.RANKSEG.HIDDEN_DIM // 64,
                cfg.MODEL.RANKSEG.HIDDEN_DIM * 4,
                0.1,
                cfg.MODEL.RANKSEG.SHARE_EMBEDDING,
            )
            ml_loss = AsymmetricLoss(2, 0)
            ml_loss_weight = cfg.MODEL.RANKSEG.LOSS_WEIGHT
            ml_norm = nn.LayerNorm(cfg.MODEL.RANKSEG.TOPK + 1)
        else:
            ml_head = None
            ml_loss = None
            ml_loss_weight = None
            ml_norm = None

        criterion = SetCriterion(
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
            "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
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
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        # mask classification target
        if "instances" in batched_inputs[0]:
            gt_instances = [
                x["instances"].to(self.device) for x in batched_inputs
            ]

        else:
            targets = None

        features= self.backbone(images.tensor)

        # use the query features from the 3-rd transformer decoder layer
        outputs= self.sem_seg_head(features)

        B = len(batched_inputs)

        # generate ground-truth mask
        gt_classes = [
            torch.unique(gt_instance.gt_classes)
            for gt_instance in gt_instances
        ]
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
            targets = self.prepare_targets(gt_instances, images)

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

            # upsample masks
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

            del outputs

            processed_results = []
            for (
                    mask_cls_result,
                    mask_pred_result,
                    input_per_image,
                    image_size,
                    i,
            ) in zip(
                    mask_cls_results,
                    mask_pred_results,
                    batched_inputs,
                    images.image_sizes,
                    range(B),
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                processed_results.append({})

                if self.sem_seg_postprocess_before_inference:
                    mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                        mask_pred_result, image_size, height, width)
                    mask_cls_result = mask_cls_result.to(mask_pred_result)

                # semantic segmentation inference
                if self.semantic_on:
                    r = retry_if_cuda_oom(self.semantic_inference)(
                        mask_cls_result, mask_pred_result)
                    if not self.sem_seg_postprocess_before_inference:
                        r = retry_if_cuda_oom(sem_seg_postprocess)(r,
                                                                   image_size,
                                                                   height,
                                                                   width)
                    processed_results[-1]["sem_seg"] = r

                # panoptic segmentation inference
                if self.panoptic_on:
                    panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(
                        mask_cls_result, mask_pred_result)
                    processed_results[-1]["panoptic_seg"] = panoptic_r

                if self.ml_head:
                    processed_results[-1]["multilabel"] = (
                        img_pred[i].unsqueeze(0),
                        topk_index[i, :-1].unsqueeze(0),
                        torch.logical_not(not_gt_mask[i, :-1]).unsqueeze(0),
                    )
                # instance segmentation inference
                if self.instance_on:
                    instance_r = retry_if_cuda_oom(self.instance_inference)(
                        mask_cls_result, mask_pred_result)
                    processed_results[-1]["instances"] = instance_r

            return processed_results

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros(
                (gt_masks.shape[0], h_pad, w_pad),
                dtype=gt_masks.dtype,
                device=gt_masks.device,
            )
            padded_masks[:, :gt_masks.shape[1], :gt_masks.shape[2]] = gt_masks
            new_targets.append({
                "labels": targets_per_image.gt_classes,
                "masks": padded_masks,
            })
        return new_targets

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def panoptic_inference(self, mask_cls, mask_pred):
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()

        keep = labels.ne(self.sem_seg_head.num_classes) & (
            scores > self.object_mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w),
                                   dtype=torch.int32,
                                   device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg, segments_info
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = (pred_class in self.metadata.
                           thing_dataset_id_to_contiguous_id.values())
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item(
                ) > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(
                                pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(
                                pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append({
                        "id": current_segment_id,
                        "isthing": bool(isthing),
                        "category_id": int(pred_class),
                    })

            return panoptic_seg, segments_info

    def instance_inference(self, mask_cls, mask_pred):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]

        # [Q, K]
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        labels = (torch.arange(self.sem_seg_head.num_classes,
                               device=self.device).unsqueeze(0).repeat(
                                   self.num_queries, 1).flatten(0, 1))
        # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(
            self.test_topk_per_image, sorted=False)
        labels_per_image = labels[topk_indices]

        topk_indices = topk_indices // self.sem_seg_head.num_classes
        # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
        mask_pred = mask_pred[topk_indices]

        # if this is panoptic segmentation, we only keep the "thing" classes
        if self.panoptic_on:
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = (lab in self.metadata.
                           thing_dataset_id_to_contiguous_id.values())

            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]

        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = (mask_pred > 0).float()
        result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
        # Uncomment the following to get boxes from masks (this is slow)
        # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

        # calculate average mask prob
        mask_scores_per_image = (
            mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)
        ).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image
        return result
