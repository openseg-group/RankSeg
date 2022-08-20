# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/sukjunhwang/IFC

import contextlib
import copy
import io
import itertools
import json
import logging
import numpy as np
import os
from collections import OrderedDict
import pycocotools.mask as mask_util
import torch

from detectron2.evaluation.sem_seg_evaluation import SemSegEvaluator
from .datasets.ytvis_api.ytvos import YTVOS
from .datasets.ytvis_api.ytvoseval import YTVOSeval
from tabulate import tabulate

import detectron2.utils.comm as comm
from detectron2.config import CfgNode
from detectron2.data import MetadataCatalog
from detectron2.evaluation import DatasetEvaluator, SemSegEvaluator
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import create_small_table

_CV2_IMPORTED = True
try:
    import cv2  # noqa
except ImportError:
    # OpenCV is an optional dependency at the moment
    _CV2_IMPORTED = False
    
class VSPWEvaluator(SemSegEvaluator):
    """
    Evaluate AR for object proposals, AP for instance detection/segmentation, AP
    for keypoint detection outputs using COCO's metrics.
    See http://cocodataset.org/#detection-eval and
    http://cocodataset.org/#keypoints-eval to understand its metrics.

    In addition to COCO, this evaluator is able to support any bounding box detection,
    instance segmentation, or keypoint detection dataset.
    """
    def __init__(
        self,
        dataset_name,
        distributed=True,
        output_dir=None,
        *,
        num_classes=None,
        ignore_label=None,
    ):
        self._logger = logging.getLogger(__name__)
        if num_classes is not None:
            self._logger.warn(
                "SemSegEvaluator(num_classes) is deprecated! It should be obtained from metadata."
            )
        if ignore_label is not None:
            self._logger.warn(
                "SemSegEvaluator(ignore_label) is deprecated! It should be obtained from metadata."
            )
        self._dataset_name = dataset_name
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")


        meta = MetadataCatalog.get(dataset_name)
        # Dict that maps contiguous training ids to COCO category ids
        try:
            c2d = meta.stuff_dataset_id_to_contiguous_id
            self._contiguous_id_to_dataset_id = {v: k for k, v in c2d.items()}
        except AttributeError:
            self._contiguous_id_to_dataset_id = None
        self._class_names = meta.stuff_classes
        self._num_classes = len(meta.stuff_classes)
        if num_classes is not None:
            assert self._num_classes == num_classes, f"{self._num_classes} != {num_classes}"
        self._ignore_label = ignore_label if ignore_label is not None else meta.ignore_label

        # This is because cv2.erode did not work for int datatype. Only works for uint8.
        self._compute_boundary_iou = True
        if not _CV2_IMPORTED:
            self._compute_boundary_iou = False
            self._logger.warn(
                """Boundary IoU calculation requires OpenCV. B-IoU metrics are
                not going to be computed because OpenCV is not available to import."""
            )
        if self._num_classes >= np.iinfo(np.uint8).max:
            self._compute_boundary_iou = False
            self._logger.warn(
                f"""SemSegEvaluator(num_classes) is more than supported value for Boundary IoU calculation!
                B-IoU metrics are not going to be computed. Max allowed value (exclusive)
                for num_classes for calculating Boundary IoU is {np.iinfo(np.uint8).max}.
                The number of classes of dataset {self._dataset_name} is {self._num_classes}"""
            )


    def process(self, inputs, outputs):
        input_ = inputs[0]
        output = outputs
        for frame_idx in range(len(input_['sem_seg'])):
            pred = output["sem_seg"][:,frame_idx,:,:].argmax(dim=0).to(self._cpu_device)
            pred = np.array(pred, dtype=np.int)
            gt = input_['sem_seg'][frame_idx].numpy()
            gt[gt == self._ignore_label] = self._num_classes
            self._conf_matrix += np.bincount(
                (self._num_classes + 1) * pred.reshape(-1) + gt.reshape(-1),
                minlength=self._conf_matrix.size,
            ).reshape(self._conf_matrix.shape)
