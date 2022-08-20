# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/sukjunhwang/IFC

import copy
import logging
import random
import numpy as np
from typing import List, Union
import torch
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Instances,
)

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

from .augmentation import build_augmentation

__all__ = ["VSPWDatasetMapper"]


class VSPWDatasetMapper:
    """
    A callable which takes a dataset dict in YouTube-VIS Dataset format,
    and map it into a format used by the model.
    """

    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        use_instance_mask: bool = False,
        sampling_frame_num: int = 2,
        sampling_frame_range: int = 5,
        sampling_frame_shuffle: bool = False,
        num_classes: int = 124,
        sampling_random = True,
        sampling_interval = 0,
        ignore_label = 255,
        size_divisibility = 480,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            use_instance_mask: whether to process instance segmentation annotations, if available
        """
        # fmt: off
        self.is_train               = is_train
        self.augmentations          = T.AugmentationList(augmentations)
        self.image_format           = image_format
        self.use_instance_mask      = use_instance_mask
        self.sampling_frame_num     = sampling_frame_num
        self.sampling_frame_range   = sampling_frame_range
        self.sampling_frame_shuffle = sampling_frame_shuffle
        self.sampling_random        = sampling_random
        self.sampling_interval      = sampling_interval
        self.num_classes            = num_classes
        self.ignore_label = ignore_label
        self.size_divisibility = size_divisibility
        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        augs = build_augmentation(cfg, is_train)

        sampling_frame_num = cfg.INPUT.SAMPLING_FRAME_NUM
        sampling_frame_range = cfg.INPUT.SAMPLING_FRAME_RANGE
        sampling_frame_shuffle = cfg.INPUT.SAMPLING_FRAME_SHUFFLE

        
        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "sampling_frame_num": sampling_frame_num,
            "sampling_frame_range": sampling_frame_range,
            "sampling_frame_shuffle": sampling_frame_shuffle,
            "sampling_random": cfg.INPUT.SAMPLING_RANDOM,
            "sampling_interval": cfg.INPUT.SAMPLING_INTERVAL,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "ignore_label": 255,
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,
        }

        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one video, in YTVIS Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        # TODO consider examining below deepcopy as it costs huge amount of computations.
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        video_length = dataset_dict["length"]
        if self.is_train:
            if self.sampling_random:
                ref_frame = random.randrange(video_length)

                start_idx = max(0, ref_frame-self.sampling_frame_range)
                end_idx = min(video_length, ref_frame+self.sampling_frame_range + 1)

                selected_idx = np.random.choice(
                    np.array(list(range(start_idx, ref_frame)) + list(range(ref_frame+1, end_idx))),
                    self.sampling_frame_num - 1,
                )
                selected_idx = selected_idx.tolist() + [ref_frame]
            else:
                temp = random.randrange(self.sampling_interval)
                temp_2 = random.randrange(max((video_length - temp) // self.sampling_interval - self.sampling_frame_num + 2, 1))
                selected_idx = [((temp_2 + i) * self.sampling_interval + temp) % video_length for i in range(self.sampling_frame_num)]
            selected_idx = sorted(selected_idx)
            if self.sampling_frame_shuffle:
                random.shuffle(selected_idx)
        else:
            selected_idx = range(video_length)


        sem_seg_files_name = dataset_dict.pop("sem_seg_files_name", None)
        image_files_name = dataset_dict.pop("image_files_name", None)


        dataset_dict["image"] = []
        dataset_dict["instances"] = []
        dataset_dict["sem_seg"] = []
        dataset_dict["image_files_name"] = [image_files_name[frame_idx] for frame_idx in selected_idx]
        
        for frame_idx in selected_idx:
            # Read image
            image = utils.read_image(image_files_name[frame_idx], format=self.image_format)
            sem_seg = utils.read_image(sem_seg_files_name[frame_idx]).astype(np.uint8)
            
                    
            sem_seg = sem_seg - 1
            sem_seg[sem_seg>=self.num_classes] = 255
            sem_seg = sem_seg.astype("double")
            
            utils.check_image_size(dataset_dict, image)
            utils.check_image_size(dataset_dict, sem_seg)

            aug_input = T.AugInput(image, sem_seg=sem_seg)
            transforms = self.augmentations(aug_input)
            image = aug_input.image
            sem_seg = aug_input.sem_seg
            
            image_shape = image.shape[:2]  # h, w
            # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
            # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
            # Therefore it's important to use torch.Tensor.
            
            sem_seg = torch.as_tensor(sem_seg.astype("long"))
            dataset_dict["image"].append(torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))))
            dataset_dict["sem_seg"].append(sem_seg.long())
            
        
        classes = np.unique(torch.stack(dataset_dict["sem_seg"]).numpy())
        # remove ignored region
        classes = classes[classes != self.ignore_label]
        
        for i in range(len(selected_idx)):
            sem_seg = dataset_dict["sem_seg"][i]
            instances = Instances(image_shape)
            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)
            masks = []
            for class_id in classes:
                masks.append(sem_seg.numpy() == class_id)

            if len(masks) == 0:
                # Some image does not have annotation (all ignored)
                instances.gt_masks = torch.zeros((0, sem_seg.shape[-2], sem_seg.shape[-1]))
            else:
                masks = BitMasks(
                    torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
                )
                instances.gt_masks = masks

            dataset_dict["instances"].append(instances)
        return dataset_dict