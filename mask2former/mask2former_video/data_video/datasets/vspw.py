# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/sukjunhwang/IFC

import contextlib
import io
import json
import logging
import numpy as np
import os
import pycocotools.mask as mask_util
from fvcore.common.file_io import PathManager
from fvcore.common.timer import Timer
from detectron2.data import detection_utils as utils
from detectron2.structures import Boxes, BoxMode, PolygonMasks
from detectron2.data import DatasetCatalog, MetadataCatalog
from tqdm import tqdm

# Copyright (c) Facebook, Inc. and its affiliates.
import json
import os
import pickle
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.file_io import PathManager

logger = logging.getLogger(__name__)


def load_video_vspw_json(name, text_file, root):
    """
    Args:
        json_file (str): path to the json file. e.g., "~/coco/annotations/panoptic_train2017.json".
    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    """
    with open(text_file, 'r') as f:
        files = f.readlines()

    ret = []
    
    import hashlib
    hash = hashlib.sha1(root.encode()).hexdigest()
    pickle_path = os.path.join(root, 'VSPW_480p',
                               hash[:10] + "_" + name + '.pkl')
    print(pickle_path)
    if os.path.exists(pickle_path):
        with open(pickle_path, "rb") as fp:
            ret = pickle.load(fp)
    else:
        for video_name in tqdm([file[:-1] for file in files]):
            image_dir = os.path.join(root, 'VSPW_480p/data', video_name,
                                     'origin')
            sem_seg_dir = os.path.join(root, 'VSPW_480p/data', video_name,
                                       'mask')
            image_files_name = os.listdir(image_dir)
            sem_seg_files_name = os.listdir(sem_seg_dir)
            if len(image_files_name) != len(sem_seg_files_name):
                logger.warning(
                    '{} images and {} annotations found for {}'.format(
                        len(image_files_name), len(sem_seg_files_name),
                        video_name))

            files_name = set([
                name.split('.')[0] for name in image_files_name
            ]) & set([name.split('.')[0] for name in sem_seg_files_name])
            files_name = sorted(files_name)
            image = utils.read_image(os.path.join(image_dir,
                                                  image_files_name[0]),
                                     format='BGR')
            h, w, _ = (image.shape)
            
            if 'val' in text_file:
                for i in range((len(files_name) - 1) // 20 + 1):       
                    info = {
                        'width':                w,
                        'height':               h,
                        'file_name':            video_name,
                        'length':               len(files_name[i * 20: i * 20 + 20]),
                        'image_files_name': [
                            os.path.join(image_dir, file_name + '.jpg')
                            for file_name in files_name[i * 20: i * 20 + 20]
                        ],
                        'sem_seg_files_name': [
                            os.path.join(sem_seg_dir, file_name + '.png')
                            for file_name in files_name[i * 20: i * 20 + 20]
                        ],
                    }
                    ret.append(info)
            else:
                info = {
                    'width':                w,
                    'height':               h,
                    'file_name':            video_name,
                    'length':               len(files_name),
                    'image_files_name': [
                        os.path.join(image_dir, file_name + '.jpg')
                        for file_name in files_name
                    ],
                    'sem_seg_files_name': [
                        os.path.join(sem_seg_dir, file_name + '.png')
                        for file_name in files_name
                    ],
                }
                ret.append(info)
        with open(pickle_path, 'wb') as fp:
            pickle.dump(ret, fp)
            print(f'Saved to {pickle_path}')

    return ret


def register_video_vspw_json(
    name,
    meta,
    root,
    text_file,
):
    """
    Register a "standard" version of ADE20k panoptic segmentation dataset named `name`.
    The dictionaries in this registered dataset follows detectron2's standard format.
    Hence it's called "standard".
    Args:
        name (str): the name that identifies a dataset,
            e.g. "ade20k_panoptic_train"
        metadata (dict): extra metadata associated with this dataset.
        imageroot (str): directory which contains all the images
        root (str): directory which contains panoptic annotation images in COCO format
        json (str): path to the json panoptic annotation file in COCO format
        sem_segroot (none): not used, to be consistent with
            `register_coco_panoptic_separated`.
        instances_json (str): path to the json instance annotation file
    """
    DatasetCatalog.register(
        name,
        lambda: load_video_vspw_json(name, text_file, root),
    )
    MetadataCatalog.get(name).set(text_file=text_file,
                                  evaluator_type=None,
                                  categories=124,
                                  ignore_label=(255),
                                  root=root,
                                  **meta)


_PREDEFINED_SPLITS_VSPW = {
    "vspw_train": ("VSPW_480p/train.txt"),
    "vspw_val": ("VSPW_480p/val.txt"),
    "vspw_test": ("VSPW_480p/test.txt"),
}


def get_meta():
    VSPW_THING_CLASSES = [
        'flag', 'parasol_or_umbrella', 'cushion_or_carpet', 'tent',
        'roadblock', 'car', 'bus', 'truck', 'bicycle', 'motorcycle',
        'wheeled_machine', 'ship_or_boat', 'raft', 'airplane', 'tyre',
        'traffic_light', 'lamp', 'person', 'cat', 'dog', 'horse', 'cattle',
        'other_animal', 'tree', 'flower', 'other_plant', 'toy', 'ball_net',
        'backboard', 'skateboard', 'bat', 'ball',
        'cupboard_or_showcase_or_storage_rack', 'box',
        'traveling_case_or_trolley_case', 'basket', 'bag_or_package',
        'trash_can', 'cage', 'plate', 'tub_or_bowl_or_pot', 'bottle_or_cup',
        'barrel', 'fishbowl', 'bed', 'pillow', 'table_or_desk',
        'chair_or_seat', 'bench', 'sofa', 'shelf', 'bathtub', 'gun', 'commode',
        'roaster', 'other_machine', 'refrigerator', 'washing_machine',
        'Microwave_oven', 'fan', 'curtain', 'textiles', 'clothes',
        'painting_or_poster', 'mirror', 'flower_pot_or_vase', 'clock', 'book',
        'tool', 'blackboard', 'tissue', 'screen_or_television', 'computer',
        'printer', 'Mobile_phone', 'keyboard', 'other_electronic_product',
        'fruit', 'food', 'instrument', 'train'
    ]
    VSPW_STUFF_CLASSES = [
        'wall', 'ceiling', 'door', 'stair', 'ladder', 'escalator',
        'Playground_slide', 'handrail_or_fence', 'window', 'rail', 'goal',
        'pillar', 'pole', 'floor', 'ground', 'grass', 'sand', 'athletic_field',
        'road', 'path', 'crosswalk', 'building', 'house', 'bridge', 'tower',
        'windmill', 'well_or_well_lid', 'other_construction', 'sky',
        'mountain', 'stone', 'wood', 'ice', 'snowfield', 'grandstand', 'sea',
        'river', 'lake', 'waterfall', 'water', 'billboard_or_Bulletin_Board',
        'sculpture', 'pipeline', 'flag', 'parasol_or_umbrella',
        'cushion_or_carpet', 'tent', 'roadblock', 'car', 'bus', 'truck',
        'bicycle', 'motorcycle', 'wheeled_machine', 'ship_or_boat', 'raft',
        'airplane', 'tyre', 'traffic_light', 'lamp', 'person', 'cat', 'dog',
        'horse', 'cattle', 'other_animal', 'tree', 'flower', 'other_plant',
        'toy', 'ball_net', 'backboard', 'skateboard', 'bat', 'ball',
        'cupboard_or_showcase_or_storage_rack', 'box',
        'traveling_case_or_trolley_case', 'basket', 'bag_or_package',
        'trash_can', 'cage', 'plate', 'tub_or_bowl_or_pot', 'bottle_or_cup',
        'barrel', 'fishbowl', 'bed', 'pillow', 'table_or_desk',
        'chair_or_seat', 'bench', 'sofa', 'shelf', 'bathtub', 'gun', 'commode',
        'roaster', 'other_machine', 'refrigerator', 'washing_machine',
        'Microwave_oven', 'fan', 'curtain', 'textiles', 'clothes',
        'painting_or_poster', 'mirror', 'flower_pot_or_vase', 'clock', 'book',
        'tool', 'blackboard', 'tissue', 'screen_or_television', 'computer',
        'printer', 'Mobile_phone', 'keyboard', 'other_electronic_product',
        'fruit', 'food', 'instrument', 'train'
    ]
    return {
        "thing_classes": VSPW_THING_CLASSES,
        "stuff_classes": VSPW_STUFF_CLASSES,
    }


def register_all_video_vspw(root):
    for (
            prefix,
        (text_file),
    ) in _PREDEFINED_SPLITS_VSPW.items():
        # The "standard" version of COCO panoptic segmentation dataset,
        # e.g. used by Panoptic-DeepLab
        register_video_vspw_json(
            prefix,
            get_meta(),
            root,
            os.path.join(root, text_file),
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
# register_all_video_vspw(_root)

if __name__ == "__main__":
    """
    Test the YTVIS json dataset loader.
    """
    from detectron2.utils.logger import setup_logger
    from detectron2.utils.visualizer import Visualizer
    import detectron2.data.datasets  # noqa # add pre-defined metadata
    import sys
    from PIL import Image

    logger = setup_logger(name=__name__)
    #assert sys.argv[3] in DatasetCatalog.list()

    dicts = load_video_vspw_json("vspw_val", os.path.join(_root, "VSPW_480p/val.txt"), _root)
    logger.info("Done loading {} samples.".format(len(dicts)))