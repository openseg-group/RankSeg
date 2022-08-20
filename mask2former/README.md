# H-Deformable-DETR

This is the official implementation of the paper "[RankSeg: Adaptive Pixel Classification with
Image Category Ranking for Segmentation
](https://arxiv.org/abs/2203.04187)". 

Authors: Haodi He, Yuhui Yuan, Xiangyu Yue, Han Hu


## Installation

See [installation instructions](INSTALL.md).

## Getting Started

See [Preparing Datasets for Mask2Former](datasets/README.md).

See [Getting Started with Mask2Former](GETTING_STARTED.md).

To train a model with 8 gpus:
```
python train_net.py --num-gpus 8 --config-file <CONFIG_FILE>

```

To evaluate a model with 8 gpus:
```
python train_net.py --num-gpus 8 --config-file CONFIG_FILE --eval-only MODEL.WEIGHTS <CHECKPOINT>

```

Please use `train_net_video.py` instead of `train_net.py` for `YoutubeVis2019` and `VSPW` datasets.

## Model Zoo and Baselines
We provide trained models of baseline and our method in [Model Zoo](MODEL_ZOO.md).


## Acknowledgement
Code is largely based on Mask2Former (https://github.com/facebookresearch/Mask2Former).