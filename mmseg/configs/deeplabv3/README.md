# MLSeg + DeepLabV3

## Results and models

* Batch-size of all models is set as 16 following the original setting of Deeplabv3. All models are trained on 8 V-100 GPUs.

| Method | Dataset | Backbone | Crop Size | Lr schd |  mIoU | mIoU(ms+flip) | config | download |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| DeepLabV3 (Official) | COCO-Stuff | R101 | 512x512 | 20000 | 37.3 | 38.4 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/deeplabv3_r101-d8_512x512_4x4_20k_coco-stuff10k.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3/deeplabv3_r101-d8_512x512_4x4_80k_coco-stuff164k/deeplabv3_r101-d8_512x512_4x4_80k_coco-stuff164k_20210709_201252-13600dc2.pth)
| DeepLabV3 + MLSeg | COCO-Stuff | R101 | 512x512 | 20000 | 38.4 | 39.8 | [config](deeplabv3_mlseg_r101-d8_512x512_20k_coco-stuff10k.py) | -
| DeepLabV3 (Official) | ADE20K | R101 | 512x512 | 80000 | 44.1 | 45.2 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/deeplabv3/deeplabv3_r101-d8_512x512_80k_ade20k.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3/deeplabv3_r101-d8_512x512_80k_ade20k/deeplabv3_r101-d8_512x512_80k_ade20k_20200615_021256-d89c7fa4.pth)
| DeepLabV3 + MLSeg | ADE20K | R101 | 512x512 | 80000 | 45.5 | 46.6 | - | -
| DeepLabV3 | COCO+LVIS | R101 | 512x512 | 160000 | 11.0 | - | - | -
| DeepLabV3 + MLSeg | COCO+LVIS | R101 | 512x512 | 160000 | 12.8 | - | - | -
