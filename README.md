# MLSeg: Image and Video Segmentation as Multi-Label Classification and Selected-Label Pixel Classification



## Introduction
For a long period of time, research studies on segmentation have typically formulated the task as pixel classification that predicts a class for each pixel from a set of predefined, fixed number of semantic categories. Yet standard architectures following this formulation will inevitably encounter various challenges under more realistic settings where the total number of semantic categories scales up (e.g., beyond 1k classes). On the other hand, a standard image or video usually contains only a small number of semantic categories from the entire label set. Motivated by this intuition, in this paper, we propose to decompose segmentation into two sub-problems: (i) image-level or video-level multi-label classification and (ii) pixel-level selected-label classification. Given an input image or video, our framework first conducts multi-label classification over the large complete label set and selects a small set of labels according to the class confidence scores. Then the follow-up pixel-wise classification is only performed among the selected subset of labels. Our approach is conceptually general and can be applied to various existing segmentation frameworks by simply adding a lightweight multi-label classification branch. We demonstrate the effectiveness of our framework with competitive experimental results across four tasks including image semantic segmentation, image panoptic segmentation, video instance segmentation, and video semantic segmentation. Especially, with our MLSeg, Mask2Former gains +0.8%/+0.7%/+0.7% on ADE20K panoptic segmentation/YouTubeVIS 2019 video instance segmentation/VSPW video semantic segmentation benchmarks respectively.

- The MLSeg architecture:

![teaser](./MLSeg.png)


## Image Semantic Segmentation based on DeepLabV3/Segmenter/Swin/BEiT + MLSeg

### MLSeg + DeepLabV3

| Method | Dataset | Backbone | Crop Size | Lr schd |  mIoU | mIoU(ms+flip) | config | download |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| DeepLabV3 (Official) | COCO-Stuff | R101 | 512x512 | 20000 | 37.3 | 38.4 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/deeplabv3_r101-d8_512x512_4x4_20k_coco-stuff10k.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3/deeplabv3_r101-d8_512x512_4x4_80k_coco-stuff164k/deeplabv3_r101-d8_512x512_4x4_80k_coco-stuff164k_20210709_201252-13600dc2.pth)
| DeepLabV3 + MLSeg | COCO-Stuff | R101 | 512x512 | 20000 | 38.4 | 39.8 | [config](deeplabv3_mlseg_r101-d8_512x512_20k_coco-stuff10k.py) | -
| DeepLabV3 (Official) | ADE20K | R101 | 512x512 | 80000 | 44.1 | 45.2 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/deeplabv3/deeplabv3_r101-d8_512x512_80k_ade20k.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3/deeplabv3_r101-d8_512x512_80k_ade20k/deeplabv3_r101-d8_512x512_80k_ade20k_20200615_021256-d89c7fa4.pth)
| DeepLabV3 + MLSeg | ADE20K | R101 | 512x512 | 80000 | 45.5 | 46.6 | - | -
| DeepLabV3 | COCO+LVIS | R101 | 512x512 | 160000 | 11.0 | - | - | -
| DeepLabV3 + MLSeg | COCO+LVIS | R101 | 512x512 | 160000 | 12.8 | - | - | -


### MLSeg + Segmenter

* Multi-Scale test is not conducted on ADE20KFull and COCO+LVIS datasets because of memory limits.

| Method | Dataset | Backbone | Crop Size | Lr schd |  mIoU | mIoU(ms+flip) | config | download |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Segmenter | COCO-Stuff | ViT-B | 512x512 | 40000 | 41.9 | 43.8 | [config](segmenter-ori_vit-b16_512x512_40k_coco-stuff10k.py) | -
| Segmenter + MLSeg | COCO-Stuff | ViT-B | 512x512 | 40000 | 44.9 | 46.2 | [config](segmenter-ori_mlseg_vit-b16_512x512_40k_coco-stuff10k.py) | -
| Segmenter | COCO-Stuff | ViT-B | 512x512 | 80000 | 43.4 | 45.2 | [config](segmenter-ori_vit-b16_512x512_80k_coco-stuff10k.py) | -
| Segmenter + MLSeg | COCO-Stuff | ViT-B | 512x512 | 80000 | 45.7 | 46.7 | [config](segmenter-ori_mlseg_vit-b16_512x512_80k_coco-stuff10k.py) | -
| Segmenter | COCO-Stuff | ViT-L | 640x640 | 40000 | 45.5 | 47.1 | - | -
| Segmenter + MLSeg | COCO-Stuff | ViT-B | 640x640 | 40000 | 46.7 | 47.9 | - | -
| Segmenter | Pascal-Context60 | ViT-B | 480x480 | 80000 | 53.8 | 54.6 | [config](segmenter-ori_vit-b16_480x480_80k_pascal-context.py) | -
| Segmenter + MLSeg | Pascal-Context60 | ViT-B | 480x480 | 80000 | 54.7 | 55.4 | [config](segmenter-ori_mlseg_vit-b16_480x480_80k_pascal-context.py) | - 
| Segmenter | ADE20K | ViT-B | 512x512 | 160000 | 48.8 | 50.7 | [config](segmenter-ori_vit-b16_512x512_160k_ade20k.py) | -
| Segmenter + MLSeg | ADE20K | ViT-B | 512x512 | 160000 | 49.7 | 51.4 | [config](segmenter-ori_mlseg_vit-b16_512x512_160k_ade20k.py) | -
| Segmenter | ADE20K | ViT-L | 640x640 | 160000 | 52.0 | 53.6 | [config](segmenter-ori_vit-l16_640x640_160k_ade20k.py) | -
| Segmenter + MLSeg | ADE20K | ViT-L | 640x640 | 160000 | 52.6 | 54.4 | [config](segmenter-ori_mlseg_vit-b16_512x512_160k_ade20k.py) | -
| Segmenter | ADE20KFull | ViT-B | 512x512 | 160000 | 17.8 | - | [config](segmenter-ori_vit-b16_512x512_160k_ade20kfull.py) | -
| Segmenter + MLSeg | ADE20KFull | ViT-B | 512x512 | 160000 | 18.8 | - | [config](segmenter-ori_mlseg_vit-b16_512x512_160k_ade20kfull.py) | -
| Segmenter | COCO+LVIS | ViT-B | 512x512 | 320000 | 19.4 | - | [config](segmenter-ori_vit-b16_512x512_320k_lvis.py) | -
| Segmenter + MLSeg | COCO+LVIS | ViT-B | 512x512 | 320000 | 21.3 | - | [config](segmenter-ori_mlseg_vit-b16_512x512_320k_lvis.py) | -
| Segmenter | COCO+LVIS | ViT-B | 640x640 | 320000 | 23.7 | - | - | -
| Segmenter + MLSeg | COCO+LVIS | ViT-B | 640x640 | 320000 | 24.9 | - | - | -

### MLSeg + Swin

| Method | Dataset | Backbone | Crop Size | Lr schd |  mIoU | mIoU(ms+flip) | config | download |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Swin | COCO-Stuff | Swin-B | 512x512 | 40000 | 45.7 | 47.2 | - | -
| Swin + MLSeg | COCO-Stuff | Swin-B | 512x512 | 40000 | 46.6 | 47.9 | - | -
| Swin (Official) | ADE20K | Swin-B | 512x512 | 160000 | 50.8 | 52.4 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/swin/upernet_swin_base_patch4_window12_512x512_160k_ade20k_pretrain_384x384_22K.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/swin/upernet_swin_base_patch4_window12_512x512_160k_ade20k_pretrain_384x384_22K/upernet_swin_base_patch4_window12_512x512_160k_ade20k_pretrain_384x384_22K_20210531_125459-429057bf.pth)
| Swin + MLSeg | ADE20K | Swin-B | 512x512 | 160000 | 51.4 | 53.0 | [config](upernet_swin_mlseg_base_patch4_window12_512x512_160k_ade20k_pretrain_384x384_22K.py) | -
| Swin | COCO+LVIS | Swin-B | 512x512 | 160000 | 20.3 | - | - | -
| Swin + MLSeg | COCO+LVIS | Swin-B | 512x512 | 160000 | 20.8 | - | - | -

### MLSeg + BEiT

| Method | Dataset | Backbone | Crop Size | Lr schd |  mIoU | mIoU(ms+flip) | config | download |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| BEiT (Official) | ADE20K | BEiT-L | 640x640 | 160000 | 56.7 | 57.0 | [config](upernet-unilm_beit-l16_640x640_slide_320k_ade20k.py) | -
| MLSeg + BEiT | ADE20K | BEiT-L | 640x640 | 160000 | 57.0 | 57.8| [config](upernet-unilm_mlseg_beit-l16_640x640_slide_320k_ade20k.py) | -


## Image Semantic & Panoptic Segmentation based on MaskFormer +  MLSeg


### Semantic Segmentation
| Method | Dataset | Backbone | Crop Size | Lr schd |  mIoU | mIoU(ms+flip) | config | download |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| MaskFormer | ADE20K | Swin-B | 512x512 | 160000 | 52.7 | 53.9 | [config](configs/ade20k-150/swin/maskformer_swin_base_IN21k_384_bs16_160k_res640.yaml) | -
| MaskFormer + MLSeg | ADE20K | Swin-B | 512x512 | 160000 | 53.9 | 55.1 | [config](configs/ade20k-150/swin/maskformer_swin_base_IN21k_384_bs16_160k_res640-mlseg.yaml) | -
| MaskFormer | ADE20K | Swin-L | 512x512 | 160000 | 54.1 | 55.6 | [config](configs/ade20k-150/swin/maskformer_swin_large_IN21k_384_bs16_160k_res640.yaml) | -
| MaskFormer + MLSeg | ADE20K | Swin-L | 512x512 | 160000 | 54.6 | 55.8 | [config](maskformer_swin_large_IN21k_384_bs16_160k_res640-mlseg.yaml) | -


### Panoptic Segmantation
| Method | Dataset | Backbone | Crop Size | Lr schd | PQ | PQ-th | PQ-st | RQ | RQ-th | RQ-st | SQ | SQ-th | SQ-st | config | download |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| MaskFormer | ADE20K | R50 | 640x640 | 720000 | 34.7 | 32.2 | 39.7 | 42.8 | 40.1 | 48.1 | 76.7 | 76.9 | 76.3 | [config](configs/ade20k-150-panoptic/maskformer_panoptic_R50_bs16_720k.yaml) | -
| MaskFormer + MLSeg | ADE20K | R50 | 640x640 | 720000 | 36.5 | 34.5 | 40.6 | 44.9 | 42.8 | 48.9 | 76.8 | 77.1 | 76.0 | [config](configs/ade20k-150-panoptic/maskformer_panoptic_R50_bs16_720k-mlseg.yaml) | -
| MaskFormer + MLSeg + GT| ADE20K | R50 | 640x640 | 720000 | 44.3 | 39.7 | 53.5 | 54.5 | 49.5 | 64.6 | 79.6 | 78.6 | 81.7 | [config](configs/ade20k-150-panoptic/maskformer_panoptic_R50_bs16_720k-gt.yaml) | -


## Image Semantic & Image Panoptic & Video Semantic & Video Instance Segmentation based on Mask2Former +  MLSeg


## Image Semantic based on SeMask +  MLSeg


## Citation

If you find this project useful in your research, please consider cite:

```
@article{HYYH2022MLSeg,
  title={MLSeg: Image and Video Segmentation as Multi-Label Classification and Selected-Label Pixel Classification},
  author={Haodi He and Yuhui Yuan and Xiangyu Yue and Han Hu},
  booktitle={arXiv},
  year={2022}
}
```


```
git diff-index HEAD
git subtree add -P pose <url to sub-repo> <sub-repo branch>
```
