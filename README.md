# MLSeg: Image and Video Segmentation as Multi-Label Classification and Selected-Label Pixel Classification



## Introduction
For a long period of time, research studies on segmentation have typically formulated the task as pixel classification that predicts a class for each pixel from a set of predefined, fixed number of semantic categories. Yet standard architectures following this formulation will inevitably encounter various challenges under more realistic settings where the total number of semantic categories scales up (e.g., beyond 1k classes). On the other hand, a standard image or video usually contains only a small number of semantic categories from the entire label set. Motivated by this intuition, in this paper, we propose to decompose segmentation into two sub-problems: (i) image-level or video-level multi-label classification and (ii) pixel-level selected-label classification. Given an input image or video, our framework first conducts multi-label classification over the large complete label set and selects a small set of labels according to the class confidence scores. Then the follow-up pixel-wise classification is only performed among the selected subset of labels. Our approach is conceptually general and can be applied to various existing segmentation frameworks by simply adding a lightweight multi-label classification branch. We demonstrate the effectiveness of our framework with competitive experimental results across four tasks including image semantic segmentation, image panoptic segmentation, video instance segmentation, and video semantic segmentation. Especially, with our MLSeg, Mask2Former gains +0.8%/+0.7%/+0.7% on ADE20K panoptic segmentation/YouTubeVIS 2019 video instance segmentation/VSPW video semantic segmentation benchmarks respectively.

- The MLSeg architecture:

![teaser](./figures/MLSeg.png)


## Image Semantic Segmentation based on DeepLabV3/Segmenter/Swin/BEiT + MLSeg

### MLSeg + DeepLabV3

| Method | Dataset | Backbone | Crop Size | Lr schd |  mIoU | mIoU(ms+flip) | config | download |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| DeepLabV3 (Official) | COCO-Stuff | R101 | 512x512 | 20000 | 37.3 | 38.4 | - | -
| DeepLabV3 + MLSeg | COCO-Stuff | R101 | 512x512 | 20000 | 38.4 | 39.8 | - | -
| DeepLabV3 (Official) | ADE20K | R101 | 512x512 | 80000 | 44.1 | 45.2 | - | -
| DeepLabV3 + MLSeg | ADE20K | R101 | 512x512 | 80000 | 45.5 | 46.6 | - | -
| DeepLabV3 | COCO+LVIS | R101 | 512x512 | 160000 | 11.0 | - | - | -
| DeepLabV3 + MLSeg | COCO+LVIS | R101 | 512x512 | 160000 | 12.8 | - | - | -


### MLSeg + Segmenter

* Multi-Scale test is not conducted on ADE20KFull and COCO+LVIS datasets because of memory limits.

| Method | Dataset | Backbone | Crop Size | Lr schd |  mIoU | mIoU(ms+flip) | config | download |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Segmenter | COCO-Stuff | ViT-B | 512x512 | 40000 | 41.9 | 43.8 |- | -
| Segmenter + MLSeg | COCO-Stuff | ViT-B | 512x512 | 40000 | 44.9 | 46.2 | - | -
| Segmenter | COCO-Stuff | ViT-B | 512x512 | 80000 | 43.4 | 45.2 | - | -
| Segmenter + MLSeg | COCO-Stuff | ViT-B | 512x512 | 80000 | 45.7 | 46.7 | - | -
| Segmenter | COCO-Stuff | ViT-L | 640x640 | 40000 | 45.5 | 47.1 | - | -
| Segmenter + MLSeg | COCO-Stuff | ViT-B | 640x640 | 40000 | 46.7 | 47.9 | - | -
| Segmenter | Pascal-Context60 | ViT-B | 480x480 | 80000 | 53.8 | 54.6 | - | -
| Segmenter + MLSeg | Pascal-Context60 | ViT-B | 480x480 | 80000 | 54.7 | 55.4 | - | - 
| Segmenter | ADE20K | ViT-B | 512x512 | 160000 | 48.8 | 50.7 | - | -
| Segmenter + MLSeg | ADE20K | ViT-B | 512x512 | 160000 | 49.7 | 51.4 | - | -
| Segmenter | ADE20K | ViT-L | 640x640 | 160000 | 52.0 | 53.6 | - | -
| Segmenter + MLSeg | ADE20K | ViT-L | 640x640 | 160000 | 52.6 | 54.4 | - | -
| Segmenter | ADE20KFull | ViT-B | 512x512 | 160000 | 17.8 | - | - | -
| Segmenter + MLSeg | ADE20KFull | ViT-B | 512x512 | 160000 | 18.8 | - | - | -
| Segmenter | COCO+LVIS | ViT-B | 512x512 | 320000 | 19.4 | - | - | -
| Segmenter + MLSeg | COCO+LVIS | ViT-B | 512x512 | 320000 | 21.3 | - | - | -
| Segmenter | COCO+LVIS | ViT-B | 640x640 | 320000 | 23.7 | - | - | -
| Segmenter + MLSeg | COCO+LVIS | ViT-B | 640x640 | 320000 | 24.6 | - | - | -

### MLSeg + Swin

| Method | Dataset | Backbone | Crop Size | Lr schd |  mIoU | mIoU(ms+flip) | config | download |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Swin | COCO-Stuff | Swin-B | 512x512 | 40000 | 45.7 | 47.2 | - | -
| Swin + MLSeg | COCO-Stuff | Swin-B | 512x512 | 40000 | 46.6 | 47.9 | - | -
| Swin (Official) | ADE20K | Swin-B | 512x512 | 160000 | 50.8 | 52.4 | - |-
| Swin + MLSeg | ADE20K | Swin-B | 512x512 | 160000 | 51.4 | 53.0 | - | -
| Swin | COCO+LVIS | Swin-B | 512x512 | 160000 | 20.3 | - | - | -
| Swin + MLSeg | COCO+LVIS | Swin-B | 512x512 | 160000 | 20.8 | - | - | -

### MLSeg + BEiT

| Method | Dataset | Backbone | Crop Size | Lr schd |  mIoU | mIoU(ms+flip) | config | download |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| BEiT (Official) | ADE20K | BEiT-L | 640x640 | 160000 | 56.7 | 57.0 | - | -
| MLSeg + BEiT | ADE20K | BEiT-L | 640x640 | 160000 | 57.0 | 57.8| - | -
| BEiT (Official) | COCO-Stuff | BEiT-L | 640x640 | 160000 | 49.7 | 49.9 | - | -
| MLSeg + BEiT | COCO-Stuff | BEiT-L | 640x640 | 160000 | 49.9 | 50.3| - | -

## Image Semantic & Panoptic Segmentation based on MaskFormer +  MLSeg

### Semantic Segmentation
| Method | Dataset | Backbone | Crop Size | Lr schd |  mIoU | mIoU(ms+flip) | config | download |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| MaskFormer | ADE20K | Swin-B | 512x512 | 160000 | 52.7 | 53.9 | - | -
| MaskFormer + MLSeg | ADE20K | Swin-B | 512x512 | 160000 | 53.9 | 55.1 | - | -

### Panoptic Segmentation
| Method | Dataset | Backbone | Crop Size | Lr schd | PQ | PQ-th | PQ-st | RQ | RQ-th | RQ-st | SQ | SQ-th | SQ-st | config | download |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| MaskFormer | ADE20K | R50 | 640x640 | 720000 | 34.7 | 32.2 | 39.7 | 42.8 | 40.1 | 48.1 | 76.7 | 76.9 | 76.3 | - | -
| MaskFormer + MLSeg | ADE20K | R50 | 640x640 | 720000 | 36.5 | 34.5 | 40.6 | 44.9 | 42.8 | 48.9 | 76.8 | 77.1 | 76.0 | - | -
| MaskFormer + MLSeg + GT| ADE20K | R50 | 640x640 | 720000 | 44.3 | 39.7 | 53.5 | 54.5 | 49.5 | 64.6 | 79.6 | 78.6 | 81.7 | - | -


## Image Semantic & Image Panoptic & Video Semantic & Video Instance Segmentation based on Mask2Former +  MLSeg
### Semantic Segmentation
| Method | Dataset | Backbone | Crop Size | Lr schd |  mIoU | mIoU(ms+flip) | config | download |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Mask2Former | ADE20K | Swin-B | 512x512 | 160000 | 53.9 | 55.1 | - | -
| Mask2Former + MLSeg | ADE20K | Swin-B | 512x512 | 160000 | 54.9 | 55.6 | - | -
| Mask2Former | ADE20K | Swin-L | 512x512 | 160000 | 56.1 | 57.3 | - | -
| Mask2Former + MLSeg | ADE20K | Swin-L | 512x512 | 160000 | 56.5 | 58.0 | - | -

### Panoptic Segmentation
| Method | Dataset | Backbone | Crop Size | Lr schd | PQ | config | download |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Mask2Former | ADE20K | Swin-L | 512x512 | 160000 | 48.1 | - | -
| Mask2Former + MLSeg | ADE20K | Swin-L | 512x512 | 160000 | 48.9 | - | -

### Video Semantic Segmentation
| Method | Dataset | Backbone | Crop Size | Lr schd | mIoU | config | download |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Mask2Former | VSPW | R101 | 512x512 | 6000 | 45.9 | - | -
| Mask2Former + MLSeg | VSPW | R101 | 512x512 | 6000 | 47.0 | - | -
| Mask2Former | VSPW | Swin-L | 512x512 | 6000 | 59.4 | - | -
| Mask2Former + MLSeg | VSPW | Swin-L | 512x512 | 6000 | 60.1 | - | -

### Video Instance Segmentation
| Method | Dataset | Backbone | Crop Size | Lr schd | AP | config | download |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Mask2Former | YoutubeVIS2019 | R101 | 512x512 | 6000 | 49.2 | - | -
| Mask2Former + MLSeg | YoutubeVIS2019 | R101 | 512x512 | 6000 | 50.5 | - | -
| Mask2Former | YoutubeVIS2019 | Swin-B | 512x512 | 6000 | 59.5 | - | -
| Mask2Former + MLSeg | YoutubeVIS2019 | Swin-B | 512x512 | 6000 | 60.3 | - | -
| Mask2Former | YoutubeVIS2019 | Swin-L | 512x512 | 6000 | 60.4 | - | -
| Mask2Former + MLSeg | YoutubeVIS2019 | Swin-L | 512x512 | 6000 | 61.1 | - | -

## Image Semantic based on SeMask +  MLSeg
# Semantic Segmentation
| Method | Dataset | Backbone | Crop Size | Lr schd | mIoU | mIoU(ms) | config | download |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| SeMask + Mask2Former MSFAPN | ADE20K | Swin-L | 512x512 | 160000 | 56.54 | 58.22 | - | -
| Mask2Former + MLSeg | YoutubeVIS2019 | Swin-L | 512x512 | 160000 | 56.82 | 58.48 | - | -

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
