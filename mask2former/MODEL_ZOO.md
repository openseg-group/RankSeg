# Mask2Former Model Zoo and Baselines

## Introduction

This file documents a collection of models reported in our paper.
All numbers were obtained on [Big Basin](https://engineering.fb.com/data-center-engineering/introducing-big-basin-our-next-generation-ai-hardware/)
servers with 8 NVIDIA V100 GPUs & NVLink (except Swin-L models are trained with 16 NVIDIA V100 GPUs).

#### How to Read the Tables
* The "Name" column contains a link to the config file. Running `train_net.py --num-gpus 8` with this config file
  will reproduce the model (except Swin-L models are trained with 16 NVIDIA V100 GPUs with distributed training on two nodes).
* The *model id* column is provided for ease of reference.
  To check downloaded file integrity, any model on this page contains its md5 prefix in its file name.

#### Detectron2 ImageNet Pretrained Models

It's common to initialize from backbone models pre-trained on ImageNet classification tasks. The following backbone models are available:

* [R-50.pkl (torchvision)](https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/torchvision/R-50.pkl): converted copy of [torchvision's ResNet-50](https://pytorch.org/docs/stable/torchvision/models.html#torchvision.models.resnet50) model.
  More details can be found in [the conversion script](tools/convert-torchvision-to-d2.py).
* [R-103.pkl](https://dl.fbaipublicfiles.com/detectron2/DeepLab/R-103.pkl): a ResNet-101 with its first 7x7 convolution replaced by 3 3x3 convolutions. This modification has been used in most semantic segmentation papers (a.k.a. ResNet101c in our paper). We pre-train this backbone on ImageNet using the default recipe of [pytorch examples](https://github.com/pytorch/examples/tree/master/imagenet).

Note: below are available pretrained models in Detectron2 that we do not use in our paper.
* [R-50.pkl](https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MSRA/R-50.pkl): converted copy of [MSRA's original ResNet-50](https://github.com/KaimingHe/deep-residual-networks) model.
* [R-101.pkl](https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MSRA/R-101.pkl): converted copy of [MSRA's original ResNet-101](https://github.com/KaimingHe/deep-residual-networks) model.
* [X-101-32x8d.pkl](https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/FAIR/X-101-32x8d.pkl): ResNeXt-101-32x8d model trained with Caffe2 at FB.

#### Third-party ImageNet Pretrained Models

Our paper also uses ImageNet pretrained models that are not part of Detectron2, please refer to [tools](https://github.com/facebookresearch/MaskFormer/tree/master/tools) to get those pretrained models.

#### License

All models available for download through this document are licensed under the
[Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/).


## ADE20K Model Zoo

### Panoptic Segmentation

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">iterations</th>
<th valign="bottom">PQ</th>
<th valign="bottom">AP</th>
<th valign="bottom">mIoU</th>
<th valign="bottom">model id</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: maskformer2_swin_large_IN21k_384_bs16_160k -->
 <tr><td align="left"><a href="configs/ade20k/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k.yaml">Mask2Former (200 queries)</a></td>
<td align="center">Swin-L (IN21k)</td>
<td align="center">160k</td>
<td align="center">48.1</td>
<td align="center">34.2</td>
<td align="center">54.5</td>
<td align="center">48267279</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/maskformer/mask2former/ade20k/panoptic/maskformer2_swin_large_IN21k_384_bs16_160k/model_final_e0c58e.pkl">model</a></td>
</tr>

 <tr><td align="left"><a href="configs/ade20k/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k-rankseg.yaml">Mask2Former (200 queries) + RankSeg</a></td>
<td align="center">Swin-L (IN21k)</td>
<td align="center">160k</td>
<td align="center">48.9</td>
<td align="center">-</td>
<td align="center">56.2</td>
<td align="center">-</td>
<td align="center"><a href="https://github.com/openseg-group/RankSeg/releases/download/v1.0.0/ade20k_pan_maskformer2_swin_large_IN21k_384_bs16_160k-rankseg.pth">model</a></td>
</tr>
</tbody></table>


### Semantic Segmentation

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">iterations</th>
<th valign="bottom">mIoU</th>
<th valign="bottom">mIoU (ms+flip)</th>
<th valign="bottom">model id</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: maskformer2_swin_base_IN21k_384_bs16_160k_res640 -->
 <tr><td align="left"><a href="configs/ade20k/semantic-segmentation/swin/maskformer2_swin_base_IN21k_384_bs16_160k_res640.yaml">Mask2Former</a></td>
<td align="center">Swin-B (IN21k)</td>
<td align="center">160k</td>
<td align="center">53.9</td>
<td align="center">55.1</td>
<td align="center">48333157_5</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/maskformer/mask2former/ade20k/semantic/maskformer2_swin_base_IN21k_384_bs16_160k_res640/model_final_7e47bf.pkl">model</a></td>
</tr>

 <tr><td align="left"><a href="configs/ade20k/semantic-segmentation/swin/maskformer2_swin_base_IN21k_384_bs16_160k_res640-rankseg.yaml">Mask2Former + RankSeg</a></td>
<td align="center">Swin-B (IN21k)</td>
<td align="center">160k</td>
<td align="center">54.9</td>
<td align="center">55.6</td>
<td align="center">-</td>
<td align="center"><a href="https://github.com/openseg-group/RankSeg/releases/download/v1.0.0/ade20k_maskformer2_swin_base_IN21k_384_bs16_160k_res640-rankseg.pth">model</a></td>
</tr>

 <tr><td align="left"><a href="configs/ade20k/semantic-segmentation/swin/maskformer2_swin_base_IN21k_384_bs16_160k_res640-gt.yaml">Mask2Former + GT</a></td>
<td align="center">Swin-B (IN21k)</td>
<td align="center">160k</td>
<td align="center">68.0</td>
<td align="center">-</td>
<td align="center">-</td>
<td align="center"><a href="https://github.com/openseg-group/RankSeg/releases/download/v1.0.0/ade20k_maskformer2_swin_base_IN21k_384_bs16_160k_res640-gt.pth">model</a></td>
</tr>
<!-- ROW: maskformer2_swin_large_IN21k_384_bs16_160k_res640 -->
 <tr><td align="left"><a href="configs/ade20k/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k_res640.yaml">Mask2Former</a></td>
<td align="center">Swin-L (IN21k)</td>
<td align="center">160k</td>
<td align="center">56.1</td>
<td align="center">57.3</td>
<td align="center">48004474_0</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/maskformer/mask2former/ade20k/semantic/maskformer2_swin_large_IN21k_384_bs16_160k_res640/model_final_6b4a3a.pkl">model</a></td>
</tr>

 <tr><td align="left"><a href="configs/ade20k/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k_res640-rankseg.yaml">Mask2Former + RankSeg</a></td>
<td align="center">Swin-L (IN21k)</td>
<td align="center">160k</td>
<td align="center">56.5</td>
<td align="center">58.0</td>
<td align="center">-</td>
<td align="center"><a href="https://github.com/openseg-group/RankSeg/releases/download/v1.0.0/ade20k_maskformer2_swin_large_IN21k_384_bs16_160k_res640-rankseg.pth">model</a></td>
</tr>
</tbody></table>


## Video Instance Segmentation
### YouTubeVIS 2019

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">iterations</th>
<th valign="bottom">AP</th>
<th valign="bottom">model id</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: maskformer2_R101_bs16_50ep -->
 <tr><td align="left"><a href="configs/youtubevis_2019/video_maskformer2_R101_bs16_8ep.yaml">Mask2Former</a></td>
<td align="center">R101</td>
<td align="center">6k</td>
<td align="center">49.2</td>
<td align="center">50897581_1</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/maskformer/video_mask2former/ytvis_2019/video_maskformer2_R101_bs16_8ep/model_final_a34dca.pkl">model</a></td>
</tr>

 <tr><td align="left"><a href="configs/youtubevis_2019/video_maskformer2_R101_bs16_8ep-rankseg.yaml">Mask2Former + RankSeg</a></td>
<td align="center">R101</td>
<td align="center">6k</td>
<td align="center">50.5</td>
<td align="center">-</td>
<td align="center"><a href="https://github.com/openseg-group/RankSeg/releases/download/v1.0.0/ytvis2019_video_maskformer2_R101_bs16_8ep-rankseg.pth">model</a></td>
</tr>
<!-- ROW: maskformer2_swin_base_IN21k_384_bs16_50ep -->
 <tr><td align="left"><a href="configs/youtubevis_2019/swin/video_maskformer2_swin_base_IN21k_384_bs16_8ep.yaml">Mask2Former</a></td>
<td align="center">Swin-B (IN21k)</td>
<td align="center">6k</td>
<td align="center">59.5</td>
<td align="center">50897733_2</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/maskformer/video_mask2former/ytvis_2019/video_maskformer2_swin_base_IN21k_384_bs16_8ep/model_final_221a8a.pkl">model</a></td>
</tr>

 <tr><td align="left"><a href="configs/youtubevis_2019/swin/video_maskformer2_swin_base_IN21k_384_bs16_8ep-rankseg.yaml">Mask2Former + RankSeg</a></td>
<td align="center">Swin-B (IN21k)</td>
<td align="center">6k</td>
<td align="center">60.3</td>
<td align="center">-</td>
<td align="center"><a href="https://github.com/openseg-group/RankSeg/releases/download/v1.0.0/ytvis2019_video_maskformer2_swin_base_IN21k_384_bs16_8ep.pth">model</a></td>
</tr>
<!-- ROW: maskformer2_swin_large_IN21k_384_bs16_100ep -->
 <tr><td align="left"><a href="configs/youtubevis_2019/swin/video_maskformer2_swin_large_IN21k_384_bs16_8ep.yaml">Mask2Former (200 queries)</a></td>
<td align="center">Swin-L (IN21k)</td>
<td align="center">6k</td>
<td align="center">60.4(60.7)</td>
<td align="center">50908813_0</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/maskformer/video_mask2former/ytvis_2019/video_maskformer2_swin_large_IN21k_384_bs16_8ep/model_final_c5c739.pkl">model</a></td>
</tr>

 <tr><td align="left"><a href="configs/youtubevis_2019/swin/video_maskformer2_swin_large_IN21k_384_bs16_8ep-rankseg.yaml">Mask2Former (200 queries) + RankSeg</a></td>
<td align="center">Swin-L (IN21k)</td>
<td align="center">6k</td>
<td align="center">61.1(61.4)</td>
<td align="center">-</td>
<td align="center"><a href="https://github.com/openseg-group/RankSeg/releases/download/v1.0.0/ytvis2019_video_maskformer2_swin_large_IN21k_384_bs16_8ep-rankseg.pth">model</a></td>
</tr>
</tbody></table>

\* Upload `result.json` to the [online server](https://competitions.codalab.org/competitions/20128) to evaluate YoutubeVis2019 model. Considering the variance in result, We report the avarage result of 3 models for our methods.

## Video Semantic Segmentation
### VSPW

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">iterations</th>
<th valign="bottom">mIoU</th>
<th valign="bottom">model id</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: maskformer2_R101_bs16_50ep -->
 <tr><td align="left"><a href="configs/vspw/video_maskformer2_R101_bs16_8ep.yaml">Mask2Former</a></td>
<td align="center">R101</td>
<td align="center">6k</td>
<td align="center">45.9</td>
<td align="center">-</td>
<td align="center"><a href="https://github.com/openseg-group/RankSeg/releases/download/v1.0.0/vspw_video_maskformer2_R101_bs16_8ep.pth">model</a></td>
</tr>

 <tr><td align="left"><a href="configs/vspw/video_maskformer2_R101_bs16_8ep-rankseg.yaml">Mask2Former + RankSeg</a></td>
<td align="center">R101</td>
<td align="center">6k</td>
<td align="center">47.0</td>
<td align="center">-</td>
<td align="center"><a href="https://github.com/openseg-group/RankSeg/releases/download/v1.0.0/vspw_video_maskformer2_R101_bs16_8ep-rankseg.pth">model</a></td>
</tr>

 <tr><td align="left"><a href="configs/vspw/video_maskformer2_R101_bs16_8ep-gt.yaml">Mask2Former + GT</a></td>
<td align="center">R101</td>
<td align="center">6k</td>
<td align="center">62.3</td>
<td align="center">-</td>
<td align="center"><a href="https://github.com/openseg-group/RankSeg/releases/download/v1.0.0/vspw_video_maskformer2_R101_bs16_8ep-gt.pth">model</a></td>
</tr>

<!-- ROW: maskformer2_swin_base_IN21k_384_bs16_50ep -->
 <tr><td align="left"><a href="configs/vspw/swin/video_maskformer2_swin_base_IN21k_384_bs16_8ep.yaml">Mask2Former</a></td>
<td align="center">Swin-B (IN21k)</td>
<td align="center">6k</td>
<td align="center">59.4</td>
<td align="center">-</td>
<td align="center">-</td>
</tr>

 <tr><td align="left"><a href="configs/vspw/swin/video_maskformer2_swin_base_IN21k_384_bs16_8ep-rankseg.yaml">Mask2Former + RankSeg</a></td>
<td align="center">Swin-B (IN21k)</td>
<td align="center">6k</td>
<td align="center">60.1</td>
<td align="center">-</td>
<td align="center">-</td>
</tr>
</tbody></table>

\* Considering the variance in result, We report the avarage result of 3 models for baseline and our methods.


