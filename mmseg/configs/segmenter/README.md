# RankSeg + Segmenter

## Results and models

* Batch-size of all models is set as 8 following the original setting of Segmenter. All models are trained on 8 V-100 GPUs.
* Multi-Scale test is not conducted on ADE20KFull and COCO+LVIS datasets because of memory limits.

| Method | Dataset | Backbone | Crop Size | Lr schd |  mIoU | mIoU(ms+flip) | config | download |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Segmenter | COCO-Stuff | ViT-B | 512x512 | 40000 | 41.9 | 43.8 | [config](segmenter-ori_vit-b16_512x512_40k_coco-stuff10k.py) | [ckpt](https://github.com/openseg-group/RankSeg/releases/download/v1.0.2/segmenter_vitb_cocostuff_baseline.pth)
| Segmenter + RankSeg | COCO-Stuff | ViT-B | 512x512 | 40000 | 44.9 | 46.2 | [config](segmenter-ori_rankseg_vit-b16_512x512_40k_coco-stuff10k.py) | [ckpt](https://github.com/openseg-group/RankSeg/releases/download/v1.0.2/segmenter_vitb_cocostuff_rankseg.pth)
| Segmenter | COCO-Stuff | ViT-B | 512x512 | 80000 | 43.4 | 45.2 | [config](segmenter-ori_vit-b16_512x512_80k_coco-stuff10k.py) | [ckpt](https://github.com/openseg-group/RankSeg/releases/download/v1.0.2/segmenter_vitb_cocostuff_baseline_80k.pth)
| Segmenter + RankSeg | COCO-Stuff | ViT-B | 512x512 | 80000 | 45.7 | 46.7 | [config](segmenter-ori_rankseg_vit-b16_512x512_80k_coco-stuff10k.py) | [ckpt](https://github.com/openseg-group/RankSeg/releases/download/v1.0.2/segmenter_vitb_cocostuff_rankseg_80k.pth)
| Segmenter | COCO-Stuff | ViT-L | 640x640 | 40000 | 45.5 | 47.1 | [config](segmenter-ori_vit-l16_640x640_40k_coco-stuff10k.py) | [ckpt](https://github.com/openseg-group/RankSeg/releases/download/v1.0.2/segmenter_vitl_cocostuff_baseline.pth)
| Segmenter + RankSeg | COCO-Stuff | ViT-L | 640x640 | 40000 | 46.7 | 47.9 | [config](segmenter-ori_rankseg_vit-l16_640x640_40k_coco-stuff10k.py) | [ckpt](https://github.com/openseg-group/RankSeg/releases/download/v1.0.2/segmenter_vitl_cocostuff_rankseg.pth)
| Segmenter | Pascal-Context60 | ViT-B | 480x480 | 80000 | 53.8 | 54.6 | [config](segmenter-ori_vit-b16_480x480_80k_pascal-context.py) | -
| Segmenter + RankSeg | Pascal-Context60 | ViT-B | 480x480 | 80000 | 54.7 | 55.4 | [config](segmenter-ori_rankseg_vit-b16_480x480_80k_pascal-context.py) | [ckpt](https://github.com/openseg-group/RankSeg/releases/download/v1.0.2/segmenter_vitb_pascal_rankseg.pth) 
| Segmenter | ADE20K | ViT-B | 512x512 | 160000 | 48.8 | 50.7 | [config](segmenter-ori_vit-b16_512x512_160k_ade20k.py) | [ckpt](https://github.com/openseg-group/RankSeg/releases/download/v1.0.2/segmenter_vitb_ade20k_baseline.pth)
| Segmenter + RankSeg | ADE20K | ViT-B | 512x512 | 160000 | 49.7 | 51.4 | [config](segmenter-ori_rankseg_vit-b16_512x512_160k_ade20k.py) | [ckpt](https://github.com/openseg-group/RankSeg/releases/download/v1.0.2/segmenter_vitb_ade20k_rankseg.pth)
| Segmenter | ADE20K | ViT-L | 640x640 | 160000 | 52.0 | 53.6 | [config](segmenter-ori_vit-l16_640x640_160k_ade20k.py) | [ckpt (official)](https://www.rocq.inria.fr/cluster-willow/rstrudel/segmenter/checkpoints/ade20k/seg_large_mask_640/checkpoint.pth)
| Segmenter + RankSeg | ADE20K | ViT-L | 640x640 | 160000 | 52.6 | 54.4 | [config](segmenter-ori_rankseg_vit-b16_512x512_160k_ade20k.py) | [ckpt](https://github.com/openseg-group/RankSeg/releases/download/v1.0.2/segmenter_vitl_ade20k_rankseg.pth)
| Segmenter | ADE20KFull | ViT-B | 512x512 | 160000 | 17.8 | - | - | -
| Segmenter + RankSeg | ADE20KFull | ViT-B | 512x512 | 160000 | 18.8 | - | - | -
| Segmenter | COCO+LVIS | ViT-B | 512x512 | 320000 | 19.4 | - | - | -
| Segmenter + RankSeg | COCO+LVIS | ViT-B | 512x512 | 320000 | 21.3 | - | - | -
| Segmenter | COCO+LVIS | ViT-B | 640x640 | 320000 | 23.7 | - | - | -
| Segmenter + RankSeg | COCO+LVIS | ViT-B | 640x640 | 320000 | 24.6 | - | - | -


### Ablation Experiments

#### Ground-Truth Experiments
| Method | Dataset | Backbone | Crop Size | Lr schd | mIoU | config | download |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Segmenter + RankSeg + GT | COCO-Stuff | ViT-B | 512x512 | 40000 | 66.8 | [config](segmenter-ori_rankseg_gt_vit-b16_512x512_40k_coco-stuff10k.py) | [ckpt](https://github.com/openseg-group/RankSeg/releases/download/v1.0.2/segmenter_vitb_cocostuff_gt.pth)
| Segmenter + RankSeg + GT | Pascal-Context60 | ViT-B | 480x480 | 80000 | 70.8 | [config](segmenter-ori_rankseg_vit-b16_480x480_80k_pascal-context.py) | [ckpt](https://github.com/openseg-group/RankSeg/releases/download/v1.0.2/segmenter_vitb_pascal_gt.pth)
| Segmenter + RankSeg + GT | ADE20K | ViT-B | 512x512 | 160000 | 63.6 | [config](segmenter-ori_rankseg_gt_vit-b16_512x512_160k_ade20k.py) | [ckpt](https://github.com/openseg-group/RankSeg/releases/download/v1.0.2/segmenter_vitb_ade_gt.pth)
| Segmenter + RankSeg + GT| ADE20KFull | ViT-B | 512x512 | 160000 | 37.0 | - | -
| Segmenter + RankSeg + GT| COCO+LVIS | ViT-B | 512x512 | 320000 | 46.8 | - | -

#### Ablation of Multi-Label Classification Head
| Method | Dataset | Backbone | Crop Size | Lr schd | mIoU | config | download |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 TranEnc Layer | COCO-Stuff | ViT-B | 512x512 | 40000 | 44.9 | [config](segmenter-ori_rankseg_vit-b16_512x512_40k_coco-stuff10k.py) | [ckpt](https://github.com/openseg-group/RankSeg/releases/download/v1.0.2/segmenter_vitb_cocostuff_rankseg.pth)
| 2 TranDec Layers | COCO-Stuff | ViT-B | 512x512 | 40000 | 44.1 | [config](segmenter-ori_rankseg_vit-b16_512x512_40k_coco-stuff10k.py) | [ckpt](https://github.com/openseg-group/RankSeg/releases/download/v1.0.2/segmenter_vitb_cocostuff_transdec.pth)
| Global Pooling | COCO-Stuff | ViT-B | 512x512 | 40000 | 43.2 | [config](segmenter-ori_rankseg_vit-b16_512x512_40k_coco-stuff10k.py) | [ckpt](https://github.com/openseg-group/RankSeg/releases/download/v1.0.2/segmenter_vitb_cocostuff_pooling.pth)

#### Ablation of Class Embedding from Backbone
We additionally found that the performance of Segmenter and Segmenter + RankSeg may be furtherly improved if we introduce class embeddings before the last layer of the backbone and process it jointly with patch encodings. 

We mark it as OCE(Optimized Class Embeddings) and list the related result here. 

| Method | Dataset | Backbone | Crop Size | Lr schd | mIoU | config | download |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Segmenter + OCE | COCO-Stuff | ViT-B | 512x512 | 40000 | 44.1 | [config](segmenter-ori_backbone-cls-emb_vit-b16_512x512_40k_coco-stuff10k.py) | -
| Segmenter + RankSeg + OCE | COCO-Stuff | ViT-B | 512x512 | 40000 | 46.0 | [config](segmenter-ori_rankseg_backbone-cls-emb_vit-b16_512x512_40k_coco-stuff10k.py) | -
| Segmenter + OCE | ADE20K | ViT-B | 512x512 | 160000 | 48.8 | [config](segmenter-ori_backbone-cls-emb_vit-b16_512x512_160k_ade20k.py) | -
| Segmenter + RankSeg + OCE | ADE20K | ViT-B | 512x512 | 160000 | 50.1 | [config](segmenter-ori_rankseg_backbone-cls-emb_vit-b16_512x512_160k_ade20k.py) | -
