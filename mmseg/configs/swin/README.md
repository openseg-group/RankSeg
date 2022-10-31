# MLSeg + Swin

## Results and models

* Batch-size of all models is set as 16 following the original setting of Swin. All models are trained on 8 V-100 GPUs.

| Method | Dataset | Backbone | Crop Size | Lr schd |  mIoU | mIoU(ms+flip) | config | download |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Swin | COCO-Stuff | Swin-B | 512x512 | 40000 | 45.7 | 47.2 | - | -
| Swin + MLSeg | COCO-Stuff | Swin-B | 512x512 | 40000 | 46.6 | 47.9 | - | -
| Swin (Official) | ADE20K | Swin-B | 512x512 | 160000 | 50.8 | 52.4 | [config](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/swin/upernet_swin_base_patch4_window12_512x512_160k_ade20k_pretrain_384x384_22K.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/swin/upernet_swin_base_patch4_window12_512x512_160k_ade20k_pretrain_384x384_22K/upernet_swin_base_patch4_window12_512x512_160k_ade20k_pretrain_384x384_22K_20210531_125459-429057bf.pth)
| Swin + MLSeg | ADE20K | Swin-B | 512x512 | 160000 | 51.4 | 53.0 | [config](upernet_swin_mlseg_base_patch4_window12_512x512_160k_ade20k_pretrain_384x384_22K.py) | -
| Swin | COCO+LVIS | Swin-B | 512x512 | 160000 | 20.3 | - | - | -
| Swin + MLSeg | COCO+LVIS | Swin-B | 512x512 | 160000 | 20.8 | - | - | -
