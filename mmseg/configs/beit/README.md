# MLSeg + BEiT

## Results and models

* Batch-size of all models is set as 16 following the original setting of BEiT. All models are trained on 8 V-100 GPUs.

| Method | Dataset | Backbone | Crop Size | Lr schd |  mIoU | mIoU(ms+flip) | config | download |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| BEiT (Official) | ADE20K | BEiT-L | 640x640 | 160000 | 56.7 | 57.0 | [config](upernet-unilm_beit-l16_640x640_slide_320k_ade20k.py) | -
| MLSeg + BEiT | ADE20K | BEiT-L | 640x640 | 160000 | 57.0 | 57.8| [config](upernet-unilm_mlseg_beit-l16_640x640_slide_320k_ade20k.py) | -