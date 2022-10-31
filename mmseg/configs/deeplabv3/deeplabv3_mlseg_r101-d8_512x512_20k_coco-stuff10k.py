_base_ = [
    '../_base_/models/deeplabv3_r50-d8_mlseg.py',
    '../_base_/datasets/coco-stuff10k.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py'
]
model = dict(
    pretrained='open-mmlab://resnet101_v1c',
    backbone=dict(depth=101),
    decode_head=dict(num_classes=171, mlseg=dict(head=dict(num_classes=171))),
    auxiliary_head=dict(num_classes=171))
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
)
