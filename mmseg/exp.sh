#!/bin/bash

# NOTE: this script should be run from inside the container!
#
# In entrypoint of AML experiment, invoke this script like:
# bash exp.sh path/to/config.py --auto-resume

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
echo TEST
ls /luoxiao_storage
ls /luoxiao_storage/dataset
ls /luoxiao_storage/dataset/original_cityscapes_minimal

# Change to the mounted path inside the container
if [[ "$@" =~ .*ade20kfull.* ]]; then
    ROOT=/luoxiao_storage/dataset/ade20k_full
elif [[ "$@" =~ .*ade20k.* ]]; then
    ROOT=/luoxiao_storage/dataset/ADEChallengeData2016
elif [[ "$@" =~ .*cityscapes.* ]]; then
    ROOT=/luoxiao_storage/dataset/original_cityscapes_minimal
elif [[ "$@" =~ .*pascal_context.* ]]; then
    ROOT=/luoxiao_storage/dataset/VOCdevkit/VOC2010
elif [[ "$@" =~ .*coco-stuff10k.* ]]; then
    ROOT=/luoxiao_storage/dataset/coco_stuff_10k/coco_stuff10k
elif [[ "$@" =~ .*lvis.* ]]; then
    ROOT=/luoxiao_storage/dataset/coco
else
    echo unsupported config $@
    exit 1
fi

echo $ROOT

echo bash tools/dist_train.sh "$@" --auto-resume\
    --options \
    data.train.data_root=$ROOT \
    data.test.data_root=$ROOT \
    data.val.data_root=$ROOT
bash tools/dist_train.sh "$@" --auto-resume\
    --options \
    data.train.data_root=$ROOT \
    data.test.data_root=$ROOT \
    data.val.data_root=$ROOT