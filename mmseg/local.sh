#!/bin/bash

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

IMAGE="hsfzxjy/mmseg:pytorch1.8.1-cuda10.2-cudnn7-devel"

TRY_CONFIG=$1
if [[ "$TRY_CONFIG"x =~ ^configs/.* ]]; then
    TASK_IS_EXPERIMENT=true
else
    TASK_IS_EXPERIMENT=false
fi

ARGS=("$@")
QUOTED_ARGS=$(printf "'%s' " "${ARGS[@]}")

read -d '' CMD <<EOF
export CUDA_VISIBLE_DEVICES=2,3
ln -s /mdata/ade /data/ade
ln -s /mdata/cityscapes /data/cityscapes
ln -s /mdata/VOCdevkit /data/VOCdevkit
ln -s /mdata/coco_stuff10k /data/coco_stuff10k
ln -s /mdata/ade20k_full /data/ade20k_full
ln -s /mdata/lvis /data/lvis

cd /workspace
if $TASK_IS_EXPERIMENT; then
    bash tools/dist_train.sh $QUOTED_ARGS
else
    $QUOTED_ARGS
fi
EOF

if [ $# == 0 ]; then
    CMD=bash
fi

if [ -t 1 ]; then
    FLAG="-it"
fi

echo "=========== COMMAND ==========="
echo "$CMD"
echo "==============================="

sudo nvidia-docker run \
    --rm --ipc=host ${FLAG} \
    -v "$PWD":/workspace \
    -v ~/openseg_blob_fix/dataset/ADEChallengeData2016:/mdata/ade/ADEChallengeData2016/ \
    -v ~/openseg_blob_fix/dataset/original_cityscapes:/mdata/cityscapes/ \
    -v ~/openseg_blob_fix/dataset/VOCdevkit:/mdata/VOCdevkit/ \
    -v ~/openseg_blob_fix/dataset/coco_stuff_10k/coco_stuff10k:/mdata/coco_stuff10k/ \
    -v ~/openseg_blob_fix/dataset/ade20k_full:/mdata/ade20k_full \
    -v ~/openseg_blob_fix/dataset/coco:/mdata/lvis \
    -v /mnt:/mnt \
    -u $(id -u):$(id -g) \
    "${IMAGE}" \
    bash -c "$CMD"
