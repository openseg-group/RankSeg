#!/bin/bash

# NOTE: this script should be run from inside the container!
# One should use docker image hsfzxjy/mmseg:pytorch1.8.1-cuda10.2-cudnn7-devel-mpi
#
# In entrypoint of AML experiment, invoke this script like:
# bash exp_mnode.sh <NNODE> <NPROC_PER_NODE> path/to/config.py --auto-resume --options data.samples_per_gpu=1
#
# e.g. for 2 nodes with 8 GPUs on each, NNODE=2 NPROC_PER_NODE=8

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

# Change to the mounted path inside the container
ADE_ROOT=/path/to/ADEChallengeData2016

bash tools/dist_train.sh "$@" \
    --options \
    data.train.data_root=$ADE_ROOT \
    data.test.data_root=$ADE_ROOT \
    data.val.data_root=$ADE_ROOT
