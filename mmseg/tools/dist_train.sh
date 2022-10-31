#!/usr/bin/env bash

CONFIG=$1
GPUS=$(awk -F ',' '{ print NF '} <<<"${CUDA_VISIBLE_DEVICES}")
PORT=${PORT:-29500}

ARGS=()
SEEN_OPTIONS=false
for ARG in "${@:2}"; do
    if [ "$ARG"x == "--options"x ]; then
        if ! $SEEN_OPTIONS; then
            SEEN_OPTIONS=true
            ARGS+=("$ARG")
        fi
    else
        ARGS+=("$ARG")
    fi
done

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
    python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch "${ARGS[@]}"
