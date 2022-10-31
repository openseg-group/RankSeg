#!/usr/bin/env bash

HERE=$(dirname "$0")

NNODE=$1 && echo NNODE: $NNODE
shift

NPROC_PER_NODE=$1 && echo NPROC_PER_NODE: $NPROC_PER_NODE
shift

CONFIG=$1
echo CONFIG: $CONFIG
shift

export NCCL_SOCKET_IFNAME=ib0
export MKL_THREADING_LAYER=GNU
export PYTHONPATH="$HERE/..":$PYTHONPATH

MASTER_IP=${MASTER_IP}
MASTER_PORT=12345
NODE_RANK=${OMPI_COMM_WORLD_RANK} && echo NODE_RANK: ${NODE_RANK}

ARGS=()
SEEN_OPTIONS=false
for ARG in "${@:1}"; do
    if [ "$ARG"x == "--options"x ]; then
        if ! $SEEN_OPTIONS; then
            SEEN_OPTIONS=true
            ARGS+=("$ARG")
        fi
    else
        ARGS+=("$ARG")
    fi
done

python -m torch.distributed.launch \
    --nnodes=$NNODE \
    --node_rank=$NODE_RANK \
    --nproc_per_node=$NPROC_PER_NODE \
    --master_addr=$MASTER_IP \
    --master_port=$MASTER_PORT \
    $HERE/train.py $CONFIG --launcher pytorch "${ARGS[@]}"
