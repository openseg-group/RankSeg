#!/bin/bash

ROOT_DIR=$(cd $(dirname $0) &>/dev/null && pwd)

cp ${ROOT_DIR}/mmseg/ops/ssn/ docker/ -r

sudo docker build \
    -t hsfzxjy/mmseg:pytorch1.8.1-cuda10.2-cudnn7-devel \
    docker/

rm docker/ssn/ -rf
