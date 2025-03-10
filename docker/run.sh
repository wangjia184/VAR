#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

IMAGE_URL=everymatrix.jfrog.io/emlab-docker/ayida/pytorch:2.6.0-cuda12.4-cudnn9-runtime-face128x128
docker rm -f var
docker run -ti --name var --network=host --gpus all --runtime=nvidia \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v $SCRIPT_DIR/../:/app \
    -v /mnt/e/real_faces_128/:/data \
    $IMAGE_URL $@


