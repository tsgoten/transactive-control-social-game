#!/bin/bash

UNAME=tc
DOCKER_BUILDKIT=1 docker build . -t tc-temp --build-arg UID=$(id -u) --build-arg UNAME=$UNAME
WORKDIR=/home/$UNAME

GPU_FLAGS=""
if command -v nvidia-smi &> /dev/null
then
  GPU_FLAGS="--gpus=all"
fi

docker run -it $GPU_FLAGS --rm --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --net=host -v "$(pwd):$WORKDIR" tc-temp
docker tag tc-temp lucasspangher/socialgame_v1:austin
docker push lucasspangher/socialgame_v1:austin