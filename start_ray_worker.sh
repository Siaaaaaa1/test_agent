#!/bin/bash

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

export http_proxy=http://hk-mmhttpproxy.woa.com:11113
export https_proxy=http://hk-mmhttpproxy.woa.com:11113
export HTTP_PROXY=$http_proxy
export HTTPS_PROXY=$https_proxy
export NCCL_SOCKET_IFNAME=bond1
export GLOO_SOCKET_IFNAME=bond1
export TP_SOCKET_IFNAME=bond1
HOST_IP=$(hostname -I | awk '{print $1}')
export no_proxy="localhost,127.0.0.1,::1,0.0.0.0,29.209.112.175,$HOST_IP,29.0.0.0/8,10.0.0.0/8,172.16.0.0/12,.woa.com"
export NO_PROXY=$no_proxy

export RAY_worker_register_timeout_seconds=600
export RAY_maximum_startup_concurrency=16

ray stop --force
pkill -f "env_service.env_service"
rm -rf /tmp/ray/*

ray start --address='29.209.112.175:6379' \
    --num-cpus=64 \
    --num-gpus=8 \
    --disable-usage-stats