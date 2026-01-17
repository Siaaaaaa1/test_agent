ray stop --force 2>/dev/null
pkill -9 -f ray
pkill -9 -f vllm
pkill -9 -f python
export SETUPTOOLS_USE_DISTUTILS=local
# ray start --head --disable-usage-stats --num-cpus=192

# export RAY_ADDRESS="auto" 
# export MASTER_ADDRESS=$(ip route get 1.1.1.1 | grep -oP 'src \K\S+')
# export RAY_TEMP_DIR=/tmp/ray_local

# unset http_proxy
# unset https_proxy
# unset HTTP_PROXY
# unset HTTPS_PROXY
# export NO_PROXY=localhost,127.0.0.1,::1
export http_proxy=http://hk-mmhttpproxy.woa.com:11113
export https_proxy=http://hk-mmhttpproxy.woa.com:11113

bash ./Our_examples/run_api_driven_0113.sh