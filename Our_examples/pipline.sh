ray stop --force 2>/dev/null
pkill -9 -f ray
pkill -9 raylet
pkill -9 gcs_server
pkill -9 -f vllm
pkill -9 -f python
export SETUPTOOLS_USE_DISTUTILS=local
export http_proxy=http://hk-mmhttpproxy.woa.com:11113
export https_proxy=http://hk-mmhttpproxy.woa.com:11113

bash ./Our_examples/run_api_driven_H20.sh