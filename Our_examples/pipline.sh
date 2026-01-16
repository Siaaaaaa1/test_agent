ray stop --force 2>/dev/null

# ray start --head --disable-usage-stats --num-cpus=192

# export RAY_ADDRESS="auto" 

# export RAY_TEMP_DIR=/tmp/ray_local

# unset http_proxy
# unset https_proxy
# unset HTTP_PROXY
# unset HTTPS_PROXY
# export NO_PROXY=localhost,127.0.0.1,::1
export http_proxy=http://hk-mmhttpproxy.woa.com:11113
export https_proxy=http://hk-mmhttpproxy.woa.com:11113

bash ./Our_examples/run_api_driven_0113.sh