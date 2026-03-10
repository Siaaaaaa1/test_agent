import os
from huggingface_hub import snapshot_download

# 1. 环境变量设置（使用镜像站）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 2. 待下载的模型列表
model_ids = [
    "Qwen/Qwen3-VL-4B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-9B"
]

# 3. 基础保存路径
base_dir = "./models"
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

print(f"开始批量下载任务，目标镜像站: {os.environ['HF_ENDPOINT']}\n")

for model_id in model_ids:
    # 自动处理文件夹名称，例如 Qwen/Qwen3-VL-4B-Instruct -> ./models/Qwen3-VL-4B-Instruct
    local_dir_name = model_id.split('/')[-1]
    local_dir = os.path.join(base_dir, local_dir_name)
    
    print(f"正在下载: {model_id} ...")
    print(f"保存路径: {local_dir}")
    
    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
            etag_timeout=100
        )
        print(f"✅ {model_id} 下载完成！\n")
    except Exception as e:
        print(f"❌ {model_id} 下载出错: {e}\n")

print("所有任务处理完毕！")