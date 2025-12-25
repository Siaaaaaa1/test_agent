# agentevolver/preprocess/main.py

import os
import sys

# 强制要求以模块方式运行
if __package__ is None:
    print("❌ 错误: 请以模块方式运行此脚本。")
    print("✅ 正确用法: 在项目根目录下运行: python -m agentevolver.preprocess.main")
    sys.exit(1)

try:
    from agentevolver.preprocess.generators import ToolManualGenerator, TaskAppLabeler
except ImportError as e:
    print(f"❌ 导入模块失败: {e}")
    print("请确保已安装 appworld 并正确配置了环境。")
    sys.exit(1)

def main():
    print("🚀 AppWorld 数据预处理流水线启动")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, "output")
    
    # 1. 生成工具手册
    # 注意：这会依赖 generators.py 中设置的 path
    manual_gen = ToolManualGenerator(output_dir=output_dir)
    manual_gen.generate(filename="appworld_tool_manual.json")

    # 2. 标注任务
    # 根据您提供的文件列表，我们指定具体的 split 名称
    # train.txt -> "train"
    # dev.txt -> "dev"
    # test_challenge.txt -> "test_challenge"
    # test_normal.txt -> "test_normal"
    target_splits = ["train", "dev", "test_challenge", "test_normal"]
    
    labeler = TaskAppLabeler(output_dir=output_dir)
    labeler.run(
        splits=target_splits, 
        filename_prefix="task_app_labels"
    )

    print("\n✨ 所有任务执行完毕。")

if __name__ == "__main__":
    main()