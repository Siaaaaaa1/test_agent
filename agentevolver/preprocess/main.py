# agentevolver/preprocess/main.py

import os
import sys

# --- 移除旧的路径注入代码 (sys.path hack) ---
# 强制要求以模块方式运行，以保证包结构引用的正确性
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
    
    # 获取当前文件所在目录作为基准
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, "output")
    
    # 1. 生成工具手册
    # 包含 API 分类 (Executive/Informational)
    manual_gen = ToolManualGenerator(output_dir=output_dir)
    manual_gen.generate(filename="appworld_tool_manual.json")

    # 2. 标注任务
    # 读取所有任务，调用 LLM 识别所需 App
    # 默认处理 'train' 和 'dev'
    labeler = TaskAppLabeler(output_dir=output_dir)
    labeler.run(splits=["train", "dev", "test"], filename="task_app_labels.json")

    print("\n✨ 所有任务执行完毕。")

if __name__ == "__main__":
    main()