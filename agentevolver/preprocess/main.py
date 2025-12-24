# agentevolver/preprocess/main.py

import os
import sys

# --- 动态添加项目根目录到 sys.path ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from agentevolver.preprocess.generators import ToolManualGenerator, TaskAppLabeler
except ImportError as e:
    print(f"❌ 导入模块失败: {e}")
    print("请确保已安装 appworld 并正确配置了 agentevolver 路径。")
    sys.exit(1)

def main():
    print("🚀 AppWorld 数据预处理流水线启动")
    
    # 默认输出目录
    output_dir = os.path.join(current_dir, "output")
    
    # 1. 生成工具手册
    # 包含 API 分类 (Executive/Informational)
    manual_gen = ToolManualGenerator(output_dir=output_dir)
    manual_gen.generate(filename="appworld_tool_manual.json")

    # 2. 标注任务
    # 读取所有任务，调用 LLM 识别所需 App
    # 默认处理 'train' 和 'dev' (test 集通常无标准答案，视需求可加)
    labeler = TaskAppLabeler(output_dir=output_dir)
    labeler.run(splits=["train", "dev", "test"], filename="task_app_labels.json")

    print("\n✨ 所有任务执行完毕。")

if __name__ == "__main__":
    main()