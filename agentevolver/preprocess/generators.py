# agentevolver/preprocess/generators.py

import json
import os
import time
from tqdm import tqdm
from appworld import AppWorld, load_task_ids
from agentevolver.client.llm_client import DashScopeClient
from agentevolver.preprocess.prompts import APP_SELECTION_SYSTEM_PROMPT, APP_SELECTION_USER_TEMPLATE

class ToolManualGenerator:
    """
    功能 1: 生成 AppWorld 工具手册 (包含五元组信息: Name, Inputs, Outputs, Description, ActionType)
    """
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def _classify_action_type(self, api_name: str) -> str:
        """
        基于规则对 API 进行二分类：
        - Informational Action (资源/信息获取): get, list, show, search, check 等
        - Executive Action (锚点/副作用): create, send, delete, update, play, pay 等
        """
        api_name_lower = api_name.lower()
        
        # 信息型关键词 (只读、查询、检查)
        info_keywords = [
            "get", "list", "show", "search", "find", "read", "check", 
            "count", "is_", "peek", "status", "help", "view"
        ]
        
        # 如果名字以这些关键词开头，归为信息型
        if any(api_name_lower.startswith(k) for k in info_keywords):
            return "Informational Action"
        
        # 默认为执行型 (通常涉及状态改变，如 create, send, update, delete)
        return "Executive Action"

    def generate(self, filename="appworld_tool_manual.json"):
        print(f"\n[ToolManualGenerator] 正在生成工具手册...")
        output_path = os.path.join(self.output_dir, filename)
        
        # 初始化环境以读取文档 (使用 train 集第一个任务)
        task_ids = load_task_ids("train")
        world = AppWorld(task_id=task_ids[0])
        manual_data = {}

        try:
            # 1. 获取所有 App
            apps_json = world.execute("print(apis.api_docs.show_app_descriptions())")
            apps_list = json.loads(apps_json)

            for app in tqdm(apps_list, desc="解析应用文档"):
                app_name = app['name']
                manual_data[app_name] = {
                    "description": app['description'],
                    "apis": {}
                }

                # 2. 获取该 App 下的所有 API
                api_list_cmd = f"print(apis.api_docs.show_api_descriptions(app_name='{app_name}'))"
                api_descriptions = json.loads(world.execute(api_list_cmd))

                for api_short_desc in api_descriptions:
                    # 提取 API 名称 (格式通常为 "name : description")
                    api_name = api_short_desc.split(" : ")[0] if " : " in api_short_desc else api_short_desc
                    
                    # 3. 获取 API 详细文档 (入参、出参)
                    api_doc_cmd = f"print(apis.api_docs.show_api_doc(app_name='{app_name}', api_name='{api_name}'))"
                    api_doc = json.loads(world.execute(api_doc_cmd))

                    # 4. 判定动作类型
                    action_type = self._classify_action_type(api_name)

                    # 5. 组装数据
                    # Key 为 API 名称 (作为唯一标识)
                    manual_data[app_name]["apis"][api_name] = {
                        "call_name": f"apis.{app_name}.{api_name}",  # 完整的调用名
                        "action_type": action_type,                   # 第五个属性: 分类
                        "description": api_doc.get("description", ""),
                        "parameters": api_doc.get("parameters", []),  # 入参
                        "returns": api_doc.get("response_schemas", {}) # 出参
                    }
            
            # 保存
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(manual_data, f, indent=4, ensure_ascii=False)
            print(f"✅ 工具手册已生成: {output_path}")

        except Exception as e:
            print(f"❌ 生成手册时出错: {e}")
        finally:
            world.close()


class TaskAppLabeler:
    """
    功能 2: 读取 Task 并调用 LLM 标注所需 App
    """
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # 检查 API Key，若无则仅仅打印警告，初始化时不崩溃
        if not os.environ.get("DASHSCOPE_API_KEY"):
            print("⚠️ 警告: 未检测到 DASHSCOPE_API_KEY，LLM 调用将失败。")
        
        # 初始化 Client
        try:
            self.client = DashScopeClient(model_name="qwen-plus", temperature=0.0)
        except Exception as e:
            print(f"Client 初始化失败: {e}")
            self.client = None

    def _get_apps_context(self):
        """获取环境中所有 App 名称的列表字符串"""
        task_ids = load_task_ids("train")
        world = AppWorld(task_id=task_ids[0])
        try:
            apps_json = world.execute("print(apis.api_docs.show_app_descriptions())")
            apps_list = [app['name'] for app in json.loads(apps_json)]
            return json.dumps(apps_list) # 返回 JSON 格式的字符串列表
        finally:
            world.close()

    def run(self, splits=["train", "dev", "test"], filename="task_app_labels.json"):
        print(f"\n[TaskAppLabeler] 开始标注任务 (Splits: {splits})...")
        if not self.client:
            print("❌ 无法运行: LLM Client 未就绪。")
            return

        output_path = os.path.join(self.output_dir, filename)

        # 1. 准备上下文
        try:
            apps_context = self._get_apps_context()
        except Exception as e:
            print(f"❌ 获取 App 列表失败: {e}")
            return

        all_results = []

        # 2. 遍历所有指定的 Split
        for split in splits:
            try:
                task_ids = load_task_ids(split)
            except Exception:
                print(f"⚠️ 跳过 split '{split}': 无法加载或不存在。")
                continue
                
            print(f"处理 {split} 集，共 {len(task_ids)} 个任务...")
            
            for tid in tqdm(task_ids, desc=f"Labeling {split}"):
                try:
                    # 获取 Query
                    world = AppWorld(task_id=tid)
                    query = world.task.instruction
                    world.close()

                    # 构建 Prompt
                    messages = [
                        {"role": "system", "content": APP_SELECTION_SYSTEM_PROMPT},
                        {"role": "user", "content": APP_SELECTION_USER_TEMPLATE.format(
                            apps_context=apps_context,
                            query=query
                        )}
                    ]

                    # 调用 LLM
                    response = self.client.chat(messages, sampling_params={"stream": False})
                    
                    # 简单清洗
                    cleaned_resp = response.replace("```json", "").replace("```", "").strip()
                    
                    try:
                        needed_apps = json.loads(cleaned_resp)
                        # 确保是列表
                        if not isinstance(needed_apps, list):
                            needed_apps = [str(needed_apps)]
                    except json.JSONDecodeError:
                        needed_apps = ["PARSE_ERROR", cleaned_resp]

                    # 构造返回对象
                    all_results.append({
                        "TaskID": tid,
                        "Query": query,
                        "NeededApps": needed_apps
                    })

                except Exception as e:
                    all_results.append({"TaskID": tid, "Error": str(e)})

        # 3. 保存
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=4, ensure_ascii=False)
        print(f"✅ 任务标注完成，结果已保存: {output_path}")