# agentevolver/preprocess/generators.py

import json
import os
import re
import time
from tqdm import tqdm
from appworld import AppWorld, load_task_ids
from agentevolver.client.llm_client import DashScopeClient
from agentevolver.preprocess.prompts import APP_SELECTION_SYSTEM_PROMPT, APP_SELECTION_USER_TEMPLATE

# ... (保留 extract_json_from_str 和 ToolManualGenerator 类不变) ...

def extract_json_from_str(text: str):
    """
    从字符串中提取 JSON (对象或数组)。
    优先尝试直接解析，失败则尝试正则提取第一个 {...} 或 [...] 块。
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 尝试提取 JSON 对象或数组
    pattern = r"(\{.*\}|\[.*\])"
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        json_str = match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
            
    raise ValueError(f"无法从输出中提取有效的 JSON: {text[:200]}...")


class ToolManualGenerator:
    # ... (保持原有的 ToolManualGenerator 代码不变) ...
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def _classify_action_type(self, api_name: str) -> str:
        api_name_lower = api_name.lower()
        info_keywords = [
            "get", "list", "show", "search", "find", "read", "check", 
            "count", "is_", "peek", "status", "help", "view"
        ]
        if any(api_name_lower.startswith(k) for k in info_keywords):
            return "Informational Action"
        return "Executive Action"

    def generate(self, filename="appworld_tool_manual.json"):
        print(f"\n[ToolManualGenerator] 正在生成工具手册...")
        output_path = os.path.join(self.output_dir, filename)
        # 初始化环境以读取文档
        # 这里使用一个简单的 try-catch 防止 task_id 加载失败
        try:
            task_ids = load_task_ids("train")
            world = AppWorld(task_id=task_ids[0])
        except Exception as e:
            print(f"❌ 初始化 AppWorld 失败: {e}")
            return

        manual_data = {}

        try:
            raw_apps_output = world.execute("print(apis.api_docs.show_app_descriptions())")
            apps_list = extract_json_from_str(raw_apps_output)

            for app in tqdm(apps_list, desc="解析应用文档"):
                app_name = app['name']
                manual_data[app_name] = {
                    "description": app['description'],
                    "apis": {}
                }

                api_list_cmd = f"print(apis.api_docs.show_api_descriptions(app_name='{app_name}'))"
                raw_apis_output = world.execute(api_list_cmd)
                api_descriptions = extract_json_from_str(raw_apis_output)

                for api_short_desc in api_descriptions:
                    api_name_token = api_short_desc.split(" : ")[0] if " : " in api_short_desc else api_short_desc
                    
                    api_doc_cmd = f"print(apis.api_docs.show_api_doc(app_name='{app_name}', api_name='{api_name_token}'))"
                    raw_doc_output = world.execute(api_doc_cmd)
                    
                    try:
                        api_doc = extract_json_from_str(raw_doc_output)
                    except ValueError as e:
                        print(f"⚠️ 跳过 {app_name}.{api_name_token}: 文档解析失败")
                        continue

                    action_type = self._classify_action_type(api_name_token)

                    manual_data[app_name]["apis"][api_name_token] = {
                        "call_name": f"apis.{app_name}.{api_name_token}",
                        "action_type": action_type,
                        "description": api_doc.get("description", ""),
                        "parameters": api_doc.get("parameters", []),
                        "returns": api_doc.get("response_schemas", {})
                    }
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(manual_data, f, indent=4, ensure_ascii=False)
            print(f"✅ 工具手册已生成: {output_path}")

        except Exception as e:
            print(f"❌ 生成手册时出错: {e}")
        finally:
            world.close()


class TaskAppLabeler:
    """
    功能 2: 读取 Task 并调用 LLM 标注所需 App。
    支持按 Dataset Split (train, dev, test) 分别存储文件。
    """
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        if not os.environ.get("DASHSCOPE_API_KEY"):
            print("⚠️ 警告: 未检测到 DASHSCOPE_API_KEY，LLM 调用将失败。")
        
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
            raw_output = world.execute("print(apis.api_docs.show_app_descriptions())")
            apps_data = extract_json_from_str(raw_output)
            apps_list = [app['name'] for app in apps_data]
            return json.dumps(apps_list) 
        finally:
            world.close()

    def run(self, splits=["train", "dev", "test"], filename_prefix="task_app_labels"):
        """
        按 split 分别生成文件，例如:
        task_app_labels_train.json
        task_app_labels_dev.json
        """
        print(f"\n[TaskAppLabeler] 开始标注任务 (Splits: {splits})...")
        if not self.client:
            print("❌ 无法运行: LLM Client 未就绪。")
            return

        # 1. 准备上下文 (只需获取一次)
        try:
            apps_context = self._get_apps_context()
        except Exception as e:
            print(f"❌ 获取 App 列表失败: {e}")
            return

        # 2. 遍历所有指定的 Split
        for split in splits:
            try:
                task_ids = load_task_ids(split)
            except Exception:
                print(f"⚠️ 跳过 split '{split}': 无法加载或不存在。")
                continue
                
            print(f"处理 {split} 集，共 {len(task_ids)} 个任务...")
            split_results = []
            
            for tid in tqdm(task_ids, desc=f"Labeling {split}"):
                try:
                    world = AppWorld(task_id=tid)
                    query = world.task.instruction
                    world.close()

                    messages = [
                        {"role": "system", "content": APP_SELECTION_SYSTEM_PROMPT},
                        {"role": "user", "content": APP_SELECTION_USER_TEMPLATE.format(
                            apps_context=apps_context,
                            query=query
                        )}
                    ]

                    response = self.client.chat(messages, sampling_params={"stream": False})
                    cleaned_resp = response.replace("```json", "").replace("```", "").strip()
                    
                    try:
                        needed_apps = extract_json_from_str(cleaned_resp)
                        if not isinstance(needed_apps, list):
                            needed_apps = [str(needed_apps)]
                    except Exception:
                        needed_apps = ["PARSE_ERROR", cleaned_resp]

                    split_results.append({
                        "TaskID": tid,
                        "Query": query,
                        "NeededApps": needed_apps
                    })

                except Exception as e:
                    split_results.append({"TaskID": tid, "Error": str(e)})

            # 3. 单独保存该 Split 的结果
            output_filename = f"{filename_prefix}_{split}.json"
            output_path = os.path.join(self.output_dir, output_filename)
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(split_results, f, indent=4, ensure_ascii=False)
            
            print(f"✅ {split} 集标注完成，结果已保存: {output_path}")

        print("\n✨ 所有指定的数据集处理完毕。")