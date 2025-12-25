# agentevolver/preprocess/generators.py

import json
import os
import re
import ast
import sys
from tqdm import tqdm

# --- [CRITICAL] 配置 AppWorld 路径 ---
# 必须在 import appworld 之前设置
CUSTOM_APPWORLD_ROOT = "./env_service/environments/appworld"

if os.path.exists(CUSTOM_APPWORLD_ROOT):
    os.environ["APPWORLD_ROOT"] = CUSTOM_APPWORLD_ROOT
else:
    print(f"⚠️ 警告: 未找到指定的 AppWorld 路径: {CUSTOM_APPWORLD_ROOT}")
# ------------------------------------

from appworld import AppWorld, load_task_ids
from agentevolver.client.llm_client import DashScopeClient
from agentevolver.preprocess.prompts import APP_SELECTION_SYSTEM_PROMPT, APP_SELECTION_USER_TEMPLATE

def extract_json_from_str(text: str):
    """
    增强版解析器：
    1. 尝试标准 JSON (json.loads)
    2. 尝试 Python 字典字面量 (ast.literal_eval) -> 解决单引号问题
    3. 尝试正则提取
    """
    text = text.strip()
    
    # 1. 尝试直接解析 JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. 尝试解析 Python 字典/列表字符串 (例如 {'a': 1})
    try:
        result = ast.literal_eval(text)
        if isinstance(result, (dict, list)):
            return result
    except (ValueError, SyntaxError):
        pass

    # 3. 尝试正则提取 JSON/Dict 块
    pattern = r"(\{.*\}|\[.*\])"
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        clean_str = match.group(1)
        # 再次尝试 JSON
        try:
            return json.loads(clean_str)
        except json.JSONDecodeError:
            pass
        # 再次尝试 ast
        try:
            result = ast.literal_eval(clean_str)
            if isinstance(result, (dict, list)):
                return result
        except (ValueError, SyntaxError):
            pass
            
    raise ValueError(f"无法解析数据: {text[:100]}...")


class ToolManualGenerator:
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
        
        try:
            task_ids = load_task_ids("train")
            if not task_ids:
                raise ValueError("未加载到任何 Task ID")
            world = AppWorld(task_id=task_ids[0])
        except Exception as e:
            print(f"❌ 初始化 AppWorld 失败: {e}")
            return

        manual_data = {}

        try:
            # 1. 获取所有 APP 列表
            raw_apps_output = world.execute("print(apis.api_docs.show_app_descriptions())")
            apps_list = extract_json_from_str(raw_apps_output)

            for app in tqdm(apps_list, desc="解析应用文档"):
                # 兼容处理：有些版本返回 dict，有些可能是字符串
                if isinstance(app, dict):
                    app_name = app.get('name')
                    app_desc = app.get('description', '')
                else:
                    print(f"⚠️ 无法解析 APP 数据结构: {app}")
                    continue

                if not app_name:
                    continue

                manual_data[app_name] = {
                    "description": app_desc,
                    "apis": {}
                }

                # 2. 获取该 APP 下的所有 API 简介
                api_list_cmd = f"print(apis.api_docs.show_api_descriptions(app_name='{app_name}'))"
                raw_apis_output = world.execute(api_list_cmd)
                
                try:
                    api_descriptions = extract_json_from_str(raw_apis_output)
                except ValueError:
                    print(f"⚠️ 跳过 {app_name}: API 列表解析失败")
                    continue

                for api_item in api_descriptions:
                    # [修复关键点] 此时 api_item 是一个字典，不是字符串
                    # 例如: {'name': 'login', 'description': 'Log in...'}
                    if isinstance(api_item, dict):
                        api_name_token = api_item.get('name')
                    elif isinstance(api_item, str):
                        # 旧逻辑兼容
                        api_name_token = api_item.split(" : ")[0]
                    else:
                        continue

                    if not api_name_token:
                        continue
                    
                    # 3. 获取具体 API 的详细文档
                    api_doc_cmd = f"print(apis.api_docs.show_api_doc(app_name='{app_name}', api_name='{api_name_token}'))"
                    raw_doc_output = world.execute(api_doc_cmd)
                    
                    try:
                        api_doc = extract_json_from_str(raw_doc_output)
                    except ValueError:
                        print(f"⚠️ 跳过 {app_name}.{api_name_token}: 文档解析失败 (Raw: {raw_doc_output[:50]}...)")
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
            import traceback
            traceback.print_exc()
        finally:
            world.close()


class TaskAppLabeler:
    """
    功能 2: 读取 Task 并调用 LLM 标注所需 App。
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
            # 这里同样使用增强版解析器
            apps_data = extract_json_from_str(raw_output)
            
            # 数据清洗：确保提取正确的 name
            apps_list = []
            for app in apps_data:
                if isinstance(app, dict) and 'name' in app:
                    apps_list.append(app['name'])
                elif isinstance(app, str):
                    apps_list.append(app)
            
            return json.dumps(apps_list) 
        finally:
            world.close()

    def run(self, splits, filename_prefix="task_app_labels"):
        print(f"\n[TaskAppLabeler] 开始标注任务 (Splits: {splits})...")
        if not self.client:
            print("❌ 无法运行: LLM Client 未就绪。")
            return

        try:
            apps_context = self._get_apps_context()
        except Exception as e:
            print(f"❌ 获取 App 列表失败: {e}")
            return

        for split in splits:
            try:
                task_ids = load_task_ids(split)
            except Exception as e:
                print(f"⚠️ 跳过 split '{split}': 无法加载 (Error: {e})。")
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

            output_filename = f"{filename_prefix}_{split}.json"
            output_path = os.path.join(self.output_dir, output_filename)
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(split_results, f, indent=4, ensure_ascii=False)
            
            print(f"✅ {split} 集标注完成，结果已保存: {output_path}")

        print("\n✨ 所有指定的数据集处理完毕。")