import torch
import copy
import re
import json
import random
import time
from typing import List, Callable
from agentevolver.schema.trajectory import Sample
from best_logger import print_dict, print_listofdict
from agentevolver.module.context_manager.cmt_linear_think import ExtendedMessage, Linear_CMT, LinearThinkCMT
from agentevolver.module.context_manager.cmt_linear import find_sublist_indices, replace_token_ids
from best_logger import register_logger, print_dict, print_nested, NestedJsonItem, SeqItem
from textwrap import dedent
from openai import OpenAI
from loguru import logger


def construct_alien_llm_chat_fn(config, rollout_config):
    """
    构造用于上下文管理的“异形”LLM (Alien LLM) 的聊天函数。
    这个模型专门用于执行“元认知”任务（如判断哪些上下文重要、压缩文本），
    而不是用于生成 Agent 的主要回复。这样可以避免污染主要训练流程，或者利用更廉价的模型。
    """
    def alien_llm_chat_fn(messages, request_id=""):
        max_try = 4
        # 从配置中获取异形模型的名称和响应长度限制
        alien_model_name = config.actor_rollout_ref.rollout.context_template_alien_llm_model
        alien_model_response_length = config.actor_rollout_ref.rollout.context_template_alien_model_response_length
        
        # 定义 API Key 列表，包含常规 Key 和备用 Key，用于负载均衡和容错
        regular_key_list = ["sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"]
        backup_key_list = ["sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"]
        
        for n_try in range(max_try):
            try:
                # 简单的负载均衡策略：前几次尝试使用常规 Key，中间尝试使用备用 Key，最后混合使用
                if n_try < max_try // 2:
                    api_key=random.choice(regular_key_list)
                elif n_try == max_try // 2:
                    api_key=random.choice(backup_key_list)
                else:
                    api_key=random.choice(regular_key_list + backup_key_list)
                
                # 初始化 OpenAI 客户端（这里指向阿里云 DashScope 的兼容接口）
                client = OpenAI(
                    api_key=api_key,
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                )
                
                # 设置采样参数，temperature=0 确保输出的确定性
                sampling_params = dict(
                    n=1,
                    max_completion_tokens=alien_model_response_length,
                )
                sampling_params["temperature"] = 0
                
                # 发送请求
                completion = client.chat.completions.create(
                    model=alien_model_name,
                    messages=messages,
                    extra_body=sampling_params
                )
                
                # 解析响应
                message = completion.choices[0].message.model_dump(exclude_unset=True, exclude_none=True)
                if "content" not in message: message["content"] = ""
                return {"role": message["role"], "content": message['content']}
            except Exception as e:
                # 异常处理与重试机制
                logger.bind(exception=True).exception(f"Error calling alien llm: {e}")
                time.sleep(5)
                print(f"Error calling alien llm: {e}, retrying...")
        
        # 如果所有尝试都失败，抛出运行时错误
        raise RuntimeError(f"Failed to get response from alien llm after {max_try} attempts")
    return alien_llm_chat_fn


class SelfContextClipCMT(LinearThinkCMT):
    """
    非线性上下文管理器模板 (Non-linear Context Manager Template)。
    它不仅处理 LLM 和环境之间的对话流，还具备自我剪裁上下文的能力。
    继承自 LinearThinkCMT，支持思维链 (Thinking Process)。
    """

    def __init__(self, config, tokenizer, llm_chat_fn):
        self.llm_chat_fn = llm_chat_fn
        # 构造用于上下文压缩的辅助 LLM 函数
        self.alien_llm_chat_fn: Callable = construct_alien_llm_chat_fn(config, config.actor_rollout_ref.rollout)
        self.latest_env_response_id = ""
        self.latest_env_response_content = ""
        self.console_debug_mode = False
        self.force_think = config.actor_rollout_ref.rollout.force_think
        self.env_feedin_preference = config.env_service.env_feedin_preference
        # 决定是否使用主模型进行上下文压缩操作（通常为 False，使用 Alien LLM）
        self.train_sp_action = config.actor_rollout_ref.rollout.context_template_train_sp_action
        self.clipped_before = False # 标记是否已经进行过剪裁
        
        # 根据环境偏好设置强制思考的 Prompt
        if self.env_feedin_preference == "box":
            # 适用于数学或逻辑题，要求用 \box{} 包裹最终答案
            self.force_think_prompt = dedent("""
                Additional requirements: Think before action! You must think step by step before your next action, and you must use <think>...</think> to wrap your thinking process before finally produce your answer with \\box{}.
                Your thought (<think>...</think>) should be as short and concise as possible.
                For example:
                <think>...your thinking process...</think>
                \\box{...your final answer...}
            """)
        elif self.env_feedin_preference == "code":
            # 适用于代码生成任务，要求输出 Python 代码块
            self.force_think_prompt = dedent("""
                Additional requirements: Think before action! You must think step by step before your next action, and you must use <think>...</think> to wrap your thinking process before finally produce the next-step action.
                Your thought (<think>...</think>) should be as short and concise as possible.
                For example:
                <think>...your thinking process...</think>
                ```python
                # your action here
                ```
            """)
        super().__init__(config, tokenizer)

    def post_tag_init_message_context(self, content, is_last) -> str:
        """
        处理初始消息内容（如 System Prompt 或 User Query）。
        如果这是最后一条消息且开启了强制思考，会追加 force_think_prompt。

        Args:
            content (str): 消息内容。
            is_last (bool): 是否是序列中的最后一条消息。

        Returns:
            str: 处理后的内容。
        """
        if is_last:
            content = content.strip()  # ⭐ 确保内容去除首尾空格
        if is_last and self.force_think:
            content += self.force_think_prompt  # ⭐ 如果是最后一条且开启强制思考，追加提示词
        return content.strip()  # ⭐ 返回最终处理后的内容

    def post_tag_env_message_context(self, content, turn, is_last) -> str:
        """
        处理环境返回的消息（Observation）。
        
        关键功能：
        1. **打标签 (Tagging)**：给每条环境消息加上 ID，例如 `[Environment Response, id=ER001]`。
           这对于后续让 LLM 引用并决定是否删除该消息至关重要。
        2. 记录最新的环境响应 ID，用于后续逻辑判断。

        Args:
            content (str): 环境消息内容。
            turn (int): 当前轮次（0-99）。
            is_last (bool): 是否是最后一条。

        Returns:
            str: 格式化并打好标签的消息内容。
        """
        from textwrap import dedent
        assert 0 <= turn < 99, "turn must be in the range [0, 99)"
        turn_id = f"{turn:03d}"
        self.latest_env_response_id = f"ER{turn_id}"  # ⭐ 更新最新的环境响应 ID
        self.latest_env_response_content = content.strip()  # ⭐ 更新最新的环境响应内容
        
        # 添加头部标签
        content = dedent(f"""
            [Environment Response, id=ER{turn_id}]
            ---
        """).strip() + '\n' + content.strip()  # ⭐ 格式化并标记消息内容
        if is_last and self.force_think:
            content += self.force_think_prompt  # ⭐ 必要时追加强制思考提示
        return content

    def post_tag_llm_message_context(self, content, turn, is_last) -> str:
        """
        处理 LLM 生成的消息（Assistant Response）。
        
        关键功能：
        1. **打标签 (Tagging)**：给每条助手消息加上 ID，例如 `[Assistant Response, id=AR001]`。

        Args:
            content (str): LLM 消息内容。
            turn (int): 轮次。
            is_last (bool): 是否是最后一条（通常 LLM 消息后会紧跟环境反馈，所以不应是最后）。

        Returns:
            str: 格式化并打好标签的消息内容。
        """
        from textwrap import dedent
        assert not is_last, "llm message should never be last"  # ⭐ 确保 LLM 消息不是最后一条
        assert 0 <= turn < 99, "turn must be in the range [0, 99)"  # ⭐ 验证轮次范围
        turn_id = f"{turn:03d}"
        content = dedent(f"""
            [Assistant Response, id=AR{turn_id}]
            ---
        """).strip() + '\n' + content.strip()  # ⭐ 添加标签和轮次标识符
        return content

    def strip_think_tags(self, text: str) -> str:
        """
        从文本中移除 <think>...</think> 标签及其包裹的思考过程内容。
        这通常用于将之前的历史对话中的思考过程压缩掉，以节省 Token，
        或者在将消息喂给 Alien LLM 进行分析时简化内容。

        Args:
            text (str): 包含 <think> 标签的文本。

        Returns:
            str: 移除思考过程后的文本。
        """
        new_ext_msg_content = re.sub(r'\<think\>.*?\<\/think\>', '', text, flags=re.DOTALL).strip()  # ⭐ 移除 <think> 标签内的内容
        new_ext_msg_content = new_ext_msg_content.replace("<think>", "")  # 移除残留的标签
        new_ext_msg_content = new_ext_msg_content.replace("</think>", "")  # 移除残留的标签
        return new_ext_msg_content

    def prepare_next_llm_context(self):
        """
        为 LLM 准备下一轮的上下文 (Prompt)。
        
        流程：
        1. 获取所有非丢弃状态的历史消息。
        2. 根据消息作者类型（LLM, Env, Init），分别调用对应的 `post_tag_xxx` 方法进行格式化和打标签。
        3. 对于历史中的旧 LLM 消息，移除其 `<think>` 思考过程以节省 Token。
        4. 将 ExtendedMessage 对象转换为 LLM 可接受的 dict 格式。

        Returns:
            list: 更新后的上下文消息字典列表。
        """
        self.latest_llm_interaction_socket = []

        # 获取所有之前的上下文（非丢弃的）
        # 顺序如：`init_message -> user -> llm -> user -> llm` 或 `init_message -> llm -> user -> llm -> user`
        self.latest_llm_interaction_socket = self.filter_context_via_authors(["initialization", "llm", "env"])  # ⭐ 过滤上下文，只保留相关作者

        env_turn = 1
        llm_turn = 1
        for index, ext_msg in enumerate(list(self.latest_llm_interaction_socket)):
            is_last = (index == len(self.latest_llm_interaction_socket) - 1)
            # 根据消息类型处理
            if ext_msg.author == "llm":
                # 如果是之前的 LLM 消息，移除 think 标签 (压缩历史)
                new_ext_msg_content = self.strip_think_tags(ext_msg.content)
                author_override = "llm(do_not_train)" # 标记为不参与训练计算 loss
                self.latest_llm_interaction_socket[index] = ExtendedMessage(
                    author=author_override,
                    role=ext_msg.role,
                    content=self.post_tag_llm_message_context(new_ext_msg_content, turn=llm_turn, is_last=is_last),
                    token_generator='auto',
                    tokenizer=self.tokenizer,
                    build_from_uuid=ext_msg.uuid,
                )
                llm_turn += 1

            # 处理环境消息
            elif ext_msg.author == "env":
                self.latest_llm_interaction_socket[index] = ExtendedMessage(
                    author=ext_msg.author,
                    role=ext_msg.role,
                    content=self.post_tag_env_message_context(content=ext_msg.content_for_future, turn=env_turn, is_last=is_last),
                    token_generator='auto',
                    tokenizer=self.tokenizer,
                    build_from_uuid=ext_msg.uuid,
                )
                env_turn += 1

            # 处理初始化消息
            elif ext_msg.author in ["initialization"]:
                self.latest_llm_interaction_socket[index] = ExtendedMessage(
                    author=ext_msg.author,
                    role=ext_msg.role,
                    content=self.post_tag_init_message_context(content=ext_msg.content_for_future, is_last=is_last),
                    token_generator='auto',
                    tokenizer=self.tokenizer,
                    build_from_uuid=ext_msg.uuid,
                )

            else:
                raise RuntimeError(f"Unknown author {ext_msg.author} in latest_llm_interaction_socket")

        listofdict_context = self.to_role_content(self.latest_llm_interaction_socket)  # ⭐ 将处理后的上下文转换为字典列表
        return listofdict_context


    def save_init_input(self, init_input_arr:list, add_nothink):
        """
        保存初始输入数组。调用父类方法。

        Args:
            init_input_arr (list): 初始输入数组。
            add_nothink: 传递给父类的额外参数。
        """
        super().save_init_input(init_input_arr, add_nothink)  # ⭐ 调用父类方法保存初始输入
        return


    def impl_new_request_from_previous_interaction(self, new_message,  this_interaction, strip_think=False):
        """
        基于先前的交互历史，发起一个新的请求（通常是给 Alien LLM）。
        这用于在不改变当前主要上下文的情况下，让模型执行一些辅助任务（如分析历史、压缩文本）。

        Args:
            new_message: 要添加到交互中的新消息（通常是 Prompt，要求 LLM 执行压缩/分析）。
            this_interaction: 先前的交互历史列表。
            strip_think (bool, optional): 是否移除历史消息中的 think 标签。默认为 False。

        Returns:
            tuple: 包含更新后的交互列表和 LLM 的输出内容。
        """
        latest_llm_interaction_socket_additional = copy.deepcopy(this_interaction)
        if strip_think:
            # 移除历史消息中的思考过程，减少 Input Token
            for index, ext_msg in enumerate(latest_llm_interaction_socket_additional):
                if ext_msg.author == "llm(do_not_train)" or ext_msg.author == "llm":
                    latest_llm_interaction_socket_additional[index] = ExtendedMessage(
                        author=ext_msg.author,
                        role=ext_msg.role,
                        content=self.strip_think_tags(ext_msg.content),
                        token_generator='auto',
                        tokenizer=self.tokenizer,
                        build_from_uuid=ext_msg.build_from_uuid if ext_msg.build_from_uuid else ext_msg.uuid,
                    )  # ⭐ 移除 content 中的 think 标签
                else:
                    continue
        
        # 添加新指令消息
        latest_llm_interaction_socket_additional += [new_message]
        dict_context = self.to_role_content(latest_llm_interaction_socket_additional)
        
        # 选择使用哪个模型进行生成
        if self.train_sp_action:
            llm_output = self.llm_chat_fn(dict_context, request_id="")  # ⭐ 使用当前的主模型生成
        else:
            llm_output = self.alien_llm_chat_fn(dict_context, request_id="")  # ⭐ 使用外部 Alien 模型生成
        
        # 记录这次辅助交互（注意：这里用的是 save_llm_output_do_not_register_full_context，避免污染主线 Full Context）
        latest_llm_interaction_socket_additional += [self.save_llm_output_do_not_register_full_context(llm_output, dict_context)]  # ⭐ 添加 LLM 输出到临时交互历史
        
        if self.train_sp_action:
            this_interaction = copy.deepcopy(latest_llm_interaction_socket_additional)
            self.grouped_steps += [this_interaction]  # ⭐ 如果是在训练 SP Action，则记录到分组步骤中
        
        # 日志记录
        if self.console_debug_mode:
            print_listofdict(
                dict_context + [{'role': 'llm_latest', 'content': llm_output['content']}], mod='c'
            )  # ⭐ 打印到控制台
        else:
            print_listofdict(
                dict_context + [{'role': 'llm_latest', 'content': llm_output['content']}], mod='env_clip'
            )  # ⭐ 记录到 env_clip 日志文件
        
        output_llm_content = llm_output['content'].strip()
        return latest_llm_interaction_socket_additional, output_llm_content


    def after_save_llm_output(self, this_interaction):
        """
        在保存 LLM 输出后调用的钩子函数。
        这是**上下文自我剪裁**的核心逻辑所在。
        
        流程：
        1. 检查当前上下文长度是否超过触发阈值 (`clip_trigger_token_num`)。
        2. 如果超过，构造一个 Prompt，要求 Alien LLM 检查历史消息（根据 ID）。
        3. Alien LLM 返回一个 JSON 列表，指示每条消息是 Keep（保留）、Remove（删除）还是 Compress（压缩）。
        4. 根据 JSON 指令修改 `self.full_context`：
           - Remove: 将消息标记为 discard。
           - Compress: 再次调用 Alien LLM 对该条特定消息进行摘要压缩。
        
        Args:
            this_interaction: 当前的完整交互历史列表。
        """
        from textwrap import dedent
        if not self.latest_env_response_id:
            return

        # 获取剪裁触发阈值
        clip_token_cnt = self.config.actor_rollout_ref.rollout.context_template_clip_trigger_token_num
        this_interaction = copy.deepcopy(this_interaction)
        
        # 检查是否达到触发条件
        if self._get_seq_length(this_interaction) < clip_token_cnt:
            return

        # 防止重复剪裁（简单策略：一次 Rollout 只剪裁一次，或者是标记位控制）
        if self.clipped_before:
            return
        self.clipped_before = True

        # 第一步：请求 Alien LLM 分析每条消息的有用性
        _, generated_content = self.impl_new_request_from_previous_interaction(
            new_message=ExtendedMessage(
                author='user',
                role='user',
                content=dedent("""
                    Your new task is to inspect each `Environment Response` and `Assistant Response` messages,
                    and determine whether each message is useful for the next-step decision-making.
                    Generate a json structure following the format below:
                    ```json
                    [
                        {"id":"ARXXX or ERXXX", "useful":true or false, "action": "keep or remove or compress"},
                        ...,
                        {"id":"ARXXX or ERXXX", "useful":true or false, "action": "keep or remove or compress"},
                    ]
                    ```

                    For example:
                    ```json
                    [
                        {"id":"ER001", "useful":true, "action": "keep"},
                        {"id":"AR001", "useful":false, "action": "remove"},
                        ...
                    ]
                    ```

                    Rules:
                    - If the message contains useful information for future decisions, set "useful":true and "action":"keep".
                    - If the message records important previous action or environment feedback, set "useful":true and "action":"keep".
                    - If the message is very long and very redundant, set "useful":true and "action":"compress".
                    - If the message is completely irrelevant, set "useful":false and "action":"remove". Note that important failures should be preserved, because learning from past is vital.
                    - Ignore messages without id=XXX tags, where XXX is a 3-digit number.
                    - Ensure the JSON is properly formatted and valid.
                    - Remove or compress at least one message, because token limit is already reached.
                    - At least remove (or compress) one message.
                    - There must be no more than 2 "compress" actions in total, because "compress" action will cost considerable amount of time.

                """),
                token_generator='auto',
                tokenizer=self.tokenizer,
            ),
            this_interaction=this_interaction,
            strip_think=True, # 给 Alien LLM 看的历史不需要包含复杂的思考过程
        )

        try:
            # 解析 Alien LLM 返回的 JSON
            llm_output_content = generated_content = generated_content.strip()
            if llm_output_content.count("```") == 2:
                extracted_content: str = llm_output_content.split("```")[1].strip()
            else:
                raise RuntimeError(f"Cannot find ``` in llm_output content: {llm_output_content}")
            if extracted_content.startswith('json'):
                extracted_content = extracted_content[len('json'):].strip()
            extracted_json = json.loads(extracted_content)
            
            # 遍历指令执行操作
            for item in extracted_json:
                if 'id' not in item or 'useful' not in item or 'action' not in item:
                    raise RuntimeError(f"Each item must contain 'id', 'useful', and 'action' fields. Error in item: {item}")
                message_id = item['id']
                message_action = item['action']
                
                # 在 this_interaction 中查找对应的 UUID
                from_uuid = None
                for ext_msg in this_interaction:
                    if message_id in ext_msg.content_for_future:
                        from_uuid = ext_msg.build_from_uuid
                        break
                if from_uuid is None:
                    raise ValueError(f"Cannot find message_id {message_id} in `this_interaction`")
                
                # 在 self.full_context 中找到对应的目标消息
                target_msg = None
                target_index = -1
                for index, msg in enumerate(self.full_context):
                    if msg.uuid == from_uuid:
                        target_msg = msg
                        target_index = index
                        break
                if target_msg is None or target_index == -1:
                    raise ValueError(f"Cannot find message_id {message_id} in full_context")

                ## 执行具体动作
                if message_action == 'remove':
                    # 标记为 discard，实际上是在 author 后加后缀，这会在 filter_context 时被过滤掉
                    self.full_context[target_index] = ExtendedMessage(
                        author=target_msg.author+"(discard)",
                        role=target_msg.role,
                        content=target_msg.content,  # 保持原始内容
                        token_generator='auto',
                        tokenizer=self.tokenizer,
                    )
                elif message_action == 'compress':
                    target_id = message_id
                    # 第二步：对于需要压缩的消息，再次发起请求让 Alien LLM 进行摘要
                    _, generated_compressed_content = self.impl_new_request_from_previous_interaction(
                        new_message=ExtendedMessage(
                            author='user',
                            role='user',
                            content=dedent(f"""
                                Your new task is to inspect {target_id}, and filter out all redundant information, and only keep the most important information that is useful for future decision-making.
                                For example, if the content is a long text with multiple paragraphs, you should only preserve the key paragraphs and use ... to replace the rest.
                                If the content is a long list of data / dict / json, you should only preserve the key items and use ... to replace the rest.
                                Be careful to preserve all information that might be useful in the future. You should at least reduce 50% of {target_id}.
                                Remember： wrap your answer with ```

                                Your response should be like:
                                ```
                                ...content after filtering...
                                ```
                            """),
                            token_generator='auto',
                            tokenizer=self.tokenizer,
                        ),
                        this_interaction=this_interaction[:-1], # 排除最新的 LLM 消息（因为通常我们压缩的是历史）
                        strip_think=True,
                    )
                    if generated_compressed_content.count("```") != 2:
                        raise RuntimeError(f"Cannot find ``` in llm_output content: {generated_compressed_content}")
                    compressed_content = generated_compressed_content.split("```")[1].strip()
                    
                    # 用压缩后的内容替换原消息
                    self.full_context[target_index] = ExtendedMessage(
                        author=target_msg.author,
                        role=target_msg.role,
                        content=compressed_content,
                        token_generator='auto',
                        tokenizer=self.tokenizer,
                    )
                elif message_action == 'keep':
                    continue
                else:
                    raise RuntimeError(f"Unknown action {message_action}, must be one of ['remove', 'keep', 'compress']")

        except Exception as e:
            logger.bind(exception=True).exception(f"Error processing llm_output: {e}")
            print(f"Error processing llm_output")
            return


    def replace_full_context_item(self, match_content: str, new_content: str):
        """
        在 full_context 中查找并替换内容。

        Args:
            match_content (str): 要匹配的内容。
            new_content (str): 新的内容。

        Returns:
            bool: 如果替换成功返回 True，否则 False。
        """
        success = False
        for index in range(len(self.full_context)):
            ext_msg = self.full_context[index]
            if match_content in ext_msg.content_for_future:
                success = True
                self.full_context[index] = ExtendedMessage(
                    author=ext_msg.author,
                    role=ext_msg.role,
                    content=new_content,
                    token_generator='auto',
                    tokenizer=self.tokenizer,
                )  # ⭐ 用新内容替换 ExtendedMessage
                # print_dict({match_content: new_content})
                return success
        return success


    def save_llm_output(self, llm_output, input_msg_ref):
        """
        保存 LLM 的输出，并触发上下文管理钩子。

        Args:
            llm_output (str): LLM 生成的输出。
            input_msg_ref (str): 触发该响应的输入消息引用。

        Returns:
            None
        """
        # 调用父类方法保存输出
        ext_msg = Linear_CMT.save_llm_output(self, llm_output, input_msg_ref)  # ⭐ 保存 LLM 输出并获取扩展消息对象
        
        # 构造当前的交互快照
        this_interaction = copy.deepcopy(self.latest_llm_interaction_socket + [ext_msg])  # ⭐ 创建包含新消息的最新交互深拷贝
        
        # 记录到 grouped_steps
        self.grouped_steps += [this_interaction]  # ⭐ 将新交互追加到分组步骤中
        
        # 触发钩子：检查是否需要进行上下文剪裁
        self.after_save_llm_output(this_interaction)  # ⭐ 调用保存后钩子进行额外处理
        
        # 重置交互 socket
        self.latest_llm_interaction_socket = []  # ⭐ 重置最新 LLM 交互 socket