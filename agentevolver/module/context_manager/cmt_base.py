from agentevolver.schema.trajectory import Reward, Trajectory
from typing import List, Dict
import uuid as uuid_gen


class ContextManagerBase:
    """
    上下文管理基类 (ContextManagerBase)
    这是一个抽象基类，提供了一个处理和管理消息上下文的框架。
    它定义了处理初始输入、准备 LLM 上下文、检查 Token 数量安全、准备环境交互、保存 LLM/环境输出以及生成日志和分词的标准接口。
    具体的策略（如线性上下文、带思维链的上下文）需要继承此类并实现这些方法。

    Methods:
        - save_init_input: 保存初始输入。
        - prepare_next_llm_context: 为 LLM 准备下一轮的上下文。
        - check_context_token_num_safe: 检查上下文 Token 数量是否安全（未超限）。
        - prepare_world_interaction: 准备与环境的交互动作。
        - save_llm_output: 保存 LLM 的输出。
        - save_env_output: 保存环境的输出。
        - remove_last_context: 移除最后的上下文。
        - generate_log: 生成日志。
        - group_tokenize: 对一组消息进行分词。
    """

    def save_init_input(self, init_input_arr: List):
        """
        保存初始输入数组。此方法应由子类实现。
        通常用于初始化对话历史，例如保存 System Prompt 和 User Query。

        Args:
            init_input_arr (List): 要保存的初始输入数组（通常是消息列表）。

        Raises:
            NotImplementedError: 此方法需要在子类中实现。
        """
        raise NotImplementedError

    def prepare_next_llm_context(self, **kwargs) -> List:
        """
        为 LLM 准备下一轮的上下文 (Prompt)。此方法应由子类实现。
        子类需要根据当前的对话历史，构建适合 LLM 输入的消息列表（可能包含 System, User, Assistant, Tool 等角色的消息）。

        Args:
            **kwargs: 上下文准备可能需要的任意关键字参数。

        Returns:
            List: 准备好的 LLM 上下文列表。

        Raises:
            NotImplementedError: 此方法需要在子类中实现。
        """
        raise NotImplementedError

    def prepare_world_interaction(self, **kwargs) -> str:
        """
        准备与环境交互的内容（动作）。此方法应由子类覆盖以提供具体实现。
        通常是从 LLM 的输出中提取代码块或 API 调用指令，准备发送给环境。

        Args:
            **kwargs: 子类用于自定义环境交互准备的任意关键字参数。

        Returns:
            str: 表示准备好的交互内容或状态的字符串。
        """
        raise NotImplementedError

    def save_llm_output(self, llm_output, **kwargs):
        """
        保存语言模型 (LLM) 的输出。此方法必须由子类实现。
        将 LLM 生成的回复追加到对话历史中。

        Args:
            llm_output: 语言模型生成的输出（通常包含 content 和 role）。
            **kwargs: 子类可能使用的其他关键字参数。

        Raises:
            NotImplementedError: 如果方法未被子类覆盖。
        """
        raise NotImplementedError

    def save_env_output(self, env_output, **kwargs):
        """
        保存环境的输出。此方法应由子类实现。
        将环境执行动作后的反馈（Observation）追加到对话历史中。

        Args:
            env_output: 需要保存的环境输出（通常包含 content 和 role，可能还有 tool_calls）。
            **kwargs: 保存输出可能需要的其他关键字参数。

        Raises:
            NotImplementedError: 如果方法未被子类覆盖。
        """
        raise NotImplementedError

    def group_tokenize(self):
        """
        对一组轨迹进行分词的占位符方法。此方法应在子类中实现。
        用于将完整的对话历史转换为 Token ID 序列，以便进行训练或评估。

        Raises:
            NotImplementedError: 如果方法未被子类覆盖。
        """
        raise NotImplementedError



class ExtendedMessage:
    """
    扩展消息类 (ExtendedMessage)
    封装了一条消息的详细信息，不仅包括内容和角色，还包括 Token 信息、作者身份、剪裁逻辑以及用于训练的 Loss Mask 计算。
    """

    def __init__(
            self,
            author,
            role="assistant",
            content="",
            token_arr=[],
            token_begin_index=-1,
            token_end_index=-1,
            clip=False,
            clip_token_limit=8192,
            tokenizer=None,
            token_generator="manual",
            build_from_uuid="",
            uuid=None,
        ):
        """
        初始化 ExtendedMessage 对象，设置与消息及其 Token 相关的各种属性。

        Args:
            author (str): 消息的作者（例如 "llm", "user", "env", "system"）。用于确定是否计算 Loss。
            role (str, optional): 消息发送者的角色（例如 "assistant", "user"）。默认为 "assistant"。
            content (str, optional): 消息的内容文本。默认为 ""。
            token_arr (list, optional): Token 数组。默认为 []。
            token_begin_index (int, optional): Token 的起始索引（在整个序列中）。默认为 -1。
            token_end_index (int, optional): Token 的结束索引。默认为 -1。
            clip (bool, optional): 是否剪裁内容（如果过长）。默认为 False。
            clip_token_limit (int, optional): 剪裁时的最大 Token 数限制。默认为 8192。
            tokenizer (Tokenizer, optional): 用于分词的分词器。默认为 None。
            token_generator (str, optional): 生成 Token 的方法（"manual" 手动传入或 "auto" 自动生成）。默认为 "manual"。
            build_from_uuid (str, optional): 该消息构建源的 UUID。默认为 ""。
            uuid (str, optional): 消息的唯一标识符。默认为 None（自动生成）。
        """
        self.author = author
        self.role = role
        self.content = content
        self.token_arr = token_arr
        self.token_begin_index = token_begin_index
        self.token_end_index = token_end_index
        # 使用属性来确保在使用前内容是安全的（经过处理的）
        self._content_for_future = ""
        self._info = ""
        self.clip = clip
        if uuid is None:
            self.uuid = uuid_gen.uuid4().hex  # 如果未提供，生成随机 UUID
        else:
            self.uuid = uuid
        self.build_from_uuid = build_from_uuid

        if not clip:
            self.generate_content_for_future(tokenizer=None, clip=False)
        else:
            self.generate_content_for_future(tokenizer=tokenizer, clip=True, clip_token_limit=clip_token_limit)  # ⭐ 生成未来使用的内容，根据需要进行剪裁
        self.eos_token_id = tokenizer.eos_token_id
        
        # 如果指定自动生成 Token，则调用 tokenizer 计算增量 Token
        if token_generator == 'auto':
            dummy_msg = [ {"role": "assistant",  "content": "dummy text"} ]
            self.token_arr, _ = self.get_inc_simple(
               text_frag_from=tokenizer.apply_chat_template(dummy_msg, tokenize=False),
               text_frag_to=tokenizer.apply_chat_template(dummy_msg +
                    [ {"role": self.role,  "content": self.content_for_future} ], tokenize=False),
               tokenizer=tokenizer
            )  # ⭐ 自动为消息生成 Token 数组


    @property
    def content_for_future(self):
        """
        属性：返回用于未来使用的内容。如果内容为空，设置一个默认值。
        这确保了即使原始内容为空（这可能会导致某些模型出错），也有占位符文本。

        Returns:
            str: 用于未来使用的内容。
        """
        if self._content_for_future == "":
            self._content_for_future = "(Empty Content)"
            # raise ValueError("content_for_future is not set, or previous llm output is empty!")
        return self._content_for_future  # ⭐ 确保内容不为空


    @property
    def need_training(self):
        """
        属性：根据作者判断该消息是否需要进行训练（计算 Loss）。
        
        策略：
        - NEED_TRAIN_AUTHORS: 需要训练的角色（通常只有 "llm"）。
        - NON_TRAIN_AUTHORS: 不需要训练的角色（"env", "user" 等）。
        
        Returns:
            bool: 如果消息需要训练返回 True，否则返回 False。
        """
        NEED_TRAIN_AUTHORS = ["llm"]
        NON_TRAIN_AUTHORS = ["env", "initialization", "user", "memory", "llm(do_not_train)"]
        assert (self.author in NEED_TRAIN_AUTHORS) or (self.author in NON_TRAIN_AUTHORS) or (self.author.endswith('(discard)')), f"author {self.author} is not identified"
        return (self.author in NEED_TRAIN_AUTHORS)  # ⭐ 根据作者判断是否需要训练


    def generate_content_for_future(self, tokenizer, clip, clip_token_limit=-1):
        """
        生成适合未来使用的内容版本，可能会为了适应 Token 限制而进行剪裁。
        
        这通常用于处理过长的环境观测结果（Observation），避免超出模型上下文窗口。

        Args:
            tokenizer (Tokenizer): 用于计算内容 Token 数的分词器。
            clip (bool): 标志位，指示如果超出 Token 限制是否进行剪裁。
            clip_token_limit (int, optional): 允许的最大 Token 数。默认为 -1（不剪裁）。

        Returns:
            None: 结果存储在 `_content_for_future` 属性中。
        """
        _content: str = self.content
        if clip:
            assert clip_token_limit > 0, "clip_token_limit must be set when clip is True"
            n_token = len(tokenizer(_content, return_tensors="pt", padding=False)["input_ids"][0])  # ⭐ 计算内容中的 Token 数量
            if n_token > clip_token_limit:
                n_char = len(_content)  # 例如 10,000 字符
                eps = 100   # 预留 Token 缓冲
                # 估算保留比例：(限制 - 缓冲) / 总 Token 数
                preserve_percent = (clip_token_limit - eps) / n_token  # 例如 3900 / 8000
                n_char_to_preserve = int(n_char * preserve_percent)
                # 截断字符串并添加省略号标记
                _content = _content[:n_char_to_preserve] + "... truncate ..."  # ⭐ 截断内容并添加省略号
        self._content_for_future = _content


    def get_loss_mask(self, blackout_token_combo):
        """
        为 Token 数组生成 Loss Mask (损失掩码)。
        Mask 为 1 表示该位置计算 Loss，为 0 表示忽略。
        
        逻辑：
        1. 如果需要训练 (need_training=True)：初始化全 1。
           - 将特定的 Token 组合 (blackout_token_combo) 设为 0（如某些特殊的 Prompt 标记）。
           - 将 EOS Token 之后的所有位置设为 0（除了 EOS 本身可能保留）。
        2. 如果不需要训练：全 0。

        Args:
            blackout_token_combo (list): 在第一次遇到时应该被抹黑（设为 0）的 Token ID 列表。

        Returns:
            list: 二进制掩码列表。
        """
        def blackout_specific_token_ids_first_encounter(mask, arr, token_ids):
            """
            在掩码中抹黑特定 Token ID 序列的第一次出现。
            """
            index = find_sublist_indices(arr, token_ids, reverse=False)
            if index >= 0:
                for i in range(index, index+len(token_ids)): mask[i] = 0  # ⭐ 抹黑特定的 Token IDs
            return mask

        def blackout_everything_after_eos_but_keep_eos(mask, token_arr, eos_token_id):
            """
            抹黑 EOS Token 之后的所有内容，但保留 EOS Token 本身。
            """
            eos_position = token_arr.index(eos_token_id) if eos_token_id in token_arr else -1
            if eos_position != -1:
                for i in range(eos_position + 1, len(mask)):
                    mask[i] = 0  # ⭐ 抹黑 EOS 之后的所有内容
            return mask

        if self.need_training:
            msg_token_mask = [1] * len(self.token_arr)
            msg_token_mask = blackout_specific_token_ids_first_encounter(msg_token_mask, self.token_arr, blackout_token_combo)
            msg_token_mask = blackout_everything_after_eos_but_keep_eos(mask=msg_token_mask, token_arr=self.token_arr, eos_token_id=self.eos_token_id)
            return msg_token_mask
        else:
            msg_token_mask = [0] * len(self.token_arr)
            return msg_token_mask

    def get_inc_simple(self, text_frag_from, text_frag_to, tokenizer):
        """
        获取从 `text_frag_from` 到 `text_frag_to` 的增量 Token 数组。
        
        用于计算当向对话历史添加新消息时，新增了哪些 Token。

        Args:
            text_frag_from (str): 原始文本片段。
            text_frag_to (str): 目标文本片段（通常是原始片段 + 新消息）。
            tokenizer: 用于将文本转换为 Token 的分词器。

        Returns:
            tuple: 包含增量 Token ID 列表和一条调试消息的元组。
        """
        tokenizer_output = tokenizer(text_frag_from, return_tensors="pt", padding=False)
        tokenizer_input_ids = tokenizer_output["input_ids"][0].tolist()
        token_ids_acc = tokenizer_input_ids  # ⭐ 累积原始文本的 Token ID

        tokenizer_output = tokenizer(text_frag_to, return_tensors="pt", padding=False)
        input_ids = tokenizer_output["input_ids"][0].tolist()
        # 获取新增的 Token (切片操作)
        input_id_increment = input_ids[len(token_ids_acc):]  # ⭐ 获取此步骤中添加的新 Token
        
        # 计算重叠长度（仅用于调试或验证）
        overlap_length = 0
        for i in range(len(token_ids_acc)):
            if i < len(token_ids_acc) and input_ids[i] == token_ids_acc[i]: overlap_length += 1
            else: break
        msg = f"previous token length: {len(token_ids_acc)}, overlap token length: {(overlap_length)}, increment token length: {len(input_id_increment)}"
        # print(msg)
        return input_id_increment, msg


def find_sublist_indices(large_list, small_list, reverse=False):
    """
    在 `large_list` 中查找 `small_list` 第一次出现的起始索引。
    如果 `reverse` 为 True，则从 `large_list` 的末尾开始搜索。

    Args:
        large_list (list): 要在其中搜索的列表。
        small_list (list): 要搜索的子列表。
        reverse (bool, optional): 如果为 True，则反向搜索。默认为 False。

    Returns:
        int: `small_list` 在 `large_list` 中第一次出现的起始索引，如果未找到则返回 -1。
    """
    small_len = len(small_list)
    if reverse:
        for i in reversed(range(len(large_list) - small_len + 1)):
            if large_list[i: i+small_len] == small_list:  # ⭐ 检查当前切片是否匹配 small_list
                return i
    for i in range(len(large_list) - small_len + 1):
        if large_list[i: i+small_len] == small_list:  # ⭐ 检查当前切片是否匹配 small_list
            return i
    return -1


def replace_token_ids(place_holder, replace_with, begin, end):
    """
    用 `replace_with` 替换 `place_holder` 中由 `begin` 和 `end` 标记确定的片段。
    如果 `replace_with` 中包含 `begin` 和 `end` 标记，则确保它们不包含在最终结果中（即替换掉包括标记在内的整个区间）。

    Args:
        place_holder (list): 将发生替换的原始列表。
        replace_with (list): 用于替换 `begin` 和 `end` 之间片段的列表。
        begin (list): 指示要替换片段开始的标记。
        end (list): 指示要替换片段结束的标记。

    Returns:
        list: 替换后的修改列表。
    """
    # 找到 begin 标记之后的索引
    _begin_index = find_sublist_indices(place_holder, begin) + len(begin)  # ⭐ 找到 `begin` 标记之后的位置
    # 找到 end 标记的索引（反向搜索）
    _end_index = find_sublist_indices(place_holder, end, reverse=True)  # ⭐ 找到 `end` 标记的位置

    # 如果替换内容包含结束标记，移除它
    if replace_with[-len(end):] == end:  # remove end token
        replace_with = replace_with[:-len(end)]
    # 如果替换内容包含开始标记，移除它
    if replace_with[:len(begin)] == begin:  # remove begin token
        replace_with = replace_with[len(begin):]

    # 构建最终列表：前半部分 + 替换内容 + 后半部分
    final = place_holder[:_begin_index] + replace_with + place_holder[_end_index:]  # ⭐ 构建最终替换后的列表
    return final