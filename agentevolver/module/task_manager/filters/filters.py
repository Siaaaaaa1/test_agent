import abc
from typing import Sequence

from agentevolver.module.task_manager.filters import TaskPostFilter
from agentevolver.schema.task import TaskObjective


class NaiveTaskPostFilter(TaskPostFilter):
    """
    基础后置过滤器：基于置信度排序以及字符相似度去重。
    """
    def filter(self, tasks: Sequence[TaskObjective]) -> list[TaskObjective]:
        """
        对提取出来的 RL 训练任务按置信度排序，并通过计算 Query 的词汇交并比进行去重。
        """
        tasks = list(tasks)
        tasks.sort(key=lambda x: x.confidence or 0, reverse=True)  # ⭐ 按置信度降序排序

        unique_tasks = []
        seen_queries = set()

        for i, task in enumerate(tasks):
            query = task.objective
            assert query is not None
            normalized_query = (
                query.lower().strip()
            )  # FIXME: this only supports English

            is_duplicate = False
            # 遍历已经保留的 Query 集合，检查当前句是否和之前的高度雷同
            for seen_query in seen_queries:
                if self._check_similarity(normalized_query, seen_query):
                    is_duplicate = True
                    break

            # ⭐ 只有非重复且拥有标准参考答案 (ground_truth) 的才被放行
            if task.ground_truth != "" and not is_duplicate:
                unique_tasks.append(task)  
                seen_queries.add(normalized_query)

        return unique_tasks

    def _check_similarity(
        self, query1: str, query2: str, threshold: float = 0.8
    ) -> bool:
        """
        计算两个句子的词汇级别交并比 (IoU)，判断是否雷同。
        """
        words1 = set(query1.split())
        words2 = set(query2.split())

        if not words1 or not words2:
            return False

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        # ⭐ 计算相似度分数：交集大小 / 并集大小
        similarity = len(intersection) / len(union) if union else 0  
        return similarity >= threshold  # ⭐ 若大于等于设定的阈值 (默认 0.8)，判定为雷同