#!/usr/bin/env python
# coding: utf-8

import re
from typing import List

from swift.plugin.orm import ORM


class SpatialReasoningAccuracyORM(ORM):
    """
    Reward function for spatial reasoning tasks that checks correctness of answers
    """

    def __call__(self, completions, answer=None, **kwargs) -> List[float]:
        """
        Args:
            completions (list[str]): Generated outputs from model
            answer (list[str]): Ground truth answers from dataset
        Returns:
            list[float]: Reward scores (1.0 for correct, 0.0 for incorrect)
        """
        rewards = []

        if answer is None:
            # 如果没有提供答案，则只返回格式化奖励
            return self.format_reward(completions)

        for content, sol in zip(completions, answer):
            reward = 0.0

            try:
                # 提取模型生成的答案
                extracted_answer = self.extract_answer(content)

                # 对比答案（不区分大小写）
                if extracted_answer.upper() == sol.upper():
                    reward = 1.0

                # 如果模型提供了推理过程，给予额外奖励
                if "<reasoning>" in content and "</reasoning>" in content:
                    reward += 0.1

            except Exception as e:
                print(f"Error in reward calculation: {e}")

            rewards.append(min(1.0, reward))  # 奖励上限为1.0

        return rewards

    def extract_answer(self, text: str) -> str:
        """从生成文本中提取答案（A/B/C/D）"""
        # 尝试从XML标签中提取
        answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
        if answer_match:
            answer = answer_match.group(1).strip()
            # 如果答案包含选项标识，只返回选项
            for option in ["A", "B", "C", "D"]:
                if option in answer:
                    return option
            return answer

        # 尝试匹配"答案是X"模式
        for pattern in [r'答案是\s*([A-D])', r'answer is\s*([A-D])', r'选择\s*([A-D])']:
            match = re.search(pattern, text)
            if match:
                return match.group(1)

        # 最后尝试直接查找选项
        for option in ["A", "B", "C", "D"]:
            if f" {option} " in text or f"{option}." in text or text.strip().endswith(option):
                return option

        return ""

    def format_reward(self, completions) -> List[float]:
        """基于格式提供小额奖励"""
        rewards = []

        for content in completions:
            reward = 0.0

            # 检查是否有推理过程
            if "<reasoning>" in content and "</reasoning>" in content:
                reward += 0.25

            # 检查是否有答案标签
            if "<answer>" in content and "</answer>" in content:
                reward += 0.25

            rewards.append(reward)

        return rewards


# 将这个函数作为独立模块使用
if __name__ == "__main__":
    # 测试函数
    reward_func = SpatialReasoningAccuracyORM()
    completions = ["<reasoning>我看到图形在旋转</reasoning><answer>A</answer>"]
    answers = ["A"]
    print(reward_func(completions, answers))
