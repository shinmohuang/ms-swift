import os
import re
from typing import Dict, List, Union

import json

from swift.llm import InferRequest


class ORM:

    def __call__(self, **kwargs) -> List[float]:
        raise NotImplementedError


class ReactORM(ORM):

    @staticmethod
    def evaluate_action_reward(action_pred: list, action_ref: list, cand_list: list, ref_list: list):
        f1 = []
        for i in range(len(action_pred)):
            ref_action = action_ref[i]
            pred_action = action_pred[i]

            ref_input = ref_list[i]
            cand_input = cand_list[i]

            ref_is_json = False
            try:
                ref_input_json = json.loads(ref_input)
                ref_is_json = True
            except Exception:
                ref_input_json = ref_input

            cand_is_json = False
            try:
                cand_input_json = json.loads(cand_input)
                cand_is_json = True
            except Exception:
                cand_input_json = cand_input

            if ref_action != pred_action or (ref_is_json ^ cand_is_json):
                f1.append(0)
            elif not ref_is_json and not cand_is_json:
                rougel = ReactORM.evaluate_rougel([ref_input_json], [cand_input_json])
                if rougel is None or rougel < 10:
                    f1.append(0)
                elif 10 <= rougel < 20:
                    f1.append(0.1)
                else:
                    f1.append(1)
            else:
                if not isinstance(ref_input_json, dict) or not isinstance(cand_input_json, dict):
                    # This cannot be happen, but:
                    # line 62, in evaluate_action_reward
                    # for k, v in ref_input_json.items():
                    # AttributeError: 'str' object has no attribute 'items'
                    # print(f'>>>>>>ref_input_json: {ref_input_json}, cand_input_json: {cand_input_json}')
                    f1.append(0)
                    continue

                half_match = 0
                full_match = 0
                if ref_input_json == {}:
                    if cand_input_json == {}:
                        f1.append(1)
                    else:
                        f1.append(0)
                else:
                    for k, v in ref_input_json.items():
                        if k in cand_input_json.keys():
                            if cand_input_json[k] == v:
                                full_match += 1
                            else:
                                half_match += 1

                    recall = (0.5 * half_match + full_match) / (len(ref_input_json) + 1e-30)
                    precision = (0.5 * half_match + full_match) / (len(cand_input_json) + 1e-30)
                    try:
                        f1.append((2 * recall * precision) / (recall + precision))
                    except Exception:
                        f1.append(0.0)

        if f1[0] == 1.0:
            return True
        else:
            return False

    @staticmethod
    def parse_action(text):
        if 'Action Input:' in text:
            input_idx = text.rindex('Action Input:')
            action_input = text[input_idx + len('Action Input:'):].strip()
        else:
            action_input = '{}'

        if 'Action:' in text:
            action_idx = text.rindex('Action:')
            action = text[action_idx + len('Action:'):].strip()
            if 'Action Input:' in action:
                input_idx = action.index('Action Input:')
                action = action[:input_idx].strip()
        else:
            action = 'none'
        return action, action_input

    @staticmethod
    def parse_output(text):
        action, action_input = ReactORM.parse_action(text)
        return action, action_input

    def __call__(self, infer_requests: List[Union[InferRequest, Dict]], solution: List[str], **kwargs) -> List[float]:
        rewards = []
        if not isinstance(infer_requests[0], str):
            predictions = [request['messages'][-1]['content'] for request in infer_requests]
        else:
            predictions = infer_requests
        for prediction, ground_truth in zip(predictions, solution):
            if prediction.endswith('Observation:'):
                prediction = prediction[:prediction.index('Observation:')].strip()
            action_ref = []
            action_input_ref = []
            action_pred = []
            action_input_pred = []
            reference = ground_truth
            prediction = prediction.replace('<|endoftext|>', '').replace('<|im_end|>', '').strip()
            ref_action, ref_input = ReactORM.parse_output(reference)
            pred_action, pred_input = ReactORM.parse_output(prediction)
            action_ref.append(ref_action)
            action_input_ref.append(ref_input)
            if pred_action is None:
                action_pred.append('none')
            else:
                action_pred.append(pred_action)

            if pred_input is None:
                action_input_pred.append('{}')
            else:
                action_input_pred.append(pred_input)

            reward = ReactORM.evaluate_action_reward(action_pred, action_ref, action_input_pred, action_input_ref)
            rewards.append(float(reward))
        return rewards

    @staticmethod
    def evaluate_rougel(cand_list: list, ref_list: list):
        if len(ref_list) == 0:
            return None
        try:
            from rouge import Rouge
            rouge = Rouge()
            rouge_score = rouge.get_scores(hyps=cand_list, refs=ref_list, avg=True)
            rougel = rouge_score['rouge-l']['f']
            return rougel
        except Exception:
            return None


class MathORM(ORM):

    def __init__(self):
        from transformers.utils import strtobool
        self.use_opencompass = strtobool(os.environ.get('USE_OPENCOMPASS_EVALUATOR', 'False'))
        if self.use_opencompass:
            from opencompass.datasets.math import MATHEvaluator
            self.evaluator = MATHEvaluator()

    @staticmethod
    def check_terminate(answers: Union[str, List[str]]) -> List[bool]:
        if isinstance(answers, str):
            answers = [answers]
        results = []
        for answer in answers:
            results.append('\\boxed' in answer)
        return results

    @staticmethod
    def extract_boxed_result(text):
        pattern = r'\\boxed{([^}]*)}'
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
        else:
            return text

    @staticmethod
    def clean_latex(latex_str):
        latex_str = re.sub(r'\\\(|\\\)|\\\[|\\]', '', latex_str)
        latex_str = latex_str.replace('}}', '}').replace('{', '').replace('}', '')
        return latex_str.strip()

    @staticmethod
    def parse_expression(latex_str):
        from sympy import simplify
        from sympy.parsing.latex import parse_latex
        try:
            expr = parse_latex(latex_str)
            return simplify(expr)
        except Exception:
            return None

    @staticmethod
    def compare_consecutive(first, second):
        cleaned_list = [MathORM.clean_latex(latex) for latex in [first, second]]
        parsed_exprs = [MathORM.parse_expression(latex) for latex in cleaned_list]
        if hasattr(parsed_exprs[0], 'equals') and hasattr(parsed_exprs[1], 'equals'):
            value = parsed_exprs[0].equals(parsed_exprs[1])
        else:
            value = parsed_exprs[0] == parsed_exprs[1]
        if value is None:
            value = False
        return value

    def __call__(self, infer_requests: List[Union[InferRequest, Dict]], ground_truths: List[str],
                 **kwargs) -> List[float]:
        rewards = []
        predictions = [request['messages'][-1]['content'] for request in infer_requests]
        for prediction, ground_truth in zip(predictions, ground_truths):
            if '# Answer' in prediction:
                prediction = prediction.split('# Answer')[1]
            if '# Answer' in ground_truth:
                ground_truth = ground_truth.split('# Answer')[1]
            prediction = prediction.strip()
            ground_truth = ground_truth.strip()
            prediction = MathORM.extract_boxed_result(prediction)
            ground_truth = MathORM.extract_boxed_result(ground_truth)
            if self.use_opencompass:
                reward = self.evaluator.is_equiv(prediction, ground_truth)
            else:
                reward = MathORM.compare_consecutive(prediction, ground_truth)
            rewards.append(float(reward))
        return rewards


class MathAccuracy(ORM):

    def __init__(self):
        import importlib.util
        assert importlib.util.find_spec('math_verify') is not None, (
            "The math_verify package is required but not installed. Please install it using 'pip install math_verify'.")

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        from latex2sympy2_extended import NormalizationConfig
        from math_verify import LatexExtractionConfig, parse, verify
        rewards = []
        for content, sol in zip(completions, solution):
            gold_parsed = parse(sol, extraction_mode='first_match')
            if len(gold_parsed) != 0:
                # We require the answer to be provided in correct latex (no malformed operators)
                answer_parsed = parse(
                    content,
                    extraction_config=[
                        LatexExtractionConfig(
                            normalization_config=NormalizationConfig(
                                nits=False,
                                malformed_operators=False,
                                basic_latex=True,
                                equations=True,
                                boxed=True,
                                units=True,
                            ),
                            # Ensures that boxed is tried first
                            boxed_match_priority=0,
                            try_extract_without_anchor=False,
                        )
                    ],
                    extraction_mode='first_match',
                )
                # edge case
                try:
                    reward = float(verify(gold_parsed, answer_parsed))
                except Exception:
                    reward = 0.0
            else:
                # If the gold solution is not parseable, we reward 0 to skip this example
                reward = 0.0
            rewards.append(reward)
        return rewards


class Format(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion has a specific format."""
        pattern = r'^<think>.*?</think>\s*<answer>.*?</answer>(?![\s\S])'
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
        return [1.0 if match else 0.0 for match in matches]


class ReActFormat(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion has a specific format."""
        pattern = r'^<think>.*?</think>\s*Action:.*?Action Input:.*?$'
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
        return [1.0 if match else 0.0 for match in matches]


class CosineReward(ORM):
    # https://arxiv.org/abs/2502.03373
    def __init__(self,
                 tokenizer=None,
                 cosine_min_len_value_wrong: float = -0.5,
                 cosine_max_len_value_wrong: float = 0.0,
                 cosine_min_len_value_correct: float = 1.0,
                 cosine_max_len_value_correct: float = 0.5,
                 cosine_max_len: int = 1000,
                 accuracy_orm=None):
        self.tokenizer = tokenizer
        self.min_len_value_wrong = cosine_min_len_value_wrong
        self.max_len_value_wrong = cosine_max_len_value_wrong
        self.min_len_value_correct = cosine_min_len_value_correct
        self.max_len_value_correct = cosine_max_len_value_correct
        self.max_len = cosine_max_len
        self.accuracy_orm = accuracy_orm or MathAccuracy()

    @staticmethod
    def cosfn(t, T, min_value, max_value):
        import math
        return max_value - (max_value - min_value) * (1 - math.cos(t * math.pi / T)) / 2

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        acc_rewards = self.accuracy_orm(completions, solution, **kwargs)
        rewards = []
        for content, acc_reward in zip(completions, acc_rewards):
            is_correct = acc_reward >= 1.
            if is_correct:
                # Swap min/max for correct answers
                min_value = self.max_len_value_correct
                max_value = self.min_len_value_correct
            else:
                min_value = self.max_len_value_wrong
                max_value = self.min_len_value_wrong
            gen_len = len(self.tokenizer.encode(content))
            reward = self.cosfn(gen_len, self.max_len, min_value, max_value)
            rewards.append(reward)
        return rewards


class RepetitionPenalty(ORM):
    # https://arxiv.org/abs/2502.03373
    def __init__(self, repetition_n_grams: int = 3, repetition_max_penalty: float = -1.0):
        self.ngram_size = repetition_n_grams
        self.max_penalty = repetition_max_penalty

    @staticmethod
    def zipngram(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    def __call__(self, completions, **kwargs) -> List[float]:
        """
        reward function the penalizes repetitions

        Args:
            completions: List of model completions
        """
        rewards = []
        for completion in completions:
            if completion == '':
                rewards.append(0.0)
                continue
            if len(completion.split()) < self.ngram_size:
                rewards.append(0.0)
                continue

            ngrams = set()
            total = 0
            for ng in self.zipngram(completion, self.ngram_size):
                ngrams.add(ng)
                total += 1

            scaling = 1 - len(ngrams) / total
            reward = scaling * self.max_penalty
            rewards.append(reward)
        return rewards


class SoftOverlong(ORM):

    def __init__(self, tokenizer, soft_max_length, soft_cache_length):
        self.tokenizer = tokenizer
        assert soft_cache_length < soft_max_length
        self.soft_max_length = soft_max_length
        self.soft_cache_length = soft_cache_length

    def __call__(self, completions, **kwargs) -> List[float]:
        rewards = []
        for completion in completions:
            completion_length = len(self.tokenizer.encode(completion))
            expected_len = self.soft_max_length - self.soft_cache_length
            exceed_len = completion_length - expected_len
            rewards.append(min(-exceed_len / self.soft_cache_length, 0))
        return rewards

class SpatialReasoningStrictFormat(ORM):
    """
    Reward function that checks if the completion has the correct format.
    Strict version requires specific line formatting.
    """

    def __call__(self, completions, **kwargs) -> List[float]:
        rewards = []

        # 处理不同格式的completions
        processed_completions = []
        for comp in completions:
            if isinstance(comp, dict) and 'content' in comp:
                processed_completions.append(comp['content'])
            elif isinstance(comp, list) and len(comp) > 0:
                if isinstance(comp[0], dict) and 'content' in comp[0]:
                    processed_completions.append(comp[0]['content'])
                else:
                    processed_completions.append(str(comp[0]))
            elif isinstance(comp, str):
                processed_completions.append(comp)
            else:
                processed_completions.append(str(comp))

        pattern = r"^<think>\n.*?\n</think>\n<response>\n.*?\n</response>\n<answer>.*?</answer>\n$"

        for content in processed_completions:
            try:
                match = re.search(pattern, content, re.DOTALL)
                if match:
                    print(f"SpatialReasoningStrictFormat - Matched content: {match.group(0)}")
                rewards.append(0.5 if match else 0.0)
            except Exception as e:
                print(f"Error in format check: {e}")
                print(f"Content type: {type(content)}, Content: {content[:100]}...")
                rewards.append(0.0)

        return rewards


class SpatialReasoningLooseFormat(ORM):
    """
    Reward function that checks if the completion has the correct format.
    Supports multiple format patterns with different reward values.
    """

    def __call__(self, completions, **kwargs) -> List[float]:
        rewards = []

        # 处理不同格式的completions
        processed_completions = []
        for comp in completions:
            try:
                if isinstance(comp, dict) and 'content' in comp:
                    processed_completions.append(comp['content'])
                elif isinstance(comp, list) and len(comp) > 0:
                    if isinstance(comp[0], dict) and 'content' in comp[0]:
                        processed_completions.append(comp[0]['content'])
                    else:
                        processed_completions.append(str(comp[0]))
                elif isinstance(comp, str):
                    processed_completions.append(comp)
                else:
                    processed_completions.append(str(comp))
            except Exception as e:
                print(f"Error processing completion: {e}")
                processed_completions.append("")

        # 定义不同的格式模式和对应的奖励值
        format_patterns = [
            # 最佳格式: <think>...</think><answer>...</answer>
            {
                'pattern': r"<think>\s*.*?\s*</think>\s*<answer>\s*.*?\s*</answer>",
                'reward': 1,
                'description': "标准标签格式"
            },
            # "The Answer is **X**" 格式
            {
                'pattern': r"The Answer is\s*\*\*\s*[A-E]\s*\*\*",
                'reward': 0.3,
                'description': "The Answer is** 格式"
            },
            # 仅包含加粗答案 **X**
            {
                'pattern': r"\*\*\s*[A-E]\s*\*\*",
                'reward': 0.2,
                'description': "加粗答案格式"
            },
            # 有推理过程但没有正确使用标签
            {
                'pattern': r"(think|reasoning|分析|推理).*?(answer|conclusion|答案|结论)",
                'reward': 0.1,
                'description': "非标准推理和答案段落"
            }
        ]

        for content in processed_completions:
            try:
                if not isinstance(content, str):
                    content = str(content)

                # 默认奖励为0
                max_reward = 0.0
                matched_format = None

                # 检查所有格式模式
                for format_spec in format_patterns:
                    match = re.search(format_spec['pattern'], content, re.DOTALL | re.IGNORECASE)
                    if match and format_spec['reward'] > max_reward:
                        max_reward = format_spec['reward']
                        matched_format = format_spec['description']

                if matched_format:
                    print(f"Matched format: {matched_format}, reward: {max_reward}")

                rewards.append(max_reward)

            except Exception as e:
                print(f"Error in format check: {e}")
                print(f"Content type: {type(content)}, Content preview: {str(content)[:100]}...")
                rewards.append(0.0)

        return rewards


class SpatialReasoningAccuracyORM(ORM):
    """
    Reward function for spatial reasoning tasks that checks correctness of answers
    """

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        """
        Args:
            completions (list): Generated outputs from model
            solution (list): Ground truth answers from dataset
        Returns:
            list[float]: Reward scores (1.0 for correct, 0.0 for incorrect)
        """
        rewards = []

        # 输入验证
        if not completions or not solution:
            print("Warning: Empty completions or solution received")
            return [0.0] * max(len(completions) if completions else 0, len(solution) if solution else 0, 1)

        # GRPO训练时，completions可能有不同的格式，需要适当处理
        processed_completions = []
        for comp in completions:
            try:
                if isinstance(comp, dict) and 'content' in comp:
                    processed_completions.append(comp['content'])
                elif isinstance(comp, list) and len(comp) > 0:
                    if isinstance(comp[0], dict) and 'content' in comp[0]:
                        processed_completions.append(comp[0]['content'])
                    else:
                        processed_completions.append(str(comp[0]))
                elif isinstance(comp, str):
                    processed_completions.append(comp)
                else:
                    # 如果无法识别格式，转换为字符串
                    processed_completions.append(str(comp))
            except Exception as e:
                print(f"Error processing completion: {e}")
                processed_completions.append("")

        # 确保solution也是字符串列表
        processed_solutions = []
        for sol in solution:
            try:
                if isinstance(sol, str):
                    processed_solutions.append(sol)
                else:
                    processed_solutions.append(str(sol))
            except Exception as e:
                print(f"Error processing solution: {e}")
                processed_solutions.append("")

        # 确保处理后的列表长度匹配
        if len(processed_completions) != len(processed_solutions):
            print(
                f"Warning: Mismatched lengths - completions: {len(processed_completions)}, solutions: {len(processed_solutions)}")
            # 补齐短的列表
            if len(processed_completions) < len(processed_solutions):
                processed_completions.extend([""] * (len(processed_solutions) - len(processed_completions)))
            else:
                processed_solutions.extend([""] * (len(processed_completions) - len(processed_solutions)))

        # 正式的奖励计算逻辑
        for content, sol in zip(processed_completions, processed_solutions):
            reward = 0.0

            try:
                # Extract answer from the solution
                try:
                    sol_match = re.search(r'<answer>(.*?)</answer>', sol, re.DOTALL)
                    if sol_match:
                        solution_answer = sol_match.group(1).strip().upper()
                        print(f"SpatialReasoningAccuracyORM - Solution match: <answer>{sol_match.group(1)}</answer>")
                    else:
                        solution_answer = sol.strip().upper()
                except Exception as e:
                    print(f"Error in solution pattern matching: {e}")
                    solution_answer = sol.strip().upper()

                # Extract answer from completion using multiple methods in priority order
                completion_answer = ""

                # Method 1: Try standard tag format first
                try:
                    ans_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
                    if ans_match:
                        completion_answer = ans_match.group(1).strip().upper()
                        print(
                            f"SpatialReasoningAccuracyORM - Completion match (tag): <answer>{ans_match.group(1)}</answer>")
                except Exception as e:
                    print(f"Error in standard tag format matching: {e}")

                # Method 2: Try "The Answer is**" pattern
                if not completion_answer:
                    try:
                        answer_is_pattern = re.search(r'answer is\s*\*\*\s*([A-E])\s*\*\*', content, re.IGNORECASE)
                        if answer_is_pattern:
                            completion_answer = answer_is_pattern.group(1).upper()
                            print(
                                f"SpatialReasoningAccuracyORM - Completion match (pattern): answer is **{answer_is_pattern.group(1)}**")
                    except Exception as e:
                        print(f"Error in 'answer is' pattern matching: {e}")

                # Method 3: Try bold format as last resort
                if not completion_answer:
                    try:
                        bold_match = self._extract_bold_answer(content)
                        if bold_match and bold_match.strip():
                            completion_answer = bold_match.upper()
                            print(f"SpatialReasoningAccuracyORM - Completion match (bold): **{bold_match}**")
                    except Exception as e:
                        print(f"Error extracting bold answer: {e}")

                # Method 4: Try " option is**" pattern
                if not completion_answer:
                    option_is_pattern = re.search(r'option is\s*\*\*\s*([A-E])\s*\*\*', content, re.IGNORECASE)
                    if option_is_pattern:
                        completion_answer = option_is_pattern.group(1).upper()
                        print(
                            f"SpatialReasoningAccuracyORM - Completion match (pattern): option is **{option_is_pattern.group(1)}**")

                # Check if the answer is correct
                if completion_answer and completion_answer == solution_answer:
                    reward = 1.0

                # Give extra points for reasoning
                if "<think>" in content and "</think>" in content:
                    reward += 0.1

                # Cap reward at 1.0
                reward = min(1.0, reward)

            except Exception as e:
                print(f"Error in reward calculation: {e}")
                print(f"Content type: {type(content)}, Content: {content[:100]}...")
                print(f"Solution type: {type(sol)}, Solution: {sol[:100]}...")
                reward = 0.0

            rewards.append(reward)

        return rewards

    def _extract_bold_answer(self, text):
        """
        Extract answer from bold format like **A**, **B**, **C**, **D**, **E**
        Returns the LAST bold letter found in the text.

        Args:
            text (str): The text to extract bold answer from

        Returns:
            str: The extracted answer letter or empty string if not found
        """
        try:
            if not isinstance(text, str):
                return ""

            if not text:
                return ""

            # 匹配 Markdown 加粗格式：**X** 其中 X 是一个字母
            result = ""
            try:
                bold_pattern = r'\*\*([A-E])\*\*'
                matches = re.findall(bold_pattern, text)
                if matches and len(matches) > 0:
                    last_match = matches[-1]
                    print(f"_extract_bold_answer - Found last bold match: **{last_match}**")
                    result = last_match
            except Exception as e:
                print(f"Error in first pattern matching: {e}")

            # 如果第一种模式没有匹配到，尝试匹配更宽松的格式
            if not result:
                try:
                    bold_pattern_loose = r'\*\*\s*([A-E])\s*\*\*'
                    matches = re.findall(bold_pattern_loose, text)
                    if matches and len(matches) > 0:
                        last_match = matches[-1]
                        print(f"_extract_bold_answer - Found last loose bold match: **{last_match}**")
                        result = last_match
                except Exception as e:
                    print(f"Error in second pattern matching: {e}")

            return result
        except Exception as e:
            print(f"Unexpected error in _extract_bold_answer: {e}")
            return ""


orms = {
    'toolbench': ReactORM,
    'math': MathORM,
    'accuracy': MathAccuracy,
    'format': Format,
    'react_format': ReActFormat,
    'cosine': CosineReward,
    'repetition': RepetitionPenalty,
    'soft_overlong': SoftOverlong,
    'spatial_reasoning_strict_format': SpatialReasoningStrictFormat,
    'spatial_reasoning_loose_format': SpatialReasoningLooseFormat,
    'spatial_reasoning_acc': SpatialReasoningAccuracyORM,
}