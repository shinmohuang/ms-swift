import asyncio
import re
import os
from typing import List, Union, Dict

import json
from openai import OpenAI
from dotenv import load_dotenv

from swift.plugin import ORM, orms
from swift.utils import get_logger

logger = get_logger()


# Code borrowed from plugin/orm.py
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
            gold_parsed = parse(sol, extraction_mode='first_match', extraction_config=[LatexExtractionConfig()])
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
                # Reward 1 if the content is the same as the ground truth, 0 otherwise
                reward = float(verify(answer_parsed, gold_parsed))
            else:
                # If the gold solution is not parseable, we reward 1 to skip this example
                reward = 1.0
            rewards.append(reward)
        return rewards


class MathFormat(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion has a specific format."""
        pattern = r'^<think>.*?</think>\s*<answer>.*?</answer>(?![\s\S])'
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
        return [1.0 if match else 0.0 for match in matches]


class CountdownORM(ORM):

    def __call__(self, completions, target, nums, **kwargs) -> List[float]:
        """
        Evaluates completions based on Mathematical correctness of the answer

        Args:
            completions (list[str]): Generated outputs
            target (list[str]): Expected answers
            nums (list[str]): Available numbers

        Returns:
            list[float]: Reward scores
        """
        rewards = []
        for completion, gt, numbers in zip(completions, target, nums):
            try:
                # Check if the format is correct
                match = re.search(r'<answer>(.*?)<\/answer>', completion)
                if match is None:
                    rewards.append(0.0)
                    continue
                # Extract the "answer" part from the completion
                equation = match.group(1).strip()
                if '=' in equation:
                    equation = equation.split('=')[0]
                # Extract all numbers from the equation
                used_numbers = [int(n) for n in re.findall(r'\d+', equation)]

                # Check if all numbers are used exactly once
                if sorted(used_numbers) != sorted(numbers):
                    rewards.append(0.0)
                    continue
                # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
                allowed_pattern = r'^[\d+\-*/().\s]+$'
                if not re.match(allowed_pattern, equation):
                    rewards.append(0.0)
                    continue

                # Evaluate the equation with restricted globals and locals
                result = eval(equation, {"__builti'ns__": None}, {})
                # Check if the equation is correct and matches the ground truth
                if abs(float(result) - float(gt)) < 1e-5:
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
            except Exception:
                # If evaluation fails, reward is 0
                rewards.append(0.0)
        return rewards


class MultiModalAccuracyORM(ORM):

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        """
        Reward function that checks if the completion is correct.
        Args:
            completions (list[str]): Generated outputs
            solution (list[str]): Ground Truths.

        Returns:
            list[float]: Reward scores
        """
        rewards = []
        from math_verify import parse, verify
        for content, sol in zip(completions, solution):
            reward = 0.0
            # Try symbolic verification first
            try:
                answer = parse(content)
                if float(verify(answer, parse(sol))) > 0:
                    reward = 1.0
            except Exception:
                pass  # Continue to next verification method if this fails

            # If symbolic verification failed, try string matching
            if reward == 0.0:
                try:
                    # Extract answer from solution if it has think/answer tags
                    sol_match = re.search(r'<answer>(.*?)</answer>', sol)
                    ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

                    # Extract answer from content if it has think/answer tags
                    content_match = re.search(r'<answer>(.*?)</answer>', content)
                    student_answer = content_match.group(1).strip() if content_match else content.strip()

                    # Compare the extracted answers
                    if student_answer == ground_truth:
                        reward = 1.0
                except Exception:
                    pass  # Keep reward as 0.0 if both methods fail
            rewards.append(reward)
        return rewards


# ref implementation: https://github.com/huggingface/open-r1/blob/main/src/open_r1/rewards.py
class CodeReward(ORM):

    def __init__(self):
        import importlib.util
        assert importlib.util.find_spec('e2b') is not None, (
            "The e2b package is required but not installed. Please install it using 'pip install e2b-code-interpreter'."
        )
        from dotenv import load_dotenv
        load_dotenv()

    @staticmethod
    def extract_code(completion: str, language: str) -> str:
        pattern = re.compile(rf'```{language}\n(.*?)```', re.DOTALL)
        matches = pattern.findall(completion)
        extracted_answer = matches[-1] if len(matches) >= 1 else ''
        return extracted_answer

    def run_async_from_sync(self, scripts: List[str], languages: List[str]) -> List[float]:
        """Function wrapping the `run_async` function."""
        # Create a new event loop and set it
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Run the async function and get the result
            rewards = loop.run_until_complete(self.run_async(scripts, languages))
        finally:
            loop.close()

        return rewards

    async def run_async(self, scripts: List[str], languages: List[str]) -> List[float]:
        try:
            from e2b_code_interpreter import AsyncSandbox
        except ImportError:
            logger.warning("e2b_code_interpreter not installed. Please install with: pip install e2b-code-interpreter")
            return [0.0] * len(scripts)

        # Create the sandbox by hand, currently there's no context manager for this version
        try:
            sbx = await AsyncSandbox.create(timeout=30, request_timeout=3)
        except Exception as e:
            logger.warning(f'Error from E2B executor: {e}')
            return [0.0] * len(scripts)
        # Create a list of tasks for running scripts concurrently
        tasks = [self.run_script(sbx, script, language) for script, language in zip(scripts, languages)]

        # Wait for all tasks to complete and gather their results as they finish
        results = await asyncio.gather(*tasks)
        rewards = list(results)  # collect results

        # Kill the sandbox after all the tasks are complete
        await sbx.kill()

        return rewards

    async def run_script(self, sbx, script: str, language: str) -> float:
        try:
            execution = await sbx.run_code(script, language=language, timeout=30)
        except Exception as e:
            logger.warning(f'Error from E2B executor: {e}')
            return 0.0
        try:
            return float(execution.text)
        except (TypeError, ValueError):
            return 0.0

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that evaluates code snippets using the E2B code interpreter.

        Assumes the dataset contains a `verification_info` column with test cases.
        """
        evaluation_script_template = """
        import subprocess
        import json

        def evaluate_code(code, test_cases):
            passed = 0
            total = len(test_cases)
            exec_timeout = 5

            for case in test_cases:
                process = subprocess.run(
                    ["python3", "-c", code],
                    input=case["input"],
                    text=True,
                    capture_output=True,
                    timeout=exec_timeout
                )

                if process.returncode != 0:  # Error in execution
                    continue

                output = process.stdout.strip()
                if output.strip() == case["output"].strip():
                    passed += 1

            success_rate = (passed / total)
            return success_rate

        code_snippet = {code}
        test_cases = json.loads({test_cases})

        evaluate_code(code_snippet, test_cases)
        """
        verification_info = kwargs['verification_info']
        languages = [info['language'] for info in verification_info]
        code_snippets = [
            self.extract_code(completion, language) for completion, language in zip(completions, languages)
        ]
        scripts = [
            evaluation_script_template.format(
                code=json.dumps(code), test_cases=json.dumps(json.dumps(info['test_cases'])))
            for code, info in zip(code_snippets, verification_info)
        ]
        try:
            rewards = self.run_async_from_sync(scripts, languages)

        except Exception as e:
            logger.warning(f'Error from E2B executor: {e}')
            rewards = [0.0] * len(completions)

        return rewards


class CodeFormat(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        verification_info = kwargs['verification_info']
        rewards = []
        for content, info in zip(completions, verification_info):
            pattern = r'^<think>.*?</think>\s*<answer>.*?```{}.*?```.*?</answer>(?![\s\S])'.format(info['language'])
            match = re.match(pattern, content, re.DOTALL | re.MULTILINE)
            reward = 1.0 if match else 0.0
            rewards.append(reward)
        return rewards


class SpatialReasoningStrictFormat(ORM):
    """
    Reward function that checks if the completion has the correct format.
    The strict version requires the text to contain exactly the following structure:

    <think>
      ... (任意内容) ...
    </think>
    <answer>
      A-F (单独的字母)
    </answer>

    如果完全匹配且<answer>标签内是单独的A-F字母则返回 0.5，如果在标签外有内容则返回 -0.5，否则返回 0.0。
    """

    def __call__(self, completions, **kwargs) -> List[float]:
        rewards = []

        # 处理不同格式的 completions，将其统一为字符串内容
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

        # 检查是否符合要求格式 - 只保留think和answer标签，且answer内容必须是A-F单独字母
        standard_pattern = r'^<think>(.*?)</think>\s*<answer>\s*([A-F])\s*</answer>$'
        for content in processed_completions:
            try:
                # 使用 fullmatch 确保整个字符串符合该模式
                match = re.fullmatch(standard_pattern, content, re.DOTALL)
                if match:
                    answer_content = match.group(2).strip()
                    # 确保answer标签内确实是单独的A-F字母
                    if answer_content in ['A', 'B', 'C', 'D', 'E', 'F']:
                        print(f"SpatialReasoningStrictFormat - Matched content with answer: {answer_content}")
                        rewards.append(0.5)
                    else:
                        print(f"SpatialReasoningStrictFormat - Answer content not A-F: {answer_content}")
                        rewards.append(0.0)
                else:
                    # 检查是否包含所有必要标签
                    has_all_tags = (
                        '<think>' in content and '</think>' in content and
                        '<answer>' in content and '</answer>' in content
                    )

                    if has_all_tags:
                        # 有所有标签但不符合格式，说明有标签外内容或answer内容不对
                        print("SpatialReasoningStrictFormat - Found content outside tags or invalid answer format.")
                        rewards.append(-0.5)
                    else:
                        # 缺少必要标签
                        print("SpatialReasoningStrictFormat - Missing required tags.")
                        rewards.append(0.0)
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
                'pattern': r"The Answer is\s*\*\*\s*[A-F]\s*\*\*",
                'reward': 0.3,
                'description': "The Answer is** 格式"
            },
            # 仅包含加粗答案 **X**
            {
                'pattern': r"\*\*\s*[A-F]\s*\*\*",
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
                        answer_is_pattern = re.search(r'answer is\s*\*\*\s*([A-F])\s*\*\*', content, re.IGNORECASE)
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
                    try:
                        option_is_pattern = re.search(r'option is\s*\*\*\s*([A-F])\s*\*\*', content, re.IGNORECASE)
                        if option_is_pattern:
                            completion_answer = option_is_pattern.group(1).upper()
                            print(
                                f"SpatialReasoningAccuracyORM - Completion match (pattern): option is **{option_is_pattern.group(1)}**")
                    except Exception as e:
                        print(f"Error in 'option is' pattern matching: {e}")

                # Check if the answer is correct
                if completion_answer and completion_answer == solution_answer:
                    reward = 1.0

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
        Extract answer from bold format like **A**, **B**, **C**, **D**, **E**, **F**
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
                bold_pattern = r'\*\*([A-F])\*\*'
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
                    bold_pattern_loose = r'\*\*\s*([A-F])\s*\*\*'
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


SYSTEM = """
You are an Expert Process Reward Evaluator specializing in assessing reasoning quality and answer correctness. Your role is to provide precise, objective evaluations by comparing student responses against reference solutions.

## Evaluation Framework

### Scoring Scale: -1.0 to 1.0
- **1.0**: Excellent - Correct answer with clear, logical reasoning
- **0.7-0.9**: Good - Mostly correct with minor issues in reasoning or presentation
- **0.3-0.6**: Fair - Partially correct but with significant gaps or errors
- **0.0-0.2**: Poor - Incorrect answer or severely flawed reasoning
- **-0.3 to -0.1**: Bad - Incorrect with misleading or harmful reasoning
- **-1.0**: Terrible - Completely wrong and potentially harmful

### Primary Evaluation Criteria (in order of importance):

1. **Correctness (35%)**: Does the final answer match or align with the ground truth?
2. **Reasoning Quality (45%)**: Is the logical chain coherent, complete, and well-structured?
3. **Spatial Reasoning (15%)**: Does the response demonstrate proper spatial understanding and reasoning?
4. **Clarity & Organization (5%)**: Is the response well-presented and easy to follow?

### Detailed Assessment Guidelines:

**For Correctness:**
- Exact match with ground truth: +0.3 to +0.35 points
- Conceptually correct but minor differences: +0.2 to +0.25 points
- Partially correct: +0.1 to +0.15 points
- Incorrect: -0.2 to 0 points

**For Reasoning Quality (CRITICAL - Focus on logical coherence):**
- **Step-by-step consistency**: Each reasoning step must logically follow from the previous step
- **Internal coherence**: No contradictions within the reasoning chain
- **Evidence-based progression**: Each conclusion must be supported by prior statements or given information
- **Complete logical flow**: The reasoning should form a complete path from problem to solution

Scoring for Reasoning:
- Perfect logical chain with clear step-by-step progression: +0.4 to +0.45 points
- Good reasoning with minor logical gaps or inconsistencies: +0.3 to +0.35 points
- Reasonable approach but significant logical jumps or unclear connections: +0.15 to +0.25 points
- Some logical elements but major flaws in reasoning chain: +0.05 to +0.1 points
- Incoherent or contradictory reasoning: -0.15 to 0 points
- Completely illogical or harmful reasoning: -0.3 to -0.1 points

**For Spatial Reasoning:**
- **Spatial relationship understanding**: Correctly interprets positions, directions, and spatial configurations
- **Geometric reasoning**: Properly applies concepts like distance, angles, shapes, and transformations
- **Visualization accuracy**: Demonstrates correct mental modeling of spatial scenarios
- **Spatial logic**: Uses appropriate spatial reasoning principles (e.g., transitivity of spatial relations)

Scoring for Spatial Reasoning:
- Excellent spatial understanding and reasoning: +0.12 to +0.15 points
- Good spatial reasoning with minor errors: +0.08 to +0.11 points
- Basic spatial understanding but significant gaps: +0.04 to +0.07 points
- Poor spatial reasoning or major misunderstandings: -0.05 to +0.03 points
- Completely incorrect spatial interpretation: -0.1 to -0.05 points

**For Clarity:**
- Well-organized, clear presentation: +0.04 to +0.05 points
- Adequate presentation: 0 to +0.03 points
- Poor organization or unclear: -0.02 to 0 points

### Special Considerations:
- **Incomplete Responses**: Evaluate only what is provided; do not penalize for incompleteness unless it significantly impacts understanding
- **Alternative Solutions**: Accept valid alternative approaches that differ from the reference
- **Partial Credit**: Give appropriate partial credit for correct intermediate steps
- **Format Adherence**: Consider whether the response follows required formatting (e.g., <think>...</think>, <answer>...</answer>)

### Output Requirements:
- Provide your assessment as a single numerical value
- Round to one decimal place (e.g., 0.7, -0.3)
- Output ONLY in this exact JSON format: {"Reward": your-score}
- Do not include any additional text, explanation, or formatting

### Quality Assurance:
- Ensure your score reflects the weighted criteria above
- Double-check that the score is within the valid range [-1.0, 1.0]
- Consider the overall utility and helpfulness of the response
- Be consistent and objective in your evaluation

Remember: Your evaluation directly impacts model training. Strive for accuracy, consistency, and fairness in every assessment.
"""  # noqa

QUERY = """
The original question or the previous conversation:

#query#

Here is the ground truth as the reference:

#ground_truth#

Given the upper information, give your reward(-1.0~1.0) of the following answer:

#response#
"""


class DeepSeekORM(ORM):
    """
    使用DeepSeek API进行过程奖励评估的ORM类

    Features:
    - 多API key轮询使用
    - 自动重试机制
    - 结果缓存
    - 智能并发控制
    - 详细错误处理
    """

    def __init__(self, api_keys=None, max_workers=None, timeout=60, max_retries=3, cache_size=1000, **kwargs):
        """
        初始化DeepSeekORM评估器

        Args:
            api_keys: 可选，API key列表或逗号分隔的字符串
            max_workers: 最大并发工作数，默认为API keys数量（降低并发避免超时）
            timeout: API请求超时时间（秒），默认60秒（增加到60秒）
            max_retries: 最大重试次数，默认3次
            cache_size: 缓存大小，默认1000条记录
            **kwargs: 其他参数
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.api_keys = self._init_api_keys(api_keys)
        # 降低默认并发数，避免超时问题
        self.max_workers = max_workers or min(len(self.api_keys), 3)  # 最大3个并发

        # 初始化结果缓存（基于内容hash的简单缓存）
        from collections import OrderedDict
        self.cache = OrderedDict()
        self.cache_size = cache_size

        # 统计信息
        self.stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'api_errors': 0,
            'successful_requests': 0,
            'timeout_errors': 0,
            'rate_limit_errors': 0
        }

        # 速率限制控制
        import time
        self.last_request_time = {}  # 每个API key的最后请求时间
        self.min_request_interval = 1.0  # 最小请求间隔（秒）

        logger.info(f"DeepSeekORM初始化完成 - API keys: {len(self.api_keys)}, 最大并发: {self.max_workers}, 超时: {self.timeout}秒")

    def _init_api_keys(self, api_keys):
        """初始化API keys"""
        keys = []

        # 处理直接传入的API keys
        if api_keys:
            if isinstance(api_keys, str):
                keys = [key.strip() for key in api_keys.split(',') if key.strip()]
            elif isinstance(api_keys, list):
                keys = [key for key in api_keys if key]

        # 从环境变量获取API keys
        if not keys:
            env_vars = ['JUDGE', 'DEEPSEEK_API_KEY', 'DEEPSEEK_EXTRACTION_API_KEY']
            for env_name in env_vars:
                env_value = os.getenv(env_name, '')
                if env_value:
                    if ',' in env_value:  # 多个key用逗号分隔
                        keys.extend([key.strip() for key in env_value.split(',') if key.strip()])
                    else:
                        keys.append(env_value)
                    break  # 找到第一个有效的环境变量就停止

        # 验证API keys
        if not keys:
            raise ValueError(
                "未找到有效的API key。请通过以下方式之一提供：\n"
                "1. 参数 api_keys='key1,key2,key3'\n"
                "2. 环境变量 JUDGE='key1,key2,key3'\n"
                "3. 环境变量 DEEPSEEK_API_KEY='your_key'"
            )

        # 去重并验证格式
        unique_keys = list(dict.fromkeys(keys))  # 保持顺序的去重
        valid_keys = [key for key in unique_keys if key.startswith('sk-')]

        if not valid_keys:
            logger.warning("提供的API key格式可能不正确（应以'sk-'开头）")
            valid_keys = unique_keys  # 仍然使用，但记录警告

        return valid_keys

    def _get_cache_key(self, content, solution):
        """生成缓存键"""
        import hashlib
        combined = f"{content}|{solution}"
        return hashlib.md5(combined.encode()).hexdigest()

    def _get_cached_result(self, cache_key):
        """获取缓存结果"""
        if cache_key in self.cache:
            # 移到末尾（LRU）
            value = self.cache.pop(cache_key)
            self.cache[cache_key] = value
            self.stats['cache_hits'] += 1
            return value
        return None

    def _set_cache(self, cache_key, result):
        """设置缓存"""
        if len(self.cache) >= self.cache_size:
            # 删除最老的条目
            self.cache.popitem(last=False)
        self.cache[cache_key] = result

    def _extract_content(self, comp):
        """提取completion内容的统一方法"""
        if isinstance(comp, dict):
            content = comp.get("content", "")
            if not content:
                messages = comp.get("messages", [])
                if messages and isinstance(messages[-1], dict):
                    content = messages[-1].get("content", "")
        elif isinstance(comp, str):
            content = comp
        else:
            content = str(getattr(comp, "content", comp))

        return content.strip()

    def _parse_reward(self, response_text):
        """解析API响应中的奖励值"""
        try:
            # 尝试直接解析JSON
            reward_data = json.loads(response_text)
            return float(reward_data.get("Reward", 0.0))
        except json.JSONDecodeError:
            # 如果JSON解析失败，尝试正则提取
            import re
            # 查找 "Reward": 数字 或 {"Reward": 数字} 格式
            patterns = [
                r'"Reward":\s*([+-]?\d*\.?\d+)',
                r'[Rr]eward["\']?:\s*([+-]?\d*\.?\d+)',
                r'[Rr]eward["\']?\s*=\s*([+-]?\d*\.?\d+)'
            ]

            for pattern in patterns:
                match = re.search(pattern, response_text)
                if match:
                    return float(match.group(1))

            logger.warning(f"无法解析奖励值: {response_text[:100]}...")
            return 0.0

    def _wait_for_rate_limit(self, api_key):
        """等待速率限制"""
        import time
        current_time = time.time()
        if api_key in self.last_request_time:
            time_since_last = current_time - self.last_request_time[api_key]
            if time_since_last < self.min_request_interval:
                wait_time = self.min_request_interval - time_since_last
                time.sleep(wait_time)
        self.last_request_time[api_key] = time.time()

    def _call_api_with_retry(self, content, solution, api_key, idx):
        """带重试的API调用"""
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                # 速率限制控制
                self._wait_for_rate_limit(api_key)

                # 构建查询
                query = QUERY.replace("#ground_truth#", solution).replace("#response#", content)

                # 调用API
                client = OpenAI(
                    api_key=api_key,
                    base_url='https://api.deepseek.com/v1',
                    timeout=self.timeout
                )

                response = client.chat.completions.create(
                    model="deepseek-reasoner",
                    messages=[
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": query}
                    ]
                )

                # 解析结果
                reward = self._parse_reward(response.choices[0].message.content)
                reward = max(-1.0, min(1.0, reward))  # 确保范围正确

                self.stats['successful_requests'] += 1
                return reward

            except Exception as e:
                last_exception = e
                self.stats['api_errors'] += 1

                # 分类错误类型
                error_str = str(e).lower()
                if 'timeout' in error_str or 'timed out' in error_str:
                    self.stats['timeout_errors'] += 1
                elif 'rate limit' in error_str or 'too many requests' in error_str:
                    self.stats['rate_limit_errors'] += 1

                if attempt < self.max_retries:
                    # 根据错误类型调整等待时间
                    if 'timeout' in error_str:
                        wait_time = (2 ** attempt) * 2  # 超时错误等待更久
                    elif 'rate limit' in error_str:
                        wait_time = (2 ** attempt) * 5  # 速率限制等待更久
                    else:
                        wait_time = (2 ** attempt)  # 标准指数退避

                    logger.warning(f"样本 {idx} 第 {attempt + 1} 次尝试失败 ({type(e).__name__})，{wait_time}秒后重试: {str(e)}")
                    import time
                    time.sleep(wait_time)
                else:
                    logger.error(f"样本 {idx} 所有重试均失败 ({type(e).__name__}): {str(e)}")

        return 0.0  # 所有重试失败时返回默认值

    def _process_single_sample(self, idx, comp, sol, api_key):
        """处理单个样本的评估"""
        self.stats['total_requests'] += 1

        try:
            # 提取内容
            content = self._extract_content(comp)
            if not content:
                logger.warning(f"样本 {idx} 内容为空")
                return 0.0

            # 检查缓存
            cache_key = self._get_cache_key(content, sol)
            cached_result = self._get_cached_result(cache_key)
            if cached_result is not None:
                logger.debug(f"样本 {idx} 使用缓存结果: {cached_result}")
                return cached_result

            # 调用API
            reward = self._call_api_with_retry(content, sol, api_key, idx)

            # 缓存结果
            self._set_cache(cache_key, reward)

            logger.debug(f"样本 {idx} 评估完成: {reward}")
            return reward

        except Exception as e:
            logger.error(f"处理样本 {idx} 时发生未预期错误: {str(e)}")
            return 0.0

    def __call__(self, completions: List[Union[str, Dict]], solution: List[str], **kwargs) -> List[float]:
        """
        使用多个API keys并发处理评估请求

        Args:
            completions: 模型生成的完成结果列表
            solution: 参考答案列表

        Returns:
            rewards: 奖励值列表
        """
        import concurrent.futures

        num_samples = len(completions)
        logger.info(f"开始DeepSeek评估 - 样本数: {num_samples}, API keys: {len(self.api_keys)}, 并发数: {self.max_workers}")

        # 验证输入
        if len(solution) != num_samples:
            raise ValueError(f"completions和solution长度不匹配: {num_samples} vs {len(solution)}")

        # 初始化结果
        rewards = [0.0] * num_samples

        # 准备任务
        tasks = []
        for i in range(num_samples):
            api_key = self.api_keys[i % len(self.api_keys)]  # 轮询使用API keys
            tasks.append((i, completions[i], solution[i], api_key))

        # 并发执行
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交任务
            future_to_idx = {
                executor.submit(self._process_single_sample, idx, comp, sol, api_key): idx
                for idx, comp, sol, api_key in tasks
            }

            # 收集结果
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    rewards[idx] = future.result()
                except Exception as e:
                    logger.error(f"获取样本 {idx} 结果失败: {str(e)}")
                    rewards[idx] = 0.0

        # 输出统计信息
        self._log_stats()

        return rewards

    def _log_stats(self):
        """输出统计信息"""
        stats = self.stats
        total = stats['total_requests']
        if total > 0:
            cache_rate = stats['cache_hits'] / total * 100
            success_rate = stats['successful_requests'] / total * 100
            logger.info(
                f"DeepSeek评估完成 - "
                f"总请求: {total}, "
                f"缓存命中率: {cache_rate:.1f}%, "
                f"成功率: {success_rate:.1f}%, "
                f"API错误: {stats['api_errors']}, "
                f"超时错误: {stats['timeout_errors']}, "
                f"速率限制错误: {stats['rate_limit_errors']}"
            )

    def clear_cache(self):
        """清空缓存"""
        self.cache.clear()
        logger.info("DeepSeek缓存已清空")

    def get_stats(self):
        """获取统计信息"""
        return self.stats.copy()


orms['external_math_acc'] = MathAccuracy
orms['external_math_format'] = MathFormat
orms['external_countdown'] = CountdownORM
orms['external_r1v_acc'] = MultiModalAccuracyORM
orms['external_code_reward'] = CodeReward
orms['external_code_format'] = CodeFormat
orms['external_code_reward_by_judge0'] = CodeRewardByJudge0
orms['spatial_reasoning_strict_format'] = SpatialReasoningStrictFormat
orms['spatial_reasoning_loose_format'] = SpatialReasoningLooseFormat
orms['spatial_reasoning_acc'] = SpatialReasoningAccuracyORM
orms['deepseek_orm'] = DeepSeekORM

# For genrm you can refer to swift/llm/plugin/rm_plugin/GenRMPlugin


class CustomizedRMPlugin:
    """
    Customized Reward Model Plugin, same to DefaultRMPlugin

    It assumes that `self.model` is a classification model with a value head(output dimmension 1).
    The first logits value from the model's output is used as the reward score.
    """

    def __init__(self, model, template):
        self.model = model
        self.template: Template = template

    def __call__(self, inputs):
        batched_inputs = [self.template.encode(deepcopy(infer_request)) for infer_request in inputs]
        reward_inputs = to_device(self.template.data_collator(batched_inputs), self.model.device)
        reward_inputs.pop('labels')

        with torch.inference_mode():
            return self.model(**reward_inputs).logits[:, 0]


rm_plugins['my_rmplugin'] = CustomizedRMPlugin
