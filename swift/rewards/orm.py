# Copyright (c) ModelScope Contributors. All rights reserved.
# Outcome Reward Model (ORM) implementations for GRPO training.

import json
import os
import re
from typing import TYPE_CHECKING, Dict, List, Optional, Union
import math
import requests
import time


from swift.infer_engine import InferRequest

if TYPE_CHECKING:
    from swift.megatron.arguments import MegatronArguments
    from swift.rlhf_trainers import GRPOConfig


class ORM:
    """Base class for synchronous outcome reward models (ORM).

    Subclasses should implement the __call__ method to compute rewards.

    Example:
        class MyReward(ORM):
            def __call__(self, completions, **kwargs) -> List[float]:
                return [1.0 if len(c) > 100 else 0.0 for c in completions]
    """

    def __init__(self, args: Optional[Union['GRPOConfig', 'MegatronArguments']] = None, **kwargs):
        self.args = args

    def __call__(self, **kwargs) -> List[float]:
        raise NotImplementedError


class AsyncORM:
    """Base class for asynchronous outcome reward models (ORM).

    Use this for reward functions that involve I/O operations (e.g., API calls,
    database queries) that can benefit from async execution.

    Async reward functions are executed in parallel using asyncio.gather,
    which can significantly speed up reward computation when multiple async
    reward functions are used or when the reward function involves network calls.

    Example:
        class MyAsyncReward(AsyncORM):
            async def __call__(self, completions, **kwargs) -> List[float]:
                # Use asyncio.gather for parallel execution of all API calls
                import asyncio
                import aiohttp

                async def score_single(session, text):
                    async with session.post(api_url, json={'text': text}) as resp:
                        result = await resp.json()
                        return result['score']

                async with aiohttp.ClientSession() as session:
                    tasks = [score_single(session, c) for c in completions]
                    rewards = await asyncio.gather(*tasks)
                    return list(rewards)
    """

    def __init__(self, args: Optional[Union['GRPOConfig', 'MegatronArguments']] = None, **kwargs):
        self.args = args

    async def __call__(self, **kwargs) -> List[float]:
        raise NotImplementedError


class MathAccuracy(ORM):

    def __init__(self, args=None, **kwargs):
        super().__init__(args, **kwargs)
        import importlib.util
        assert importlib.util.find_spec('math_verify') is not None, (
            'The math_verify package is required but not installed. '
            "Please install it using 'pip install math_verify'.")

    def __call__(self, completions, solution=None, **kwargs) -> List[float]:
        from latex2sympy2_extended import NormalizationConfig
        from math_verify import LatexExtractionConfig, parse, verify
        if solution is None:
            solution = kwargs.get('solution')
        if solution is None:
            return [0.0] * len(completions)

        rewards = []
        for content, sol in zip(completions, solution):
            if not sol:
                rewards.append(0.0)
                continue
            content_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
            content_to_parse = content_match.group(1).strip() if content_match else content
            has_answer_tag = content_match is not None

            sol_match = re.search(r'<answer>(.*?)</answer>', sol, re.DOTALL)
            sol_to_parse = sol_match.group(1).strip() if sol_match else sol

            gold_parsed = parse(sol_to_parse, extraction_mode='first_match')
            if len(gold_parsed) != 0:
                if has_answer_tag:
                    answer_parsed = parse(content_to_parse, extraction_mode='first_match')
                else:
                    answer_parsed = parse(
                        content_to_parse,
                        extraction_config=[
                            LatexExtractionConfig(
                                normalization_config=NormalizationConfig(
                                    nits=False,
                                    malformed_operators=False,
                                    basic_latex=True,
                                    boxed=True,
                                    units=True,
                                ),
                                boxed_match_priority=0,
                                try_extract_without_anchor=False,
                            )
                        ],
                        extraction_mode='first_match',
                    )
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
    def __init__(self, args: Optional[Union['GRPOConfig', 'MegatronArguments']] = None, accuracy_orm=None):
        super().__init__(args)
        self.min_len_value_wrong = args.cosine_min_len_value_wrong
        self.max_len_value_wrong = args.cosine_max_len_value_wrong
        self.min_len_value_correct = args.cosine_min_len_value_correct
        self.max_len_value_correct = args.cosine_max_len_value_correct
        self.max_len = args.cosine_max_len
        self.accuracy_orm = accuracy_orm or MathAccuracy()

    @staticmethod
    def cosfn(t, T, min_value, max_value):
        import math
        return max_value - (max_value - min_value) * (1 - math.cos(t * math.pi / T)) / 2

    def __call__(self, completions, solution=None, **kwargs) -> List[float]:
        acc_rewards = self.accuracy_orm(completions, solution, **kwargs)
        response_token_ids = kwargs.get('response_token_ids')
        if response_token_ids is None:
            response_token_ids = [completion.split() for completion in completions]
        rewards = []
        for ids, acc_reward in zip(response_token_ids, acc_rewards):
            is_correct = acc_reward >= 1.
            if is_correct:
                # Swap min/max for correct answers
                min_value = self.max_len_value_correct
                max_value = self.min_len_value_correct
            else:
                min_value = self.max_len_value_wrong
                max_value = self.min_len_value_wrong
            gen_len = len(ids)
            reward = self.cosfn(gen_len, self.max_len, min_value, max_value)
            rewards.append(reward)
        return rewards


class RepetitionPenalty(ORM):
    # https://arxiv.org/abs/2502.03373
    def __init__(self, args: Optional[Union['GRPOConfig', 'MegatronArguments']] = None, **kwargs):
        super().__init__(args)
        self.ngram_size = args.repetition_n_grams
        self.max_penalty = args.repetition_max_penalty

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

    def __init__(self, args: Optional[Union['GRPOConfig', 'MegatronArguments']] = None, **kwargs):
        super().__init__(args)
        assert args.soft_cache_length < args.soft_max_length
        self.soft_max_length = args.soft_max_length
        self.soft_cache_length = args.soft_cache_length

    def __call__(self, completions, **kwargs) -> List[float]:
        rewards = []
        response_token_ids = kwargs.get('response_token_ids')
        for ids in response_token_ids:
            completion_length = len(ids)
            expected_len = self.soft_max_length - self.soft_cache_length
            exceed_len = completion_length - expected_len
            rewards.append(min(-exceed_len / self.soft_cache_length, 0))
        return rewards


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

    def __call__(self, infer_requests: List[Union['InferRequest', Dict]], solution: Optional[List[str]] = None,
                 **kwargs) -> List[float]:
        rewards = []
        if solution is None:
            solution = kwargs.get('solution')
        if solution is None:
            return [0.0] * len(infer_requests)

        if not isinstance(infer_requests[0], str):
            predictions = [request['messages'][-1]['content'] for request in infer_requests]
        else:
            predictions = infer_requests
        for prediction, ground_truth in zip(predictions, solution):
            if not ground_truth:
                rewards.append(0.0)
                continue
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

    def __init__(self, args=None, **kwargs):
        super().__init__(args)
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

    def __call__(self, infer_requests: List[Union['InferRequest', Dict]], ground_truths: Optional[List[str]] = None,
                 **kwargs) -> List[float]:
        rewards = []
        if ground_truths is None:
            ground_truths = kwargs.get('ground_truths') or kwargs.get('solution')
        if ground_truths is None:
            return [0.0] * len(infer_requests)

        if isinstance(infer_requests[0], str):
            predictions = infer_requests
        elif isinstance(infer_requests[0], dict):
            predictions = [request['messages'][-1]['content'] for request in infer_requests]
        else:
            predictions = [request.messages[-1]['content'] for request in infer_requests]
        for prediction, ground_truth in zip(predictions, ground_truths):
            if not ground_truth:
                rewards.append(0.0)
                continue
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

class GeoAnswerFormat(ORM):
    @staticmethod
    def _extract_final_answer(answer_content: str) -> Optional[str]:
        try:
            data = json.loads(answer_content)
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict):
            return None
        final_answer = data.get('FinalAnswer')
        return str(final_answer).strip() if final_answer is not None else None

    def __call__(self, completions, **kwargs) -> List[float]:
        """
        奖励函数：只检查 <answer> 标签内是否为可解析 JSON，并且 FinalAnswer 正好由分号隔成三段。
        """
        results = []
        for content in completions:
            content = content.strip()

            answer_matches = re.findall(r'<answer>\s*(.*?)\s*</answer>', content, re.DOTALL | re.I)
            if not answer_matches:
                results.append(0.0)
                continue

            answer_content = answer_matches[-1].strip()
            final_answer = self._extract_final_answer(answer_content)
            if not final_answer:
                results.append(0.0)
                continue

            parts = re.split(r'[;；]', final_answer)
            parts = [part.strip() for part in parts if part.strip()]  # 去除空白部分

            if len(parts) == 3:
                results.append(1.0)
            else:
                results.append(0.0)
                
        return results


class GeoNoUnknown(ORM):
    FINAL_BANNED_RE = re.compile(
        r'\b(?:unknown|n/?a|unspecified|undetermined|unidentified|uncertain|unclear|'
        r'placeholder|generic)\b'
        r'|not enough|cannot be determined|can not be determined|not possible to determine|'
        r'hard to determine|difficult to determine',
        re.I,
    )
    REFUSAL_RE = re.compile(
        r'\b(?:cannot|can not|can\'t|unable to)\s+(?:be\s+)?(?:\w+\s+){0,3}'
        r'(?:determin(?:e|ed)|identif(?:y|ied)|infer(?:red)?|locate(?:d)?|'
        r'geolocate(?:d)?|pinpoint(?:ed)?)\b'
        r'|\b(?:impossible|not possible|hard|difficult)\s+to\s+(?:\w+\s+){0,3}'
        r'(?:determine|identify|infer|locate|geolocate|pinpoint)\b'
        r'|\b(?:insufficient|not enough)\s+(?:visual\s+)?'
        r'(?:information|evidence|clues|detail|details|context)\b'
        r'|\bno identifiable\s+(?:geographic\s+)?'
        r'(?:information|features|landmarks|location|place|clues)\b'
        r'|\black(?:s|ing)?\s+(?:any\s+)?identifiable\s+(?:geographic\s+)?'
        r'(?:information|features|landmarks|location|place|clues)\b',
        re.I,
    )

    @staticmethod
    def _strip_special_tokens(content: str) -> str:
        return re.sub(r'\s*<\|im_end\|>\s*$', '', content.strip())

    @classmethod
    def _has_unknown_or_refusal(cls, content: str) -> bool:
        content = cls._strip_special_tokens(content)
        answer_matches = re.findall(r'<answer>\s*(.*?)\s*</answer>', content, re.DOTALL | re.I)
        if answer_matches:
            final_answer = GeoAnswerFormat._extract_final_answer(answer_matches[-1].strip())
            if final_answer and cls.FINAL_BANNED_RE.search(final_answer):
                return True

        return bool(cls.REFUSAL_RE.search(content))

    def __call__(self, completions, **kwargs) -> List[float]:
        rewards = []
        for content in completions:
            rewards.append(0.0 if self._has_unknown_or_refusal(content) else 1.0)
        return rewards


class GeoStrictFormat(ORM):
    BLOCK_RE = re.compile(r'^<think>\s*(.*?)\s*</think>\s*<answer>\s*(.*?)\s*</answer>\s*$', re.S | re.I)

    @staticmethod
    def _strip_special_tokens(content: str) -> str:
        return re.sub(r'\s*<\|im_end\|>\s*$', '', content.strip())

    def __call__(self, completions, **kwargs) -> List[float]:
        rewards = []
        for content in completions:
            content = self._strip_special_tokens(content)
            if GeoNoUnknown._has_unknown_or_refusal(content):
                rewards.append(0.0)
                continue

            match = self.BLOCK_RE.match(content)
            if not match:
                rewards.append(0.0)
                continue

            try:
                think_json = json.loads(match.group(1).strip())
                answer_json = json.loads(match.group(2).strip())
            except json.JSONDecodeError:
                rewards.append(0.0)
                continue

            if set(think_json) != {'Clues', 'Reasoning'} or set(answer_json) != {'FinalAnswer'}:
                rewards.append(0.0)
                continue

            clues = think_json.get('Clues')
            reasoning = think_json.get('Reasoning')
            final_answer = str(answer_json.get('FinalAnswer', '')).strip()
            parts = [part.strip() for part in re.split(r'[;；]', final_answer) if part.strip()]
            valid = (
                isinstance(clues, list)
                and bool(clues)
                and all(isinstance(item, str) and item.strip() for item in clues)
                and isinstance(reasoning, str)
                and bool(reasoning.strip())
                and len(parts) == 3
            )
            rewards.append(1.0 if valid else 0.0)
        return rewards


class GeoScoreAccuracy(ORM):
    """基于OpenCage API的GeoScore准确性奖励函数。

    The prediction is still a textual place name. For GeoBench GRPO data, the
    ground truth can be passed directly as latitude/longitude columns, avoiding
    an OpenCage lookup for the reference side. The older `solution` text path is
    kept as a fallback for backward compatibility.
    """
    
    def __init__(self, args: Optional[Union['GRPOConfig', 'MegatronArguments']] = None,
                 api_keys: Optional[List[str]] = None,
                 max_distance: float = 2000.0,
                 confidence_threshold: int = 2,
                 timeout: int = 10,
                 cache_file: Optional[str] = None,
                 **kwargs
                 ):
        """
        初始化GeoScore准确性奖励函数
        
        Args:
            api_keys: OpenCage API密钥列表（支持多GPU环境）
            max_distance: 最大距离阈值（公里），用于GeoScore计算
            confidence_threshold: 置信度阈值（1-10），低于此值直接返回0分，建议3-7
            timeout: API请求超时时间（秒）
        """
        super().__init__(args, **kwargs)
        api_keys = self._resolve_api_keys(args, api_keys)
        self.api_list = api_keys
        self.max_distance = self._resolve_max_distance(args, max_distance)
        self.confidence_threshold = confidence_threshold
        self.timeout = timeout
        self.base_url = "https://api.opencagedata.com/geocode/v1/json"
        self.query_count = 0  # 查询计数器
        self.daily_limit = 100000  # OpenCage每日限制
        self.max_qps = 15  # 每秒最大请求数
        self.quota_retry_seconds = float(os.environ.get('GEOSCORE_QUOTA_RETRY_SECONDS', '600'))
        self.rate_limit_retry_seconds = float(os.environ.get('GEOSCORE_RATE_LIMIT_RETRY_SECONDS', '30'))
        self.current_gpu = self._get_current_gpu()
        key_index = self.current_gpu if self.current_gpu >= 0 else 0
        self.api_key = api_keys[key_index % len(api_keys)]
        self.cache_file = os.environ.get('GEOSCORE_CACHE_FILE', cache_file)
        self.cache = {}
        print(
            f"GPU{self.current_gpu},AK={self._mask_api_key(self.api_key)},"
            f"max_distance={self.max_distance},cache={self.cache_file or '<memory-only>'}")

    @staticmethod
    def _split_api_keys(api_keys: Union[str, List[str], None]) -> List[str]:
        if api_keys is None:
            return []
        if isinstance(api_keys, str):
            raw_keys = [api_keys]
        else:
            raw_keys = api_keys
        keys = []
        for raw_key in raw_keys:
            for key in str(raw_key).replace(',', ' ').split():
                key = key.strip()
                if key:
                    keys.append(key)
        return keys

    def _resolve_api_keys(self, args, api_keys: Optional[List[str]]) -> List[str]:
        keys = self._split_api_keys(api_keys)
        if not keys and args is not None:
            keys = self._split_api_keys(getattr(args, 'geoscore_api_keys', None))
        if not keys:
            keys = self._split_api_keys(os.environ.get('GEOSCORE_API_KEYS'))
        if not keys:
            raise ValueError(
                'GeoScoreAccuracy requires OpenCage API keys. Pass --geoscore_api_keys to swift rlhf, '
                'or set GEOSCORE_API_KEYS.')
        return keys

    @staticmethod
    def _resolve_max_distance(args, max_distance: float) -> float:
        if args is not None and getattr(args, 'geoscore_max_distance', None) is not None:
            return float(getattr(args, 'geoscore_max_distance'))
        return float(max_distance)

    @staticmethod
    def _mask_api_key(api_key: str) -> str:
        if len(api_key) <= 8:
            return '***'
        return f'***{api_key[-4:]}'

    def _sanitize_api_error(self, error: Exception) -> str:
        message = str(error)
        for key in self.api_list:
            if key:
                message = message.replace(key, self._mask_api_key(key))
        return re.sub(r'([?&]key=)[^&\s]+', r'\1***', message)

    @staticmethod
    def _cache_key(address: str) -> str:
        normalized = address.replace(';', ',').replace('；', ',').replace('，', ',')
        normalized = re.sub(r'\s*,\s*', ',', normalized)
        normalized = re.sub(r'\s+', ' ', normalized).strip().lower()
        return normalized

    def _read_cache_file_unlocked(self):
        if not self.cache_file or not os.path.exists(self.cache_file):
            return {}
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"[GeoScoreAccuracy] 读取动态cache失败，将使用空cache: {self.cache_file}, error={e}")
            return {}

    def _get_cached_geocoding(self, address: str):
        key = self._cache_key(address)
        if key in self.cache:
            return self.cache[key]
        disk_cache = self._read_cache_file_unlocked()
        if key in disk_cache:
            self.cache[key] = disk_cache[key]
            return disk_cache[key]
        return None

    def _write_cache_once(self, address: str, result: Dict):
        key = self._cache_key(address)
        if key in self.cache:
            return
        if not self.cache_file:
            self.cache[key] = result
            return

        import fcntl

        cache_dir = os.path.dirname(self.cache_file)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        lock_file = f'{self.cache_file}.lock'
        with open(lock_file, 'w', encoding='utf-8') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            disk_cache = self._read_cache_file_unlocked()
            if key not in disk_cache:
                disk_cache[key] = result
                tmp_file = f'{self.cache_file}.tmp.{os.getpid()}'
                with open(tmp_file, 'w', encoding='utf-8') as f:
                    json.dump(disk_cache, f, ensure_ascii=False, indent=2)
                os.replace(tmp_file, self.cache_file)
            self.cache = disk_cache
            fcntl.flock(lock, fcntl.LOCK_UN)

    @staticmethod
    def _parse_json_object(text: str):
        if not text:
            return None
        try:
            data = json.loads(text.strip())
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
        for candidate in re.findall(r'\{.*?\}', text, flags=re.S):
            try:
                data = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                return data
        return None

    @staticmethod
    def _extract_answer_blocks(completion: str) -> List[str]:
        if not completion:
            return []
        return [
            block.strip()
            for block in re.findall(r"<answer>\s*(.*?)\s*</answer>", completion, flags=re.S | re.I)
        ]

    def _extract_answer_content(self, completion: str) -> str:
        """
        从模型输出中提取<answer>标签内的内容
        
        Args:
            completion: 模型生成的完整文本
            
        Returns:
            提取出的答案内容
        """
        if completion is None:
            return None

        answer_blocks = self._extract_answer_blocks(completion)
        if answer_blocks:
            answer_block = answer_blocks[-1]
            data = self._parse_json_object(answer_block)
            if isinstance(data, dict) and data.get("FinalAnswer") is not None:
                return str(data["FinalAnswer"]).strip()
            return answer_block

        data = self._parse_json_object(completion)
        if isinstance(data, dict) and data.get("FinalAnswer") is not None:
            return str(data["FinalAnswer"]).strip()

        final_answer_match = re.search(r'"FinalAnswer"\s*:\s*"([^"]+)"', completion, flags=re.S)
        if final_answer_match:
            return final_answer_match.group(1).strip()

        return None

    def _extract_pred_coordinates(self, completion: str):
        """
        从模型输出中提取预测经纬度。优先读取 <answer> 内 JSON，兼容 data_clean 的
        latlon_geoscore 输出，命中后不用再调用 OpenCage。
        """
        sources = self._extract_answer_blocks(completion)
        sources.append(completion)
        for source in sources:
            data = self._parse_json_object(source)
            if not isinstance(data, dict):
                continue
            lat = self._to_float(data.get('latitude', data.get('lat')))
            lon = self._to_float(data.get('longitude', data.get('lon', data.get('lng'))))
            if lat is not None and lon is not None and -90 <= lat <= 90 and -180 <= lon <= 180:
                return lat, lon
        return None
    
    def _smart_delay(self):
        import random
        import time
        delay = random.uniform(0,0.2)
        time.sleep(delay)

    @staticmethod
    def _is_usage_limit_status(code: int) -> bool:
        return code in {402, 429, 503}

    def _sleep_for_usage_limit(self, reason: str, retry_count: int, retry_after: Optional[Union[str, int, float]] = None):
        wait_seconds = self.rate_limit_retry_seconds
        if retry_after is not None:
            try:
                wait_seconds = max(wait_seconds, float(retry_after))
            except (TypeError, ValueError):
                pass
        if retry_count >= 5:
            wait_seconds = max(wait_seconds, self.quota_retry_seconds)
        print(
            f"[GeoScoreAccuracy3] OpenCage用量/限流限制: {reason}; "
            f"{wait_seconds:.0f}s后继续重试，不返回0奖励。"
        )
        time.sleep(wait_seconds)
    
    def _get_current_gpu(self):
        """
        在 Swift RLHF (colocate 模式) 单机多卡训练下，
        获取当前进程实际使用的 GPU 编号 (0~7)
        """
        import os
        import torch
        import torch.distributed as dist

        # 优先读取 LOCAL_RANK（torchrun/deepspeed 自动注入）
        local_rank = os.environ.get("LOCAL_RANK", None)
        if local_rank is not None:
            return int(local_rank)

        # 如果 Swift 已初始化分布式环境，从 dist.rank 推算 GPU ID
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            num_gpus = torch.cuda.device_count()
            return rank % num_gpus

        # 兜底：torch 当前设备
        if torch.cuda.is_available():
            return torch.cuda.current_device()

        # 无法确定（CPU 或初始化前）
        return -1
    
    def _call_opencage_geocoding(self, address: str, retry_count: int = 0) -> Dict:
        """
        调用OpenCage地理编码API
        
        Args:
            address: 要编码的地址
            retry_count: 重试次数
            
        Returns:
            包含坐标和置信度信息的字典
        """
        address = str(address).strip() if address is not None else ''
        if not address or re.fullmatch(r'[-+]?\d+(?:\.\d+)?', address):
            return {"status": "error", "message": "invalid geocoding query", "address": address}

        if retry_count == 0:
            cached_result = self._get_cached_geocoding(address)
            if cached_result is not None:
                return cached_result

        # 调用API
        self.query_count += 1
        if self.query_count % 500 == 0:
            print(f"[GeoScoreAccuracy3][GPU:{self.current_gpu}] 查询次数: {self.query_count}")
        
        # 检查是否超过本进程每日估算限制。不要返回error，否则会把奖励置0；等待后继续尝试。
        if self.query_count > self.daily_limit:
            self._sleep_for_usage_limit('超过本进程每日API调用估算限制', retry_count)
            self.query_count = 0
            return self._call_opencage_geocoding(address, retry_count + 1)
        
        # 智能延迟避免API限制
        self._smart_delay()
        
        params = {
            'q': address.replace(';', ',').replace('；', ',').replace('，', ','),
            'key': self.api_key,
            'language': 'en',  # 英文结果
            'limit': 1,
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=self.timeout)
            if response.status_code in {402, 429, 503}:
                self._sleep_for_usage_limit(
                    f'HTTP {response.status_code}',
                    retry_count,
                    response.headers.get('Retry-After'),
                )
                return self._call_opencage_geocoding(address, retry_count + 1)
            response.raise_for_status()
            data = response.json()
            
            # 检查配额信息
            if 'rate' in data and 'remaining' in data['rate']:
                remaining = data['rate']['remaining']
                if  ((remaining % 100) == 0) or (remaining<20):
                    print(
                        f"[GeoScoreAccuracy3] OpenCage API剩余配额 {remaining},"
                        f"API={self._mask_api_key(self.api_key)},GPU={self.current_gpu}")


            #成功查询到结果
            if data["status"]["code"]==200:
                results=data["results"]
                if len(results)==0:
                    print(f"[GeoScoreAccuracy3] 没有查询到结果，message:{data['status']['message']}, address:{address}")
                    # 尝试去掉最后一个地址组件重试
                    address_parts = address.split(';')
                    print(f"[GeoScoreAccuracy3] address_parts: {address_parts},len:{len(address_parts)}")
                    if len(address_parts) > 1:
                        # 移除最后一部分，重新组合
                        shorter_address = ';'.join(address_parts[:-1])
                        print(f"[GeoScoreAccuracy3] 没有查询结果,使用短地址{shorter_address}重试")
                        result = self._call_opencage_geocoding(shorter_address, retry_count)
                        self._write_cache_once(address, result)
                        return result
                    result = {"status": "error", "code": 200, "message": data['status']['message'], "address": address}
                    self._write_cache_once(address, result)
                    return result

                else:
                    results=results[0]
                    # 提取坐标
                    coordinates = results['geometry']
                    lat = coordinates['lat']
                    lng = coordinates['lng']
                
                    # 提取地址组件
                    components = results.get('components', {})
                    formatted_address = results.get('formatted', '')
                    confidence = results.get('confidence', 0)
                
                    api_result = {
                        "status": "success",
                        "longitude": lng,
                        "latitude": lat,
                        "confidence": confidence,
                        "formatted_address": formatted_address,
                        "components": components
                    }
                
                    self._write_cache_once(address, api_result)
                    return api_result

            # 用量/速率限制：一直等待重试，不写入错误cache，也不返回0奖励。
            if self._is_usage_limit_status(data["status"]["code"]):
                self._sleep_for_usage_limit(
                    f"{data['status']['code']}: {data['status'].get('message', '')}",
                    retry_count,
                )
                return self._call_opencage_geocoding(address, retry_count + 1)
            #地址太长
            if data["status"]["code"]==410:
                print(f"[GeoScoreAccuracy3] 地址太长: message:{data['status']['message']}, address:{address}")
                # 尝试去掉最后一个地址组件重试
                address_parts = address.split(';')
                if len(address_parts) > 1:
                    # 移除最后一部分，重新组合
                    shorter_address = ';'.join(address_parts[:-1])
                    result = self._call_opencage_geocoding(shorter_address, retry_count)
                    self._write_cache_once(address, result)
                    return result
                else:
                    result = {"status": "error", "code": 410, "message": data['status']['message'], "address": address}
                    self._write_cache_once(address, result)
                    return result
            #超时
            if data["status"]["code"]==408:
                print(f"[GeoScoreAccuracy3] 超时: message:{data['status']['message']}, address:{address}")
                if retry_count < 5:
                    print(f"[GeoScoreAccuracy3] 超时，第{retry_count+1}次重试")
                    time.sleep(0.1)
                    return self._call_opencage_geocoding(address, retry_count + 1)
                else:
                    result = {"status": "error", "code": 408, "message": data['status']['message'], "address": address}
                    self._write_cache_once(address, result)
                    return result

            #禁止
            if data["status"]["code"]==403:
                print(f"[GeoScoreAccuracy3] 禁止: message:{data['status']['message']}, address:{address}")
                result = {"status": "error", "code": 403, "message": data['status']['message'], "address": address}
                self._write_cache_once(address, result)
                return result

            #其他错误
            else:
                print(f"[GeoScoreAccuracy3] OpenCage API其他错误: {data['status']}")
                result = {
                    "status": "error",
                    "code": data["status"]["code"],
                    "message": data['status']['message'],
                    "address": address
                }
                self._write_cache_once(address, result)
                return result

                
        except requests.exceptions.RequestException as e:
            safe_error = self._sanitize_api_error(e)
            print(f"[GeoScoreAccuracy] OpenCage API请求时发生错误: {safe_error}")
            
            # 网络错误可以重试
            if retry_count < 5:
                print(f"[GeoScoreAccuracy3] OpenCage API请求时发生错误，第{retry_count+1}次重试")
                time.sleep(0.1)
                return self._call_opencage_geocoding(address, retry_count + 1)
            else:
                result = {"status": "error", "code": 0, "message": safe_error, "address": address}
                self._write_cache_once(address, result)
                return result
        except Exception as e:
            result = {"status": "error", "code": 999, "message": str(e), "address": address}
            self._write_cache_once(address, result)
            return result
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        计算两个经纬度点之间的距离（公里）
        使用Haversine公式
        
        Args:
            lat1, lon1: 第一个点的纬度和经度
            lat2, lon2: 第二个点的纬度和经度
            
        Returns:
            两点之间的距离（公里）
        """
        # 将角度转换为弧度
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine公式
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        # 地球半径（公里）
        r = 6371
        return c * r
    
    def _calculate_geoscore(self, distance: float) -> float:
        """
        根据距离计算GeoScore
        
        Args:
            distance: 距离（公里）
            
        Returns:
            GeoScore分数
        """
        if distance >= self.max_distance:
            return 0.0
        return math.exp(-10 * (distance / self.max_distance))
    
    @staticmethod
    def _as_list(value, length: int):
        if value is None:
            return [None] * length
        if isinstance(value, list):
            return value
        try:
            import numpy as np
            if isinstance(value, np.ndarray):
                return value.tolist()
        except Exception:
            pass
        try:
            import torch
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().tolist()
        except Exception:
            pass
        return [value] * length

    @staticmethod
    def _to_float(value):
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _get_indexed_value(self, value, index: int, length: int):
        values = self._as_list(value, length)
        if not values:
            return None
        if len(values) == 1 and index > 0:
            return values[0]
        if index >= len(values):
            return None
        return values[index]

    def _get_gt_coordinates(self, index: int, **kwargs):
        lat_keys = ('latitude', 'gt_latitude', 'lat')
        lon_keys = ('longitude', 'gt_longitude', 'lon', 'lng')
        lat = None
        lon = None
        for key in lat_keys:
            values = kwargs.get(key)
            if values is not None:
                lat = self._to_float(self._get_indexed_value(values, index, index + 1))
                break
        for key in lon_keys:
            values = kwargs.get(key)
            if values is not None:
                lon = self._to_float(self._get_indexed_value(values, index, index + 1))
                break
        if lat is None or lon is None:
            return None
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            print(f"[GeoScoreAccuracy] 标准坐标越界: latitude={lat}, longitude={lon}")
            return None
        return lat, lon

    def _get_gt_coordinates_from_solution(self, sol):
        if sol is None:
            return None
        if isinstance(sol, dict):
            lat = self._to_float(sol.get('latitude', sol.get('gt_latitude')))
            lon = self._to_float(sol.get('longitude', sol.get('gt_longitude', sol.get('lng'))))
            if lat is not None and lon is not None:
                return lat, lon
        try:
            data = json.loads(sol) if isinstance(sol, str) else None
            if isinstance(data, dict):
                lat = self._to_float(data.get('latitude', data.get('gt_latitude')))
                lon = self._to_float(data.get('longitude', data.get('gt_longitude', data.get('lng'))))
                if lat is not None and lon is not None:
                    return lat, lon
        except Exception:
            pass
        return None

    def __call__(self, completions: List[str], solution: Optional[Union[str, List[str]]] = None, **kwargs) -> List[float]:
        """
        计算基于API的GeoScore准确性奖励
        
        Args:
            completions: 模型生成的回答列表
            solution: 标准答案列表或单个标准答案
            **kwargs: 其他参数
            
        Returns:
            List[float]: 奖励分数列表
        """
        rewards = []

        solutions = self._as_list(solution, len(completions))
        if len(solutions) == 1 and len(completions) > 1:
            solutions = solutions * len(completions)
        elif len(solutions) != len(completions):
            print(f"[GeoScoreAccuracy] completions({len(completions)}) 与 solutions({len(solutions)}) 长度不匹配，solution回退为空")
            solutions = [None] * len(completions)

        for i, (completion, sol) in enumerate(zip(completions, solutions)):
            try:
                # 提取答案内容
                pred_answer = self._extract_answer_content(completion)
                # print(f"pred_answer: {pred_answer}")
                # exit()
                if not pred_answer:
                    rewards.append(0.0)
                    continue

                # 预测端只使用文本地点。不要信任模型自报经纬度，统一通过 OpenCage 地理编码后计算距离。
                pred_result = self._call_opencage_geocoding(pred_answer)
                if pred_result.get("status") != "success":
                    rewards.append(0.0)
                    print(f"[GeoScoreAccuracy3] 预测答案失败: {pred_answer},奖励为0")
                    continue

                pred_lng = pred_result.get("longitude")
                pred_lat = pred_result.get("latitude")

                gt_coordinates = self._get_gt_coordinates(i, **kwargs) or self._get_gt_coordinates_from_solution(sol)
                if gt_coordinates is not None:
                    gt_lat, gt_lng = gt_coordinates
                else:
                    gt_answer = self._extract_answer_content(str(sol))
                    if not gt_answer:
                        rewards.append(0.0)
                        print("[GeoScoreAccuracy] 未提供标准经纬度或solution，奖励为0")
                        continue

                    # 兼容旧数据：调用OpenCage API获取标准答案的坐标和置信度。
                    # _call_opencage_geocoding 内部会先查动态cache，未命中才请求API。
                    gt_result = self._call_opencage_geocoding(gt_answer)
                    if gt_result.get("status") != "success":
                        rewards.append(0.0)
                        print(f"[GeoScoreAccuracy3] 标准答案失败: {gt_answer},奖励为0")
                        continue

                    gt_lng = gt_result.get("longitude")
                    gt_lat = gt_result.get("latitude")
                
                # 检查坐标有效性
                if (pred_lng is None or pred_lat is None or 
                    gt_lng is None or gt_lat is None):
                    rewards.append(0.0)
                    print(f"[GeoScoreAccuracy] 坐标无效: {pred_lng}, {pred_lat}, {gt_lng}, {gt_lat},奖励为0")
                    continue
                
                # 计算距离
                distance = self._calculate_distance(pred_lat, pred_lng, gt_lat, gt_lng)
                
                # 计算GeoScore
                geoscore = self._calculate_geoscore(distance)
                
                rewards.append(float(geoscore))
                
            except Exception as e:
                print(f"[GeoScoreAccuracy] 处理单个样本时发生错误: {e}")
                rewards.append(0.0)
        
        return rewards

orms = {
    'toolbench': ReactORM,
    'math': MathORM,
    'accuracy': MathAccuracy,
    'format': Format,
    'react_format': ReActFormat,
    'cosine': CosineReward,
    'repetition': RepetitionPenalty,
    'soft_overlong': SoftOverlong,
    'geo_format': GeoAnswerFormat,
    'geo_no_unknown': GeoNoUnknown,
    'geo_strict_format': GeoStrictFormat,
    'geoscore_accuracy':GeoScoreAccuracy,
}
