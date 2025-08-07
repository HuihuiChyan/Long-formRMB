"""
语言模型响应生成和评估流水线

本脚本实现了一个完整的流水线，用于：
1. 加载和过滤对话数据集
2. 使用各种语言模型（包括基于API和本地模型）生成响应
3. 使用自动化指标评估响应质量
4. 在流水线的不同阶段保存结果

作者: HuihuiChyan
日期: 2025/05/05
"""

import pandas as pd
import numpy as np
import re
import json
from openai import OpenAI
import openai
from collections import Counter
from typing import List, Dict, Any, Tuple
import time
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import datasets
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import argparse
from whale import TextGeneration
from whale.util import Timeout
import random

#------------------------------------------------------------------------------
# API交互函数
#------------------------------------------------------------------------------
def call_gpt_api(prompt: str, model_name: str = "gpt-4o-0806", max_tokens: int = 65535, 
                 temperature: float = 0.3) -> str:
    """
    发起单个API调用以获取模型响应，包含重试机制。
    
    参数:
        prompt (str): 发送给模型的输入文本
        model_name (str): 使用的模型名称
        max_tokens (int): 响应中的最大token数
        temperature (float): 生成的采样温度
        
    返回:
        str: 模型的响应文本，如果所有重试都失败则返回None
    """
    # API调用设置
    API_KEY = os.getenv('OPENAI_API_KEY', 'put-your-key-here')
    client = OpenAI(api_key=API_KEY, base_url="https://idealab.alibaba-inc.com/api/openai/v1")
    API_RETRY_ATTEMPTS = 10  # API调用失败时的重试次数
    API_RETRY_DELAY = 5     # 重试间隔（秒）

    for attempt in range(API_RETRY_ATTEMPTS):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content
            if model_name in ["deepseek-r1", "grok-3-reason", "claude-3-7-sonnet-20250219-thinking"]:
                # 查找thinking trace确保其存在
                thinking_pattern = r"<think>.*?</think>"
                matches = re.findall(thinking_pattern, content, flags=re.DOTALL)
                assert len(matches) == 1
                # # 去掉<think>标签及其内容
                # content = re.sub(thinking_pattern, "", content, flags=re.DOTALL).strip()
                # return {"trace": matches[0], "content": content}

            return content

        except Exception as e:
            print(f"Model {model_name} API call error (attempt {attempt + 1}/{API_RETRY_ATTEMPTS}): {e}")
            if attempt < API_RETRY_ATTEMPTS - 1:
                time.sleep(API_RETRY_DELAY)
    return ""

def call_one_api(prompt: str, model_name: str, max_tokens: int = 65535, 
                 temperature: float = 0.3) -> str:
    """
    发起单个API调用以获取模型响应，包含重试机制。
    
    参数:
        prompt (str): 发送给模型的输入文本
        model_name (str): 使用的模型名称
        max_tokens (int): 响应中的最大token数
        temperature (float): 生成的采样温度
        
    返回:
        str: 模型的响应文本，如果所有重试都失败则返回None
    """

    # API调用设置
    API_KEY = os.getenv('OPENAI_API_KEY', 'put-your-key-here')
    client = OpenAI(api_key=API_KEY, base_url="https://api.shubiaobiao.cn/v1")
    API_RETRY_ATTEMPTS = 5  # API调用失败时的重试次数
    API_RETRY_DELAY = 1     # 重试间隔（秒）

    for attempt in range(API_RETRY_ATTEMPTS):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content
            if model_name in ["deepseek-r1", "grok-3-reason", "claude-3-7-sonnet-20250219-thinking"]:
                # 查找thinking trace确保其存在
                thinking_pattern = r"<think>.*?</think>"
                matches = re.findall(thinking_pattern, content, flags=re.DOTALL)
                assert len(matches) == 1
                # # 去掉<think>标签及其内容
                # content = re.sub(thinking_pattern, "", content, flags=re.DOTALL).strip()
                # return {"trace": matches[0], "content": content}

            return content

        except Exception as e:
            print(f"Model {model_name} API call error (attempt {attempt + 1}/{API_RETRY_ATTEMPTS}): {e}")
            if attempt < API_RETRY_ATTEMPTS - 1:
                time.sleep(API_RETRY_DELAY)
    return ""

def call_gpt_api_multi_process(prompts: List[str], model_name: str = "gpt-4o-0806", max_tokens: int = 65535, 
                               temperature: float = 0.3, api_type: str = "alibaba") -> List[str]:
    """
    使用ThreadPoolExecutor并行处理多个提示。
    
    参数:
        prompts (List[str]): 要处理的提示列表
        model_name (str): 使用的模型名称
        max_tokens (int): 响应中的最大token数
        temperature (float): 生成的采样温度
        
    返回:
        List[str]: 模型响应列表
    """

    results = []
    with ThreadPoolExecutor(20) as executor:
        if api_type == "alibaba":
            call_gpt_api_func = partial(call_gpt_api, model_name=model_name, max_tokens=max_tokens,
                                        temperature=temperature)
        elif api_type == "oneapi":
            call_gpt_api_func = partial(call_one_api, model_name=model_name, max_tokens=max_tokens,
                                        temperature=temperature)        
        else:
            call_gpt_api_func = partial(call_whale_api, model_name=model_name, max_tokens=max_tokens,
                                        temperature=temperature)
        for entry in tqdm(executor.map(call_gpt_api_func, prompts), total=len(prompts)):
            results.append(entry)
    return results

def analyze_token_lengths(string_list: List[str]) -> Dict[str, float]:
    """
    使用 tokenizer 切分字符串列表，并统计token长度的指定分位数。

    Args:
        string_list: 需要切分和分析的字符串列表。

    Returns:
        一个字典，包含 'quantile_0.2', 'quantile_0.4', 'quantile_0.6',
        'quantile_0.8', 'quantile_1.0' 五个键，对应统计的0.2, 0.4, 0.6, 0.8, 1.0分位数。
        如果输入列表为空，返回所有分位数为 0 的字典。
    """
    # 使用 AutoTokenizer 从本地路径加载模型对应的tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained("/home/hh456524/WorkSpace/TuningFactoryModels/Qwen2.5-7B-Instruct")
    except Exception as e:
        print(f"Error loading tokenizer from /home/hh456524/WorkSpace/TuningFactoryModels/Qwen2.5-7B-Instruct: {e}")
        print("Please ensure the path is correct and contains tokenizer files.")
        # 返回一个包含0的字典或者根据需要处理错误
        return {f'quantile_{q}': 0.0 for q in [0.2, 0.4, 0.6, 0.8, 1.0]}

    token_lengths = []

    # 遍历字符串列表，对每个字符串进行tokenization并记录长度
    for text in string_list:
        # tokenizer.encode 方法将字符串转换为token ID列表
        # 这里我们只关心token的数量，所以直接取列表长度
        # 注意：不同的 tokenizer 对于特殊 token (如 bos, eos) 处理方式可能不同，
        #       encode() 通常会包含这些，len() 会计算它们。
        #       如果需要排除，可以使用 tokenizer(text, add_special_tokens=False)['input_ids']
        #       但对于统计长度，包含特殊 token 通常是合理的。
        tokens = tokenizer.encode(text)
        token_lengths.append(len(tokens))

    # 定义需要计算的分位数
    quantiles_to_calculate = [0.2, 0.4, 0.6, 0.8, 1.0]
    results = {}

    # 计算统计数据
    if not token_lengths:
        # 处理输入列表为空的情况
        for q in quantiles_to_calculate:
             results[f'quantile_{q}'] = 0.0 # 返回浮点数 0
    else:
        # 使用 numpy 计算分位数
        token_lengths_np = np.array(token_lengths)
        # numpy.quantile() 计算指定分位数
        calculated_quantiles = np.quantile(token_lengths_np, quantiles_to_calculate)

        # 将结果存入字典
        for i, q in enumerate(quantiles_to_calculate):
             # 将分位数结果存入字典，键名为 'quantile_0.x'
             results[f'quantile_{q}'] = float(calculated_quantiles[i]) # 确保是 float 类型

    return results

def filter_responses_top_k_by_subset_length(
    dataset: List[Dict],
    responses: List[str],
    tokenizer: Any, # 使用 Any 表示可以接受多种tokenizer类型
    top_k: int = 400
) -> Tuple[List[str], List[int]]:
    """
    根据不同子集的要求，从每个子集中选择长度最长的top_k条响应。
    
    参数:
        dataset (List[Dict]): 数据集，每项应包含 subset 信息。
        responses (List[str]): 生成的响应列表。
        tokenizer: 用于计算token长度的tokenizer。
        top_k (int): 每个子集保留的最大响应数量 (按长度排序)。
        
    返回:
        Tuple[List[str], List[int]]: 过滤并按子集取top_k后的响应列表和对应的原始有效索引。
    """
    # 用于按子集存储所有有效响应及其长度和原始索引
    # 结构: {subset_name: [(token_length, original_index, response_text), ...]}
    subset_all_candidates = {}
    
    # 第一步: 按子集分组，收集所有有效响应及其长度
    for i, (response, example) in enumerate(zip(responses, dataset)):
   
        # 计算 token 长度
        tokens = tokenizer.encode(response)
        token_length = len(tokens)
        
        subset = example['subset']
        # 将响应、长度和原始索引添加到对应的子集列表中
        if subset not in subset_all_candidates:
            subset_all_candidates[subset] = []
        subset_all_candidates[subset].append((token_length, i, response))
    
    # 第二步: 对于每个子集，按长度排序并保留最长的top_k条响应
    final_filtered_responses = []
    final_valid_indices = []
    
    # 遍历按子集分组后的候选字典
    for subset, candidates_list in subset_all_candidates.items():
        # 按 token_length 降序排序当前子集的所有候选响应
        sorted_subset_candidates = sorted(candidates_list, key=lambda x: x[0], reverse=True)
        
        # 取当前子集排序后的前 top_k 条响应
        top_k_subset_candidates = sorted_subset_candidates[:top_k]
        
        # 将选出的 top_k 响应及其原始索引添加到最终结果列表
        # 这里的顺序会是先处理一个子集的前 top_k，再处理下一个子集的前 top_k
        for token_length, original_index, response_text in top_k_subset_candidates:
            final_filtered_responses.append(response_text)
            final_valid_indices.append(original_index)

    # 注意: 最终返回的列表顺序是按子集分组处理的顺序，可能与原始顺序或全局长度排序不同。
    # 如果需要保持原始顺序，需要在最后根据 final_valid_indices 对 filtered_responses 进行重排。
    # 这里我们直接返回按处理顺序构建的列表。

    return final_filtered_responses, final_valid_indices

def generate_responses(dataset: List[Dict], model_name: str, max_tokens=65536, temperature=0.7, api_type="alibaba", do_filtering=False) -> List[Dict]:
    """
    使用指定的API模型为数据集中的所有提示生成响应，并根据子集过滤长度。
    
    参数:
        dataset (List[Dict]): 包含提示的示例列表
        model_name (str): 使用的模型名称
        
    返回:
        List[Dict]: 包含模型响应的更新数据集
    """
    if 'responses' in dataset[0].keys() and model_name in dataset[0]['responses'].keys():
        return dataset

    prompts = [example['prompt'] for example in dataset]
    print(f"Generating response using {model_name}...")
    if model_name in ["gpt-4o-0806"]:
        max_tokens = 16384
    elif model_name in ["claude35_haiku", "deepseek-v3"]:
        max_tokens = 8192
    else:
        max_tokens = 32768

    results = call_gpt_api_multi_process(
        prompts, 
        model_name=model_name, 
        max_tokens=max_tokens, 
        temperature=temperature,
        api_type=api_type,
    )

    if do_filtering:
        raise Exception("Please do not do filtering!")
        # 加载tokenizer用于长度过滤
        tokenizer = AutoTokenizer.from_pretrained("./Qwen2-7b")
        filtered_results, valid_indices = filter_responses_top_k_by_subset_length(dataset, results, tokenizer)
        print(f"Kept {len(filtered_results)}/{len(results)} responses after length filtering")
    else:
        filtered_results = results
        valid_indices = range(len(results)) 

    length_info = analyze_token_lengths(filtered_results)
    print(f"For {model_name}, length info after filtering:")
    print(length_info)
    
    # 只保存通过长度过滤的响应
    for i, result in enumerate(results):
        if 'responses' not in dataset[i].keys():
            dataset[i]['responses'] = {}
        if i not in valid_indices:
            dataset[i]['responses'][model_name] = ""
        else:
            filtered_idx = valid_indices.index(i)
            dataset[i]['responses'][model_name] = filtered_results[filtered_idx]
            
    return dataset

def generate_responses_vllm(dataset: List[Dict], model_path: str, do_filtering=False) -> List[Dict]:
    """
    使用vLLM通过本地托管模型生成响应，并根据子集过滤长度。
    
    参数:
        dataset (List[Dict]): 包含提示的示例列表
        model_path (str): 本地模型的路径
        
    返回:
        List[Dict]: 包含模型响应的更新数据集
    """
    model_name = model_path.split("/")[-1]
    if 'responses' in dataset[0].keys() and model_name in dataset[0]['responses'].keys():
        return dataset

    if model_name == "Meta-Llama-3-8B-Instruct":
        max_len = 8192
    else:
        max_len = 32768

    model = LLM(model=model_path, tensor_parallel_size=torch.cuda.device_count(), trust_remote_code=True, max_model_len=max_len, max_seq_len_to_capture=max_len)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    prompts = []
    for example in dataset:
        messages = [{"role": "user", "content": example['prompt']}]
        prompts.append(model.llm_engine.tokenizer.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))

    sampling_params = SamplingParams(max_tokens=max_len, temperature=0.7, n=1,
                                     stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")])
    outputs = model.generate(prompt_token_ids=prompts, sampling_params=sampling_params)

    results = [output.outputs[0].text for output in outputs]

    if model_name == "QwQ-32B" or "DeepSeek-R1-Distill-Qwen" in model_name:
        results = ["<think>\n" + result for result in results]

    length_info = analyze_token_lengths(results)
    print(f"For {model_name}, length info is:")
    print(length_info)

    if do_filtering:
        # 加载tokenizer用于长度过滤
        filtered_results, valid_indices = filter_responses_top_k_by_subset_length(dataset, results, tokenizer, top_k=500)
        print(f"Kept {len(filtered_results)}/{len(results)} responses after length filtering")

        length_info = analyze_token_lengths(filtered_results)
        print(f"For {model_name}, length info after filtering:")
        print(length_info)

        # 只保存通过长度过滤的响应
        filtered_dataset = []
        for i, result in enumerate(results):
            if 'responses' not in dataset[i].keys():
                dataset[i]['responses'] = {}
            if i in valid_indices:
                filtered_idx = valid_indices.index(i)
                dataset[i]['responses'][model_name] = filtered_results[filtered_idx]
                filtered_dataset.append(dataset[i])
        dataset = filtered_dataset 
    else:
        for i, result in enumerate(results):
            if 'responses' not in dataset[i].keys():
                dataset[i]['responses'] = {}
            dataset[i]['responses'][model_name] = results[i]

    return dataset

def filter_prompts_by_quality(dataset: List[Dict]) -> List[Dict]:
    """
    基于LLM的质量评估过滤提示。
    使用质量评估提示对每个提示的质量进行评分，
    并过滤掉低质量提示（分数 < 5）。
    
    参数:
        dataset (List[Dict]): 输入数据集
        
    返回:
        List[Dict]: 只包含高质量提示的数据集
    """
    quality_prompt = """
Please evaluate the quality of the following prompt. Consider factors such as clarity, specificity, appropriateness, and potential to generate meaningful responses. Rate the prompt on a scale of 1-10 and provide a brief explanation.

Prompt to evaluate: "{prompt}"

Your evaluation should be in JSON format:
{{
    "explanation": "<brief explanation, not exceed 50 words>",
    "quality_score": <score between 1 and 10>,
}}
"""
    prompts = [quality_prompt.format(prompt=example['prompt']) for example in dataset]
    responses = call_gpt_api_multi_process(prompts, max_tokens=2048)
    new_dataset = []
    for i, response in enumerate(responses):
        if response.startswith("```json"):
            response = response[7:-3]
        try:
            evaluation = json.loads(response)
        except json.JSONDecodeError:
            # print(f"Failed to parse quality assessment result: {response}")
            evaluation = {
                "explanation": "Failed to parse assessment result",
                "quality_score": 0,
            }
        if evaluation['quality_score'] >= 5:
            dataset[i]["prompt_evaluation"] = evaluation
            new_dataset.append(dataset[i])

    print(f"Data before quality filtering: {len(dataset)}, after quality filtering: {len(new_dataset)}")
    return new_dataset

def generate_prompt_checklist(dataset: List[Dict]) -> List[Dict[str, Any]]:
    """
    使用LLM为每个提示生成评估标准检查表。
    创建5-8个带权重的评估项目用于对响应评分。
    
    参数:
        dataset (List[Dict]): 输入数据集
        
    返回:
        List[Dict]: 添加了评估检查表的数据集
    """
    checklist_prompt = """
Create a detailed evaluation checklist for responses to the following prompt. 
Include about 5 items to evaluate quality, accuracy, completeness, conciseness, and relevance.

Prompt: "{prompt}"

Format as JSON array:
[
    {{
        "criterion": "<criterion name>",
        "description": "<detailed description>",
        "weight": <importance 1-5>
    }},
    ...
]
Please only return the JSON array. Do not generate any other opennings or closings.
"""
    prompts = []
    for example in dataset:
        prompts.append(checklist_prompt.format(prompt=example['prompt']))

    print(f"Constructing checklist for evaluation.")
    responses = call_gpt_api_multi_process(prompts, model_name="gpt-4o-0806", max_tokens=4096)
    examples = []
    for i, response in enumerate(responses):
        try:
            if response.startswith("```json"):
                response = response[7:-3]
            checklist = json.loads(response)
            
            # Normalize weights to sum to 10
            total_weight = sum(item['weight'] for item in checklist)
            for j, item in enumerate(checklist):
                checklist[j]['weight'] = int((item['weight'] / total_weight) * 10)        
            dataset[i]['checklist'] = checklist

        except:
            print(f"Failed to parse checklist: {response}")
            dataset[i]['checklist'] = [
                {"criterion": "Relevance", "description": "Response addresses main points", "weight": 3},
                {"criterion": "Accuracy", "description": "Information is factually correct", "weight": 3},
                {"criterion": "Completeness", "description": "Covers all aspects", "weight": 2},
                {"criterion": "Clarity", "description": "Well-structured and clear", "weight": 1},
                {"criterion": "Depth", "description": "Provides sufficient detail", "weight": 1}
            ]
    return dataset

def evaluate_responses(dataset: List[Dict], model_name: str, is_reason = False) -> List[Dict]:
    """
    使用生成的检查表评估特定模型的响应质量。
    基于评估标准计算加权分数。
    
    参数:
        dataset (List[Dict]): 包含响应和检查表的数据集
        model_name (str): 要评估的模型名称
        
    返回:
        List[Dict]: 添加了评估分数的数据集
    """
    if is_reason:
        evaluation_prompt = """
Evaluate the following response and reasoning trace based on the provided checklist and golden answer.

PROMPT: {prompt}
RESPONSE: {response}
CHECKLIST: {checklist}
GOLDEN ANSWER: {reference}

Return evaluation as the following JSON format:
[
    {{
        "criterion": "<criterion name>",
        "score": <1-10>,
        "explanation": "<brief explanation>",
        "weighted_score": <score × weight>
    }},
    ...
]
Please only return the JSON array. Do not generate any other opennings or closings.
"""
    else:
        evaluation_prompt = """
Evaluate the following response based on the provided checklist.

PROMPT: {prompt}
RESPONSE: {response}
CHECKLIST: {checklist}

Return evaluation as the following JSON format:
[
    {{
        "criterion": "<criterion name>",
        "score": <1-10>,
        "explanation": "<brief explanation>",
        "weighted_score": <score × weight>
    }},
    ...
]
Only return the JSON, do not generate any other openings or closings.
"""
    prompts = []
    for i, example in enumerate(dataset):
        question = example['prompt']
        answer = example['responses'][model_name]

        if is_reason:
            # 查找thinking trace确保其存在
            thinking_pattern = r"<think>.*?</think>"
            try:
                matches = re.findall(thinking_pattern, answer, flags=re.DOTALL)
                assert len(matches) == 1
                # 去掉<think>标签及其内容
                answer = re.sub(thinking_pattern, "", answer, flags=re.DOTALL).strip()
            except:
                answer = "" # 如果没有<think>那就answer置为空
                dataset[i]['responses'][model_name] = ""

        if not is_reason:
            prompt = evaluation_prompt.format(
                prompt = question,
                response = answer,
                checklist = json.dumps(example['checklist'], indent=2))
        else:
            prompt = evaluation_prompt.format(
                prompt = question,
                response = answer,
                reference = example['reference'],
                checklist = json.dumps(example['checklist'], indent=2))

        prompts.append(prompt)
    print(f"Evaluating response of {model_name}...")
    responses = call_gpt_api_multi_process(prompts, model_name="gpt-4o-0806", max_tokens=16384, temperature=0.1)
    
    for i, response in enumerate(responses):
        if dataset[i]['responses'][model_name] == "":
            total_score = 0
        else:
            try:
                if response.startswith("```json"):
                    response = response[7:-3]
                result = json.loads(response.strip())
                total_score = sum(item.get('weighted_score', 0) for item in result)
            except:
                print(f"Failed to parse evaluation result: {response}")
                result = ""
                total_score = 0

        if "evaluations" not in dataset[i].keys():
            dataset[i]['evaluations'] = {}
        dataset[i]['evaluations'][model_name] = total_score
    
    return dataset

#------------------------------------------------------------------------------
# 文件输入输出函数
#------------------------------------------------------------------------------

def load_jsonline(file_path: str) -> List[Dict]:
    """
    从JSONL文件加载数据集。
    
    参数:
        file_path (str): JSONL文件的路径
        
    返回:
        List[Dict]: 加载的数据集
    """
    with open(file_path, "r") as fin:
        lines = [json.loads(line.strip()) for line in fin.readlines()]
    return lines

def dump_jsonline(lines: List[Dict], file_path: str) -> None:
    """
    将数据集保存到JSONL文件。
    
    参数:
        lines (List[Dict]): 要保存的数据集
        file_path (str): 输出文件路径
    """
    with open(file_path, "w") as fout:
        for line in lines:
            fout.write(json.dumps(line, ensure_ascii=False)+"\n")

#------------------------------------------------------------------------------
# 评分提取函数
#------------------------------------------------------------------------------

def separate_cot(answer: str) -> Tuple[str, str]:
    """
    从包含CoT的回答中分离出CoT部分和最终回答部分。
    
    参数:
        answer (str): 包含CoT的回答字符串
        
    返回:
        Tuple[str, str]: (cot_part, answer_part)
    """
    thinking_pattern = r""
    matches = re.findall(thinking_pattern, answer, flags=re.DOTALL)

    if len(matches) == 0:
        return "", answer

    answer_part = re.sub(thinking_pattern, "", answer, flags=re.DOTALL).strip()
    return "", answer_part

def extract_highest_score_response(json_data: Dict, id: int, min_resp_num: int, min_low_num = 1) -> Dict:
    """
    从一个JSON数据项中随机选取6个回复，从中选出分数最高的作为chosen，
    再从剩余5个中随机选取3个分数较低的作为rejected。

    参数:
        json_data (Dict): 包含evaluations, responses等字段的数据项
        id (int): 数据项ID
        
    返回:
        Dict: 包含处理后的数据的字典，如果不满足条件则返回None
    """
    subset_name = json_data.get('subset', "RAG")
    required_fields = ['evaluations', 'responses', 'prompt']
    for field in required_fields:
        if field not in json_data:
            raise ValueError(f"JSON数据中缺少必需的字段: '{field}'")

    evaluations = json_data['evaluations']
    responses = json_data['responses']
    prompt = json_data['prompt']

    # 确保至少有6个回复
    if len(evaluations) < min_resp_num:
        return None

    # 随机选取6个回复
    model_keys = list(evaluations.keys())
    selected_keys = random.sample(model_keys, min_resp_num)
    
    # 从6个中找出分数最高的
    selected_scores = [(key, evaluations[key]) for key in selected_keys]
    chosen_key, chosen_score = max(selected_scores, key=lambda x: x[1])
    
    # 从剩余5个中找出分数比chosen低的
    remaining_keys = [key for key in selected_keys if key != chosen_key]
    lower_score_keys = [key for key in remaining_keys if evaluations[key] < chosen_score]
    
    # 如果低分回复不足3个，返回None
    if len(lower_score_keys) < min_low_num:
        return None
        
    # 随机选取3个低分回复
    rejected_keys = random.sample(lower_score_keys, min_low_num)
    
    # 随机选择一个rejected作为answer_2
    answer_2_key = random.choice(rejected_keys)
    
    line = {
        "subset": subset_name,
        "id": f"{subset_name}_{id}",
        "question": prompt,
        "answer_1": responses[chosen_key],
        "answer_2": responses[answer_2_key],
        "answer_1_model": chosen_key,
        "answer_2_model": answer_2_key,
        "better": 1,  # 因为answer_1总是chosen，所以better固定为1
        "rejected_models": [key for key in rejected_keys if key != answer_2_key]  # 保存其他rejected的model
    }

    reference = json_data.get('reference')
    if reference is not None:
        line['reference'] = reference

    return line

def fetch_score(dataset: List[Dict], min_resp_num = 4) -> List[Dict]:
    """
    从数据集中提取评分数据并生成偏好对。流程：
    1. 对每个样本随机选取6个回复
    2. 从6个中选取分数最高的作为chosen
    3. 从剩余5个中随机选取3个分数较低的作为rejected
    4. 如果没有足够的低分回复，则跳过该样本
    
    参数:
        dataset (List[Dict]): 输入数据集
        
    返回:
        List[Dict]: 处理后的偏好对数据集
    """
    preference_data = []
    skipped_count = 0
    insufficient_responses_count = 0
    insufficient_low_scores_count = 0

    print("开始处理数据集...")
    for i, line_data in enumerate(tqdm(dataset)):
        del line_data['responses']['DeepSeek-R1-Distill-Qwen-14B']
        del line_data['evaluations']['DeepSeek-R1-Distill-Qwen-14B']
        processed_line = extract_highest_score_response(line_data, i, min_resp_num)
        if processed_line is not None:
            preference_data.append(processed_line)
        else:
            if len(line_data['evaluations']) < min_resp_num:
                insufficient_responses_count += 1
            else:
                insufficient_low_scores_count += 1
            skipped_count += 1

    print(f"\n成功处理 {len(preference_data)} 对数据")
    print(f"跳过 {skipped_count} 项数据，其中：")
    print(f"- 回复数量不足6个的样本：{insufficient_responses_count}项")
    print(f"- 低分回复不足3个的样本：{insufficient_low_scores_count}项")

    return preference_data

def verify_preference_with_llm(preference_data: List[Dict], model_name: str = "gemini-2.5-pro-preview-05-06", is_reason = False) -> List[Dict]:
    """
    使用LLM验证偏好对的判断结果。
    
    参数:
        preference_data (List[Dict]): 偏好对数据集
        model_name (str): 使用的模型名称
        
    返回:
        List[Dict]: 验证后的偏好对数据集
    """
    verified_data = []
    skipped_count = 0
    
    if is_reason:
        comparison_prompt = """请根据参考答案，仔细比较以下两个回答，并判断哪个回答更好。请先用中文详细说明你的判断理由，然后在最后一行输出一个数字（1或2）。

问题：{question}
参考答案：{reference}

回答1：
{answer1}

回答2：
{answer2}

请先用中文提供详细的判断理由，然后在最后一行只输出数字1或2："""
    else:
        comparison_prompt = """请仔细比较以下两个回答，并判断哪个回答更好。请先用中文详细说明你的判断理由，然后在最后一行输出一个数字（1或2）。

问题：{question}

回答1：
{answer1}

回答2：
{answer2}

请先用中文提供详细的判断理由，然后在最后一行只输出数字1或2："""        

    prompts = []
    for item in preference_data:
        if is_reason:
            prompt = comparison_prompt.format(
                question=item['question'],
                answer1=item['answer_1'],
                answer2=item['answer_2'],
                reference=item['reference']
            )
        else:
            prompt = comparison_prompt.format(
                question=item['question'],
                answer1=item['answer_1'],
                answer2=item['answer_2']
            )            
        prompts.append(prompt)
    
    print(f"使用{model_name}验证偏好对...")
    responses = call_gpt_api_multi_process(prompts, model_name=model_name)
    
    for item, response in zip(preference_data, responses):
        try:
            # 分离理由和选择
            response_lines = response.strip().split('\n')
            llm_choice = int(response_lines[-1].strip())
            reasoning = '\n'.join(response_lines[:-1]).strip()
            
            if llm_choice not in [1, 2]:
                print(f"跳过：LLM返回了无效的选择 {response}")
                skipped_count += 1
                continue
                
            # 检查LLM的判断是否与原始better标签一致
            if llm_choice == item['better']:
                item['verification_reasoning'] = reasoning  # 添加验证理由
                verified_data.append(item)
            else:
                skipped_count += 1
                
        except (ValueError, TypeError) as e:
            print(f"跳过：无法解析LLM响应 {response}")
            skipped_count += 1
            continue
    
    print(f"Verification finished! Keepping {len(verified_data)}/{len(responses)} valid preference pairs")

    return verified_data

#------------------------------------------------------------------------------
# 主执行流程
#------------------------------------------------------------------------------

if __name__ == "__main__":
    """
    主执行流水线：
    1. 基于答案长度和提示质量加载和过滤数据集
    2. 为过滤后的提示生成评估检查表
    3. 使用API和本地模型生成响应
    4. 使用生成的检查表评估响应
    5. 在每个阶段保存结果
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, default=None)
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument("--stage", type=str, default="processing", choices=("processing", "generation", "evaluation", "fetch_score", "verification"))
    parser.add_argument("--model-name", type=str, default="gpt-4o-0806")
    parser.add_argument("--model-type", type=str, default="alibaba", choices=("alibaba", "oneapi", "local", "whale"))
    parser.add_argument("--do-filtering", default=False, action="store_true")
    parser.add_argument("--is-reason", default=False, action="store_true")
    parser.add_argument("--is-debug", default=False, action="store_true")
    args = parser.parse_args()

    if os.path.exists(args.output_path):
        dataset = load_jsonline(args.output_path)
    else:
        dataset = load_jsonline(args.input_path)
    
    if args.is_debug:
        dataset = dataset[:50]

    if args.stage == "processing":  
        if 'prompt_evaluation' not in dataset[0].keys():
            # 按提示质量过滤
            dataset = filter_prompts_by_quality(dataset)
            dump_jsonline(dataset, args.output_path)

    if args.stage == "generation":
        if args.model_type == "local":
            dataset = generate_responses_vllm(dataset, args.model_name, args.do_filtering)
        else:
            dataset = generate_responses(dataset, args.model_name, api_type=args.model_type, do_filtering=args.do_filtering)
        dump_jsonline(dataset, args.output_path)        

    if args.stage == "evaluation":
        if 'checklist' not in dataset[0].keys():
            dataset = generate_prompt_checklist(dataset)
            dump_jsonline(dataset, args.output_path)

        for model_name in dataset[0]['responses'].keys():
            if 'evaluations' not in dataset[0].keys() or \
                model_name not in dataset[0]['evaluations'].keys():
                dataset = evaluate_responses(dataset, model_name, args.is_reason)
                dump_jsonline(dataset, args.output_path)

    if args.stage == "fetch_score":
        preference_data = fetch_score(dataset)
        # 保存为JSONL文件
        dump_jsonline(preference_data, args.output_path)
        print(f"Data has been written to '{args.output_path}' ({len(preference_data)} 行)")
        
    if args.stage == "verification":
        preference_data = load_jsonline(args.input_path)
        verified_data = verify_preference_with_llm(preference_data, is_reason=args.is_reason)
        dump_jsonline(verified_data, args.output_path)
        print(f"Verified data has been saved to '{args.output_path}'")
