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
import argparse
import random
import httpx

#------------------------------------------------------------------------------
# API交互函数
#------------------------------------------------------------------------------
def call_one_api(prompt: str, model_name: str, max_tokens: int, temperature: float) -> str:
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
    http_client = httpx.Client(timeout=60.0) 
    client = OpenAI(api_key=API_KEY, base_url="https://api.shubiaobiao.cn/v1", http_client=http_client)
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

        except httpx.TimeoutException as e: # Catch the specific timeout exception
            print(f"Model {model_name} API call timed out: {e}")
            return ""
        
        except Exception as e:
            print(f"Model {model_name} API call error (attempt {attempt + 1}/{API_RETRY_ATTEMPTS}): {e}")
            if attempt < API_RETRY_ATTEMPTS - 1:
                time.sleep(API_RETRY_DELAY)
    return ""

def call_gpt_api_multi_process(prompts: List[str], model_name: str, max_tokens: int, 
                               temperature: float, process_num: int=10) -> List[str]:
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
    with ThreadPoolExecutor(process_num) as executor:
        call_gpt_api_func = partial(call_one_api, model_name=model_name, max_tokens=max_tokens, temperature=temperature)        
        for entry in tqdm(executor.map(call_gpt_api_func, prompts), total=len(prompts)):
            results.append(entry)
    return results


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
Create a detailed evaluation checklist for responses to the following prompt. Include about 5 items to evaluate response quality.
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
        if example['prompt'].endswith("Your answer should be long enough to provide a comprehensive response."):
            example['prompt'] = example['prompt'][:-len("Your answer should be long enough to provide a comprehensive response.")].strip()
        if example['prompt'].endswith("Based on the references, the answer should be long enough to provide a comprehensive response."):
            example['prompt'] = example['prompt'][:-len("Based on the references, the answer should be long enough to provide a comprehensive response.")].strip()
        prompts.append(checklist_prompt.format(prompt=example['prompt']))

    print(f"Constructing checklist for evaluation.")
    responses = call_gpt_api_multi_process(prompts, model_name="gpt-4o", max_tokens=4096, temperature=0.7)
    examples = []
    for i, response in enumerate(responses):
        try:
            if response.startswith("```json"):
                response = response[7:-3]
            checklist = json.loads(response)
            
            # Normalize weights to sum to 10
            total_weight = sum(item['weight'] for item in checklist)
            for j, item in enumerate(checklist):
                checklist[j]['weight'] = round((item['weight'] / total_weight) * 10)        
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

        prompt = evaluation_prompt.format(
            prompt = question,
            response = answer,
            checklist = json.dumps(example['checklist'], indent=2))

        prompts.append(prompt)
    print(f"Evaluating response of {model_name}...")
    responses = call_gpt_api_multi_process(prompts, model_name="gpt-4o", max_tokens=16384, temperature=0.1)
    
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

def extract_highest_score_response(json_data: Dict, id: int) -> Dict:
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
    if len(evaluations) < 5:
        return None

    # 随机选取6个回复
    model_keys = list(evaluations.keys())
    selected_keys = random.sample(model_keys, 5)
    
    # 从6个中找出分数最高的
    selected_scores = [(key, evaluations[key]) for key in selected_keys]
    chosen_key, chosen_score = max(selected_scores, key=lambda x: x[1])
    
    # 从剩余5个中找出分数比chosen低的
    remaining_keys = [key for key in selected_keys if key != chosen_key]
    lower_score_keys = [key for key in remaining_keys if evaluations[key] < chosen_score]
    
    # 如果低分回复不足3个，返回None
    if len(lower_score_keys) < 3:
        return None
        
    # 随机选取3个低分回复
    rejected_keys = random.sample(lower_score_keys, 3)
    rejected_responses = [responses[key] for key in rejected_keys]
    
    line = {
        "subset": subset_name,
        "id": f"{subset_name}_{id}",
        "question": prompt,
        "answer_1": responses[chosen_key],
        "answer_2": rejected_responses,
        "answer_1_model": chosen_key,
        "answer_2_model": rejected_keys,
        "better": 1,  # 因为answer_1总是chosen，所以better固定为1
    }

    return line

def fetch_score(dataset: List[Dict]) -> List[Dict]:
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
        processed_line = extract_highest_score_response(line_data, i)
        if processed_line is not None:
            preference_data.append(processed_line)
        else:
            if len(line_data['evaluations']) < 6:
                insufficient_responses_count += 1
            else:
                insufficient_low_scores_count += 1
            skipped_count += 1

    print(f"\n成功处理 {len(preference_data)} 对数据")
    print(f"跳过 {skipped_count} 项数据，其中：")
    print(f"- 回复数量不足5个的样本：{insufficient_responses_count}项")
    print(f"- 低分回复不足3个的样本：{insufficient_low_scores_count}项")

    return preference_data

def verify_preference_with_llm(preference_data: List[Dict], model_names: List[str], is_reason: bool = False) -> List[Dict]:
    """
    使用LLM验证偏好对的判断结果。
    
    参数:
        preference_data (List[Dict]): 偏好对数据集
        model_names (List[str]): 使用的模型名称列表
        is_reason (bool): 是否包含参考答案进行判断
        
    返回:
        List[Dict]: 验证后的偏好对数据集
    """
    verified_data = []
    skipped_count = 0
    
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
            raise Exception("is_reason参数目前不支持")
        else:
            for rejected_answer in item['answer_2']:
                prompt = comparison_prompt.format(
                    question=item['question'],
                    answer1=item['answer_1'],
                    answer2=rejected_answer,
                )            
                prompts.append(prompt)
    
    all_llm_choices_per_model = {}
    all_reasonings_per_model = {}

    for model_name in model_names:
        print(f"使用{model_name}验证偏好对...")
        # Call call_gpt_api_multi_process directly for each model
        responses = call_gpt_api_multi_process(prompts, model_name=model_name, max_tokens=16384, temperature=0.1)
        
        llm_choices = []
        reasonings = []

        for response in responses:
            try:
                # 分离理由和选择
                response_lines = response.strip().split('\n')
                llm_choice = int(response_lines[-1].strip())
                reasoning = '\n'.join(response_lines[:-1]).strip()

                assert llm_choice in [1, 2], f"LLM返回了无效的选择 {response}"

                llm_choices.append(llm_choice)
                reasonings.append(reasoning)
                    
            except (ValueError, TypeError) as e:
                reasoning = ""
                llm_choice = None
                print(f"Error processing response from {model_name}: {e} for response: {response}")
                llm_choices.append(llm_choice)
                reasonings.append(reasoning)
        
        all_llm_choices_per_model[model_name] = llm_choices
        all_reasonings_per_model[model_name] = reasonings
    
    # Aggregate results from all models
    for i, item in enumerate(preference_data):
        all_models_agree_on_1 = True
        current_reasonings = {} # To store reasoning from each model for this item

        for model_name in model_names:
            model_llm_choices = all_llm_choices_per_model.get(model_name, [])
            model_reasonings = all_reasonings_per_model.get(model_name, [])

            if not model_llm_choices or not model_reasonings:
                all_models_agree_on_1 = False
                break

            current_llm_choices_for_item = [model_llm_choices[i * len(item['answer_2']) + j] for j in range(len(item['answer_2']))]
            current_reasonings_for_item = [model_reasonings[i * len(item['answer_2']) + j] for j in range(len(item['answer_2']))]

            if not all(choice == 1 for choice in current_llm_choices_for_item):
                all_models_agree_on_1 = False
                break
            
            current_reasonings[model_name] = current_reasonings_for_item

        if all_models_agree_on_1:
            verified_data.append({
                "subset": item['subset'],
                "id": item['id'],
                "question": item['question'],
                "answer_1": item['answer_1'],
                "answer_2": item['answer_2'],
                "answer_1_model": item['answer_1_model'],
                "answer_2_model": item['answer_2_model'],
                "better": 1,
                "reasoning": current_reasonings,
            })
    
    print(f"Verification finished! Keeping {len(verified_data)}/{len(preference_data)} valid preference pairs")

    return verified_data

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

    dataset = load_jsonline(args.input_path)

    if args.is_debug:
        dataset = random.sample(dataset, min(50, len(dataset)))  # 仅用于调试，随机抽取100条数据

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
        verified_data = verify_preference_with_llm(dataset, model_names=["gpt-4o"],is_reason=args.is_reason)
        dump_jsonline(verified_data, args.output_path)
        print(f"Verified data has been saved to '{args.output_path}'")
