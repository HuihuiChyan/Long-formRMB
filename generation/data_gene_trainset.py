import pandas as pd
import numpy as np
import re
import json
from openai import OpenAI
import openai
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional
import time
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import datasets
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import torch
# from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import argparse
# from whale import TextGeneration # Assuming this might be conditionally available
# from whale.util import Timeout # Assuming this might be conditionally available
import random

def call_gpt_api(prompt: str, model_name: str = 'gpt-4o-0806', max_tokens: int = 16384,
                 temperature: float = 0.3, n: int = 1) -> Optional[str]:
    """
    发起单个API调用以获取模型响应，包含重试机制。

    参数:
        prompt (str): 发送给模型的输入文本
        model_name (str): 使用的模型名称
        max_tokens (int): 响应中的最大token数
        temperature (float): 生成的采样温度

    返回:
        Optional[str]: 模型的响应文本，如果所有重试都失败则返回None
    """
    API_KEY = os.getenv('OPENAI_API_KEY', 'put-your-key-here')
    client = OpenAI(api_key=API_KEY, base_url="https://idealab.alibaba-inc.com/api/openai/v1")
    API_RETRY_ATTEMPTS = 5
    API_RETRY_DELAY = 1

    if not prompt:
        return None
    
    if 'gpt-4o' in model_name:
        for attempt in range(API_RETRY_ATTEMPTS):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n=n
                )
                contents = [choice.message.content for choice in response.choices]
                return contents
            except Exception as e:
                print(f"Model {model_name} API call error (attempt {attempt + 1}/{API_RETRY_ATTEMPTS}): {e}")
                if "Error code: 429" in str(e):
                    return [None] * n  # 如果是429错误，直接返回None
                if attempt < API_RETRY_ATTEMPTS - 1:
                    time.sleep(API_RETRY_DELAY * (attempt + 1))
                else:
                    print(f"All {API_RETRY_ATTEMPTS} retries failed for model {model_name} with prompt starting: {prompt[:50]}...")
        return [None] * n
    else:
        contents = [None] * n
        for i in range(n):
            for attempt in range(API_RETRY_ATTEMPTS):
                try:
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        max_tokens=max_tokens,
                        n=1
                    )
                    contents[i] = response.choices[0].message.content
                    break
                except Exception as e:
                    print(f"Model {model_name} API call error (attempt {attempt + 1}/{API_RETRY_ATTEMPTS}): {e}")
                    # if "Error code: 429" in str(e):
                    #     return [None] * n  # 如果是429错误，直接返回None
                    if attempt < API_RETRY_ATTEMPTS - 1:
                        time.sleep(API_RETRY_DELAY * (attempt + 1))
                    else:
                        print(f"All {API_RETRY_ATTEMPTS} retries failed for model {model_name} with prompt starting: {prompt[:50]}...")
        return contents
    

def call_one_api(prompt: str, model_name: str = 'gpt-4o-0806', max_tokens: int = 65535,
                 temperature: float = 0.3, n: int = 1) -> List[str]: # Modified to return List[str]
    """
    发起单个API调用以获取模型响应，包含重试机制。

    参数:
        prompt (str): 发送给模型的输入文本
        model_name (str): 使用的模型名称
        max_tokens (int): 响应中的最大token数
        temperature (float): 生成的采样温度
        n (int): 为每个prompt生成的完成次数

    返回:
        List[str]: 模型的响应文本列表，如果所有重试都失败则返回[None] * n
    """
    API_KEY = os.getenv('OPENAI_API_KEY', 'put-your-key-here')
    client = OpenAI(api_key=API_KEY, base_url="https://api.shubiaobiao.cn/v1")
    API_RETRY_ATTEMPTS = 5
    API_RETRY_DELAY = 1

    for attempt in range(API_RETRY_ATTEMPTS):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                n=n
            )
            contents = [choice.message.content for choice in response.choices]
            return contents
            # Similar post-processing as call_gpt_api if needed
        except Exception as e:
            print(f"Model {model_name} API call error (attempt {attempt + 1}/{API_RETRY_ATTEMPTS}): {e}")
            if attempt < API_RETRY_ATTEMPTS - 1:
                time.sleep(API_RETRY_DELAY)
    return [None] * n


def call_gpt_api_multi_process(prompts: List[str], model_name: str = "gpt-4o-0806", max_tokens: int = 65535,
                               temperature: float = 0.3, api_type: str = "alibaba", n: int = 1) -> List[List[str]]: # Returns List of Lists
    """
    使用ThreadPoolExecutor并行处理多个提示。
    
    参数:
        prompts (List[str]): 要处理的提示列表
        model_name (str): 使用的模型名称
        max_tokens (int): 响应中的最大token数
        temperature (float): 生成的采样温度
        api_type (str): API类型 ("alibaba", "oneapi", "whale")
        n (int): 每个prompt生成的完成次数
        
    返回:
        List[List[str]]: 模型响应列表的列表 (outer list for prompts, inner list for n responses per prompt)
    """
    results = []
    with ThreadPoolExecutor(20) as executor:
        if api_type == "alibaba":
            call_api_func = partial(call_gpt_api, model_name=model_name, max_tokens=max_tokens,
                                    temperature=temperature, n=n)
        elif api_type == "oneapi":
            call_api_func = partial(call_one_api, model_name=model_name, max_tokens=max_tokens,
                                    temperature=temperature, n=n)
        elif api_type == "whale":
            if not WHALE_AVAILABLE:
                print("Whale SDK not available. Cannot use 'whale' api_type.")
                return [[None]*n for _ in prompts] # Return list of lists of Nones
            call_api_func = partial(call_whale_api, model_name=model_name, max_tokens=max_tokens,
                                    temperature=temperature, n=n)
        else:
            raise ValueError(f"Unsupported api_type: {api_type}")

        for entry_list in tqdm(executor.map(call_api_func, prompts), total=len(prompts)):
            results.append(entry_list) # entry_list is a list of n responses
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
        # Consider making tokenizer path configurable or using a default one
        tokenizer_path = "/home/hh456524/WorkSpace/TuningFactoryModels/Qwen2.5-3B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception as e:
        print(f"Error loading tokenizer from {tokenizer_path}: {e}")
        print("Please ensure the path is correct and contains tokenizer files, or a valid Hugging Face model name.")
        # 返回一个包含0的字典或者根据需要处理错误
        return {f'quantile_{q}': 0.0 for q in [0.2, 0.4, 0.6, 0.8, 1.0]}

    token_lengths = []
    if not string_list: # Handle empty input list early
        return {f'quantile_{q}': 0.0 for q in [0.2, 0.4, 0.6, 0.8, 1.0]}

    for text in string_list:
        if text is None: # Handle None strings that might come from failed API calls
            token_lengths.append(0)
            continue
        tokens = tokenizer.encode(text)
        token_lengths.append(len(tokens))

    quantiles_to_calculate = [0.2, 0.4, 0.6, 0.8, 1.0]
    results = {}

    if not token_lengths: # Should be caught by the earlier check, but good for safety
        for q in quantiles_to_calculate:
             results[f'quantile_{q}'] = 0.0
    else:
        token_lengths_np = np.array(token_lengths)
        calculated_quantiles = np.quantile(token_lengths_np, quantiles_to_calculate)
        for i, q in enumerate(quantiles_to_calculate):
             results[f'quantile_{q}'] = float(calculated_quantiles[i])

    return results

def filter_responses_top_k_by_subset_length(
    dataset: List[Dict],
    responses_list_of_list: List[List[str]], # Each inner list has 'x' responses for a prompt
    original_indices: List[int], # Indices in the original dataset these responses correspond to
    tokenizer: Any,
    top_k: int = 400
) -> Tuple[List[List[str]], List[int]]:
    """
    根据不同子集的要求，从每个子集中选择长度最长的top_k条响应对应的 *prompt*。
    If a prompt is selected, all its 'x' responses are kept. Length is averaged over 'x' responses.
    
    参数:
        dataset (List[Dict]): 完整数据集.
        responses_list_of_list (List[List[str]]): 生成的响应列表的列表.
        original_indices (List[int]): responses_list_of_list 中每个条目对应的原始数据集索引.
        tokenizer: 用于计算token长度的tokenizer.
        top_k (int): 每个子集保留的最大响应数量 (按长度排序).
        
    返回:
        Tuple[List[List[str]], List[int]]: 过滤并按子集取top_k后的响应列表的列表和对应的原始有效索引.
    """
    subset_all_candidates = {} # {subset_name: [(avg_token_length, original_index, list_of_x_responses), ...]}
    
    for i, (x_responses, original_idx) in enumerate(zip(responses_list_of_list, original_indices)):
        if not x_responses or all(r is None for r in x_responses):
            avg_token_length = 0
        else:
            current_prompt_token_lengths = []
            for response_text in x_responses:
                if response_text:
                    tokens = tokenizer.encode(response_text)
                    current_prompt_token_lengths.append(len(tokens))
            avg_token_length = np.mean(current_prompt_token_lengths) if current_prompt_token_lengths else 0
        
        subset = dataset[original_idx]['subset']
        if subset not in subset_all_candidates:
            subset_all_candidates[subset] = []
        subset_all_candidates[subset].append((avg_token_length, original_idx, x_responses))
    
    final_filtered_responses_lol = []
    final_valid_indices = []
    
    for subset, candidates_list in subset_all_candidates.items():
        sorted_subset_candidates = sorted(candidates_list, key=lambda x_item: x_item[0], reverse=True)
        top_k_subset_candidates = sorted_subset_candidates[:top_k]
        
        for avg_len, original_idx, x_responses_for_prompt in top_k_subset_candidates:
            final_filtered_responses_lol.append(x_responses_for_prompt)
            final_valid_indices.append(original_idx)

    return final_filtered_responses_lol, final_valid_indices


def generate_responses(dataset: List[Dict], model_name: str, nums_to_process: int, num_responses_per_prompt: int,
                       max_tokens=65536, temperature=1.0, api_type="alibaba", do_filtering=False) -> List[Dict]:
    """
    使用指定的API模型为数据集中选定的提示生成响应。
    当前模型从第一个未被任何模型处理的数据开始，处理接下来的 nums_to_process 条数据。
    为每个选定的 prompt 生成 num_responses_per_prompt 个回复。
    """
    prompts_for_this_model = []
    original_indices_for_this_model = []
    
    # Find the first item not processed by any model
    start_processing_idx = -1
    for i, item in enumerate(dataset):
        if 'responses' not in item or not item['responses']: # No responses from any model yet
            start_processing_idx = i
            break
    
    if start_processing_idx == -1:
        print(f"Model {model_name}: No prompts found needing generation by any model.")
        return dataset # All prompts already have some responses

    # Collect nums_to_process prompts starting from start_processing_idx
    # that haven't been processed by any model yet.
    count_collected = 0
    for i in range(start_processing_idx, len(dataset)):
        if count_collected >= nums_to_process:
            break
        item = dataset[i]

        # Double check: ensure this item truly has no responses yet by any model
        # And also ensure this specific model hasn't somehow already processed it (e.g. in a re-run with different nums)
        if ('responses' not in item or not item['responses']) and (model_name not in item.get('responses', {})):
            prompts_for_this_model.append(item['prompt'])
            original_indices_for_this_model.append(i)
            count_collected += 1
            
    if not prompts_for_this_model:
        print(f"Model {model_name}: No suitable prompts found for generation in the assigned batch.")
        return dataset

    print(f"Model {model_name} generating {num_responses_per_prompt} responses for {len(prompts_for_this_model)} prompts...")
    
    # Adjust max_tokens based on model
    if model_name in ["gpt-4o", "gpt-4o-0806"]: 
        max_tokens = 16384
    elif model_name in ["claude35_haiku", "deepseek-v3"]: 
        max_tokens = 8192
    else: 
        max_tokens = 32768 # Default

    # results_lol is List[List[str]], outer list for prompts, inner for num_responses_per_prompt
    results_lol = call_gpt_api_multi_process(
        prompts_for_this_model,
        model_name=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
        api_type=api_type,
        n=num_responses_per_prompt
    )

    # Flatten results for length analysis if needed, or analyze per prompt
    all_individual_responses = [resp for sublist in results_lol for resp in sublist if resp]
    
    if do_filtering:
        tokenizer_path = "/mnt/bn/deepresearch/Models/Qwen3-4B" # Or make configurable
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # filter_responses_top_k_by_subset_length expects original_indices relative to the full dataset
        filtered_results_lol, final_valid_original_indices = filter_responses_top_k_by_subset_length(
            dataset, results_lol, original_indices_for_this_model, tokenizer
        )
        print(f"Kept {len(filtered_results_lol)} prompts (each with {num_responses_per_prompt} responses) out of {len(results_lol)} after length filtering for model {model_name}")
        all_individual_responses_after_filter = [resp for sublist in filtered_results_lol for resp in sublist if resp]
        length_info = analyze_token_lengths(all_individual_responses_after_filter)
    else:
        filtered_results_lol = results_lol
        final_valid_original_indices = original_indices_for_this_model
        length_info = analyze_token_lengths(all_individual_responses)

    print(f"For model {model_name}, length info (num_responses_per_prompt={num_responses_per_prompt}):")
    print(length_info)
    
    # Store responses back into the dataset
    for i, list_of_x_responses in enumerate(filtered_results_lol):
        original_dataset_idx = final_valid_original_indices[i]
        if 'responses' not in dataset[original_dataset_idx]:
            dataset[original_dataset_idx]['responses'] = {}
        dataset[original_dataset_idx]['responses'][model_name] = list_of_x_responses
    
    # For prompts that were candidates but got filtered out (if filtering is aggressive)
    # or failed API calls, ensure they don't have an entry for this model or have Nones
    for original_idx in original_indices_for_this_model:
        if original_idx not in final_valid_original_indices:
            if 'responses' not in dataset[original_idx]:
                dataset[original_idx]['responses'] = {}
            # Mark as processed with empty or None if it was attempted but filtered/failed
            # This logic might need refinement based on how "processed" is defined for filtered items
            dataset[original_idx]['responses'][model_name] = [None] * num_responses_per_prompt


    return dataset

def generate_responses_vllm(dataset: List[Dict], model_path: str, nums_to_process: int, 
                            num_responses_per_prompt: int, do_filtering=False) -> List[Dict]:
    """
    使用vLLM为数据集中选定的提示生成响应。
    逻辑与 generate_responses 类似。
    """
    model_name = model_path.split("/")[-1]

    prompts_for_this_model = []
    original_indices_for_this_model = []
    
    start_processing_idx = -1
    for i, item in enumerate(dataset):
        if 'responses' not in item or not item['responses']:
            start_processing_idx = i
            break
    
    if start_processing_idx == -1:
        print(f"vLLM Model {model_name}: No prompts found needing generation.")
        return dataset

    count_collected = 0
    for i in range(start_processing_idx, len(dataset)):
        if count_collected >= nums_to_process:
            break
        item = dataset[i]

        if ('responses' not in item or not item['responses']) and \
           (model_name not in item.get('responses', {})):
            
            # Prepare prompt for vLLM
            if any(word in model_name for word in ["Qwen3"]):
                content = item['prompt'] + "/no_think" # Specific handling for Qwen3
            else:
                content = item['prompt']
            # This part needs the tokenizer from the model to apply chat template
            # Temporarily load tokenizer here, or pass it in.
            # For now, assuming prompt is ready or simple user role.
            # messages = [{"role": "user", "content": content}]
            # This step should be done right before model.generate if templating is complex
            prompts_for_this_model.append(content) # This should be tokenized IDs or templated string
            original_indices_for_this_model.append(i)
            count_collected += 1

    if not prompts_for_this_model:
        print(f"vLLM Model {model_name}: No suitable prompts found for generation.")
        return dataset

    print(f"vLLM Model {model_name} generating {num_responses_per_prompt} responses for {len(prompts_for_this_model)} prompts...")

    max_len = 8192 if model_name == "Meta-Llama-3-8B-Instruct" else 32768
    
    # Initialize LLM and Tokenizer
    # This should ideally be done once if script handles multiple models or stages serially
    llm = LLM(model=model_path, tensor_parallel_size=torch.cuda.device_count(), trust_remote_code=True, max_model_len=max_len)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Apply chat template to prompts
    final_prompts_for_vllm = []
    for p_text in prompts_for_this_model:
        messages = [{"role": "user", "content": p_text}]
        templated_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        final_prompts_for_vllm.append(templated_prompt)


    sampling_params = SamplingParams(
        max_tokens=max_len, # Max new tokens
        temperature=1.0,
        n=num_responses_per_prompt,
        stop_token_ids=[tokenizer.eos_token_id] + ([tokenizer.convert_tokens_to_ids("<|eot_id|>")] if "<|eot_id|>" in tokenizer.vocab else [])
    )
    
    # outputs is List[RequestOutput]
    # each RequestOutput has .outputs, which is List[CompletionOutput] of length n
    request_outputs = llm.generate(final_prompts_for_vllm, sampling_params=sampling_params)

    # results_lol is List[List[str]]
    results_lol = []
    for req_output in request_outputs:
        prompt_responses = [comp_output.text for comp_output in req_output.outputs]
        # if model_name == "QwQ-32B" or "DeepSeek-R1-Distill-Qwen" in model_name: # Specific post-processing
        #     prompt_responses = ["<think>\n" + resp for resp in prompt_responses]
        if "Qwen3" in model_name:
            prompt_responses = [resp.split("<think>\n\n</think>\n\n") for resp in prompt_responses]
        results_lol.append(prompt_responses)

    all_individual_responses = [resp for sublist in results_lol for resp in sublist if resp]

    if do_filtering:
        # Tokenizer already loaded for vLLM
        filtered_results_lol, final_valid_original_indices = filter_responses_top_k_by_subset_length(
            dataset, results_lol, original_indices_for_this_model, tokenizer
        )
        print(f"Kept {len(filtered_results_lol)} prompts for vLLM model {model_name} after filtering.")
        all_individual_responses_after_filter = [resp for sublist in filtered_results_lol for resp in sublist if resp]
        length_info = analyze_token_lengths(all_individual_responses_after_filter)
    else:
        filtered_results_lol = results_lol
        final_valid_original_indices = original_indices_for_this_model
        length_info = analyze_token_lengths(all_individual_responses)
        
    print(f"For vLLM model {model_name}, length info (num_responses_per_prompt={num_responses_per_prompt}):")
    print(length_info)

    for i, list_of_x_responses in enumerate(filtered_results_lol):
        original_dataset_idx = final_valid_original_indices[i]
        if 'responses' not in dataset[original_dataset_idx]:
            dataset[original_dataset_idx]['responses'] = {}
        dataset[original_dataset_idx]['responses'][model_name] = list_of_x_responses
    
    for original_idx in original_indices_for_this_model:
        if original_idx not in final_valid_original_indices:
            if 'responses' not in dataset[original_idx]:
                dataset[original_idx]['responses'] = {}
            dataset[original_idx]['responses'][model_name] = [None] * num_responses_per_prompt
            
    # del llm # Release VRAM
    # torch.cuda.empty_cache()
    return dataset


def back_translation(dataset: List[Dict]) -> List[Dict]:
    """
    基于LLM产生doc对应的prompt
    
    参数:
        dataset (List[Dict]): 输入数据集
        
    返回:
        List[Dict]: 包含prompt的数据集
    """
    quality_prompt = """Please generate a single instruction that would lead to the given text as a response. 
- The instruction should not be a question. Instead, it should be a more general task.
- The instruction should be a complex instruction.

Here are some example instructions:
Example Instruction 1: In the context of episodes of a hypothetical legal/courtroom drama produced for British television called "County Court",  State 10 fictional civil cases which could plausibly have come before the fictional Millsbuy County Court.   The cases should reflect the typical civil matters such a court what deal with. In the generated descriptions, give a plausible case name,  3 sentence summary,  plaintiff counsel's argument, the defense counsel's argument, and a likely verdict. You can mention specific points of law relevant to each case if you like.
Example Instruction 2: Write 2500-word unique article about the topic "When you’re at your lowest, look to the highest." that is inspiring, easy-to-understand. Make the content punchy and engaging by using a conversational tone. incorporating relevant inspiring quotes.  Use HTML for headings and bullets. Don't use HTML code "<p>" for paragraphs (Leave them as plain text).
Example Instruction 3: Write a detailed literature review on the title "Design and develop an ontology driven model for the classification and analysis of public policy problems using deep learning techniques from social media tweets in the case of Ethiopia.". Use tables and add citations too.

Text: {doc}
Instruction: """
    prompts_to_llm = [quality_prompt.format(doc=example['document']) for example in dataset if 'prompt' not in example] # Process only if needed
    
    if not prompts_to_llm:
        return dataset

    # Assuming call_gpt_api_multi_process returns List[List[str]], take the first response from each inner list
    responses_lol = call_gpt_api_multi_process(prompts_to_llm, max_tokens=2048, n=1) # n=1 for this task
    
    new_dataset_indices = [i for i, example in enumerate(dataset) if 'prompt' not in example]
    
    processed_count = 0
    for i, (original_idx, resp_list) in enumerate(zip(new_dataset_indices, responses_lol)):
        response = resp_list[0] if resp_list and resp_list[0] else ""
        if len(response) > 5:
            dataset[original_idx]["prompt"] = response
            processed_count +=1
        else:
            dataset[original_idx]["prompt"] = "" # Mark as processed but failed

    print(f"Data before back_trans: {len(dataset)}, prompts generated/updated: {processed_count}")
    return dataset # Return the modified dataset

def filter_prompts_by_quality(dataset: List[Dict]) -> List[Dict]:
    """
    基于LLM的质量评估过滤提示。
    使用质量评估提示对每个提示的质量进行评分，
    并过滤掉低质量提示（分数 < 5）。
    
    参数:
        dataset (List[Dict]): 输入数据集
        
    返回:
        List[Dict]: 只包含高质量提示的数据集 (actually marks low quality, doesn't filter list)
    """
    quality_prompt_template = """
Please evaluate the quality of the following prompt. Consider factors such as clarity, specificity, appropriateness, and potential to generate meaningful responses. Rate the prompt on a scale of 1-10 and provide a brief explanation.

Prompt to evaluate: "{prompt}"

Your evaluation should be in JSON format:
{{
    "explanation": "<brief explanation, not exceed 50 words>",
    "quality_score": <score between 1 and 10>
}}
"""
    prompts_to_evaluate = []
    original_indices_to_evaluate = []

    for i, example in enumerate(dataset):
        if 'prompt' in example and example['prompt'] and 'prompt_evaluation' not in example : # Only if prompt exists and not evaluated
            prompts_to_evaluate.append(quality_prompt_template.format(prompt=example['prompt']))
            original_indices_to_evaluate.append(i)
    
    if not prompts_to_evaluate:
        return dataset

    responses_lol = call_gpt_api_multi_process(prompts_to_evaluate, max_tokens=2048, n=1) # n=1 for this

    updated_count = 0
    dataset_updated = []
    for i, (original_idx, resp_list) in enumerate(zip(original_indices_to_evaluate, responses_lol)):
        response_str = resp_list[0] if resp_list and resp_list[0] else None
        evaluation = {"explanation": "Failed to get or parse assessment", "quality_score": 0}
        if response_str:
            if response_str.startswith("```json"):
                response_str = response_str[7:-3]
            try:
                evaluation = json.loads(response_str)
            except json.JSONDecodeError:
                print(f"Failed to parse quality assessment result: {response_str}")
        
        dataset[original_idx]["prompt_evaluation"] = evaluation
        if evaluation['quality_score'] >= 8:
            updated_count += 1
            dataset_updated.append(dataset[original_idx])
            
    # The original function returned a new_dataset. To keep modification in-place:
    # We don't filter out here, just add the evaluation. Filtering can be a separate step if needed.
    print(f"Prompts evaluated for quality: {len(prompts_to_evaluate)}. Prompts meeting quality threshold (>=5): {updated_count}")
    return dataset_updated


def generate_prompt_checklist(dataset: List[Dict]) -> List[Dict[str, Any]]:
    """
    使用LLM为每个提示生成评估标准检查表。
    创建5-8个带权重的评估项目用于对响应评分。
    
    参数:
        dataset (List[Dict]): 输入数据集
        
    返回:
        List[Dict]: 添加了评估检查表的数据集
    """
    checklist_prompt_template = """
Create a detailed evaluation checklist for responses to the following prompt. 
Include about 5 items to evaluate quality, accuracy, completeness, and relevance.

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
    prompts_for_checklist = []
    original_indices_for_checklist = []

    for i, example in enumerate(dataset):
        # Only generate if prompt exists, is good quality (if evaluated), and checklist doesn't exist
        prompt_ok = 'prompt' in example and example['prompt']
        quality_ok = 'prompt_evaluation' not in example or example['prompt_evaluation'].get('quality_score', 0) >= 5
        checklist_missing = 'checklist' not in example

        if prompt_ok and quality_ok and checklist_missing:
            prompts_for_checklist.append(checklist_prompt_template.format(prompt=example['prompt']))
            original_indices_for_checklist.append(i)

    if not prompts_for_checklist:
        return dataset
        
    print(f"Constructing checklist for {len(prompts_for_checklist)} prompts.")
    # n=1 for this task
    responses_lol = call_gpt_api_multi_process(prompts_for_checklist, model_name="gpt-4o-0806", max_tokens=4096, n=1)
    
    default_checklist = [
        {"criterion": "Relevance", "description": "Response addresses main points", "weight": 3},
        {"criterion": "Accuracy", "description": "Information is factually correct", "weight": 3},
        {"criterion": "Completeness", "description": "Covers all aspects", "weight": 2},
        {"criterion": "Clarity", "description": "Well-structured and clear", "weight": 1},
        {"criterion": "Depth", "description": "Provides sufficient detail", "weight": 1}
    ]

    for i, (original_idx, resp_list) in enumerate(zip(original_indices_for_checklist, responses_lol)):
        response_str = resp_list[0] if resp_list and resp_list[0] else None
        checklist = default_checklist # Default if parsing fails
        if response_str:
            try:
                if response_str.startswith("```json"):
                    response_str = response_str[7:-3]
                parsed_checklist = json.loads(response_str)
                
                total_weight = sum(item.get('weight', 0) for item in parsed_checklist if isinstance(item, dict))
                if total_weight > 0: # Avoid division by zero
                    for item_idx, item in enumerate(parsed_checklist):
                        if isinstance(item, dict) and 'weight' in item:
                             parsed_checklist[item_idx]['weight'] = int((item['weight'] / total_weight) * 10)
                    checklist = parsed_checklist
                else: # if weights are missing or all zero, use default
                    print(f"Warning: Checklist for prompt {original_idx} had zero total weight or invalid items. Using default.")

            except Exception as e: # Catch broader exceptions during parsing/processing
                print(f"Failed to parse or process checklist for prompt {original_idx}: {response_str}. Error: {e}. Using default.")
        
        dataset[original_idx]['checklist'] = checklist
    return dataset


def evaluate_responses(dataset: List[Dict], model_name: str, is_reason=False) -> List[Dict]:
    """
    使用生成的检查表评估特定模型的响应质量。
    """
    evaluation_prompt_template_no_reason = """
Evaluate the following response based on the provided checklist.
PROMPT: {prompt}
RESPONSE: {response}
CHECKLIST: {checklist}
Return evaluation as the following JSON format:
[ {{ "criterion": "<criterion name>", "score": <1-10>, "explanation": "<brief explanation>", "weighted_score": <score × weight> }}, ... ]
Only return the JSON, do not generate any other openings or closings.
"""
    prompts_for_llm_eval = []
    # Store tuples of (original_dataset_idx, response_idx_within_model_responses)
    # to map results back correctly.
    map_back_info = [] 
    
    processed_prompt_count = 0

    for i, example in enumerate(dataset):

        # Check if this model has responses for this prompt
        if model_name not in example.get('responses', {}):
            continue
        
        # Check if this model's responses for this prompt have already been evaluated
        if model_name in example.get('evaluations', {}):
            continue # Already evaluated by this model

        # Check if checklist exists
        if 'checklist' not in example:
            print(f"Skipping evaluation for item {i}, model {model_name}: checklist missing.")
            continue

        # This prompt is a candidate for evaluation by this model
        model_responses_list = example['responses'][model_name] # This is a list of x responses
        if not isinstance(model_responses_list, list):
            print(f"Warning: Responses for item {i}, model {model_name} is not a list. Skipping.")
            continue

        question = example['prompt']
        checklist_json = json.dumps(example.get('checklist'), indent=2) # Use default if missing

        for resp_idx, single_response_text in enumerate(model_responses_list):
            if single_response_text is None: # Skip if a response is None (e.g. API failure)
                continue

            current_response_to_eval = single_response_text

            prompt_text_for_evaluator = evaluation_prompt_template_no_reason.format(
                prompt=question,
                response=current_response_to_eval,
                checklist=checklist_json
            )

            prompts_for_llm_eval.append(prompt_text_for_evaluator)
            map_back_info.append({'original_idx': i, 'response_idx': resp_idx})
        
        processed_prompt_count += 1 # Counted one prompt (with all its x responses)

    if not prompts_for_llm_eval:
        print(f"Model {model_name}: No responses found needing evaluation by this model in the assigned batch.")
        return dataset

    print(f"Model {model_name} evaluating {len(prompts_for_llm_eval)} individual responses (from {processed_prompt_count} prompts)...")
    # n=1 as each call to evaluator is for one response
    llm_eval_results_lol = call_gpt_api_multi_process(prompts_for_llm_eval, model_name="gpt-4o-0806", max_tokens=16384, n=1)
    # llm_eval_results_lol = call_gpt_api_multi_process(prompts_for_llm_eval, model_name="gpt-4o", api_type="oneapi", max_tokens=16384, n=1)

    # Initialize evaluations structure in dataset items if not present
    for info in map_back_info:
        item = dataset[info['original_idx']]
        if 'evaluations' not in item:
            item['evaluations'] = {}
        if model_name not in item['evaluations']:
            # Initialize with Nones for all x responses, to be filled in
            num_x_responses = len(item['responses'].get(model_name, []))
            item['evaluations'][model_name] = [None] * num_x_responses

    for i, eval_result_list in enumerate(llm_eval_results_lol):
        eval_result_str = eval_result_list[0] if eval_result_list and eval_result_list[0] else None
        info = map_back_info[i]
        original_dataset_idx = info['original_idx']
        response_idx_in_model_list = info['response_idx']

        item_data = dataset[original_dataset_idx]
        
        # If the original response text was empty or None, score is 0
        original_response_text = item_data['responses'][model_name][response_idx_in_model_list]
        if not original_response_text: # Handles None or empty string
            total_score = 0
        elif not eval_result_str: # Evaluator failed
            total_score = 0
        else:
            try:
                if eval_result_str.startswith("```json"):
                    eval_result_str = eval_result_str[7:-3]
                parsed_eval = json.loads(eval_result_str.strip())
                total_score = sum(eval_item.get('weighted_score', 0) for eval_item in parsed_eval if isinstance(eval_item, dict))
            except Exception as e: # Catch JSONDecodeError and others
                print(f"Failed to parse evaluation result for item {original_dataset_idx}, model {model_name}, response_idx {response_idx_in_model_list}: {eval_result_str}. Error: {e}")
                total_score = 0
        
        # Ensure the list is long enough (should be due to pre-initialization)
        if len(item_data['evaluations'][model_name]) > response_idx_in_model_list:
            item_data['evaluations'][model_name][response_idx_in_model_list] = total_score
        else:
            # This case should ideally not happen if pre-initialization is correct
            print(f"Warning: evaluation list for item {original_dataset_idx}, model {model_name} too short.")
            # Pad with Nones and then add score (less ideal)
            padding_needed = response_idx_in_model_list + 1 - len(item_data['evaluations'][model_name])
            item_data['evaluations'][model_name].extend([None] * padding_needed)
            item_data['evaluations'][model_name][response_idx_in_model_list] = total_score
            
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
    with open(file_path, "r", encoding='utf-8') as fin: # Added encoding
        lines = [json.loads(line.strip()) for line in fin.readlines()]
    return lines

def dump_jsonline(lines: List[Dict], file_path: str) -> None:
    """
    将数据集保存到JSONL文件。
    
    参数:
        lines (List[Dict]): 要保存的数据集
        file_path (str): 输出文件路径
    """
    with open(file_path, "w", encoding='utf-8') as fout: # Added encoding
        for line in lines:
            fout.write(json.dumps(line, ensure_ascii=False)+"\n")

#------------------------------------------------------------------------------
# 评分提取函数
#------------------------------------------------------------------------------

def extract_responses_for_prompt(item_data: Dict, response_list: List, score_list: List, model_names: List) -> Dict:
    """
    从一个JSON数据项中，针对特定模型生成的多个回复中，提取评分最高和最低的两个回复。
    如果评分相同，则随机选择。如果回复少于2个，则返回None。

    参数:
        json_data_item (Dict): 包含 responses, evaluations, prompt 等字段的数据项.
                               responses[model_to_focus_on] is a list of strings.
                               evaluations[model_to_focus_on] is a list of scores.
        model_to_focus_on (str): 当前负责挑选回复的模型名称.
        item_id_suffix (str): 用于构成唯一ID的后缀 (e.g., original dataset index).
        
    返回:
        Dict: 包含处理后的偏好对数据的字典，如果无法形成对则返回None.
    """

    # Sort by score
    indexed_data = [(value, index) for index, value in enumerate(score_list)]
    sorted_by_score = sorted(indexed_data)

    sorted_indices = [item[1] for item in sorted_by_score]

    best_index = sorted_indices[-1]
    worst_index = sorted_indices[0]
    
    chosen_response_text = response_list[best_index]
    reject_response_text = response_list[worst_index]

    chosen_model = model_names[best_index]
    reject_model = model_names[worst_index]

    if len(list(set(score_list))) == 1:
        return None

    # Randomize order for preference pair
    if random.random() > 0.5:
        ans_1_resp = chosen_response_text
        ans_2_resp = reject_response_text
        better_flag = 1
    else:
        ans_1_resp = reject_response_text
        ans_2_resp = chosen_response_text
        better_flag = 2

    line = {
        "subset": "train",
        "question": item_data["prompt"],
        "answer_1": ans_1_resp,
        "answer_2": ans_2_resp,
        "answer_1_model": chosen_model, # Both from the same model
        "answer_2_model": reject_model, # Both from the same model
        "better": better_flag,
    }

    return line


def fetch_score(dataset: List[Dict]) -> List[Dict]:
    """
    从数据集中提取评分数据并生成偏好对。
    当前模型负责为 nums_to_process 个 prompt 创建偏好对，
    条件是该模型为这些 prompt 生成了 >=2 个回复且已评估。
    """
    preference_data = []
    processed_prompt_count = 0
    skipped_count = 0

    for i, item_data in enumerate(dataset):

        if 'responses' not in item_data:
            continue
        if 'evaluations' not in item_data:
            continue

        model_keys = list(item_data['responses'].keys())

        model_responses = []
        model_evals = []
        model_names = []
        for model_key in model_keys:
            # Check conditions: model has responses, evaluations, and at least 2 responses
            model_responses.extend(item_data['responses'][model_key]) 
            model_evals.extend(item_data['evaluations'][model_key])
            model_names.extend([model_key] * len(item_data['responses'][model_key]))

        if not (len(model_responses) >= 2 and len(model_evals) == len(model_responses)):
            continue # Conditions not met for this model and item

        # This item is a candidate for this model to process for fetch_score
        # Pass original index 'i' as item_id_suffix for unique preference ID
        processed_line = extract_responses_for_prompt(item_data, model_responses, model_evals, model_names)
        if processed_line is not None:
            preference_data.append(processed_line)
        else:
            skipped_count += 1 # extract_responses_from_model_for_prompt decided to skip

        processed_prompt_count += 1
            
    print(f"\nFetch_score: Successfully processed {len(preference_data)} preference pairs from {processed_prompt_count} prompts considered.")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} items during pair extraction.")

    return preference_data

#------------------------------------------------------------------------------
# 主执行流程
#------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--stage", type=str, default="generation", 
                        choices=("back_trans", "processing", "generation", "evaluation", "fetch_score", "verification"))
    parser.add_argument("--model-name", type=str, default="gpt-4o-0806", 
                        help="Name of the model for generation/evaluation, or path for local vLLM models. For verification, this is the verifier LLM.")
    parser.add_argument("--model-type", type=str, default="alibaba", choices=("alibaba", "oneapi", "local", "whale"))
    parser.add_argument("--nums", type=int, default=3, help="Number of data items the current model instance should process in this stage.")
    parser.add_argument("-x", "--num-responses-per-prompt", type=int, default=5, 
                        help="Number of responses to generate per prompt in the generation stage. Must be >= 2 for fetch_score to pick two from the same model.")
    parser.add_argument("--do-filtering", default=False, action="store_true", help="Enable length-based filtering after generation.")
    parser.add_argument("--is-reason", default=False, action="store_true", help="Enable reason/CoT specific logic in evaluation and verification.")
    parser.add_argument("--is-debug", default=False, action="store_true", help="Run on a small random sample of data for debugging.") # Changed default to False
    args = parser.parse_args()

    # Ensure num_responses_per_prompt is valid for fetch_score if that's the stage
    if args.stage == "fetch_score" and args.num_responses_per_prompt < 2:
        print(f"Warning: For 'fetch_score' stage, '--num-responses-per-prompt' (x) should be >= 2 for the model '{args.model_name}' to select two of its own responses. Current x = {args.num_responses_per_prompt}. Fetch score might not produce pairs.")

    # Load dataset: If output exists, load it (continuation). Else, load input.
    if os.path.exists(args.output_path):
        print(f"Loading existing data from output path: {args.output_path}")
        dataset = load_jsonline(args.output_path)
    else:
        print(f"Loading new data from input path: {args.input_path}")
        dataset = load_jsonline(args.input_path)
    
    if args.is_debug:
        sample_size = min(len(dataset), 20)
        dataset = random.sample(dataset, sample_size)
        print(f"Debug mode: Sampled {len(dataset)} items.")

    dataset = dataset

    if args.stage == "back_trans":  
        # Assuming back_trans processes all items that need it.
        dataset = back_translation(dataset) # Modifies in place
        dump_jsonline(dataset, args.output_path)
    
    elif args.stage == "processing":  
        # Similar to back_trans, processes items needing it.
        dataset = filter_prompts_by_quality(dataset) # Modifies in place
        dump_jsonline(dataset, args.output_path)

    elif args.stage == "generation":
        if args.model_type == "local":
            dataset = generate_responses_vllm(dataset, args.model_name, args.nums, args.num_responses_per_prompt, args.do_filtering)
        else:
            dataset = generate_responses(dataset, args.model_name, args.nums, args.num_responses_per_prompt, 
                                         api_type=args.model_type, do_filtering=args.do_filtering)
        dump_jsonline(dataset, args.output_path)        

    elif args.stage == "evaluation":
        # Checklist generation is a pre-requisite, typically done once for all prompts
        if not any('checklist' in item for item in dataset if item.get('prompt')): # Check if any relevant item has checklist
            print("Generating checklists as they seem to be missing...")
            dataset = generate_prompt_checklist(dataset) # Modifies in place

        dataset = evaluate_responses(dataset, args.model_name, args.is_reason)
        dump_jsonline(dataset, args.output_path)

    elif args.stage == "fetch_score":
        # `dataset` here is the one with responses and evaluations
        # `fetch_score` will generate new preference data based on `args.model_name`'s responses
        # The output is a NEW list of preference pairs, not modifying `dataset` in place.
        preference_data_this_model = fetch_score(dataset)
        
        # Append or overwrite logic for preference data
        # If output file exists, load existing preference data and append. 
        # Otherwise, save new preference data
        if os.path.exists(args.output_path):
            print(f"Appending new preference data to existing file: {args.output_path}")
            existing_preference_data = load_jsonline(args.output_path)
            combined_preference_data = existing_preference_data + preference_data_this_model
            # Deduplicate if necessary, based on 'id'
            seen_ids = set()
            unique_preference_data = []
            for item in combined_preference_data:
                if item['id'] not in seen_ids:
                    unique_preference_data.append(item)
                    seen_ids.add(item['id'])
                else:
                    print(f"Duplicate preference ID found and skipped: {item['id']}")
            dump_jsonline(unique_preference_data, args.output_path)
            print(f"Data has been updated in '{args.output_path}' ({len(unique_preference_data)} total rows)")
        else:
            dump_jsonline(preference_data_this_model, args.output_path)
            print(f"New preference data has been written to '{args.output_path}' ({len(preference_data_this_model)} rows)")