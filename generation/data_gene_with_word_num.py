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
import argparse
import random
import pandas as pd
import tiktoken
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

def call_gpt_api(prompt: str, model_name: str = 'gpt-4o-0806', max_tokens: int = 16384,
                 temperature: float = 0.3, n: int = 1) -> Optional[List[str]]:
    """
    发起单个API调用以获取模型响应，包含重试机制。

    参数:
        prompt (str): 发送给模型的输入文本
        model_name (str): 使用的模型名称
        max_tokens (int): 响应中的最大token数
        temperature (float): 生成的采样温度

    返回:
        Optional[List[str]]: 模型的响应文本列表，如果所有重试都失败则返回[None] * n
    """
    API_KEY = os.getenv('OPENAI_API_KEY', 'put-your-key-here')
    client = OpenAI(api_key=API_KEY, base_url="https://idealab.alibaba-inc.com/api/openai/v1")
    API_RETRY_ATTEMPTS = 5
    API_RETRY_DELAY = 1

    if not prompt:
        return [None] * n

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

def call_one_api(prompt: str, model_name: str = 'gpt-4o-0806', max_tokens: int = 65535,
                 temperature: float = 0.3, n: int = 1) -> List[str]:
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
        except Exception as e:
            print(f"Model {model_name} API call error (attempt {attempt + 1}/{API_RETRY_ATTEMPTS}): {e}")
            if attempt < API_RETRY_ATTEMPTS - 1:
                time.sleep(API_RETRY_DELAY)
    return [None] * n

def call_gpt_api_multi_process(prompts: List[str], model_name: str = "gpt-4o-0806", max_tokens: int = 65535,
                               temperature: float = 0.3, api_type: str = "alibaba", n: int = 1) -> List[List[str]]:
    """
    使用多线程处理多个提示。
    
    参数:
        prompts (List[str]): 要处理的提示列表
        model_name (str): 使用的模型名称
        max_tokens (int): 响应中的最大token数
        temperature (float): 生成的采样温度
        api_type (str): API类型 ("alibaba", "oneapi")
        n (int): 每个prompt生成的完成次数
        
    返回:
        List[List[str]]: 模型响应列表的列表 (outer list for prompts, inner list for n responses per prompt)
    """
    results = []
    with ThreadPoolExecutor(10) as executor:
        if api_type == "alibaba":
            call_api_func = partial(call_gpt_api, model_name=model_name, max_tokens=max_tokens,
                                    temperature=temperature, n=n)
        elif api_type == "oneapi":
            call_api_func = partial(call_one_api, model_name=model_name, max_tokens=max_tokens,
                                    temperature=temperature, n=n)
        for entry_list in tqdm(executor.map(call_api_func, prompts), total=len(prompts)):
            results.append(entry_list) # entry_list is a list of n responses
    return results

def calculate_token_count(text: str, encoding_name: str = "o200k_base") -> int:
    """
    Calculate the number of tokens in a text using tiktoken.
    
    Parameters:
        text (str): The text to tokenize
        encoding_name (str): The encoding to use
        
    Returns:
        int: The number of tokens in the text
    """
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)
    return len(tokens)

def remove_word_count_requirements(instruction: str) -> str:
    """
    移除指令中的字数、长度等要求
    
    参数:
        instruction (str): 原始指令
        
    返回:
        str: 移除字数要求后的指令
    """
    # 构建移除字数要求的提示
    remove_prompt = f"""
Please remove any word count or length requirements from the following instruction, including: 
1. A specific word count (e.g., "write 500 words", "in 1000 words", etc.); 
2. Length requirements (e.g., "write a detailed response", "provide a comprehensive answer", "write a long essay", etc.);
3. Any other requirement that can decisively influence the response length (e.g., "include 50 cases", etc.);

Afther that, summarize the instruction in 10 words or less while preserving the core request.

Instruction: {instruction}

Return only the summarized instruction without any explanations.
"""
    
    # 调用API进行处理
    response = call_gpt_api(remove_prompt, model_name="gpt-4o-0806", max_tokens=1000, temperature=0.0, n=1)
    
    if response and response[0]:
        return response[0].strip()
    return instruction

def add_word_count_requirement(instruction: str, word_count: int) -> str:
    """
    在指令末尾添加字数要求
    
    参数:
        instruction (str): 原始指令
        word_count (int): 要求的字数
        
    返回:
        str: 添加了字数要求的指令
    """
    # 确保指令末尾有适当的标点符号
    instruction = instruction.rstrip()
    if not instruction[-1] in ['.', '!', '?', '。', '！', '？']:
        instruction += '.'
    
    # 添加字数要求
    return f"{instruction} Your response should be with {word_count} tokens."

def prepare_instructions_with_word_count(dataset: List[Dict], word_counts: List[int]) -> List[Dict]:
    """
    第一阶段：处理指令并添加字数要求
    
    参数:
        dataset (List[Dict]): 输入数据集
        word_counts (List[int]): 字数要求列表
        
    返回:
        List[Dict]: 包含所有字数要求变体的指令数据列表
    """
    # 初始化结果列表
    instructions_with_variations = []

    instruction_count = 0
    
    # 处理每个数据项
    for item in tqdm(dataset):
        instruction = item.get('question', '')
        subset = item.get('subset', '')

        if subset not in ['wildchat', 'writing']:
            continue 
        
        # 移除字数、长度等要求，将指令内容概括为10个词以内
        summarized_instruction = remove_word_count_requirements(instruction)
        
        # 创建结果项
        result_item = {
            'original_id': item.get('id', ''),
            'instruction': summarized_instruction
        }
        
        # 为每个字数要求生成指令变体
        for word_count in word_counts:
            modified_instruction = add_word_count_requirement(summarized_instruction, word_count)
            
            # 添加到对应字数要求的变体
            result_item[str(word_count)] = {
                'instruction': modified_instruction
            }
        
        # 添加到结果列表
        instructions_with_variations.append(result_item)

        instruction_count += 1
        # if instruction_count >= 5:
        #     break
    
    return instructions_with_variations


def generate_extension_prompt(instruction: str, current_response: str, required_word_count: int, current_word_count: int) -> str:
    """
    生成用于延长回复的提示
    
    参数:
        instruction (str): 原始指令
        current_response (str): 当前回复
        required_word_count (int): 要求的字数
        current_word_count (int): 当前字数
        
    返回:
        str: 延长回复的提示
    """
    return f"""
Your previous response to the following instruction was too short:

Instruction: {instruction}

Your response was {current_word_count} tokens, but it needs to be at least {required_word_count} tokens.

Here is your previous response:
{current_response}

Please modify your previous response to extend it into more than {required_word_count} tokens. 
ONLY return the extended response. Do not generate any other openings, closings or extra explanations.
"""

def process_single_instruction(instruction: str, model_name: str, api_type: str, 
                               max_tokens: int, temperature: float, max_attempts: int = 3) -> str:
    """
    处理单个指令，确保回复达到要求的字数
    
    参数:
        instruction (str): 指令
        model_name (str): 模型名称
        api_type (str): API类型
        max_tokens (int): 最大token数
        temperature (float): 生成的采样温度
        max_attempts (int): 最大尝试次数
        
    返回:
        str: 达到字数要求的回复
    """
    # 从指令中提取字数要求
    word_count_match = re.search(r'with (\d+) tokens.$', instruction)
    if not word_count_match:
        # 如果没有找到字数要求，直接生成回复
        if api_type == "alibaba":
            responses = call_gpt_api(instruction, model_name=model_name, max_tokens=max_tokens, temperature=temperature)
        else:
            responses = call_one_api(instruction, model_name=model_name, max_tokens=max_tokens, temperature=temperature)
        return responses[0] if responses and responses[0] else None
    
    required_word_count = int(word_count_match.group(1))
    
    # 初始生成回复
    if api_type == "alibaba":
        responses = call_gpt_api(instruction, model_name=model_name, max_tokens=max_tokens, temperature=temperature)
    else:
        responses = call_one_api(instruction, model_name=model_name, max_tokens=max_tokens, temperature=temperature)
    
    current_response = responses[0] if responses and responses[0] else ""
    if not current_response:
        return None
    
    # 计算当前字数
    current_word_count = calculate_token_count(current_response)
    
    # 如果字数不足，循环生成扩展内容
    attempts = 0
    tolerance_coeff_dict = {
        "1000": 1.5,
        "2000": 1.8,
        "3000": 1.8,
        "4000": 1.5,
        "5000": 1.2,
        "6000": 1.2,
        "7000": 1.1,
        "8000": 1.1,
    }
    tolerance_coeff = tolerance_coeff_dict[str(required_word_count)]
    while current_word_count * tolerance_coeff < required_word_count and attempts < max_attempts:

        # print(f"Response for model {model_name} has {current_word_count} words, extending to reach {required_word_count} words (attempt {attempts+1}/{max_attempts})...")
        
        # 生成扩展提示
        extension_prompt = generate_extension_prompt(
            instruction, 
            current_response, 
            required_word_count, 
            current_word_count,
        )
        # 获取扩展内容
        if api_type == "alibaba":
            extension_responses = call_gpt_api(extension_prompt, model_name=model_name, max_tokens=max_tokens, temperature=temperature)
        else:
            extension_responses = call_one_api(extension_prompt, model_name=model_name, max_tokens=max_tokens, temperature=temperature)
        
        extended_response = extension_responses[0] if extension_responses and extension_responses[0] else ""
        if not extended_response:
            break
        
        # 合并回复
        current_response = extended_response
        current_word_count = calculate_token_count(current_response)
        attempts += 1

        if current_word_count < 100:
            # print(f"Model {model_name} refuses to make a response, failed to reach {required_word_count} words (attempt {attempts+1}/{max_attempts})!")
            return None
    
    if current_word_count * tolerance_coeff > required_word_count:
        # print(f"Response for model {model_name} has {current_word_count} words, reaching {required_word_count} words already (attempt {attempts}/{max_attempts})!")
        return current_response
    else:
        # print(f"Response for model {model_name} has {current_word_count} words, failed to reach {required_word_count} words (attempt {attempts}/{max_attempts})!")
        return None

def process_instruction_item(item: Dict, word_counts: List[int], model_name: str, api_type: str,
                            max_tokens: int, temperature: float, max_attempts: int = 3, 
                            output_file: str = None) -> Dict:
    """
    处理单个指令项的所有字数变体
    
    参数:
        item (Dict): 指令项
        word_counts (List[int]): 字数要求列表（降序排列）
        model_name (str): 模型名称
        api_type (str): API类型
        max_tokens (int): 最大token数
        temperature (float): 生成的采样温度
        max_attempts (int): 最大尝试次数
        output_file (str): 输出文件路径，用于追加写入结果
        
    返回:
        Dict: 处理后的指令项，如果处理失败则返回包含valid_item=False的字典
    """
    item_copy = item.copy()
    valid_item = True
    
    # 按照字数要求的倒序处理每个变体
    for word_count in word_counts:
        word_count_str = str(word_count)
        if word_count_str not in item:
            continue
            
        instruction = item[word_count_str]['instruction']
        
        # 处理单个指令
        response = process_single_instruction(
            instruction=instruction,
            model_name=model_name,
            api_type=api_type,
            max_tokens=max_tokens,
            temperature=temperature,
            max_attempts=max_attempts
        )
        
        # 如果生成失败，标记整个指令为无效
        if response is None:
            print(f"Failed to generate response for instruction with word count {word_count}, discarding entire instruction.")
            valid_item = False
            break
        
        # 添加响应到结果中
        item_copy[word_count_str]['response'] = response
    
    # 添加valid_item标记
    item_copy["valid_item"] = valid_item
    
    with open(output_file, "a+", encoding='utf-8') as fout:
        fout.write(json.dumps(item_copy, ensure_ascii=False) + "\n")
    
    return item_copy

def generate_responses_for_instructions(input_dataset: List[Dict], model_name: str, api_type: str,
                                        max_tokens: int, temperature: float, max_attempts: int = 3,
                                        num_workers: int = 4, output_file: str = None) -> List[Dict]:
    """
    第二阶段：为带有字数要求的指令生成回复，按照字数要求的倒序处理
    
    参数:
        input_dataset (List[Dict]): 带有字数要求变体的指令数据集
        model_name (str): 模型名称
        api_type (str): API类型
        max_tokens (int): 最大token数
        temperature (float): 生成的采样温度
        max_attempts (int): 最大尝试次数
        num_workers (int): 并行处理的工作线程数
        output_file (str): 输出文件路径，用于追加写入结果
        
    返回:
        List[Dict]: 包含回复的结果数据
    """    
    print(f"Processing instructions for model {model_name} using {api_type} API with {num_workers} parallel workers...")
    
    # 调整max_tokens以适应不同模型
    adjusted_max_tokens = max_tokens
    if model_name in ["gpt-4o", "gpt-4o-0806"]:
        adjusted_max_tokens = min(16384, max_tokens)

    # 提取所有字数要求并按降序排序
    word_counts = []
    for item in input_dataset:
        for key in item.keys():
            if key.isdigit():
                word_count = int(key)
                if word_count not in word_counts:
                    word_counts.append(word_count)
    
    word_counts.sort(reverse=True)  # 按降序排序
    
    # 如果提供了输出文件，检查已经处理过的指令
    processed_instructions = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding='utf-8') as fin:
            for line in fin:
                item = json.loads(line.strip())
                if "instruction" in item:
                    processed_instructions.add(item["instruction"])
    
    # 过滤掉已经处理过的指令
    input_dataset_filtered = []
    for item in input_dataset:
        if item.get("instruction") not in processed_instructions:
            input_dataset_filtered.append(item)
    
    print(f"Found {len(processed_instructions)} already processed instructions, {len(input_dataset_filtered)} remaining to process.")
    
    # 创建处理函数的偏函数，固定部分参数
    process_func = partial(
        process_instruction_item,
        word_counts=word_counts,
        model_name=model_name,
        api_type=api_type,
        max_tokens=adjusted_max_tokens,
        temperature=temperature,
        max_attempts=max_attempts,
        output_file=output_file
    )
    
    # 使用ThreadPoolExecutor并行处理所有指令
    valid_items = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # 提交所有任务
        future_to_item = {executor.submit(process_func, item): item for item in input_dataset_filtered}
        
        # 收集结果
        for future in tqdm(as_completed(future_to_item), total=len(input_dataset_filtered), desc="Processing instructions"):
            result = future.result()
            if result.get("valid_item", False):
                valid_items.append(result)
            if len(valid_items) >= 150:
                break
    
    return valid_items

def load_jsonline(file_path: str) -> List[Dict]:
    """
    从JSONL文件加载数据集。
    
    参数:
        file_path (str): JSONL文件的路径
        
    返回:
        List[Dict]: 加载的数据集
    """
    with open(file_path, "r", encoding='utf-8') as fin:
        lines = [json.loads(line.strip()) for line in fin.readlines()]
    return lines

def dump_jsonline(lines: List[Dict], file_path: str) -> None:
    """
    将数据集保存到JSONL文件。
    
    参数:
        lines (List[Dict]): 要保存的数据集
        file_path (str): 输出文件路径
    """
    with open(file_path, "w", encoding='utf-8') as fout:
        for line in lines:
            fout.write(json.dumps(line, ensure_ascii=False)+"\n")

def analyze_response_token_lengths(input_file: str) -> Dict[int, List[Dict]]:
    """
    第三阶段：计算响应的token长度并按字数要求汇总。
    
    参数:
        input_file (str): 包含响应文件的路径
        
    返回:
        Dict[int, List[Dict]]: 按字数要求分组的响应数据，每个响应包含token长度信息
    """ 
    responses = load_jsonline(input_file)
    
    # 按字数要求分组
    results_by_word_count = {}
    stats_by_word_count = {}
    
    for item in responses:
        # 跳过无效的指令项
        if not item.get("valid_item", True):
            continue
            
        # 处理每个字数要求的变体
        for key in item.keys():
            if not key.isdigit():
                continue
                
            word_count = int(key)
            variant = item[key]
            
            # 初始化该字数要求的分组
            if word_count not in results_by_word_count:
                results_by_word_count[word_count] = []
                stats_by_word_count[word_count] = {
                    'count': 0,
                    'token_counts': []
                }
            
            # 计算响应的token数并添加到结果中
            response_text = variant.get('response')
            if response_text:
                token_count = calculate_token_count(response_text)
                # 添加token数信息到响应中
                variant['token_count'] = token_count
                stats_by_word_count[word_count]['token_counts'].append(token_count)
                stats_by_word_count[word_count]['count'] += 1
                results_by_word_count[word_count].append({
                    "instruction": item["instruction"], 
                    "response": response_text,
                    "token_count": token_count,
                })
    
    # 打印汇总信息
    print("\nToken Length Summary by Word Count Requirement:")
    print("----------------------------------------------")
    print("Word Count | Responses | Avg Tokens")
    print("----------------------------------------------")
    for word_count in sorted(stats_by_word_count.keys()):
        token_counts = stats_by_word_count[word_count]['token_counts']
        if token_counts:
            avg_token_count = sum(token_counts) / len(token_counts)
            print(f"{word_count:9d} | {stats_by_word_count[word_count]['count']:9d} | {avg_token_count:.2f}")
    print("----------------------------------------------")
    
    return results_by_word_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True, help="Input JSONL file path")
    parser.add_argument("--output-file", type=str, required=True, help="Output directory for result files")
    parser.add_argument("--stage", type=str, choices=["preparation", "generation", "tokenization"], required=True, 
                        help="Processing stage: preparation for instruction preparation, generation for response generation, tokenization for token counting")
    parser.add_argument("--model-name", type=str, 
                        help="List of model names to use (required for stage 2)")
    parser.add_argument("--api-type", type=str,
                        help="List of API types (alibaba, oneapi) corresponding to models (required for stage 2)")
    parser.add_argument("--word-counts", type=int, nargs='+', default=[1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000], 
                        help="List of word counts to require (for stage 1)")
    parser.add_argument("--max-tokens", type=int, default=32768, help="Maximum tokens for generation (for stage 2)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for generation (for stage 2)")
    parser.add_argument("--num-workers", type=int, default=10, help="Number of parallel workers for generation (for stage 2)")
    args = parser.parse_args()
    
    # 第一阶段：准备带有字数要求的指令
    if args.stage == "preparation":
        # 加载数据集
        print(f"Stage 1: Loading dataset from {args.input_file}...")
        data = pd.read_excel(args.input_file)
        dataset = []
        for idx in tqdm(range(len(data)), desc="Grouping items by dimension"):
            dataset.append(data.loc[idx].to_dict())
        
        random.shuffle(dataset)
            
        # 筛选指令并添加字数要求
        print("Stage 1: Preparing instructions with word count requirements...")
        instructions_with_variations = prepare_instructions_with_word_count(dataset, args.word_counts)
        
        print(f"Stage 1: Saving all {len(instructions_with_variations)} instructions to {args.output_file}...")
        dump_jsonline(instructions_with_variations, args.output_file)
        
        print("Stage 1 completed!")
    
    # 第二阶段：为指令生成回复
    elif args.stage == "generation":
        # 验证参数
        
        # 加载指令数据集
        print(f"Stage 2: Loading instructions from {args.input_file}...")
        input_dataset = load_jsonline(args.input_file)
        
        # 生成回复
        print("Stage 2: Generating responses for instructions...")
        results = generate_responses_for_instructions(
            input_dataset,
            args.model_name,
            args.api_type,
            args.max_tokens,
            args.temperature,
            num_workers=args.num_workers,
            output_file=args.output_file
        )

        # 结果已经在处理过程中追加写入到输出文件，这里只打印统计信息
        print(f"Stage 2: Successfully processed {len(results)} valid responses.")
        
        print("Stage 2 completed!")
    
    # 第三阶段：计算响应的token长度并汇总
    elif args.stage == "tokenization":
        print(f"Stage 3: Analyzing response token lengths from {args.input_file}...")
        results_by_word_count = analyze_response_token_lengths(args.input_file)
        final_items = []
        for word_count, items in results_by_word_count.items():
            for item in items[:50]:
                item['word_count'] = word_count
                final_items.append(item)
        print(f"Stage 3: Saving {len(items)} responses with {word_count} token requirement to {args.output_file}...")
        dump_jsonline(final_items, args.output_file)
        print("Stage 3 completed!")