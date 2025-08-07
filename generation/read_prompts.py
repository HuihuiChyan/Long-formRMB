import re
import json
import datasets
from typing import List, Dict, Any, Tuple
from tqdm import tqdm

#------------------------------------------------------------------------------
# 文本处理和过滤函数
#------------------------------------------------------------------------------
def count_words(text: str) -> int:
    """
    使用正则表达式计算文本中的词数。
    
    参数:
        text (str): 输入文本
        
    返回:
        int: 词数
    """
    return len(re.findall(r'\b\w+\b', text)) if text else 0


def process_laravel_qa_data(file_path: List[str]) -> None:
    """Process WildChat data using datasets library and save as JSONL."""
    # Load dataset from Parquet files
    dataset = datasets.load_dataset('parquet', data_files=file_path)
    
    # Process the dataset and filter out None results
    processed_data = []
    for example in dataset['train']:
        answer = example["assistant"]
        result = {'subset': 'QA', "prompt": example["human"] + " Your answer should be long enough to provide a comprehensive response."}
        processed_data.append(result)

    print(f"Totally {len(processed_data)} lines for laravel_qa.")
    return processed_data

# Data Processing Functions
def process_proxy_qa_data(file_path: List[str]):

    dataset = datasets.load_from_disk(file_path)
    
    # Process the dataset and filter out None results
    processed_data = []
    for example in dataset:
        result = {'subset': 'QA', "prompt": example['meta_question'] + " Your answer should be long enough to provide a comprehensive response."}
        processed_data.append(result)

    print(f"Totally {len(processed_data)} lines for proxy_qa.")
    return processed_data
    
# Data Processing Functions
def process_hellobench_qa_data(file_path: List[str]):
    
    fin = open(file_path, "r")
    dataset = [json.loads(line.strip()) for line in fin.readlines()]
    
    # Process the dataset and filter out None results
    processed_data = []
    for example in dataset:
        result = {'subset': 'QA', "prompt": example['instruction']}
        processed_data.append(result)

    print(f"Totally {len(processed_data)} lines for hellobench_qa.")
    return processed_data

# Data Processing Functions
def process_longwriter_data(file_path: List[str]):
    """Process WildChat data from multiple files."""
    fin = open(file_path, "r")
    dataset = [json.loads(line.strip()) for line in fin.readlines()]

    # Process the dataset and filter out None results
    processed_data = []
    for example in dataset:
        messages = example["messages"]
        if len(messages) >= 2:
            if (messages[0]['role'] == 'user' and messages[1]['role'] == 'assistant'):
                result = {
                    'subset': 'writing',
                    'prompt': messages[0]['content'],
                }
                answer = messages[1]['content']
                word_count = count_words(answer)
                if word_count > 1000:
                    processed_data.append(result)

    print(f"Totally {len(processed_data)} lines for longwriter_data.")
    return processed_data

if __name__ == "__main__":
    # fin = open("./data/wildchat/processed_wildchat01.jsonl", "r")
    # wildchat_data1 = [json.loads(line.strip()) for line in fin.readlines()]

    # fin = open("./data/wildchat/processed_wildchat02.jsonl", "r")
    # wildchat_data2 = [json.loads(line.strip()) for line in fin.readlines()]

    # file_path = "./data/proxy_qa"
    # proxy_qa_data = process_proxy_qa_data(file_path)

    # file_path = "./data/hellobench_qa/open_ended_qa.jsonl"
    # hellobench_qa_data = process_hellobench_qa_data(file_path)

    file_path = "./data/longwriter/long.jsonl"
    longwriter_data = process_longwriter_data(file_path)

    # file_path = "./data/laravel_qa/train-00000-of-00001.parquet"
    # laravel_qa_data = process_laravel_qa_data(file_path)

    all_data = longwriter_data

    output_path = "./data/longwriter/processed_longwriter.jsonl"
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Processed data saved to {output_path}")