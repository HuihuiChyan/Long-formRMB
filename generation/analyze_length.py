import json
from transformers import AutoTokenizer
from typing import Dict, List, Optional
import numpy as np

def analyze_answer_lengths(
    jsonl_file_path: str,
    tokenizer_name: str,
    subset_key: str = "subset",
    answer_key_1: str = "answer_1",
    answer_key_2: str = "answer_2",
    better_key: str = "better",
    quantiles: Optional[List[float]] = None,
) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
    """
    加载 Transformers tokenizer，读取 JSON Lines 文件，根据 'better' 字段的值动态确定 chosen 和 rejected answer，
    并统计不同 subset 的 chosen 和 rejected answer 的长度分位数。

    Args:
        jsonl_file_path (str): JSON Lines 文件的路径。
        tokenizer_name (str): Transformers tokenizer 的名称 (例如, "bert-base-uncased", "gpt2").
        subset_key (str): JSON 对象中表示 subset 的键名 (默认为 "subset").
        answer_key_1 (str): JSON 对象中表示第一个 answer 的键名 (默认为 "answer_1").
        answer_key_2 (str): JSON 对象中表示第二个 answer 的键名 (默认为 "answer_2").
        better_key (str): JSON 对象中表示哪个 answer 更好的键名 (默认为 "better").
                           值应为 1 或 2。
        quantiles (Optional[List[float]]): 要计算的分位数列表 (例如, [0.2, 0.4, 0.6, 0.8, 1.0]).
                                        如果为 None，则默认为 [0.2, 0.4, 0.6, 0.8, 1.0].

    Returns:
        Dict[str, Dict[str, Dict[str, List[float]]]]: 一个嵌套字典，格式为:
        {
            "subset_value_1": {
                "chosen": {
                    "token_length_quantiles": [quantile_value_0.2, ...],
                    "char_length_quantiles": [quantile_value_0.2, ...],
                },
                "rejected": {
                    "token_length_quantiles": [quantile_value_0.2, ...],
                    "char_length_quantiles": [quantile_value_0.2, ...],
                },
            },
            ...
        }
    """

    if quantiles is None:
        quantiles = [0.2, 0.4, 0.6, 0.8, 1.0]

    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        print(f"Tokenizer '{tokenizer_name}' 加载成功.")
    except Exception as e:
        print(f"加载 tokenizer '{tokenizer_name}' 失败: {e}")
        return {}

    subset_lengths: Dict[str, Dict[str, Dict[str, List[int]]]] = {}

    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            subset = data.get(subset_key)
            chosen_answer = data.get(answer_key_1)
            rejected_answers = data.get(answer_key_2)
            assert data.get(better_key) == 1

            if subset not in subset_lengths:
                subset_lengths[subset] = {
                    "chosen": [],
                    "rejected": [],
                }

            chosen_tokens = tokenizer.tokenize(chosen_answer)
            subset_lengths[subset]["chosen"].append(len(chosen_tokens))

            rejected_lengths = []
            for rejected_answer in rejected_answers:
                if rejected_answer:
                    rejected_tokens = tokenizer.tokenize(rejected_answer)
                    rejected_lengths.append(len(rejected_tokens))
            subset_lengths[subset]["rejected"].append(sum(rejected_lengths)/len(rejected_lengths))

    subset_quantiles: Dict[str, Dict[str, Dict[str, List[float]]]] = {}

    # 计算分位数
    for subset, answer_types in subset_lengths.items():
        subset_quantiles[subset] = {'chosen': {}, 'rejected': {}}
        for answer_type, lengths in answer_types.items():
            quantiles_list = np.quantile(lengths, quantiles).tolist()
            quantiles_list = [round(q, 2) for q in quantiles_list]
            subset_quantiles[subset][answer_type]["token_length_quantiles"] = quantiles_list

    return subset_quantiles

if __name__ == "__main__":
    jsonl_file = "/Users/huihuang/Desktop/LF-RewardBench/data/processed_data_preference.jsonl"  # 替换为你的 JSON Lines 文件路径
    tokenizer_name = "/Users/huihuang/Desktop/LF-RewardBench/Qwen2.5-0.5B-Instruct"  # 替换为你想要使用的 tokenizer 名称

    length_stats = analyze_answer_lengths(jsonl_file, tokenizer_name)

    if length_stats:
        print("\n不同 Subset 的 Chosen 和 Rejected Answer 长度统计:")
        for subset, answer_types in length_stats.items():
            print(f"\nSubset: {subset}")
            for answer_type, lengths in answer_types.items():
                print(f"  {answer_type.capitalize()} Answer:")
                print(f"    Token Lengths (Count: {len(lengths)}): {lengths}")
    else:
        print("未能生成长度统计信息。")