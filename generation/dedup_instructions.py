from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
import re
import json
import datasets
from typing import List, Dict, Any, Tuple
from tqdm import tqdm

def filter_most_similar_prompts(
    data: List[Dict],
    remove_percent: float = 0.10,
    model_name: str = "hkunlp/instructor-large" # 示例使用一个较大的instructor模型
) -> List[Dict]:
    """
    加载 Sentence-BERT 模型，对字典列表中的 'prompt' 进行表示，
    然后移除那些与它们最近邻prompt相似度最高的 remove_percent 比例的字典。

    Args:
        data (List[Dict]): 包含 'prompt' 键的字典列表。
        remove_percent (float): 需要移除的最相似的字典比例 (0.0 到 1.0)。
                                 例如，0.10 表示移除最相似的 10%。
        model_name (str): 要加载的 Sentence-BERT 模型名称。

    Returns:
        List[Dict]: 移除最相似项后的字典列表。
    """
    if not data:
        print("输入数据列表为空，返回空列表。")
        return []

    # 1. 提取 prompts 并保留原始索引
    prompts_to_embed: List[str] = []
    original_indices: List[int] = []
    
    for i, item in enumerate(data):
        # 安全地获取 prompt 值，确保它是非空的字符串
        prompt = item.get("prompt")
        if isinstance(prompt, str) and prompt.strip(): # 检查是否是字符串且非空白
            prompts_to_embed.append(prompt.strip())
            original_indices.append(i)
        else:
            # 可以选择记录或忽略没有有效prompt的项
            # print(f"警告: 索引 {i} 的项没有有效的 'prompt' 键，将被跳过嵌入。")
            pass # 这些项将在最后的结果构建时被自然包含（如果它们没有被其他prompt的相似性牵连）

    if not prompts_to_embed:
        print("没有找到有效的 prompts 进行嵌入，返回原始列表。")
        return data # 如果没有可以嵌入的prompt，无法进行相似度过滤，返回原列表

    print(f"共找到 {len(prompts_to_embed)} 个有效的 prompts 进行嵌入。")

    # 2. 加载 embedding 模型
    print(f"正在加载模型: {model_name}...")
    try:
        model = SentenceTransformer(model_name)
        print("模型加载成功。")
    except Exception as e:
        print(f"加载模型失败: {e}")
        print("请确保模型名称正确且已安装 sentence-transformers 和 torch。")
        print("尝试安装: pip install sentence-transformers torch")
        return data # 模型加载失败，无法过滤，返回原列表


    # 3. 对 prompts 进行表示 (embedding)
    print("正在生成 prompts 的 embeddings...")
    # 使用 convert_to_tensor=True 将 embeddings 放在 GPU (如果可用)
    embeddings = model.encode(prompts_to_embed, convert_to_tensor=True, show_progress_bar=True)
    print("Embeddings 生成完成。")

    # 4. 计算余弦相似度矩阵
    print("正在计算余弦相似度矩阵...")
    # cos_sim 返回一个 NxN 的张量
    cosine_scores = util.cos_sim(embeddings, embeddings)
    print("相似度矩阵计算完成。")

    # 5. 找出每个 prompt 的最近邻相似度
    # 需要找到每行（每个prompt）的最大相似度，但排除自己与自己的相似度 (总是1)
    nearest_neighbor_scores: List[float] = []
    # indices_to_remove_by_similarity: List[int] = [] # 存储原始数据中的索引

    # 将对角线元素设为一个非常小的值，这样在查找最大值时会忽略掉自己与自己的相似度1
    torch.diagonal(cosine_scores).fill_(-1.0) # 或使用一个足够小的负数

    # 找到每行的最大值 (即每个prompt与其它prompt的最大相似度)
    max_sim_scores, _ = torch.max(cosine_scores, dim=1)

    # 将最近邻相似度与其原始数据索引关联起来
    # pairs: [(max_sim_score, original_data_index), ...]
    similarity_scores_with_indices = list(zip(max_sim_scores.tolist(), original_indices))

    # 6. 找出相似度最高的 remove_percent 比例的项
    # 按相似度从高到低排序
    similarity_scores_with_indices.sort(key=lambda x: x[0], reverse=True)

    # 计算要移除的数量
    num_items_to_remove = int(len(prompts_to_embed) * remove_percent)
    if num_items_to_remove >= len(prompts_to_embed):
         print(f"要移除的数量 ({num_items_to_remove}) >= 有效 prompts 数量 ({len(prompts_to_embed)})。将保留一个。")
         num_items_to_remove = len(prompts_to_embed) -1 if len(prompts_to_embed) > 0 else 0
    if num_items_to_remove < 0: num_items_to_remove = 0 # 避免负数

    print(f"根据阈值 {remove_percent*100:.0f}%，将移除最相似的 {num_items_to_remove} 项。")

    # 获取要移除的项的原始数据索引集合
    indices_to_remove_set = set([
        idx for score, idx in similarity_scores_with_indices[:num_items_to_remove]
    ])

    # 7. 构建去重后的列表
    # 遍历原始数据列表，只保留索引不在移除集合中的项
    deduplicated_data = []
    for i, item in enumerate(data):
        if i not in indices_to_remove_set:
            deduplicated_data.append(item)

    print(f"原始项数量: {len(data)}")
    print(f"移除项数量: {len(indices_to_remove_set)} (对应嵌入的 {num_items_to_remove} 个)")
    print(f"剩余项数量: {len(deduplicated_data)}")

    return deduplicated_data

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


input_path = "./data/longwriter/processed_longwriter.jsonl"
with open(input_path, "r", encoding="utf-8") as fin:
    lines = [json.loads(line.strip()) for line in fin.readlines()]
dedup_data = filter_most_similar_prompts(lines, remove_percent=0.5, model_name="/home/hh456524/WorkSpace/TuningFactoryModels/instructor-large")

output_path = "./data/longwriter/processed_longwriter_dedup.jsonl"
with open(output_path, 'w', encoding='utf-8') as f:
    for item in dedup_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"Processed data saved to {output_path}")