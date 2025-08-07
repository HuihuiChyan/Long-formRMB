import pandas as pd
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import ast
import json

def _process_data_and_plot(df_data: pd.DataFrame, current_scope_name: str, tokenizer):
    """
    内部函数：处理给定 DataFrame 的数据，并生成统计信息和图表。
    图表将直接保存到当前目录下，不显示。
    """
    print(f"\n--- 开始处理: {current_scope_name} ---")

    chosen_lengths = []
    rejected_lengths = []
    length_ratios = []

    # 定义直方图的固定 bin 边界
    fixed_bins = np.arange(1, 10001, 100) # From 1 to 10000, step of 100

    for index, row in df_data.iterrows():
        current_chosen_length = 0
        # Chosen 长度
        if 'chosen' in row and pd.notna(row['chosen']):
            chosen_tokens = tokenizer.encode(str(row['chosen']), add_special_tokens=False)
            current_chosen_length = len(chosen_tokens)
            chosen_lengths.append(current_chosen_length)

        # Rejected 长度 (合并所有 rejected 列) 和计算长度比
        for i in range(1, 4): # rejected1, rejected2, rejected3
            col_name = f'rejected{i}'
            if col_name in row and pd.notna(row[col_name]):
                current_rejected_text = str(row[col_name])
                rejected_tokens = tokenizer.encode(current_rejected_text, add_special_tokens=False)
                current_rejected_length = len(rejected_tokens)
                rejected_lengths.append(current_rejected_length)

                # 计算 chosen/rejected 长度比
                if current_chosen_length > 0 and current_rejected_length > 0:
                    length_ratios.append(current_chosen_length / current_rejected_length)
                elif current_chosen_length > 0 and current_rejected_length == 0:
                    length_ratios.append(np.nan)
                elif current_chosen_length == 0 and current_rejected_length > 0:
                    length_ratios.append(0.0)

    # Chosen 长度统计
    if chosen_lengths:
        print(f"\nChosen 文本长度 (Token 数) 统计 ({current_scope_name}):")
        print(f"  平均长度: {np.mean(chosen_lengths):.2f}")
        print(f"  最小长度: {np.min(chosen_lengths)}")
        print(f"  最大长度: {np.max(chosen_lengths)}")

        plt.figure(figsize=(10, 6))
        sns.histplot(chosen_lengths, bins=fixed_bins, kde=True, color='skyblue')
        plt.title(f'Chosen Text Token Length Distribution - {current_scope_name}')
        plt.xlabel('Token Length')
        plt.ylabel('Frequency')
        plt.xlim(1, 10000)
        plt.grid(axis='y', alpha=0.75)
        plt.tight_layout()
        plt.savefig(f"{current_scope_name.replace(' ', '_').replace(':', '')}_Chosen_Length_Distribution.png")
        plt.close() # Close the plot to free up memory
    else:
        print(f"\n{current_scope_name} 中没有找到有效的 'chosen' 文本数据进行长度统计。")

    # Rejected 长度统计
    if rejected_lengths:
        print(f"\nRejected 文本长度 (Token 数) 统计 ({current_scope_name}):")
        print(f"  平均长度: {np.mean(rejected_lengths):.2f}")
        print(f"  最小长度: {np.min(rejected_lengths)}")
        print(f"  最大长度: {np.max(rejected_lengths)}")

        plt.figure(figsize=(10, 6))
        sns.histplot(rejected_lengths, bins=fixed_bins, kde=True, color='lightcoral')
        plt.title(f'Rejected Text Token Length Distribution - {current_scope_name}')
        plt.xlabel('Token Length')
        plt.ylabel('Frequency')
        plt.xlim(1, 10000)
        plt.grid(axis='y', alpha=0.75)
        plt.tight_layout()
        plt.savefig(f"{current_scope_name.replace(' ', '_').replace(':', '')}_Rejected_Length_Distribution.png")
        plt.close()
    else:
        print(f"\n{current_scope_name} 中没有找到有效的 'rejected' 文本数据进行长度统计。")

    # 长度比分布统计
    print(f"\n--- Chosen/Rejected 长度比分布统计 ({current_scope_name}) ---")
    valid_length_ratios = [r for r in length_ratios if pd.notna(r)]

    if valid_length_ratios:
        print(f"  比值数量: {len(valid_length_ratios)}")
        print(f"  平均比值: {np.mean(valid_length_ratios):.2f}")
        print(f"  最小比值: {np.min(valid_length_ratios):.2f}")
        print(f"  最大比值: {np.max(valid_length_ratios):.2f}")

        plt.figure(figsize=(10, 6))
        sns.histplot(valid_length_ratios, bins=50, kde=True, color='purple')
        plt.title(f'Chosen/Rejected Text Token Length Ratio Distribution - {current_scope_name}')
        plt.xlabel('Length Ratio (Chosen Length / Rejected Length)')
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.75)
        plt.tight_layout()
        plt.savefig(f"{current_scope_name.replace(' ', '_').replace(':', '')}_Length_Ratio_Distribution.png")
        plt.close()
    else:
        print(f"\n{current_scope_name} 中没有找到有效的 Chosen/Rejected 长度比数据进行统计。")

    # 统计 chosen_model 的信息
    print(f"\n--- Chosen Model 统计信息 ({current_scope_name}) ---")
    if 'chosen_model' in df_data.columns:
        chosen_model_counts = df_data['chosen_model'].value_counts(dropna=False)
        print(chosen_model_counts)

        # Plot as a pie chart
        if not chosen_model_counts.empty:
            plt.figure(figsize=(10, 10))
            wedges, texts, autotexts = plt.pie(
                chosen_model_counts,
                labels=chosen_model_counts.index,
                autopct='%1.1f%%',
                startangle=140,
                pctdistance=0.85,
                colors=sns.color_palette('viridis', len(chosen_model_counts))
            )
            # 加粗标签字体
            for text in texts:
                text.set_fontweight('bold')
            plt.title(f'Chosen Model Distribution - {current_scope_name}')
            plt.axis('equal')
            plt.tight_layout()
            plt.savefig(f"{current_scope_name.replace(' ', '_').replace(':', '')}_Chosen_Model_Distribution.png")
            plt.close()
        else:
            print(f"{current_scope_name} 的 'chosen_model' 数据为空，无法生成扇形图。")
    else:
        print(f"{current_scope_name} 中不包含 'chosen_model' 列。")

    # 统计 rejected_models 的信息
    print(f"\n--- Rejected Models 统计信息 ({current_scope_name}) ---")
    if 'rejected_models' in df_data.columns:
        all_rejected_models = []
        for item in df_data['rejected_models'].dropna():
            if pd.isna(item):
                continue
            try:
                model_list = ast.literal_eval(str(item))
                if isinstance(model_list, list):
                    all_rejected_models.extend([m.strip() for m in model_list])
                else:
                    all_rejected_models.extend([m.strip() for m in str(item).split(',')])
            except (ValueError, SyntaxError):
                all_rejected_models.extend([m.strip() for m in str(item).split(',')])

        if all_rejected_models:
            rejected_model_counts = pd.Series(all_rejected_models).value_counts()
            print(rejected_model_counts)

            # Plot as a pie chart
            if not rejected_model_counts.empty:
                plt.figure(figsize=(10, 10))
                wedges, texts, autotexts = plt.pie(
                    rejected_model_counts,
                    labels=rejected_model_counts.index,
                    autopct='%1.1f%%',
                    startangle=140,
                    pctdistance=0.85,
                    colors=sns.color_palette('plasma', len(rejected_model_counts))
                )
                # 加粗标签字体
                for text in texts:
                    text.set_fontweight('bold')
                plt.title(f'Rejected Models Distribution - {current_scope_name}')
                plt.axis('equal')
                plt.tight_layout()
                plt.savefig(f"{current_scope_name.replace(' ', '_').replace(':', '')}_Rejected_Models_Distribution.png")
                plt.close()
            else:
                print(f"{current_scope_name} 的 'rejected_models' 数据为空，无法生成扇形图。")
        else:
            print(f"{current_scope_name} 中没有找到有效的 'rejected_models' 数据。")
    else:
        print(f"{current_scope_name} 中不包含 'rejected_models' 列。")

    print(f"--- 结束处理: {current_scope_name} ---")


def analyze_jsonl_file_with_subsets(file_path: str):
    """
    加载包含多个 subset 的 Excel 文件，并为每个 subset 和总览数据生成统计信息和图表。
    所有图表都将直接保存到当前目录下，不显示。

    Args:
        file_path (str): Excel 文件的路径。
    """
    with open(file_path) as fin:
        lines = [json.loads(line) for line in fin.readlines()]

    tokenizer = AutoTokenizer.from_pretrained("/Users/huihuang/Desktop/LF-RewardBench/Qwen2.5-0.5B-Instruct")

    df_full = pd.DataFrame(lines)

    # 检查是否存在 'subset' 列
    if 'subset' in df_full.columns:
        unique_subsets = df_full['subset'].dropna().unique()
        if len(unique_subsets) > 0:
            print(f"\n检测到 'subset' 列，包含 {len(unique_subsets)} 个独立子集: {unique_subsets.tolist()}")
            for subset_name in unique_subsets:
                df_subset = df_full[df_full['subset'] == subset_name].copy()
                _process_data_and_plot(df_subset, f"Subset: {subset_name}", tokenizer)
        else:
            print("\n'subset' 列存在，但没有发现有效的子集名称。将作为整体数据处理。")
            _process_data_and_plot(df_full, "Overall", tokenizer)
    else:
        print("\n未检测到 'subset' 列。将作为整体数据处理。")
        _process_data_and_plot(df_full, "Overall", tokenizer)

    # 最后，处理总览信息 (无论是否有 'subset' 列，都进行总览统计)
    print("\n" + "="*50)
    print("--- 开始处理总览 (Overall) 信息 ---")
    _process_data_and_plot(df_full, "Overall", tokenizer)
    print("="*50 + "\n")
        
if __name__ == '__main__':
    # 将下面的 'your_excel_file.xlsx' 替换为你的 Excel 文件路径。
    # 确保此 Excel 文件包含一个名为 'subset' 的列，用于标识不同的子集。
    excel_file_path = './data-longcot/preference_data_reasoning.jsonl'
    analyze_jsonl_file_with_subsets(excel_file_path)