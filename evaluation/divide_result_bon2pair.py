import sys
import json
import math
import tqdm
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer

def calculate_accuracy_by_length_and_position(file_a_path, file_b_path, num_intervals=5):
    """
    读取两个JSONL文件，首先按长度将数据划分为指定数量的区间，
    然后在每个长度区间内，再按位置划分为指定数量的子区间，
    并计算每个子区间内 result=1 的比率（accuracy）。
    """
    # 读取结果数据
    results_data = []
    with open(file_a_path, 'r', encoding='utf-8') as f_a:
        lines = [json.loads(line) for line in f_a.readlines()]
        for obj_a in lines:
            if 'chosen_scores' in obj_a:
                cs = obj_a.get('chosen_scores')[0]
                rs = obj_a.get('rejected_scores')
                results_data.append(int(all([cs>r for r in rs])))
            else:
                cs = obj_a.get('selected_idx')
                rs = obj_a.get('chosen_idx')
                results_data.append(int(cs==rs))
    
    # 读取原始数据，获取长度和位置信息
    tokenizer = AutoTokenizer.from_pretrained("/home/hh456524/WorkSpace/TuningFactoryModels/Qwen2.5-7B-Instruct")
    length_data = []
    position_data = []
    
    with open(file_b_path, 'r', encoding='utf-8') as f_b:
        lines = [json.loads(line) for line in f_b.readlines()]
        for obj_b in tqdm.tqdm(lines):
            # 获取位置信息
            position = obj_b.get('tagged_position', 0)
            position_data.append(position)
            
            word_count = obj_b['word_count']
            length_data.append(word_count)
    
    assert len(results_data) == len(length_data) == len(position_data)
    
    # 组合所有数据
    combined_data = []
    for i in range(len(results_data)):
        combined_data.append({
            'result': results_data[i],
            'length': length_data[i],
            'position': position_data[i]
        })
    
    # 按长度分桶
    length_buckets = []
    bucket_samples = []
    for i, item in enumerate(combined_data):

        bucket_samples.append(item)
        if (i != 0 and combined_data[i]['length'] != combined_data[i-1]['length']) or i == (len(combined_data) -1):

            # 在每个长度桶内，按位置排序
            bucket_samples.sort(key=lambda x: x['position'])

            # 计算每个长度桶的总体准确率
            total_in_bucket = len(bucket_samples)
            correct_in_bucket = sum(1 for d in bucket_samples if d['result'] == 1)
            bucket_accuracy = correct_in_bucket / total_in_bucket if total_in_bucket > 0 else 0.0
            
            # 在长度桶内按位置分子桶
            position_results = []
            samples_per_position_interval = math.ceil(len(bucket_samples) / num_intervals)
            
            for j in range(num_intervals):
                pos_start_idx = j * samples_per_position_interval
                pos_end_idx = min((j + 1) * samples_per_position_interval, len(bucket_samples))
                
                if pos_start_idx >= len(bucket_samples):
                    break
                    
                position_samples = bucket_samples[pos_start_idx:pos_end_idx]
                if not position_samples:
                    continue
                    
                position_start = position_samples[0]['position']
                position_end = position_samples[-1]['position']
                
                total_in_position = len(position_samples)
                correct_in_position = sum(1 for d in position_samples if d['result'] == 1)
                position_accuracy = correct_in_position / total_in_position if total_in_position > 0 else 0.0
                
                position_results.append({
                    'position_start': position_start,
                    'position_end': position_end,
                    'total_items': total_in_position,
                    'correct_items': correct_in_position,
                    'accuracy': position_accuracy
                })
            
            length_buckets.append({
                'length': combined_data[i-1]['length'],
                'total_items': total_in_bucket,
                'correct_items': correct_in_bucket,
                'accuracy': bucket_accuracy,
                'position_results': position_results
            })

            bucket_samples = []
        
    return length_buckets

def create_accuracy_heatmap(length_position_results, output_file="accuracy_heatmap.png"):
    """
    Create a heatmap visualization of accuracy by length and position.
    
    Args:
        length_position_results: Results from calculate_accuracy_by_length_and_position
        output_file: Path to save the heatmap image
    """
    # Extract lengths and prepare data for heatmap
    lengths = [bucket['length'] for bucket in length_position_results]
    
    # Create a matrix for the heatmap
    # Rows = lengths, Columns = position intervals
    max_positions = max(len(bucket['position_results']) for bucket in length_position_results)
    heatmap_data = np.zeros((len(lengths), max_positions))
    
    # Fill the matrix with accuracy values
    for i, length_bucket in enumerate(length_position_results):
        for j, pos_result in enumerate(length_bucket['position_results']):
            heatmap_data[i, j] = pos_result['accuracy']
    
    # Create position labels
    position_labels = []
    for i in range(max_positions):
        # Use the first length bucket that has this position index
        for bucket in length_position_results:
            if i < len(bucket['position_results']):
                start = bucket['position_results'][i]['position_start']
                end = bucket['position_results'][i]['position_end']
                position_labels.append(f"[{start:.1f}-{end:.1f}]")
                break
        else:
            position_labels.append(f"Pos {i+1}")
    
    # Create the heatmap
    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="YlGnBu", 
                     xticklabels=position_labels, yticklabels=lengths)
    
    plt.title("Accuracy Heatmap by Length and Position")
    plt.xlabel("Position")
    plt.ylabel("Length")
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_file)
    print(f"Heatmap saved to {output_file}")
    
    return output_file

def calculate_accuracy_by_position_intervals(file_a_path, file_b_path, num_intervals=5, interval_criterion="position"):
    """
    读取两个JSONL文件，将数据按位置划分为指定数量的区间，
    并计算每个区间内 result=1 的比率（accuracy）。
    """
    file_a_data = []
    file_b_data = []

    with open(file_a_path, 'r', encoding='utf-8') as f_a:
        for line in f_a:
            obj_a = json.loads(line.strip())
            if 'chosen_scores' in obj_a:
                cs = obj_a.get('chosen_scores')[0]
                rs = obj_a.get('rejected_scores')
                file_a_data.append(int(all([cs>r for r in rs])))
            else:
                cs = obj_a.get('selected_idx')
                rs = obj_a.get('chosen_idx')
                file_a_data.append(int(cs==rs))

    assert interval_criterion == "position"
    with open(file_b_path, 'r', encoding='utf-8') as f_b:
        lines = [json.loads(line) for line in f_b.readlines()]
        file_b_data = [line['tagged_position'] for line in lines]

    assert len(file_a_data) == len(file_b_data)

    data_combined = []
    for i in range(len(file_a_data)):
        data_combined.append({
            'result': file_a_data[i],
            'criteria': file_b_data[i]
        })

    data_combined.sort(key=lambda x: x['criteria'])

    total_samples = len(data_combined)
    samples_per_interval = math.ceil(total_samples / num_intervals) # 向上取整，确保所有样本都被包含

    results_by_interval = []
    current_index = 0

    for i in range(num_intervals):
        # 确定当前区间的样本
        interval_samples = data_combined[current_index : current_index + samples_per_interval]

        if not interval_samples: # 如果没有样本了，就跳出循环
            break

        interval_start_pos = interval_samples[0]['criteria']
        interval_end_pos = interval_samples[-1]['criteria']

        total_in_interval = len(interval_samples)
        corr_results_count = sum(1 for d in interval_samples if d['result'] == 1)

        accuracy = 0.0
        if total_in_interval > 0:
            accuracy = corr_results_count / total_in_interval

        results_by_interval.append({
            'interval_start': interval_start_pos,
            'interval_end': interval_end_pos,
            'total_items': total_in_interval,
            'corr_results_count': corr_results_count,
            'accuracy': accuracy
        })
        current_index += samples_per_interval

    return results_by_interval

# 示例用法（请根据您的实际文件路径和字段名修改）
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate accuracy metrics from a reward model inference output JSONL file.")
    parser.add_argument(
        "--data_file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--result_file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=("single", "double"),
        help="'single' for single criterion bucketing, 'double' for length+position bucketing"
    )
    parser.add_argument(
        "--criterion",
        type=str,
        default="position",
        choices=("position", "length", "length_ratio"),
        help="Criterion for single mode bucketing"
    )
    parser.add_argument(
        "--interval_num",
        type=int,
        default=10,
        help="Number of intervals for single mode or position intervals in double mode"
    )
    args = parser.parse_args()

    if args.mode == "single":
        accuracy_results = calculate_accuracy_by_position_intervals(
            args.result_file, 
            args.data_file, 
            num_intervals=args.interval_num, 
            interval_criterion=args.criterion
        )

        if accuracy_results:
            print(f"Interval Accuracy according to {args.criterion}")
            for res in accuracy_results:
                print(f" Interval [{res['interval_start']:.2f}, {res['interval_end']:.2f}]:"
                      f" Total Count={res['total_items']}, Correct Count={res['corr_results_count']},"
                      f" Accuracy={res['accuracy']:.4f}")
    else:  # double mode
        length_position_results = calculate_accuracy_by_length_and_position(
            args.result_file,
            args.data_file,
            num_intervals=args.interval_num
        )

        if length_position_results:
            print(f"Two-level bucketing: First by length, then by position ({args.interval_num} buckets)")
            print("-" * 80)
            
            for i, length_bucket in enumerate(length_position_results):
                print(f"Length Bucket {i+1}: {length_bucket['length']:.2f}")
                print(f"  Overall: Total={length_bucket['total_items']}, Correct={length_bucket['correct_items']}, Accuracy={length_bucket['accuracy']:.4f}")
                
                print("  Position sub-buckets:")
                for j, pos_result in enumerate(length_bucket['position_results']):
                    print(f"    Position {j+1}: [{pos_result['position_start']:.2f}, {pos_result['position_end']:.2f}],"
                          f" Total={pos_result['total_items']}, Correct={pos_result['correct_items']},"
                          f" Accuracy={pos_result['accuracy']:.4f}")
                print("-" * 80)
            
            # Generate and save the heatmap
            heatmap_file = create_accuracy_heatmap(length_position_results)
            print(f"Generated accuracy heatmap: {heatmap_file}")
