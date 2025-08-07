import json
import random

def merge_jsonl_responses(file_chosen, file_reject1, file_reject2):

    reject_data = {}
    with open(file_reject1, 'r', encoding='utf-8') as fin:
        for line in fin:
            data = json.loads(line.strip())
            if data["question"] not in reject_data:
                reject_data[data["question"]] = []
            reject_data[data["question"]].append(data["long_cot"])
                
    with open(file_reject2, 'r', encoding='utf-8') as fin:
        for line in fin:
            data = json.loads(line.strip())
            if data["question"] not in reject_data:
                reject_data[data["question"]] = []
            reject_data[data["question"]].append(data["long_cot"])
    
    merged_results = []
    with open(file_chosen, 'r', encoding='utf-8') as fin:
        for line in fin:
            data = json.loads(line.strip())

            question = data["question"]
            chosen_res = data["long_cot"]

            if question in reject_data:
                rejected_res = random.choice(reject_data[question])
                merged_results.append({
                        'subset': 'reasoning',
                        'id': data["id"],
                        'origin': data["origin"],
                        "prompt": question,
                        "chosen": [chosen_res],
                        "rejected": [rejected_res],
                    })
                
    return merged_results

if __name__ == "__main__":
    output_path = "data-longcot/preference_data_reasoning.jsonl"
    file_pairs = [
                    [
                        "data-longcot/math_flash所有推理结果_回答正确部分.jsonl",
                        "data-longcot/math_qwq所有推理结果_回答错误部分.jsonl",
                        "data-longcot/math_r1所有推理结果_回答错误部分.jsonl",
                    ],
                    [   
                        "data-longcot/math_qwq所有推理结果_回答正确部分.jsonl",
                        "data-longcot/math_flash所有推理结果_回答错误部分.jsonl",
                        "data-longcot/math_r1所有推理结果_回答错误部分.jsonl",
                    ],                    
                    [
                        "data-longcot/math_r1所有推理结果_回答正确部分.jsonl",
                        "data-longcot/math_flash所有推理结果_回答错误部分.jsonl",
                        "data-longcot/math_qwq所有推理结果_回答错误部分.jsonl",
                    ],
                ]
    all_results = []
    for file_pair in file_pairs:
        all_results.extend(merge_jsonl_responses(file_pair[0], file_pair[1], file_pair[2]))

    # 将结果写入新的JSON Lines文件
    with open(output_path, 'w', encoding='utf-8') as fout:
        for item in all_results:
            fout.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"处理完成！匹配到的数据已保存到 {output_path}")