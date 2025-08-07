export CUDA_VISIBLE_DEVICES=0,1

# export HF_ENDPOINT=https://hf-mirror.com
# huggingface-cli download --repo-type dataset --local-dir-use-symlinks False emozilla/dolma-v1_7-cc_en_head --local-dir ./data-train --include "000_00000.parquet"
# huggingface-cli download --repo-type dataset --local-dir-use-symlinks False emozilla/dolma-v1_7-cc_en_head --local-dir ./data-train --include "001_00000.parquet"
# huggingface-cli download --repo-type dataset --local-dir-use-symlinks False emozilla/dolma-v1_7-cc_en_head --local-dir ./data-train --include "002_00000.parquet"
# python generation/data_acquire.py --input_path ./data-train --output_path ./data-train/dolma.jsonl

DATASET="dolma"

# # 1. 处理阶段
# python3.10 generation/data_gene_trainset.py \
#     --stage "back_trans" \
#     --input-path "./data-train/${DATASET}.jsonl" \
#     --output-path "./data-train/processed_${DATASET}.jsonl"

# python3.10 generation/data_gene_trainset.py \
#     --stage "processing" \
#     --input-path "./data-train/processed_${DATASET}.jsonl" \
#     --output-path "./data-train/processed_${DATASET}.jsonl"

# 定义模型列表
VLLM_MODELS=(
    "/home/hh456524/WorkSpace/TuningFactoryModels/Qwen2.5-7B-Instruct"
    "/home/hh456524/WorkSpace/TuningFactoryModels/Meta-Llama-3.1-8B-Instruct"
    "/home/hh456524/WorkSpace/TuningFactoryModels/DeepSeek-R1-Distill-Qwen-14B"
)

# 定义模型列表
API_MODELS=(
    "gpt-4o-0806"
    "claude35_haiku"
)
# for model in "${VLLM_MODELS[@]}"; do
#     python3.10 generation/data_gene.py \
#         --stage "generation" \
#         --input-path "./data-train/processed_${DATASET}_1w.jsonl" \
#         --output-path "./data-train/processed_${DATASET}_generation.jsonl" \
#         --model-name "$model" \
#         --model-type "local"
# done

# # 生成阶段
# for model in "${API_MODELS[@]}"; do
#     python3.10 generation/data_gene.py \
#         --stage "generation" \
#         --input-path "./data-train/processed_${DATASET}_generation.jsonl" \
#         --output-path "./data-train/processed_${DATASET}_generation.jsonl" \
#         --model-name "$model" \
#         --model-type "alibaba"
# done

# python3.10 -u generation/data_gene.py \
#     --stage "evaluation" \
#     --input-path "./data-train/processed_${DATASET}_generation.jsonl" \
#     --output-path "./data-train/processed_${DATASET}_evaluation.jsonl"

python3.10 -u generation/data_gene.py \
    --stage "fetch_score" \
    --input-path "./data-train/processed_${DATASET}_evaluation.jsonl" \
    --output-path "./data-train/processed_${DATASET}_preference.jsonl"

# python3.10 -u data_gene.py \
#     --stage "verification" \
#     --input-path "./data/longwriter/processed_longwriter_dedup_preference.jsonl"  \
#     --output-path "./data/longwriter/processed_longwriter_dedup_verified_preference.jsonl"