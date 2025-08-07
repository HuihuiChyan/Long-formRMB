export CUDA_VISIBLE_DEVICES=2,3

# # # 定义模型列表
# API_MODELS=(
#     "gpt-4o-0806"
#     "claude35_haiku"
#     # "gemini-2.5-pro-preview-05-06"
# )

# # # 定义模型列表
# ONE_MODELS=(
#     "gemini-2.0-flash"
#     "grok-3-beta"
#     "deepseek-v3-1226"
# )

# # 定义模型列表
# VLLM_MODELS=(
#     "/home/hh456524/WorkSpace/TuningFactoryModels/Qwen2.5-14B-Instruct"
#     "/home/hh456524/WorkSpace/TuningFactoryModels/Qwen2.5-32B-Instruct"
# )

# model="/home/hh456524/WorkSpace/TuningFactoryModels/Qwen2.5-7B-Instruct"
# python3.10 data_gene.py \
#     --stage "generation" \
#     --input-path "./data/longwriter/processed_longwriter_dedup.jsonl" \
#     --output-path "./data/longwriter/processed_longwriter_dedup_filter.jsonl" \
#     --model-name "$model" \
#     --model-type "local" \
#     --do-filtering

# # 处理阶段
# python3.10 data_gene.py \
#     --stage "processing" \
#     --input-path "./data/longwriter/processed_longwriter_dedup_filter.jsonl" \
#     --output-path "./data/longwriter/processed_longwriter_dedup_process.jsonl"

# # 生成阶段
# for model in "${API_MODELS[@]}"; do
#     python3.10 -u data_gene.py \
#         --stage "generation" \
#         --input-path "./data/longwriter/processed_longwriter_dedup_process.jsonl" \
#         --output-path "./data/longwriter/processed_longwriter_dedup_generation.jsonl" \
#         --model-name "$model" \
#         --model-type "alibaba"
# done

# for model in "${ONE_MODELS[@]}"; do
#     python3.10 -u data_gene.py \
#         --stage "generation" \
#         --input-path "./data/longwriter/processed_longwriter_dedup_generation.jsonl" \
#         --output-path "./data/longwriter/processed_longwriter_dedup_generation.jsonl" \
#         --model-name "$model" \
#         --model-type "oneapi"
# done

# for model in "${VLLM_MODELS[@]}"; do
#     python3.10 data_gene.py \
#         --stage "generation" \
#         --input-path "./data/longwriter/processed_longwriter_dedup_generation.jsonl" \
#         --output-path "./data/longwriter/processed_longwriter_dedup_generation.jsonl" \
#         --model-name "$model" \
#         --model-type "local"
# done

# python3.10 -u data_gene.py \
#     --stage "evaluation" \
#     --input-path "./data/longwriter/processed_longwriter_dedup_generation.jsonl" \
#     --output-path "./data/longwriter/processed_longwriter_dedup_evaluation.jsonl"

python3 -u data_gene.py \
    --stage "fetch_score" \
    --input-path "./data/longwriter/processed_longwriter_dedup_evaluation.jsonl" \
    --output-path "./data/longwriter/processed_longwriter_dedup_preference.jsonl"

# python3.10 -u data_gene.py \
#     --stage "verification" \
#     --input-path "./data/longwriter/processed_longwriter_dedup_preference.jsonl"  \
#     --output-path "./data/longwriter/processed_longwriter_dedup_verified_preference.jsonl"