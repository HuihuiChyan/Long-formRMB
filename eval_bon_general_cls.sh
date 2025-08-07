export CUDA_VISIBLE_DEVICES=0

MODEL_NAME=Skywork-Reward-V2-Llama-3.1-8B
MODEL=/home/hh456524/WorkSpace/TuningFactoryModels/RewardModels/Skywork-Reward-V2-Llama-3.1-8B

# DATASET="./data/processed_data_preference_verified_bon.jsonl"
# OUTPUT="./data/processed_data_preference_verified_bon_${MODEL_NAME}.jsonl"

# python -u evaluation/infer_bon.py \
#     --model_path $MODEL \
#     --input_file $DATASET \
#     --output_file $OUTPUT \
#     --chosen_num 1 \
#     --rejected_num 3

# python -u evaluation/cal_accuracy_from_file.py \
#     --input_file $OUTPUT

DATASET="./data-longcot/preference_data_reasoning.jsonl"
OUTPUT="./data-longcot/preference_data_reasoning_${MODEL_NAME}.jsonl"

python -u evaluation/infer_bon.py \
    --model_path $MODEL \
    --input_file $DATASET \
    --output_file $OUTPUT \
    --chosen_num 1 \
    --rejected_num 1

python -u evaluation/cal_accuracy_from_file.py \
    --input_file $OUTPUT