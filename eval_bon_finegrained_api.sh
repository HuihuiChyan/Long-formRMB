export CUDA_VISIBLE_DEVICES=0

MODEL_NAME="gpt-4o-0806"

MODE="selection"

DATASET="./data-finegrained/general_rag_preference_0603_reformed_Factuality.jsonl"
OUTPUT="./data-finegrained/general_rag_preference_0603_reformed_Factuality_${MODEL_NAME}_${MODE}.jsonl"

# python evaluation/infer_bon_api.py \
#     --model_name $MODEL_NAME \
#     --input_file $DATASET \
#     --output_file $OUTPUT \
#     --num_threads 10 \
#     --chosen_num 1 \
#     --rejected_num 1 \
#     --infer_mode $MODE

python evaluation/cal_accuracy_from_file.py \
    --input_file $OUTPUT \
    --infer_mode $MODE
    # --input_length_file data/processed_data_preference_verified_bon_length.jsonl

python evaluation/divide_result_bon2pair.py \
    --result_file $OUTPUT \
    --data_file $DATASET \
    --interval_num 8 \
    --mode double