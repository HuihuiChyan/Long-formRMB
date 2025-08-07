export CUDA_VISIBLE_DEVICES=0

MODEL_NAME="gemini-2.5-flash-preview-05-20"

MODE="scoring"

DATASET="./data/processed_data_preference_verified_bon.jsonl"
OUTPUT="./data/preference_data_preference_${MODEL_NAME}_${MODE}.jsonl"

python evaluation/infer_bon_api.py \
    --model_name $MODEL_NAME \
    --input_file $DATASET \
    --output_file $OUTPUT \
    --num_threads 10 \
    --infer_mode $MODE \
    --chosen_num 1 \
    --rejected_num 3

python evaluation/cal_accuracy_from_file.py \
    --input_file $OUTPUT \
    --infer_mode $MODE