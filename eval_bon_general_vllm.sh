export CUDA_VISIBLE_DEVICES=0

MODEL_NAME=Selene-1-Mini-Llama-3.1-8B
MODEL="/home/hh456524/WorkSpace/TuningFactoryModels/RewardModels/${MODEL_NAME}"

DATASET="./data/processed_data_preference_verified_bon.jsonl"
OUTPUT="./data/processed_data_preference_verified_bon_${MODEL_NAME}.jsonl"

python -u evaluation/infer_bon_vllm.py \
    --model_path $MODEL \
    --input_file $DATASET \
    --output_file $OUTPUT

python -u evaluation/cal_accuracy_from_file.py \
    --input_file $OUTPUT