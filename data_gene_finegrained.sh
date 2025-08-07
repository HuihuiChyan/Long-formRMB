#!/bin/bash

# Define word counts
WORD_COUNTS=(8000 7000 6000 5000 4000 3000 2000 1000)

# # Stage 1: Prepare instructions with word count requirements
# echo "Running Stage 1: Preparing instructions with word count requirements..."

# # Define input and output paths
# INPUT_FILE="data-finegrained/general_rag_preference_0603.xlsx"
# OUTPUT_FILE="data-finegrained/general_rag_preference_0603_finegrained.jsonl"

# python generation/data_gene_with_word_num.py \
#   --stage "preparation" \
#   --input-file $INPUT_FILE \
#   --output-file $OUTPUT_FILE \
#   --word-counts ${WORD_COUNTS[@]}

# echo "Stage 1 completed! Instructions saved to $OUTPUT_FILE"

# # Stage 2: Generate responses for the prepared instructions
# echo "Running Stage 2: Generating responses for instructions..."

# INPUT_FILE="data-finegrained/general_rag_preference_0603_finegrained.jsonl"
# OUTPUT_FILE="data-finegrained/general_rag_preference_0603_generation.jsonl"

# python -u generation/data_gene_with_word_num.py \
#   --stage "generation" \
#   --input-file $INPUT_FILE \
#   --output-file $OUTPUT_FILE \
#   --model-name "gemini-2.5-flash-preview-05-20" \
#   --api-type "alibaba" \
#   --max-tokens 32768 \
#   --temperature 0.3

# echo "Stage 2 completed! Responses saved to $OUTPUT_FILE"

# # Stage 3: Tokenize sentences and calculate lengths

# INPUT_FILE="data-finegrained/general_rag_preference_0603_generation.jsonl"
# OUTPUT_FILE="data-finegrained/general_rag_preference_0603_reformed.jsonl"

# python3 -u generation/data_gene_with_word_num.py \
#   --stage "tokenization" \
#   --input-file $INPUT_FILE \
#   --output-file $OUTPUT_FILE \

# # Stage 4: Generation positive and negative examples

# INPUT_FILE="data-finegrained/general_rag_preference_0603_reformed.jsonl"
# OUTPUT_FILE="data-finegrained/general_rag_preference_0603_reformed_Safety.jsonl"

# python3 -u generation/data_gene_finegrained.py \
#     --dimension Safety \
#     --input_file $INPUT_FILE \
#     --output_file $OUTPUT_FILE

INPUT_FILE="data-finegrained/general_rag_preference_0603_reformed.jsonl"
OUTPUT_FILE="data-finegrained/general_rag_preference_0603_reformed_Factuality.jsonl"

python3 -u generation/data_gene_finegrained.py \
    --dimension Factuality \
    --input_file $INPUT_FILE \
    --output_file $OUTPUT_FILE