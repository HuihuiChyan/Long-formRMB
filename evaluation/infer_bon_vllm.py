import json
import logging
import re
import os
import time
import argparse
from tqdm import tqdm
from typing import List, Dict, Union, Any

from vllm import LLM, SamplingParams
from utils_prompts import model2prompt

# Set up logging for better visibility
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper function for parsing score (retained) ---
def parse_score_from_text(score_text: str) -> Union[int, None]:
    """
    Parses a numerical rating from the generated text.
    Returns the score or None if parsing fails.
    """
    # match = re.search(r"([0-9]+)\**\s*$", score_text)
    # if match is not None:
    #     rating = int(match.group(1))
    #     return rating
    # else:
    #     logger.warning(f"Could not parse score for text: '{score_text}'")
    #     return None

    match = re.search(r"\*\*Result:\*\* ([1-9]\.[0-9]*)", score_text)
    if match is not None:
        rating = float(match.group(1))
        return rating
    else:
        logger.warning(f"Could not parse score for text: '{score_text}'")
        return None

def infer_all_scores_vllm(
    model_name: str,
    vllm_model: LLM,
    sampling_params: SamplingParams,
    all_inference_data: List[Dict[str, Any]]
) -> List[Union[float, None]]:
    """
    Performs inference for ALL prompt-response pairs using a VLLM model
    and returns a list of numerical scores (ratings).
    A tqdm progress bar tracks the overall progress.

    Args:
        vllm_model: The initialized VLLM LLM object.
        sampling_params: VLLM SamplingParams for generation.
        all_inference_data: A list of dictionaries, where each dictionary contains
                            "prompt", "response", and metadata like "original_line_idx",
                            "type", and "response_idx" for mapping.

    Returns:
        A list of scores (floats) or None, maintaining the original order as in `all_inference_data`.
    """
    if not all_inference_data:
        return []

    prompts_to_generate = []
    # Prepare all prompt strings for VLLM's generate method
    for item in all_inference_data:
        if model_name not in model2prompt.keys():
            model_name = "Default"
        prompt_template = model2prompt[model_name]
        usr_prompt = prompt_template.format(prompt=item['prompt'], response=item['response'])

        prompts_to_generate.append(usr_prompt)

    logger.info(f"Generating {len(prompts_to_generate)} responses with VLLM...")

    # Perform batched inference with VLLM
    outputs = vllm_model.generate(prompts_to_generate, sampling_params)

    # Sort outputs by their request_id to ensure consistent order with input prompts
    # VLLM outputs might not be in the exact order of input prompts if some requests finish faster
    outputs_map = {output.request_id: output for output in outputs}
    
    results = [None] * len(all_inference_data)

    # Process results in the original order of all_inference_data
    for i, item_metadata in enumerate(all_inference_data):
        
        # If the order of `outputs` is guaranteed to match `prompts_to_generate`, then:
        score_text = outputs[i].outputs[0].text.strip()
        score = parse_score_from_text(score_text)
        results[i] = score

    logger.info("VLLM inference complete.")
    return results

# --- Main Inference Script ---
def run_inference_and_output_scores(
    input_file: str,
    output_file: str,
    model_path: str,
    temperature: float,
    max_tokens: int,
    chosen_num: int,
    rejected_num: int,
):
    """
    Reads a JSONL file, infers scores for chosen and rejected responses using a VLLM model,
    and saves only the scores to a new JSONL file. Uses VLLM's optimized batching.
    """
    logger.info(f"Initializing VLLM model from: {model_path}")
    # Initialize VLLM model
    # You might need to specify other VLLM parameters like `gpu_memory_utilization`
    # or `tensor_parallel_size` depending on your setup.
    vllm_model = LLM(model=model_path)
    model_name = model_path.split("/")[-1]
    
    # Initialize VLLM sampling parameters
    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)

    logger.info(f"Starting inference from '{input_file}' to '{output_file}' for chosen/rejected scores using local VLLM model.")

    with open(input_file, 'r', encoding='utf-8') as infile:
        input_lines = [json.loads(line.strip()) for line in infile.readlines()]

    all_inference_data = []
    # Prepare all prompt-response pairs for inference, keeping track of original line index
    for i, data in enumerate(input_lines):
        prompt = data.get("prompt")
        chosen_responses = data.get("chosen", [])
        rejected_responses = data.get("rejected", [])

        assert (isinstance(chosen_responses, list) and len(chosen_responses) == chosen_num)
        assert (isinstance(rejected_responses, list) and len(rejected_responses) == rejected_num)
    
        # Add metadata for mapping back to original structure
        for j, response in enumerate(chosen_responses):
            all_inference_data.append({"prompt": prompt, "response": response, "original_line_idx": i, "type": "chosen", "response_idx": j})
        
        for j, response in enumerate(rejected_responses):
            all_inference_data.append({"prompt": prompt, "response": response, "original_line_idx": i, "type": "rejected", "response_idx": j})

    # Initialize a structured dictionary to hold scores for each original line
    scores_by_original_line = {}
    for i in range(len(input_lines)):
        scores_by_original_line[i] = {
            "id": input_lines[i].get("id", f"line_{i}"),
            "subset": input_lines[i].get("subset"),
            "chosen_scores": [None] * chosen_num,
            "rejected_scores": [None] * rejected_num
        }

    # Perform all inferences in a single batched VLLM call
    all_inferred_scores = infer_all_scores_vllm(model_name, vllm_model, sampling_params, all_inference_data)
        
    # Distribute the inferred scores back to their original data structures
    for j, score in enumerate(all_inferred_scores):
        original_item_metadata = all_inference_data[j]
        original_line_idx = original_item_metadata["original_line_idx"]
        response_type = original_item_metadata["type"]
        response_idx = original_item_metadata["response_idx"]

        if response_type == "chosen":
            scores_by_original_line[original_line_idx]["chosen_scores"][response_idx] = score
        elif response_type == "rejected":
            scores_by_original_line[original_line_idx]["rejected_scores"][response_idx] = score
        
    # Collect all results from the structured dictionary, maintaining original order of input lines
    all_score_results = [scores_by_original_line[i] for i in range(len(input_lines))]

    # Save all results to the output JSONL file
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for entry in all_score_results:
            outfile.write(json.dumps(entry, ensure_ascii=False) + '\n')

    logger.info(f"Inference completed. Processed {len(input_lines)} original lines. Scores saved to '{output_file}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VLLM inference on a JSONL dataset for response scoring.")

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the local VLLM model (e.g., 'mistralai/Mistral-7B-Instruct-v0.2')."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input JSONL file containing 'prompt', 'chosen' (list), and 'rejected' (list) fields."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to the output JSONL file where reward scores will be saved."
    )
    parser.add_argument(
        "--chosen_num",
        type=int,
        default=1,
        help="Number of chosen responses per prompt in the input data."
    )
    parser.add_argument(
        "--rejected_num",
        type=int,
        default=3,
        help="Number of rejected responses per prompt in the input data."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for VLLM generation."
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1024,
        help="Maximum number of tokens to generate per response."
    )

    args = parser.parse_args()

    # Run the inference
    run_inference_and_output_scores(
        input_file=args.input_file,
        output_file=args.output_file,
        model_path=args.model_path,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        chosen_num=args.chosen_num,
        rejected_num=args.rejected_num,
    )