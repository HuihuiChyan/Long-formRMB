import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import json
import logging
import pandas as pd
import argparse
from tqdm import tqdm
from typing import List, Dict, Optional, Union
from ldlreward import LDLRewardModel27B
from qrm import LlamaForRewardModelWithGating31
from inform import INFORMForSequenceClassification

# Set up logging for better visibility
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Model Loading ---
def load_reward_model(model_path: str):
    """
    Loads the reward model and tokenizer.
    """
    logger.info(f"Loading model from: {model_path}")

    if "INF-ORM" in model_path:
        model = INFORMForSequenceClassification.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
            num_labels=1,
        )
    elif "Skywork-VL" in model_path:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        from trl import AutoModelForCausalLMWithValueHead
        from qwen_vl_utils import process_vision_info
        from transformers.utils import cached_file
        from safetensors import safe_open

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
        vhead_file = cached_file(
            path_or_repo_id=model_path, filename="value_head.safetensors"
        )
        with safe_open(vhead_file, framework="pt", device="cpu") as f:
            vhead_params = {key: f.get_tensor(key) for key in f.keys()}
        model.load_state_dict(vhead_params, strict=False)
        model.requires_grad_(False)
        model.eval()
    elif "LDL-Reward-Gemma-2-27B-v0.1" in model_path:
        model = LDLRewardModel27B.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",      
        )
    elif "QRM" in model_path:
        model = LlamaForRewardModelWithGating31.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,         
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
            trust_remote_code=True
        )
        # from trl import AutoModelForCausalLMWithValueHead
        # from transformers.utils import cached_file
        # from safetensors import safe_open
        # from transformers import AutoModelForCausalLM
        # model = AutoModelForCausalLM.from_pretrained(
        #     model_path,
        #     device_map="auto",
        #     torch_dtype=torch.bfloat16
        # )
        # model = AutoModelForCausalLMWithValueHead.from_pretrained(
        #     model,
        # )
        # vhead_file = cached_file(
        #     path_or_repo_id=model_path, filename="value_head.safetensors"
        # )
        # with safe_open(vhead_file, framework="pt", device="cpu") as f:
        #     vhead_params = {key: f.get_tensor(key) for key in f.keys()}
        # model.load_state_dict(vhead_params, strict=False)
        # model.requires_grad_(False)
        # model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    logger.info("Model and Tokenizer loaded successfully.")
    return model, tokenizer


def infer_score(
    item: Dict[str, str],
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
) -> float:
    """
    Performs inference for a batch of prompt-response pairs and returns a list of scores.

    Args:
        data: A list of dictionaries, where each dictionary contains a "prompt" and a "response" key.
              Example: [{"prompt": "P1", "response": "R1"}, {"prompt": "P2", "response": "R2"}]
        model: The loaded AutoModelForSequenceClassification model.
        tokenizer: The loaded AutoTokenizer.
        device: The device to run inference on (e.g., "cuda" or "cpu").

    Returns:
        A list of scores, one for each prompt-response pair in the input batch.
    """
    messages = [
        {"role": "user", "content": item["prompt"]},
        {"role": "assistant", "content": item["response"]}
    ]
    with torch.no_grad():
        if "QRM" in str(type(model)):
            input_ids = tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
            ).to(model.device)
            output = model(input_ids)
            import pdb;pdb.set_trace()
            score = output.score.cpu().float().item()
        elif "trl.models.modeling_value_head.AutoModelForCausalLMWithValueHead" in str(type(model)):
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            inputs = tokenizer(
                [text], return_tensors="pt", padding=True, truncation=True
            ).to('cuda')
            values = model(**inputs, return_dict=True, use_cache=False)[-1]
            scores = values.gather(
                dim=-1, index=(inputs["attention_mask"].sum(dim=-1, keepdim=True) - 1)
            )
            score = scores[0].item()
        else:
            input_ids = tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
            ).to(model.device)
            score = model(input_ids).logits[0].item()

    return score

# --- Main Inference Script ---
def run_inference_and_output_scores(
    input_file: str,
    output_file: str,
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    chosen_num: int,
    rejected_num: int,
):
    """
    Reads a JSONL file, infers scores for chosen and rejected responses,
    and saves only the scores to a new JSONL file.
    """
    logger.info(f"Starting inference from '{input_file}' to '{output_file}' for chosen/rejected scores.")

    model.eval() # Set model to evaluation mode

    all_score_results = []
    line_count = 0

    with open(input_file, 'r') as infile:
        input_lines = [json.loads(line.strip()) for line in infile.readlines()]

    all_inference_data = []
    original_line_indices = []
    for i, data in enumerate(input_lines):
        prompt = data.get("prompt")
        chosen_responses = data.get("chosen", [])
        rejected_responses = data.get("rejected", [])

        assert isinstance(chosen_responses, list) and len(chosen_responses) == chosen_num
        assert isinstance(rejected_responses, list) and len(rejected_responses) == rejected_num
    
        for response in chosen_responses:
            all_inference_data.append({"prompt": prompt, "response": response})
            original_line_indices.append(i) # Store original line index
        
        for response in rejected_responses:
            all_inference_data.append({"prompt": prompt, "response": response})
            original_line_indices.append(i) # Store original line index

    all_inferred_scores = []
    # Process all prompt-response pairs in batches
    for item in tqdm(all_inference_data, desc="Inferring Scores"):
        score = infer_score(item, model, tokenizer)
        all_inferred_scores.append(score)

    # Now, distribute the inferred scores back to their original data structures
    current_score_idx = 0
    for i, data in enumerate(input_lines):
        chosen_responses = data.get("chosen", [])
        rejected_responses = data.get("rejected", [])
        
        current_chosen_scores = all_inferred_scores[current_score_idx : current_score_idx + chosen_num]
        current_score_idx += chosen_num
        
        current_rejected_scores = all_inferred_scores[current_score_idx : current_score_idx + rejected_num]
        current_score_idx += rejected_num
        
        # Store only the scores and original ID
        all_score_results.append({
            "id": data.get("id", f"line_{i}"), # Use 'i' for line count if 'id' is missing
            "subset": data.get("subset"),
            "chosen_scores": current_chosen_scores,
            "rejected_scores": current_rejected_scores
        })

    # Save all results to the output JSONL file
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for entry in all_score_results:
            outfile.write(json.dumps(entry, ensure_ascii=False) + '\n')

    logger.info(f"Inference completed. Processed {line_count} original lines. Scores saved to '{output_file}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run reward model inference on a JSONL dataset.")

    # Define command-line arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the pre-trained reward model (e.g., 'nicolinho/QRM-Gemma-2-27B' or a local path)."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input JSONL file containing 'prompt', 'chosen', and 'rejected' fields."
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
        help="Number of chosen responses per prompt."
    )
    parser.add_argument(
        "--rejected_num",
        type=int,
        default=3,
        help="Number of rejected responses per prompt."
    )

    args = parser.parse_args()

    # Load model and tokenizer using the parsed model_path
    model, tokenizer = load_reward_model(args.model_path)

    # Run the inference with parsed input and output files
    run_inference_and_output_scores(
        input_file=args.input_file,
        output_file=args.output_file,
        model=model,
        tokenizer=tokenizer,
        chosen_num=args.chosen_num,
        rejected_num=args.rejected_num,
    )