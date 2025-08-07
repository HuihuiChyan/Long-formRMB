import openai
import json
import logging
import pandas as pd
import argparse
from tqdm import tqdm
from typing import List, Dict, Union, Tuple
import os
import time
import concurrent.futures
import httpx
import re
from functools import partial

# Set up logging for better visibility
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper functions for API calls ---
def call_openai_for_score(
    item: Dict[str, str],
    client: openai.OpenAI,
    model_name: str,
) -> Union[float, None]:
    """
    Makes a single OpenAI API call to get a numerical rating for a prompt-response pair.
    Returns the score or None if an error occurs or parsing fails.
    """
    usr_prompt = f"""### Task Description
Please act as an impartial judge and evaluate the quality of the response provided by an
AI assistant to the user query displayed below.

Notes:
1- Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response.
2- Begin your evaluation by providing a short explanation.
3- Be as objective as possible. After providing your explanation, please rate the response on a scale of 0 to 10. For your rating, only give a number between 0 and 10 (inclusive), do not use any markdown, and do not put any text after your final rating.

[Query]
{item['prompt']}

[Response]
{item['response']}

[Your judgement]"""

    messages = [{"role": "user", "content": usr_prompt},]

    max_tokens = 1024
    if "gemini" in model_name:
        max_tokens = 32768

    API_RETRY_ATTEMPTS = 5  # API调用失败时的重试次数
    API_RETRY_DELAY = 1     # 重试间隔（秒）
    for attempt in range(API_RETRY_ATTEMPTS):
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=max_tokens,
            )
            score_text = completion.choices[0].message.content.strip()
            match = re.search(r"([0-9]+)\**\s*$", score_text)

            if match is not None:
                rating = int(match.group(1))
                return rating
            else:
                raise Exception(f"Could not parse score for text: {score_text}")

        except httpx.TimeoutException as e: # Catch the specific timeout exception
            print(f"Model {model_name} API call timed out: {e}")
            return None
        
        except Exception as e:
            print(f"Model {model_name} API call error (attempt {attempt + 1}/{API_RETRY_ATTEMPTS}): {e}")
            if attempt < API_RETRY_ATTEMPTS - 1:
                time.sleep(API_RETRY_DELAY)

    return None

def call_openai_for_selection_1v3(
    item: Dict[str, str],
    client: openai.OpenAI,
    model_name: str,
) -> Union[int, None]:
    """
    Makes a single OpenAI API call to select the best response from four options (1v3 case).
    Returns the index of the selected response (0-3) or None if an error occurs.
    """
    usr_prompt = (
        "Please act as an impartial judge and evaluate the quality of the responses provided by four AI assistants to the user question displayed below. "
        "You should choose the assistant that follows the user's instructions and answers the user's question best. Your evaluation should consider "
        "factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by "
        "comparing the four responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were "
        "presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names "
        "of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: "
        '"[[A]]" if assistant A is best, "[[B]]" if assistant B is best, "[[C]]" if assistant C is best, and "[[D]]" if assistant D is best.'
        f"[User Question]\n{item['prompt']}\n\n[The Start of Assistant A's Answer]\n{item['responses'][0]}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]\n{item['responses'][1]}\n[The End of Assistant B's Answer]\n\n[The Start of Assistant C's Answer]\n{item['responses'][2]}\n[The End of Assistant C's Answer]\n\n[The Start of Assistant D's Answer]\n{item['responses'][3]}\n[The End of Assistant D's Answer]"
    )

    messages = [{"role": "user", "content": usr_prompt},]

    max_tokens = 1024
    if "gemini" in model_name:
        max_tokens = 32768

    API_RETRY_ATTEMPTS = 5
    API_RETRY_DELAY = 1
    for attempt in range(API_RETRY_ATTEMPTS):
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=max_tokens,
            )
            selection_text = completion.choices[0].message.content.strip()
            
            # Check which option the model selected
            if "[[A]]" in selection_text:
                return selection_text, 0  # Option A selected
            elif "[[B]]" in selection_text:
                return selection_text, 1  # Option B selected
            elif "[[C]]" in selection_text:
                return selection_text, 2  # Option C selected
            elif "[[D]]" in selection_text:
                return selection_text, 3  # Option D selected
            else:
                raise Exception(f"Could not parse selection for text: {selection_text}")

        except httpx.TimeoutException as e:
            print(f"Model {model_name} API call timed out: {e}")
            return None, None
        
        except Exception as e:
            print(f"Model {model_name} API call error (attempt {attempt + 1}/{API_RETRY_ATTEMPTS}): {e}")
            if attempt < API_RETRY_ATTEMPTS - 1:
                time.sleep(API_RETRY_DELAY)

    return None, None

def call_openai_for_selection_1v1(
    item: Dict[str, str],
    client: openai.OpenAI,
    model_name: str,
) -> Union[int, None]:
    """
    Makes a single OpenAI API call to select the best response from two options (1v1 case).
    Returns the index of the selected response (0-1) or None if an error occurs.
    """
    usr_prompt = (
        "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. "
        "You should choose the assistant that follows the user's instructions and answers the user's question best. Your evaluation should consider "
        "factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by "
        "comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were "
        "presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names "
        "of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: "
        '"[[A]]" if assistant A is best, "[[B]]" if assistant B is best.'
        f"[User Question]\n{item['prompt']}\n\n[The Start of Assistant A's Answer]\n{item['responses'][0]}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]\n{item['responses'][1]}\n[The End of Assistant B's Answer]"
    )

    messages = [{"role": "user", "content": usr_prompt},]
    max_tokens = 1024
    if "gemini" in model_name:
        max_tokens = 32768

    API_RETRY_ATTEMPTS = 5
    API_RETRY_DELAY = 1
    for attempt in range(API_RETRY_ATTEMPTS):
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=max_tokens,
            )
            selection_text = completion.choices[0].message.content.strip()
            
            # Check which option the model selected - for 1v1 we only have options A and B
            if "[[A]]" in selection_text:
                return selection_text, 0  # Option A selected
            elif "[[B]]" in selection_text:
                return selection_text, 1  # Option B selected
            else:
                raise Exception(f"Could not parse selection for text: {selection_text}")

        except httpx.TimeoutException as e:
            print(f"Model {model_name} API call timed out: {e}")
            return None, None
        
        except Exception as e:
            print(f"Model {model_name} API call error (attempt {attempt + 1}/{API_RETRY_ATTEMPTS}): {e}")
            if attempt < API_RETRY_ATTEMPTS - 1:
                time.sleep(API_RETRY_DELAY)

    return None, None

def infer_all_scores_multithreaded(
    client: openai.OpenAI,
    model_name: str,
    all_data_to_infer: List[Dict[str, str]], # Now takes all data, not just one batch
    num_threads: int
) -> List[Union[float, None]]:
    """
    Performs inference for ALL prompt-response pairs using OpenAI API
    with multithreading and returns a list of numerical scores (ratings).
    A tqdm progress bar tracks the overall progress.

    Args:
        client: The initialized OpenAI client.
        model_name: The name of the OpenAI model to use.
        all_data_to_infer: A list of dictionaries, where each dictionary contains a "prompt" and a "response" key,
                           plus original_line_idx, type, and response_idx for mapping.
        num_threads: The number of concurrent threads to use for API calls.

    Returns:
        A list of scores (floats) or None, maintaining the original order as in `all_data_to_infer`.
    """
    if not all_data_to_infer:
        return []

    # Use ThreadPoolExecutor for concurrent API calls
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit tasks: (index, future_object) to keep track of original order
        # Each item in all_data_to_infer is a dictionary that includes 'prompt' and 'response'
        future_to_index = {
            executor.submit(call_openai_for_score, item, client, model_name): idx
            for idx, item in enumerate(all_data_to_infer)
        }

        results = [None] * len(all_data_to_infer)

        # Collect results as they complete, with a progress bar
        for future in tqdm(concurrent.futures.as_completed(future_to_index), total=len(all_data_to_infer), desc="Inferring Scores"):
            idx = future_to_index[future]
            try:
                score = future.result()
                results[idx] = score
            except Exception as exc:
                logger.error(f"Item at index {idx} generated an exception: {exc}")
                results[idx] = None # Ensure a None if an unhandled exception occurs

    return results

def infer_all_selections_multithreaded(
    client: openai.OpenAI,
    model_name: str,
    all_data_to_infer: List[Dict[str, str]],
    num_threads: int,
    selection_mode: str = "1v3"
) -> List[Union[int, None]]:
    """
    Performs inference for ALL prompt-response sets using OpenAI API
    with multithreading and returns a list of integers indicating which response was selected.

    Args:
        client: The initialized OpenAI client.
        model_name: The name of the OpenAI model to use.
        all_data_to_infer: A list of dictionaries, where each dictionary contains "prompt", "responses" (list of responses),
                           "chosen_idx" (index of the chosen response in the responses list), and original_line_idx for mapping.
        num_threads: The number of concurrent threads to use for API calls.
        selection_mode: Either "1v1" (chosen_num=1, rejected_num=1) or "1v3" (chosen_num=1, rejected_num=3)

    Returns:
        A list of integers (0-3 for 1v3 mode, 0-1 for 1v1 mode) or None, maintaining the original order as in `all_data_to_infer`.
    """
    if not all_data_to_infer:
        return []

    # Select the appropriate selection function based on mode
    selection_function = call_openai_for_selection_1v1 if selection_mode == "1v1" else call_openai_for_selection_1v3

    # Use ThreadPoolExecutor for concurrent API calls
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit tasks: (index, future_object) to keep track of original order
        future_to_index = {
            executor.submit(selection_function, item, client, model_name): idx
            for idx, item in enumerate(all_data_to_infer)
        }

        results = [(None, None)] * len(all_data_to_infer)

        # Collect results as they complete, with a progress bar
        for future in tqdm(concurrent.futures.as_completed(future_to_index), total=len(all_data_to_infer), desc="Inferring Selections"):
            idx = future_to_index[future]
            try:
                selection = future.result()
                results[idx] = selection
            except Exception as exc:
                logger.error(f"Item at index {idx} generated an exception: {exc}")
                results[idx] = None, None  # Ensure a None if an unhandled exception occurs

    return results

# --- Main Inference Script ---
def run_inference_and_output_scores(
    input_file: str,
    output_file: str,
    client: openai.OpenAI,
    model_name: str,
    chosen_num: int,
    rejected_num: int,
    num_threads: int,
    infer_mode: str = "scoring",
):
    """
    Reads a JSONL file, infers scores or selections for chosen and rejected responses using OpenAI API,
    and saves the results to a new JSONL file. Uses multithreading for speed.
    
    If infer_mode is "scoring", it will score each response individually.
    If infer_mode is "selection", it will have the model select between chosen and rejected responses.
    """
    logger.info(f"Starting inference from '{input_file}' to '{output_file}' using OpenAI model '{model_name}' in {infer_mode} mode.")
    logger.info(f"Using {num_threads} concurrent threads for API calls.")

    with open(input_file, 'r', encoding='utf-8') as infile:
        input_lines = [json.loads(line.strip()) for line in infile.readlines()]

    if infer_mode == "scoring":
        # Scoring mode: score each response individually
        all_inference_data = []
        # Prepare all prompt-response pairs for inference, keeping track of original line index
        for i, data in enumerate(input_lines):
            prompt = data.get("prompt")
            chosen_responses = data.get("chosen", [])
            rejected_responses = data.get("rejected", [])

            # Validate input structure
            if not (isinstance(chosen_responses, list) and len(chosen_responses) == chosen_num):
                logger.warning(f"Line {i} (ID: {data.get('id', 'N/A')}) has {len(chosen_responses)} chosen responses, but expected {chosen_num}. Skipping.")
                continue
            if not (isinstance(rejected_responses, list) and len(rejected_responses) == rejected_num):
                logger.warning(f"Line {i} (ID: {data.get('id', 'N/A')}) has {len(rejected_responses)} rejected responses, but expected {rejected_num}. Skipping.")
                continue
        
            # Add metadata for mapping back to original structure
            for j, response in enumerate(chosen_responses):
                all_inference_data.append({"prompt": prompt, "response": response, "original_line_idx": i, "type": "chosen", "response_idx": j})
            
            for j, response in enumerate(rejected_responses):
                all_inference_data.append({"prompt": prompt, "response": response, "original_line_idx": i, "type": "rejected", "response_idx": j})
    elif infer_mode == "selection":  # selection mode
        # Determine which selection mode to use based on chosen_num and rejected_num
        selection_mode = "1v1" if (chosen_num == 1 and rejected_num == 1) else "1v3"
        
        if selection_mode == "1v3" and (chosen_num != 1 or rejected_num != 3):
            logger.warning(f"For 1v3 selection mode, using chosen_num=1 and rejected_num=3 (ignoring provided values: chosen_num={chosen_num}, rejected_num={rejected_num})")
            chosen_num = 1
            rejected_num = 3
        elif selection_mode == "1v1" and (chosen_num != 1 or rejected_num != 1):
            logger.warning(f"For 1v1 selection mode, using chosen_num=1 and rejected_num=1 (ignoring provided values: chosen_num={chosen_num}, rejected_num={rejected_num})")
            chosen_num = 1
            rejected_num = 1
            
        logger.info(f"Using selection mode: {selection_mode}")
            
        # Selection mode: present responses (1 chosen + N rejected) together
        all_inference_data = []
        import random
        
        # Prepare all prompt-response sets for inference, keeping track of original line index
        for i, data in enumerate(input_lines):
            prompt = data.get("prompt")
            chosen_responses = data.get("chosen", [])
            rejected_responses = data.get("rejected", [])

            # Validate input structure based on selection mode
            if not (isinstance(chosen_responses, list) and len(chosen_responses) >= 1):
                logger.warning(f"Line {i} (ID: {data.get('id', 'N/A')}) has {len(chosen_responses)} chosen responses, but expected at least 1. Skipping.")
                continue
                
            if selection_mode == "1v3":
                if not (isinstance(rejected_responses, list) and len(rejected_responses) >= 3):
                    logger.warning(f"Line {i} (ID: {data.get('id', 'N/A')}) has {len(rejected_responses)} rejected responses, but expected at least 3 for 1v3 mode. Skipping.")
                    continue
                
                # Use the first chosen response and the first 3 rejected responses
                chosen = chosen_responses[0]
                rejected_subset = rejected_responses[:3]
                
                # Combine and shuffle the responses
                all_responses = [chosen] + rejected_subset
                # Create a mapping to remember which index contains the chosen response
                chosen_idx = 0  # Initially, the chosen response is at index 0
                
                # Shuffle the responses and track where the chosen response ends up
                indices = list(range(4))
                random.shuffle(indices)
                shuffled_responses = [all_responses[idx] for idx in indices]
                # Find where the chosen response ended up after shuffling
                chosen_idx = indices.index(0)
                
            else:  # 1v1 mode
                if not (isinstance(rejected_responses, list) and len(rejected_responses) >= 1):
                    logger.warning(f"Line {i} (ID: {data.get('id', 'N/A')}) has {len(rejected_responses)} rejected responses, but expected at least 1 for 1v1 mode. Skipping.")
                    continue
                
                # Use the first chosen response and the first rejected response
                chosen = chosen_responses[0]
                rejected = rejected_responses[0]
                
                # Combine and shuffle the responses
                all_responses = [chosen, rejected]
                chosen_idx = 0  # Initially, the chosen response is at index 0
                
                # Shuffle the responses and track where the chosen response ends up
                indices = list(range(2))
                random.shuffle(indices)
                shuffled_responses = [all_responses[idx] for idx in indices]
                # Find where the chosen response ended up after shuffling
                chosen_idx = indices.index(0)
            
            all_inference_data.append({
                "prompt": prompt,
                "responses": shuffled_responses,
                "chosen_idx": chosen_idx,
                "original_line_idx": i
            })

    if infer_mode == "scoring":
        # Initialize a structured dictionary to hold scores for each original line
        scores_by_original_line = {}
        for i in range(len(input_lines)):
            # Only initialize if the line's data was included in all_inference_data (i.e., not skipped)
            if i in {item['original_line_idx'] for item in all_inference_data}:
                scores_by_original_line[i] = {
                    "id": input_lines[i].get("id", f"line_{i}"),
                    "subset": input_lines[i].get("subset"),
                    "chosen_scores": [None] * chosen_num,
                    "rejected_scores": [None] * rejected_num
                }
            else:
                # If a line was skipped, add an empty placeholder
                scores_by_original_line[i] = {
                    "id": input_lines[i].get("id", f"line_{i}"),
                    "subset": input_lines[i].get("subset"),
                    "chosen_scores": [], # Empty list if no valid data for this line
                    "rejected_scores": []
                }

        # Perform all inferences in a single multithreaded call
        all_inferred_scores = infer_all_scores_multithreaded(client, model_name, all_inference_data, num_threads)
            
        # Distribute the inferred scores back to their original data structures
        for j, score in enumerate(all_inferred_scores):
            original_item_metadata = all_inference_data[j]
            original_line_idx = original_item_metadata["original_line_idx"]
            response_type = original_item_metadata["type"]
            response_idx = original_item_metadata["response_idx"]

            # Ensure the original_line_idx is valid in scores_by_original_line
            if original_line_idx in scores_by_original_line:
                if response_type == "chosen":
                    scores_by_original_line[original_line_idx]["chosen_scores"][response_idx] = score
                elif response_type == "rejected":
                    scores_by_original_line[original_line_idx]["rejected_scores"][response_idx] = score
    else:  # selection mode
        # Initialize a structured dictionary to hold selection results for each original line
        selection_results = {}
        for i in range(len(input_lines)):
            # Only initialize if the line's data was included in all_inference_data (i.e., not skipped)
            if i in {item['original_line_idx'] for item in all_inference_data}:
                selection_results[i] = {
                    "id": input_lines[i].get("id", f"line_{i}"),
                    "subset": input_lines[i].get("subset"),
                    "selected_idx": None,  # Which response was selected (0-3)
                    "chosen_idx": None,    # Which index had the chosen response (0-3)
                    "responses": []        # The shuffled responses
                }
            else:
                # If a line was skipped, add an empty placeholder
                selection_results[i] = {
                    "id": input_lines[i].get("id", f"line_{i}"),
                    "subset": input_lines[i].get("subset"),
                    "selected_idx": None,
                    "chosen_idx": None,
                    "responses": []
                }

        # Determine which selection mode to use
        selection_mode = "1v1" if (chosen_num == 1 and rejected_num == 1) else "1v3"
        
        # Perform all inferences in a single multithreaded call
        all_inferred_selections = infer_all_selections_multithreaded(
            client, 
            model_name, 
            all_inference_data, 
            num_threads,
            selection_mode
        )
        
        for j, (selection_text, selected_idx) in enumerate(all_inferred_selections):
            if selected_idx is not None:  # Only process valid selections
                original_item_metadata = all_inference_data[j]
                original_line_idx = original_item_metadata["original_line_idx"]
                chosen_idx = original_item_metadata["chosen_idx"]
                responses = original_item_metadata["responses"]

                # Ensure the original_line_idx is valid in selection_results
                if original_line_idx in selection_results:
                    # Store the selection results
                    selection_results[original_line_idx]['selection_text'] = selection_text
                    selection_results[original_line_idx]["selected_idx"] = selected_idx
                    selection_results[original_line_idx]["chosen_idx"] = chosen_idx
                    selection_results[original_line_idx]["responses"] = responses
            
    # Collect all results from the structured dictionary, maintaining original order of input lines
    if infer_mode == "scoring":
        all_results = [scores_by_original_line[i] for i in range(len(input_lines))]
        logger.info(f"Inference completed. Processed {len(input_lines)} original lines. Scores saved to '{output_file}'.")
    else:  # selection mode
        all_results = [selection_results[i] for i in range(len(input_lines))]
        logger.info(f"Inference completed. Processed {len(input_lines)} original lines. Selection results saved to '{output_file}'.")

    # Save all results to the output JSONL file
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for entry in all_results:
            outfile.write(json.dumps(entry, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OpenAI API inference on a JSONL dataset for response scoring with multithreading.")

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the OpenAI model to use (e.g., 'gpt-4o', 'gpt-3.5-turbo')."
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
        "--batch_size",
        type=int,
        default=8,
        help="Number of prompt-response pairs to process concurrently in one batch. Higher values might lead to faster processing but could hit API rate limits."
    )
    parser.add_argument(
        "--num_threads", # New argument
        type=int,
        default=10, # Default to 5 threads
        help="Number of concurrent threads to use for OpenAI API calls within each batch."
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="Your OpenAI API key. If not provided, it will be read from OPENAI_API_KEY environment variable."
    )
    parser.add_argument(
        "--infer_mode",
        type=str,
        default="scoring",
        choices=["scoring", "selection"],
        help="Inference mode: 'scoring' to score each response individually, or 'selection' to have the model select between chosen and rejected responses."
    )

    args = parser.parse_args()

    http_client = httpx.Client(timeout=60.0)

    API_KEY = os.getenv('OPENAI_API_KEY')
    client = openai.OpenAI(api_key=API_KEY, base_url="https://api.shubiaobiao.cn/v1", http_client=http_client)

    # Run the inference
    run_inference_and_output_scores(
        input_file=args.input_file,
        output_file=args.output_file,
        client=client,
        model_name=args.model_name,
        chosen_num=args.chosen_num,
        rejected_num=args.rejected_num,
        num_threads=args.num_threads,
        infer_mode=args.infer_mode
    )
