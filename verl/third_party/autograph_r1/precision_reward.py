import json_repair
import re
from typing import Union, Dict, List

def normalize_string(s: Union[str, List]) -> str:
    """
    Normalizes a string or list of strings for consistent comparison.
    
    Args:
        s: Input string or list of strings.
        
    Returns:
        str: Normalized string with consistent formatting.
    """
    if isinstance(s, list):
        s = " ".join(str(item).strip() for item in s)
    if not isinstance(s, str):
        s = str(s)
    
    s = s.lower().strip()
    s = re.sub(r'\s+', ' ', s)
    s = s.replace("’", "'").replace("`", "'").replace("'", "'")
    s = re.sub(r'[^\w\s]', '', s)
    return s

def extract_solution(solution_str: str) -> Union[str, None]:
    try:
        assistant_parts = solution_str.split('assistant\n')
        if not assistant_parts:
            return None
        last_assistant_response = assistant_parts[-1].strip()
        solution_dict = json_repair.loads(last_assistant_response)
        if isinstance(solution_dict, dict):
            return solution_dict.get("answer", None), solution_dict.get("precision", 0)
        elif isinstance(solution_dict, list):
            # Optionally, try to get answer from first element if it's a dict
            if solution_dict and isinstance(solution_dict[0], dict):
                return solution_dict[0].get("answer", None), solution_dict[0].get("precision", 0)
            return None, 0
        else:
            return None, 0
    except Exception as e:
        print(f"Error extracting solution: {e}")
        return None, 0

def em_check(answer: Union[str, None], target: Union[str, List]) -> bool:
    """
    Performs an exact match (EM) check between the extracted answer and the ground truth.

    Args:
        answer (str): The extracted answer.
        target (str or list): The ground truth target, which can be a string or a list of aliases.

    Returns:
        bool: True if the answer matches any target alias exactly, False otherwise.
    """
    if answer is None:
        return False

    normalized_answer = normalize_string(answer)

    # If target is a list, check against all aliases
    if isinstance(target, list):
        return any(normalized_answer == normalize_string(alias) for alias in target)

    # If target is a single string, perform a direct comparison
    normalized_target = normalize_string(target)
    return normalized_answer == normalized_target

def compute_score(data_source, solution_str, ground_truth, extra_info=None) -> dict:
    """
    Computes the reward as:
        reward = em_reward + semantic_reward - edge_coverage
    clipped to be non-negative.
    """
    # Extract values
    answer, precision = extract_solution(solution_str)

    # Default
    em_reward = 0.0
    reward = 0.0

    # Handle missing answer
    if answer is None:
        return {
            'score': 0.0,
            'em_reward': 0.0,
            'precision': 0.0,
        }

    # Exact Match
    if em_check(answer, ground_truth["target"]):
        em_reward = 1.0

    try:
        precision = float(precision)
    except (TypeError, ValueError):
        precision = 0.0

    reward = em_reward + precision

    # Clip to non-negative
    reward = max(0.0, reward)

    return {
        'score': reward,
        'em_reward': em_reward,
        'precision': precision
    }

