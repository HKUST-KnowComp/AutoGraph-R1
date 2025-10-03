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
            return solution_dict.get("answer", None), solution_dict.get("edge_coverage", 1), solution_dict.get("semantic_reward", 0)
        elif isinstance(solution_dict, list):
            # Optionally, try to get answer from first element if it's a dict
            if solution_dict and isinstance(solution_dict[0], dict):
                return solution_dict[0].get("answer", None), solution_dict[0].get("edge_coverage", 1), solution_dict[0].get("semantic_reward", 0)
            return None, 1, 0
        else:
            return None, 1, 0
    except Exception as e:
        print(f"Error extracting solution: {e}")
        return None, 1, 0

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

def compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float:
    """
    Computes the score for exact match (EM) with normalized string comparison.

    Args:
        solution_str (str): The full log containing solution text and assistant response.
        ground_truth (dict): The ground truth containing the target.
        method (str): The method to extract the solution, choices are 'strict' and 'flexible'.
        format_score (float): The score for correct format (default: 0.0).
        score (float): The score for a correct answer (default: 1.0).

    Returns:
        float: The computed score based on exact match.
    """
    # Extract the answer from the last assistant message
    answer, edge_coverage, semantic_reward = extract_solution(solution_str)
    # If no answer is extracted, return format_score
    if answer is None:
        em_reward = 0.0
        edge_coverage_reward = 0
        semantic_reward = 0
        reward = em_reward + edge_coverage_reward + semantic_reward
    # Perform Exact Match (EM) check with normalization
    if em_check(answer, ground_truth["target"]):
        # print(f"Exact match found: {answer} == {ground_truth['target']}")
        em_reward = 1.0
        edge_coverage_reward = 1 - edge_coverage
        reward = em_reward + edge_coverage_reward + semantic_reward
    return reward
