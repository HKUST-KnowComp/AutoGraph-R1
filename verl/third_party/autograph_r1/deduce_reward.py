import json_repair
import re
from typing import Union, Dict, List


def extract_solution(solution_str: str) -> Union[str, None]:
    try:
        assistant_parts = solution_str.split('assistant\n')
        if not assistant_parts:
            return None
        last_assistant_response = assistant_parts[-1].strip()
        solution_dict = json_repair.loads(last_assistant_response)
        if isinstance(solution_dict, dict):
            return solution_dict.get("answer", None)
        elif isinstance(solution_dict, list):
            # Optionally, try to get answer from first element if it's a dict
            if solution_dict and isinstance(solution_dict[0], dict):
                return solution_dict[0].get("answer", None)
            return None
        else:
            return None
    except Exception as e:
        print(f"Error extracting solution: {e}")
        return None


def compute_score(data_source, solution_str, ground_truth, extra_info=None) -> dict:
    """
    Computes the reward as:
        reward = em_reward + semantic_reward - edge_coverage
    clipped to be non-negative.
    """
    # Extract values
    answer = extract_solution(solution_str)


    # Handle missing answer
    if answer is None:
        return {
            'score': 0.0,
            'deducable': 0.0,
        }
    deducable = 0.0
    if answer.lower() == "yes":
        deducable = 1.0
    elif answer.lower() == "no":
        deducable = 0.0
    else:
        deducable = 0.0  # Treat unrecognized answers as non-deducable
    
    return {
        'score': deducable,
        'deducable': deducable,
    }

