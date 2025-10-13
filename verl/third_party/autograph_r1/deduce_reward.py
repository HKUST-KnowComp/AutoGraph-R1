import json_repair
import re
from typing import Union, Dict, List


def extract_solution(solution_str: str) -> Union[str, None]:
    try:
        assistant_parts = solution_str.split('assistant\n')
        if not assistant_parts:
            return None, 0
        last_assistant_response = assistant_parts[-1].strip()
        solution_dict = json_repair.loads(last_assistant_response)
        if isinstance(solution_dict, dict):
            return solution_dict.get("answer", None), solution_dict.get("triple_repetition", 0)
        elif isinstance(solution_dict, list):
            # Optionally, try to get answer from first element if it's a dict
            if solution_dict and isinstance(solution_dict[0], dict):
                return solution_dict[0].get("answer", None), solution_dict[0].get("triple_repetition", 0)
            return None, 0
        else:
            return None, 0
    except Exception as e:
        print(f"Error extracting solution: {e}")
        return None, 0


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs) -> dict:
    """
    Computes the reward as:
        reward = em_reward + semantic_reward - edge_coverage
    clipped to be non-negative.
    """
    triple_repetition_penalty = kwargs.get('triple_repetition_penalty', 0.0)
    # Extract values
    answer, triple_repetition = extract_solution(solution_str)

    # Handle missing answer
    if answer is None:
        if triple_repetition_penalty > 0.0:
            return {
                'score': 0.0,
                'deducable': 0.0,
                'triple_repetition': 0.0
            }
        else:
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
    try:
        triple_repetition = float(triple_repetition)
    except:
        triple_repetition = 0.0
    if triple_repetition_penalty > 0.0:
        reward = deducable - triple_repetition_penalty * triple_repetition
        reward = max(reward, 0.0)
        return {
            'score': reward,
            'deducable': deducable,
            'triple_repetition': float(triple_repetition)
        }
    else:
        reward = deducable
        return {
            'score': reward,
            'deducable': deducable,
        }

