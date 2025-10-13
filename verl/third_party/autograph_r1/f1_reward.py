import json_repair
import re
from typing import Union, Dict, List, Set

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
            return None, 1, 0, 0
        last_assistant_response = assistant_parts[-1].strip()
        solution_dict = json_repair.loads(last_assistant_response)
        if isinstance(solution_dict, dict):
            return solution_dict.get("answer", None), solution_dict.get("edge_coverage", 1), solution_dict.get("semantic_reward", 0), solution_dict.get("triple_repetition", 0)
        elif isinstance(solution_dict, list):
            # Optionally, try to get answer from first element if it's a dict
            if solution_dict and isinstance(solution_dict[0], dict):
                return solution_dict[0].get("answer", None), solution_dict[0].get("edge_coverage", 1), solution_dict[0].get("semantic_reward", 0), solution_dict[0].get("triple_repetition", 0)
            return None, 1, 0, 0
        else:
            return None, 1, 0, 0
    except Exception as e:
        print(f"Error extracting solution: {e}")
        return None, 1, 0

def get_tokens(text: str) -> Set[str]:
    """
    Split text into tokens (words) for F1 computation.
    
    Args:
        text (str): Text to tokenize.
        
    Returns:
        Set[str]: Set of tokens.
    """
    normalized = normalize_string(text)
    # Split by whitespace to get individual tokens
    return set(normalized.split())

def compute_f1(answer: str, target: str) -> float:
    """
    Compute F1 score between prediction and ground truth.
    
    Args:
        answer (str): Predicted answer.
        target (str): Ground truth.
        
    Returns:
        float: F1 score between 0 and 1.
    """
    pred_tokens = get_tokens(answer)
    gold_tokens = get_tokens(target)
    
    if not pred_tokens and not gold_tokens:
        return 1.0  # Both empty strings
    if not pred_tokens or not gold_tokens:
        return 0.0  # One is empty
    
    common_tokens = pred_tokens.intersection(gold_tokens)
    
    # Precision, Recall, and F1
    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(gold_tokens)
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1

def f1_check(answer: Union[str, None], target: Union[str, List]) -> float:
    """
    Computes the F1 score between the answer and target.
    If target is a list, returns the maximum F1 across all targets.
    
    Args:
        answer (str): The extracted answer.
        target (str or list): The ground truth target.
        
    Returns:
        float: F1 score between 0 and 1.
    """
    if answer is None:
        return 0.0
    
    # If target is a list, compute F1 against each alias and take the maximum
    if isinstance(target, list):
        return max((compute_f1(answer, alias) for alias in target), default=0.0)
    
    # Direct comparison with single target
    return compute_f1(answer, target)


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs) -> dict:
    """
    Computes the reward as just the F1 score, ignoring edge_coverage and semantic_reward.
    """
    triple_repetition_penalty = kwargs.get('triple_repetition_penalty', 0.0)
    # Extract values
    answer, _, _, triple_repetition = extract_solution(solution_str)

    # Default
    f1_score = 0.0

    # Handle missing answer
    if answer is None:
        if triple_repetition_penalty > 0:
            return {
                'score': 0.0,
                'f1_score': 0.0,
                'triple_repetition': 0.0
            }
        else:
            return {
                'score': 0.0,
                'f1_score': 0.0,
            }

    # Compute F1 score
    f1_score = f1_check(answer, ground_truth["target"])

    # Just use F1 as the reward directly
    if triple_repetition_penalty >0 :
        reward = f1_score - triple_repetition_penalty * triple_repetition
        reward = max(0.0, reward)  # Clip to non-negative
        return {
            'score': reward,  # This is the main reward used by the system
            'f1_score': f1_score,
            'triple_repetition': triple_repetition
        }
    else:
        reward = f1_score

        return {
            'score': reward,  # This is the main reward used by the system
            'f1_score': f1_score,
        }