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
    s = s.replace("'", "'").replace("`", "'").replace("'", "'")
    s = re.sub(r'[^\w\s]', '', s)
    return s

def extract_solution(solution_str: str) -> Union[str, None]:
    try:
        assistant_parts = solution_str.split('assistant\n')
        if not assistant_parts:
            return None, 0, 0
        last_assistant_response = assistant_parts[-1].strip()
        solution_dict = json_repair.loads(last_assistant_response)
        if isinstance(solution_dict, dict):
            return solution_dict.get("answer", None), solution_dict.get("recall", 0), solution_dict.get("triple_repetition", 0)
        elif isinstance(solution_dict, list):
            # Optionally, try to get answer from first element if it's a dict
            if solution_dict and isinstance(solution_dict[0], dict):
                return solution_dict[0].get("answer", None), solution_dict[0].get("recall", 0), solution_dict[0].get("triple_repetition", 0)
            return None, 0, 0
        else:
            return None, 0, 0
    except Exception as e:
        print(f"Error extracting solution: {e}")
        return None, 0, 0

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

def compute_f1(answer: str, target: str) -> tuple:
    """
    Compute F1 score between prediction and ground truth.
    
    Args:
        answer (str): Predicted answer.
        target (str): Ground truth.
        
    Returns:
        tuple: (f1_score, precision, recall)
    """
    pred_tokens = get_tokens(answer)
    gold_tokens = get_tokens(target)
    
    if not pred_tokens and not gold_tokens:
        return 1.0, 1.0, 1.0  # Both empty strings
    if not pred_tokens or not gold_tokens:
        return 0.0, 0.0, 0.0  # One is empty
    
    common_tokens = pred_tokens.intersection(gold_tokens)
    
    # Precision, Recall, and F1
    precision = len(common_tokens) / len(pred_tokens) if pred_tokens else 0
    recall = len(common_tokens) / len(gold_tokens) if gold_tokens else 0
    
    if precision + recall == 0:
        return 0.0, 0.0, 0.0
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1, precision, recall

def f1_check(answer: Union[str, None], target: Union[str, List]) -> tuple:
    """
    Computes the F1 score, precision, and recall between the answer and target.
    If target is a list, returns scores for the best F1.
    
    Args:
        answer (str): The extracted answer.
        target (str or list): The ground truth target.
        
    Returns:
        tuple: (f1_score, precision, recall)
    """
    if answer is None:
        return 0.0, 0.0, 0.0
    
    # If target is a list, compute scores for each alias and take the best F1
    if isinstance(target, list):
        scores = [compute_f1(answer, alias) for alias in target]
        if not scores:
            return 0.0, 0.0, 0.0
        # Find index of best F1 score
        best_idx = max(range(len(scores)), key=lambda i: scores[i][0], default=0)
        return scores[best_idx]
    
    # Direct comparison with single target
    return compute_f1(answer, target)

def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs) -> dict:
    """
    Computes the reward as:
        reward = f1_score + model_recall
    """
    triple_repetition_penalty = kwargs.get('triple_repetition_penalty', 0.0)
    # Extract values
    answer, model_recall, triple_repetition = extract_solution(solution_str)

    # Default values
    f1_score = 0.0
    precision = 0.0
    recall = 0.0
    reward = 0.0

    # Handle missing answer
    if answer is None:
        if triple_repetition_penalty > 0:
            return {
                'score': 0.0,
                'recall': 0.0,
                'triple_repetition': 0.0
            }
        else:
            return {
                'score': 0.0,
                'recall': 0.0
            }

    # Compute F1 metrics
    # f1_score, precision, recall = f1_check(answer, ground_truth["target"])

    try:
        model_recall = float(model_recall)
        triple_repetition = float(triple_repetition)
    except (TypeError, ValueError):
        model_recall = 0.0
        triple_repetition = 0.0

    # Compute reward: f1 score + model recall
    if triple_repetition_penalty > 0:
        reward = model_recall - triple_repetition_penalty * triple_repetition
        reward = max(0.0, reward)  # Ensure non-negative
        return {
            'score': reward,
            'recall': model_recall,
            'triple_repetition': triple_repetition
        }
    else:
        reward = model_recall 

        # Clip to non-negative (should already be non-negative)
        reward = max(0.0, reward)

        return {
            'score': reward,
            'recall': model_recall
        }