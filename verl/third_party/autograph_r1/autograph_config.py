from verl.workers.config import RolloutConfig
from dataclasses import dataclass

@dataclass
class AutoGraphActorConfig(RolloutConfig):
    """Add AutoGraph-specific configuration."""
    use_api:bool =True # always set to true for AutoGraph, otherwise using original VeRL multi-turn training
    rag_method:str = 'subgraph' # available: 'subgraph', 'hipporag2', 'edge', 'tog', 'hipporag' (Use subgraph for Graph Retriever, hipporag2 for Graph-based Text Retriever)
    text_linking:bool= False # Indicate whether it is Graph-based Text Retriever or not
    freeze_answer_api:bool = False # Whether use frozen answer API or use local frozen LLM to generate answers (Set true for 3B model, false for 7B model)
    iterative:bool = True # Whether iteratively construct KG triples for each document or not
    tight:bool = True # Whether use tight retrieval or not (tight retrieval means the number of hops / the number of retrieved documents is equal to the number of hops in the question / number of supporting contexts)
    reward_function:str = 'deducible_reward' # available: f1_reward, deducible_reward, recall_reward
    filter_repetition_rollout:bool = True # Whether filter out the excessive repetition in rollout stage
    filter_repetition_threshold:float = 0.9 # Threshold for filtering excessive repetition in rollout stage