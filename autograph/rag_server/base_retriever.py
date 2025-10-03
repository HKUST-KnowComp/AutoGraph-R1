from abc import ABC, abstractmethod
from llm_api import LLMGenerator
from reranker_api import Reranker
# --- Retriever Configuration ---
class RetrieverConfig:
    def __init__(
        self,
        name: str,
        max_length: int = 256,
        temperature_exploration: float = 0.0,
        temperature_reasoning: float = 0.0,
        width: int = 3,
        depth: int = 2,
        remove_unnecessary_rel: bool = True,
        relation_prune_combination: bool = True,
        num_sents_for_reasoning: int = 9,
        topic_prune: bool = True,
        use_full_kg:bool = False,
        num_hop:int = 4
    ):
        self.name = name
        self.max_length = max_length
        self.temperature_exploration = temperature_exploration
        self.temperature_reasoning = temperature_reasoning
        self.width = width
        self.depth = depth
        self.remove_unnecessary_rel = remove_unnecessary_rel # whether to perform relation pruning during retrieval
        self.relation_prune_combination = relation_prune_combination
        self.num_sents_for_reasoning = num_sents_for_reasoning # number of paths to consider for reasoning
        self.topic_prune = topic_prune
        self.use_full_kg = use_full_kg
        self.num_hop = num_hop # number of hops to consider for subgraph retrieval

# --- Base Retriever ---
class BaseRetriever(ABC):
    def __init__(self, config: RetrieverConfig, llm_api: LLMGenerator = None, reranker: Reranker = None):
        """
        Initializes the BaseRetriever.

        Args:
            config (RetrieverConfig): Configuration for the retriever.
            llm_api (LLMGenerator): Optional LLM engine instance.
            reranker (Reranker): Optional reranker instance.
        """
        self.config = config
        self.llm_api = llm_api
        self.reranker = reranker

    @abstractmethod
    async def retrieve(self, question: str, kg, sampling_params: dict, **kwargs) -> str:
        """
        Abstract method for retrieving answers.

        Args:
            question (str): The input question.
            kg: The knowledge graph.
            sampling_params (dict): Sampling parameters.

        Returns:
            str: The retrieved answer.
        """
        pass

class DummyRetriever(BaseRetriever):
    async def retrieve(self, question: str, kg, sampling_params: dict) -> str:
        """
        Mock retrieval method that returns a static response.

        Args:
            question (str): The input question.
            kg: The knowledge graph (not used in the dummy retriever).
            sampling_params (dict): Sampling parameters (not used in the dummy retriever).

        Returns:
            str: A static response for testing purposes.
        """
        return f"Mock response for question: '{question}'"