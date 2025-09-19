import logging
import os
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4
from configparser import ConfigParser

from .base import BaseInteraction

from openai import AsyncOpenAI
from verl.third_party.autograph_r1.llm_api import LLMGenerator
from transformers import AutoTokenizer
import json_repair

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class RAGInteraction(BaseInteraction):
    """An interaction class for handling query rewrite prompts.

    - `start_interaction`: Start an interaction instance for a trajectory.
    - `generate_response`: Generate the user response based on the assistant's output.
    - `finalize_interaction`: Finalize the interaction instance.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self._instance_dict = {}

    async def start_interaction(
        self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, question: Optional[str] = None, **kwargs
    ) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "response": "",
            "plan_detected": False,
            'ground_truth': ground_truth,
            "question": question,
        }
        return instance_id

    async def generate_response(
        self, instance_id: str, messages: List[Dict[str, Any]], **kwargs
    ) -> Tuple[bool, str, float, dict]:
        """
        Generate the user response based on the assistant's output.

        If the assistant's response includes <plan>...</plan>, ask the assistant to rewrite the query.
        Otherwise, ask the assistant to generate a response based on the current retrieved context.
        """

        should_terminate_sequence = False
        iterative = kwargs.get("iterative", True)
        content = ""
        for i in range(len(messages) - 1, -1, -1):
            item = messages[i]
            if item.get("role") == "assistant":
                content = item.get("content")
                break
        # the hierarchy is 
        # plan > answer > no plan or answer
        reward = 0.0
        # Check if the assistant's response includes <plan>...</plan>
        # give format reward
        self._instance_dict[instance_id]["rag_state"] = False
        try:
            result_json = json_repair.loads(content)
        except Exception as e:
            logger.warning(f"Failed to parse assistant content as JSON: {e}\nContent: {content}")
            should_terminate_sequence = True
            response = "The response is not in the correct format."
            reward = 0.0
            return should_terminate_sequence, response, reward, {}
        if isinstance(result_json, dict) and "answer" in result_json:
            answer = result_json.get("answer", "")
            should_terminate_sequence = True
            response = "<answer>" + answer + "</answer>"
        
        elif isinstance(result_json, list) and len(result_json) > 0:
            triples_keys = ["subject", "relation", "object"]
            # check if all key are present in all items
            if all(isinstance(item, dict) and all(key in item for key in triples_keys) for item in result_json):
                if iterative:
                    should_terminate_sequence = False
                    remaining_context = kwargs["remaining_context"]
                    if len(remaining_context) > 0:
                        document = remaining_context[0]
                        response = f"Extracts for {document}"
                    else:
                        assert len(kwargs['processed_docs']) == len(kwargs['full_context'])
                        response = "You will perform graph based RAG based on your constructed knowledge graph."
                        self._instance_dict[instance_id]["rag_state"] = True
                    reward = 1.0
                else:
                    should_terminate_sequence = False
                    response = "You will perform graph based RAG based on your constructed knowledge graph."
                    reward = 1.0
                    self._instance_dict[instance_id]["rag_state"] = True
            else:
                should_terminate_sequence = True
                response = "The response is not in the correct format."
        else:
            should_terminate_sequence = True
            response = "The response is not in the correct format."
            reward = 0.0
            
        return should_terminate_sequence, response, reward, {}

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        """
        Finalize the interaction by cleaning up the instance data.
        """
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]

        