import logging
import os
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4
from configparser import ConfigParser

from .base import BaseInteraction

from openai import AsyncOpenAI
from verl.third_party.perplex_rl.llm_api import LLMGenerator
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class QueryRewriteInteraction(BaseInteraction):
    """An interaction class for handling query rewrite prompts.

    - `start_interaction`: Start an interaction instance for a trajectory.
    - `generate_response`: Generate the user response based on the assistant's output.
    - `finalize_interaction`: Finalize the interaction instance.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self._instance_dict = {}
        self.config_path = config.get("config_path", None)
        self.backend = config.get("backend", "vllm_reward")
        if self.config_path:
            config_parser = ConfigParser()
            config_parser.read(self.config_path)
            self.api_url = config_parser[self.backend]['URL']
            self.api_key = config_parser[self.backend]['KEY']
            print(f"Using config from {self.config_path} with backend {self.backend} for reward model")
            self.client = AsyncOpenAI(
                base_url=self.api_url,
                api_key=self.api_key,
                timeout=300
            )
            self.llm_api = LLMGenerator(
                client=self.client,
                model_name=self.config.get('api_model_name')
            )
            # self.tokenizer = AutoTokenizer(self.config.get('api_model_name'),use_fast=True)
            # self.tokenizer.use_chat_template = True
            # answer_tokens = ["<answer>", "</answer>", " <answer>", " </answer>"]
            # answer_start_ids = self.tokenizer("<answer>", add_special_tokens=False).input_ids
            # answer_end_ids = self.tokenizer("</answer>", add_special_tokens=False).input_ids
 

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

        current_turn = kwargs.get("current_turn", 0)
        max_user_turn = kwargs.get("max_user_turn", 5)

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
        if "<plan>" in content and "</plan>" in content:
            self._instance_dict[instance_id]["plan_detected"] = True
            response = "Please write a search query inside <tool_call> and </tool_call> for searching relevant documents to solve a subproblem in the provided plan."
            should_terminate_sequence = False
        # Check if the assistant's response includes <answer>...</answer>
        elif "<answer>" in content and "</answer>" in content:
            ground_truth = self._instance_dict[instance_id]['ground_truth']
            question = self._instance_dict[instance_id]['question']
            targets = ground_truth['target']
            # extract content from answer tokens
            start_idx = content.index("<answer>")
            end_idx = content.index("</answer>")
            answer_content = content[start_idx + len("<answer>"):end_idx].strip()
            best_target, max_intersection = await self.best_target_for_reward(targets, answer_content)
            if max_intersection > 0:
                reward = await self.calculate_reward(instance_id, best_target, question)                    
            self._instance_dict[instance_id]["plan_detected"] = False
            response = "The sequence has been completed as the assistant provided a final answer inside <answer> and </answer>."
            should_terminate_sequence = True
            # only calculate reward when answer is detected which the sequence will be terminate
        # Default case: No <plan> or <answer> detected
        else:
            self._instance_dict[instance_id]["plan_detected"] = False
            response = "Please write search query inside <tool_call> and </tool_call> for searching the relevant documents based on the plan provided in the assistant's response."
            should_terminate_sequence = False

        # No reward calculation is needed for this interaction
        
        # If last turn then prompt to ask for answer
        if current_turn == max_user_turn-1:
            response = "Please provide a final answer to the question inside <answer> and </answer> based on the retrieved documents without detailed illustrations."
            should_terminate_sequence = False
            self._instance_dict[instance_id]["plan_detected"] = False
        return should_terminate_sequence, response, reward, {}

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        """
        Finalize the interaction by cleaning up the instance data.
        """
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]

    async def best_target_for_reward(self, targets, content):
        max_intersection = 0
        best_target = None

        for target_answer in targets:
            intersection_count = len(set(target_answer) & set(content))
            if intersection_count > max_intersection:
                max_intersection = intersection_count
                best_target = target_answer

        return best_target, max_intersection

    async def calculate_reward(self, instance_id:str, ground_truth:str, question:str) -> float:
        """
        Calculate the reward based on the content and ground truth.
        This method can be overridden in subclasses to implement specific reward logic.
        """
        # For simplicity, we return a dummy reward value.
        # In practice, this should be replaced with actual reward calculation logic.
        reward_messages = [
            {
                "role": "system", # 0
                "content": "You are a helpful assistant."
            },
            {
                "role": "user", # 1
                "content": f"{question}"
            },
            {
                "role": "assistant", # API 2
                "content": f"<answer> {ground_truth} </answer>"
            }
        ]
        response = await self.llm_api.generate_response(reward_messages, return_thinking=True, temperature=0, prompt_logprobs=True)
        without_rag_log_probs = response.choices[0].logprobs.token_logprobs
        without_rag_tokens = response.choices[0].logprobs.tokens
        end_idx = None
        start_idx = None
        print(f'without rag tokens: {without_rag_tokens}')

        # for i in range(len(without_rag_tokens) - 3, -1, -1):
        #     if '<' in without_rag_tokens[i] and i + 1 < len(without_rag_tokens) and without_rag_tokens[i+1] == 'answer' and '>' in without_rag_tokens[i+2]:
        #         start_idx = i + 3  # skip past '<', 'answer', '>'
        #     elif '</' in without_rag_tokens[i] and i + 1 < len(without_rag_tokens) and without_rag_tokens[i+1] == 'answer' and '>' in without_rag_tokens[i+2]:
        #         end_idx = i  # exclude 'Ġ</'
        # target_answer_logprobs = without_rag_log_probs[start_idx:end_idx]
        # target_answer_tokens = without_rag_tokens[start_idx:end_idx]
        # print(f'target answer tokens: {target_answer_tokens}')
        # print(f'target prob: {target_answer_logprobs}')

        # rag_tokens = self._instance_dict[instance_id].get("tokens", [])
        # rag_log_probs = self._instance_dict[instance_id].get("log_probs", [])
        # for i in range(len(rag_tokens) - 3, -1, -1):
        #     if '<' in rag_tokens[i] and i + 1 < len(rag_tokens) and rag_tokens[i+1] == 'answer' and '>' in rag_tokens[i+2]:
        #         start_idx = i + 3  # skip past '<', 'answer', '>'
        #     elif '</' in rag_tokens[i] and i + 1 < len(rag_tokens) and rag_tokens[i+1] == 'answer' and '>' in rag_tokens[i+2]:
        #         end_idx = i  # exclude 'Ġ</'
        # rag_answer_logprobs = rag_log_probs[start_idx:end_idx]
        # rag_answer_tokens = rag_tokens[start_idx:end_idx]
        # print(f'rag answer tokens: {rag_answer_tokens}')
        # print(f'rag answer prob {rag_answer_logprobs}')




        