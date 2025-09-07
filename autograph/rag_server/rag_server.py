import argparse

from typing import List, Tuple

import networkx as nx
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from vllm.utils import random_uuid
import torch
import vllm
from vllm import LLM
import time
from tog import TogRetriever
from subgraph_retriever import SubgraphRetriever
from .reranker_api import Reranker
from .llm_api import LLMGenerator
import json_repair
from .base_retriever import RetrieverConfig, BaseRetriever, DummyRetriever
from openai import AsyncOpenAI, OpenAI
import configparser
import asyncio

# --- FastAPI and vLLM Setup ---
class KGQARequest(BaseModel):
    question: str
    triples_string: str = Field(
        ..., 
        description="A string containing knowledge triples, one per line, formatted as 'Subject | Relation | Object'.",
        examples=["NVIDIA | is headquartered in | Santa Clara"]
    )
    sampling_params: dict = Field(default_factory=lambda: {"temperature": 0.0, "max_tokens": 512})
    sub_queries: List[str] = Field(default_factory=list, description="List of sub-queries with answers.")
    

def build_openai_response(answer: str, model: str) -> dict:
    """Builds an OpenAI-compatible chat completion response."""
    return {
        "id": f"chatcmpl-{random_uuid()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": answer
                },
                "finish_reason": "stop"
            }
        ]
    }

def setup_retriever(retriever_config: RetrieverConfig)-> BaseRetriever:
    # print(f"Setting up retriever: {retriever_config.name}")
    if retriever_config.name == "tog":
        return TogRetriever(retriever_config, llm_api, reranker)
    if retriever_config.name == "dummy":
        return DummyRetriever(retriever_config, llm_api, reranker)
    if retriever_config.name == "subgraph":
        return SubgraphRetriever(retriever_config, llm_api, reranker)

def parse_triples(triples_string: str) -> nx.DiGraph:
    """Parses a string of triples into a directed graph (DiGraph).

    Args:
        triples_string (str): A JSON string containing a list of triples, where each triple is a dict
                              with 'subject', 'relation', and 'object' keys.

    Returns:
        nx.DiGraph: A directed graph representing the triples.
    """
    try:
        # Parse the JSON string into a Python object
        triples_json = json_repair.loads(triples_string)

        # Validate that the JSON is a list of dictionaries with the required keys
        if not isinstance(triples_json, list):
            raise ValueError("The triples_string must be a JSON array of triples.")
        
        for triple in triples_json:
            if not isinstance(triple, dict) or not all(key in triple for key in ['subject', 'relation', 'object']) or any(str(triple[key]).strip() == "" for key in ['subject', 'relation', 'object']):
                raise ValueError(f"Each triple must be a dictionary with 'subject', 'relation', and 'object' keys. Problematic triple: {triple}")

        # Create a directed graph and add edges for each triple
        graph = nx.DiGraph()
        for triple in triples_json:
            subject = str(triple['subject'])
            relation = str(triple['relation'])
            obj = str(triple['object'])
            graph.add_edge(subject, obj, relation=relation)

        return graph

    except Exception as e:
        raise ValueError(f"Failed to parse triples_string: {e}")

def parse_triples_text_kg(triples_string: str) -> nx.DiGraph:
    """Parses a string of triples into a directed graph (DiGraph).

    Args:
        triples_string (str): A string containing knowledge triples, one per line, formatted as 'Subject | Relation | Object'.

    Returns:
        nx.DiGraph: A directed graph representing the triples.
    """
    try:
        # Parse the JSON string into a Python object
        triples_json = json_repair.loads(triples_string)

        # Validate that the JSON is a list of dictionaries with the required keys
        if not isinstance(triples_json, list):
            raise ValueError("The triples_string must be a JSON array of triples.")
        
        for triple in triples_json:
            if not isinstance(triple, dict) or not all(key in triple for key in ['subject', 'relation', 'object']) or any(str(triple[key]).strip() == "" for key in ['subject', 'relation', 'object']):
                raise ValueError(f"Each triple must be a dictionary with 'subject', 'relation', and 'object' keys. Problematic triple: {triple}")

        # Create a directed graph and add edges for each triple
        graph = nx.DiGraph()
        for triple in triples_json:
            subject = str(triple['subject'])
            relation = str(triple['relation'])
            obj = str(triple['object'])
            graph.add_edge(subject, obj, relation=relation)

        return graph

    except Exception as e:
        raise ValueError(f"Failed to parse triples_string: {e}")

app = FastAPI(
    title="Knowledge Graph RAG Server",
    description="An API server that performs ToG-2 RAG on a dynamically built Knowledge Graph using vLLM.",
)

request_semaphore = asyncio.Semaphore(5)
@app.post("/v1/chat/completions")
async def generate_from_kg(request: KGQARequest):
    """OpenAI-compatible endpoint implementing ToG retrieval."""
    async with request_semaphore:
        model = "vllm-tog-rag"
        if json_repair.loads(request.triples_string) == []:
            return build_openai_response("Error parsing triples", model)
        try:
            if config.name in ["hipporag", "hipporag2"]:
                kg = parse_triples_text_kg(request.triples_string)
            else:
                kg = parse_triples(request.triples_string)
        except Exception as e:
            return build_openai_response(f"Error parsing triples: {e}", model)
        retriever = setup_retriever(config)
        answer = await retriever.retrieve(
            question=request.question,
            kg=kg,
            sampling_params=request.sampling_params,
            sub_queries=request.sub_queries
        )
        return build_openai_response(answer, model)

# --- 3. Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch the KG-RAG vLLM server.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server to.")
    parser.add_argument("--port", type=int, default=8130, help="Port to run the server on.")
    parser.add_argument("--model_dir", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Path to vLLM model directory.")
    parser.add_argument("--llm_tensor_parallel_size", type=int, default=2, help="Number of GPUs to use for tensor parallelism.")
    parser.add_argument("--llm_gpu_memory_utilization", type=float, default=0.8, help="GPU memory utilization for vLLM.")
    parser.add_argument("--reranker_gpu-memory_utilization", type=float, default=0.1, help="GPU memory utilization for vLLM.")
    parser.add_argument("--reranker_model_name", type=str, default="Qwen/Qwen3-Embedding-0.6B", help="Model name for the reranker.")
    parser.add_argument("--retriever_name", type=str, default="subgraph", choices=["tog", "tog2", "subgraph", "dummy"], help="Name of the retriever to use.")
    parser.add_argument("--reranker_tensor_parallel_size", type=int, default=2, help="Tensor parallel size for the reranker.")

    args = parser.parse_args()

    config_parser = configparser.ConfigParser()
    config_parser.read("autograph-r1/rag_server/config.ini")
    config = config_parser['vllm']
    api_url = config['URL']
    api_key = config['KEY']
    # Initialize the vLLM engine
    llm_client = AsyncOpenAI(
        base_url=api_url,
        api_key=api_key,
        timeout=300,
    )
    llm_api = LLMGenerator(
        client=llm_client,
        model_name=args.model_dir,
        backend='vllm'
    )
    # Initialize the reranker
    config = config_parser['vllm_emb']
    api_url = config['URL']
    api_key = config['KEY']
    emb_client = AsyncOpenAI(
        base_url=api_url,
        api_key=api_key,
        timeout=300,
    )
    reranker = Reranker(
        emb_client
    )

    config = RetrieverConfig(
        name = args.retriever_name,
    )


    # Launch the FastAPI server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
    )