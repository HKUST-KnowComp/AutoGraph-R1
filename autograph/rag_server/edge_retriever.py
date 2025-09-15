import asyncio
import json
import re
import json_repair
import numpy as np
import networkx as nx
from networkx import DiGraph
from collections import defaultdict
from autograph.rag_server.llm_api import LLMGenerator
from autograph.rag_server.reranker_api import Reranker
from autograph.rag_server.base_retriever import RetrieverConfig, BaseRetriever
import math
from autograph.rag_server.tog_prompt import REASONING_PROMPT, ANSWER_GENERATION_PROMPT, FEW_SHOT_EXAMPLE
import jellyfish
import logging 

def batch(iterable, n=100):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]
        
class EdgeRetriever(BaseRetriever):
    def __init__(self, config: RetrieverConfig, llm_generator: LLMGenerator, reranker: Reranker):
        self.config = config
        self.llm_generator = llm_generator
        self.reranker = reranker
        self.KG = None
        self.node_embeddings = None
        self.topN_edges = self.config.topN_retrieval_edges
    
    async def index_kg(self, query:str, kg: DiGraph, batch_size:int = 100):
        # Batched triple embeddings
        triples = [f"{src} {rel} {dst}" for src, dst, rel in kg.edges(data="relation")]
        triple_embeddings = []
        for triple_batch in batch(triples, batch_size):
            triple_embeddings.extend(await self.reranker.embed(triple_batch))
        self.triple_embeddings = np.array(triple_embeddings)
        
        assert len(self.triple_embeddings) == len(kg.edges), f"len(triple_embeddings): {len(self.triple_embeddings)}, len(kg.edges): {len(kg.edges)}"
        def get_query_instruct(sub_query: str) -> str:
            task = "Given a question, retrieve the most relevant knowledge graph triple."
            return f"Instruct: {task}\nQuery: {sub_query}"
        
        self.query_instruct = get_query_instruct(query)
        self.query_embedding = await self.reranker.embed([self.query_instruct])[0]
        


    async def retrieve(self, question, kg: DiGraph, sampling_params: dict, **kwargs) -> str:
        """Retrieve a subgraph (or full KG) and generate an answer."""
        self.KG = kg
        self.sampling_params = sampling_params
        await self.index_kg(kg, question)

        # retrieve top N edges
        edge_scores = np.dot(self.triple_embeddings, self.query_embedding)
        top_edge_indices = np.argsort(edge_scores)[-self.topN_edges:][::-1]  # Indices of top N edges
        top_edges = [list(self.KG.edges)[i] for i in top_edge_indices]
        # construct top N edges string
        edge_str_lst = []
        for src, dst in top_edges:
            relation = self.KG.edges[src, dst]['relation']
            edge_str_lst.append(f"({src}-{relation}->{dst})")
        edge_str = "\n".join(edge_str_lst)
        # Generate answer using the subgraph (or full KG)
        answer = await self.generate_answer(question, edge_str)
        return json.dumps({
            "answer": answer
        })

    async def generate_answer(self, query, edge_str: str):
        """Generate an answer using the subgraph (or full KG) with a single LLM call."""
        if not edge_str:
            edge_str = "No relevant edges found."
        prompt = ANSWER_GENERATION_PROMPT
        messages = [
            {"role": "system", "content": prompt},
        ]
        self.sampling_params["temperature"] = self.config.temperature_reasoning
        messages.append({"role": "user", "content": f"{edge_str}\n\n{query}"})
        generated_text = await self.llm_generator.generate_response(messages, **self.sampling_params)
        if "Answer:" in generated_text:
            generated_text = generated_text.split("Answer:")[-1]
        elif "answer:" in generated_text:
            generated_text = generated_text.split("answer:")[-1]
        # if answer is none
        if not generated_text:
            return "none"
        return generated_text

    