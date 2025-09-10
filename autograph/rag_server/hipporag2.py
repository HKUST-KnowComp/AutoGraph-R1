import asyncio
import json_repair
import json
import re
import numpy as np
import networkx as nx
from networkx import DiGraph
from collections import defaultdict
from autograph.rag_server.llm_api import LLMGenerator
from autograph.rag_server.reranker_api import Reranker
from autograph.rag_server.base_retriever import RetrieverConfig, BaseRetriever
from autograph.rag_server.tog_prompt import ANSWER_GENERATION_PROMPT
import math
import jellyfish
import logging
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional

def min_max_normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    range_val = max_val - min_val
    if range_val == 0:
        return np.ones_like(x)
    return (x - min_val) / range_val
import networkx as nx
import json
from tqdm import tqdm
import json
from tqdm import tqdm
from typing import Dict, List, Tuple
import networkx as nx
import numpy as np
import json_repair
from atlas_rag.vectorstore.embedding_model import BaseEmbeddingModel
from atlas_rag.llm_generator.llm_generator import LLMGenerator
from logging import Logger
from dataclasses import dataclass
from typing import Optional
from atlas_rag.retriever.base import BasePassageRetriever
from atlas_rag.retriever.inference_config import InferenceConfig

# import hashlib
import hashlib

def min_max_normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    range_val = max_val - min_val
    
    # Handle the case where all values are the same (range is zero)
    if range_val == 0:
        return np.ones_like(x)  # Return an array of ones with the same shape as x
    
    return (x - min_val) / range_val
def batch(iterable, n=100):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

class HippoRAG2Retriever(BasePassageRetriever):
    def __init__(self, config: RetrieverConfig, llm_generator: LLMGenerator, reranker: Reranker):
        self.config = config
        self.llm_generator = llm_generator
        self.reranker = reranker

    async def index_kg(self, batch_size:int = 100):
        nodes = self.node_list
        node_embeddings = []
        for node_batch in batch(nodes, batch_size):
            node_embeddings.extend(await self.reranker.embed(node_batch))
        self.node_embeddings = np.array(node_embeddings)

        # Batched triple embeddings
        triples = [f"{src} {rel} {dst}" for src, dst, rel in self.entity_KG.edges(data="relation")]
        triple_embeddings = []
        for triple_batch in batch(triples, batch_size):
            triple_embeddings.extend(await self.reranker.embed(triple_batch))
        self.triple_embeddings = np.array(triple_embeddings)
        
        assert len(self.triple_embeddings) == len(self.entity_KG.edges), f"len(triple_embeddings): {len(self.triple_embeddings)}, len(kg.edges): {len(self.entity_KG.edges)}"

        # index text
        text_embeddings = []
        for text in self.full_context:
            text_embeddings.extend(await self.reranker.embed([text]))
        self.text_embeddings = np.array(text_embeddings)



    async def filter_edges(self, query, facts_triples_json: Dict):
        # facts_triples_json is in the format of {'fact': [[head, relation, tail], ...]}
        # convert to string
        filter_messages = [
            {"role": "system", "content": "Filter out the irrelevant triples and return the relevant ones that can help answer the query only."},
            {"role": "user", "content": f"Query: {query}\nFacts: {facts_triples_json}\nRelevant facts in JSON format:\n {{\"fact\": [[subject, relation, object], ...]}}. If no relevant facts, return {{\"fact\": []}}."},
        ]
        response = await self.llm_generator.generate_response(
            filter_messages,
            **self.sampling_params
        )
        try:
            response_json = json_repair.loads(response)
            # check if all fact are triples
            if "fact" in response_json and isinstance(response_json["fact"], list) and all(isinstance(item, list) and len(item) == 3 for item in response_json["fact"]):
                return response_json["fact"]
            else:
                return facts_triples_json['fact']
        except Exception as e:
            return facts_triples_json['fact']

    async def query2edge(self, query, topN = 10):
        def get_query_instruct(sub_query: str) -> str:
            task = "Given a question with its golden answer, retrieve the most relevant knowledge graph triple."
            return f"Instruct: {task}\nQuery: {sub_query}"
        query_emb = await self.reranker.embed([get_query_instruct(query)])
        scores = min_max_normalize(self.triple_embeddings@query_emb[0].T)
        index_matrix = np.argsort(scores)[-topN:][::-1]

        for index in index_matrix:
            edge = self.edge_list[index]
            edge_str = [edge[0], self.entity_KG.edges[edge]['relation'], edge[1]]
            

        similarity_matrix = [scores[i] for i in index_matrix]
        # construct the edge list
        before_filter_edge_json = {}
        before_filter_edge_json['fact'] = []
        for index, sim_score in zip(index_matrix, similarity_matrix):
            edge = self.edge_list[index]
            edge_str = [edge[0], self.KG.edges[edge]['relation'], self.KG.nodes[edge[1]]]
            before_filter_edge_json['fact'].append(edge_str)

        filtered_facts = await self.filter_edges(query, json.dumps(before_filter_edge_json, ensure_ascii=False))
        if len(filtered_facts) == 0:
            return {}
        # use filtered facts to get the edge id and check if it exists in the original candidate list.
        node_score_dict = {}

        for edge in filtered_facts:
            edge_str = f'{edge[0]} {edge[1]} {edge[2]}'
            search_emb = self.reranker.encode([edge_str], query_type="search")
            scores = self.triple_embeddings @ search_emb[0].T
            # get the edge and the original score
            triple_index = np.argmax(scores)
            edge = self.edge_list[triple_index]
            head, tail = edge[0], edge[1]
            # check if head/tails is concept, sim_score = sim_score / # edge of that node
            sim_score = scores[triple_index]
            
            if head not in node_score_dict:
                node_score_dict[head] = [sim_score]
            else:
                node_score_dict[head].append(sim_score)
            if tail not in node_score_dict:
                node_score_dict[tail] = [sim_score]
            else:
                node_score_dict[tail].append(sim_score)
        
        # take average of the scores
        for node in node_score_dict:
            node_score_dict[node] = sum(node_score_dict[node]) / len(node_score_dict[node])
        
        return node_score_dict

    async def query2passage(self, query, weight_adjust = 0.05):
        def get_query_instruct(sub_query: str) -> str:
            task = "Given a question with its golden answer, retrieve the most relevant passage."
            return f"Instruct: {task}\nQuery: {sub_query}"
        query_emb = await self.reranker.embed([get_query_instruct(query)])
        sim_scores = self.text_embeddings @ query_emb[0].T
        sim_scores = min_max_normalize(sim_scores)*weight_adjust # converted to probability
        # create dict of passage id and score
        return dict(zip(self.text_id_list, sim_scores))
    
    async def retrieve_personalization_dict(self, query, topN=30, weight_adjust=0.05):
        node_dict = await self.query2edge(query, topN=topN)
        text_dict = await self.query2passage(query, weight_adjust=weight_adjust)
  
        return node_dict, text_dict

    async def retrieve(self, question, kg: DiGraph, sampling_params: dict, **kwargs):
        topN_edges = self.config.topN_edges
        weight_adjust = self.config.weight_adjust
        self.text_list = []
        self.node_list = []
        self.KG = kg
        for node in kg.nodes:
            if node.get('node_type') == 'text':
                self.text_list.append(node)
            else:
                self.node_list.append(node)
        self.entity_KG = kg.subgraph(self.node_list)
        self.edge_list = list(self.entity_KG.edges)
        self.sampling_params = sampling_params

        self.title_triple_dict = kwargs.get("title_triple_dict", {})
        self.supporting_context = kwargs.get("supporting_context", "")
        self.full_context = kwargs.get("full_context", "")

        self.index_kg()
        
        node_dict, text_dict = await self.retrieve_personalization_dict(question, topN=topN_edges, weight_adjust=weight_adjust)
          
        personalization_dict = {}
        if len(node_dict) == 0:
            # return topN text passages
            sorted_passages = sorted(text_dict.items(), key=lambda x: x[1], reverse=True)
            sorted_passages = sorted_passages[:self.config.topN_passages]
            sorted_passages = [passage_value_pairs[0] for passage_value_pairs in sorted_passages]
        else:
            personalization_dict.update(node_dict)
            personalization_dict.update(text_dict)
            # retrieve the top N passages
            pr = nx.pagerank(self.KG, personalization=personalization_dict)

            # get the top N passages based on the text_id list and pagerank score
            text_dict_score = {}
            for node in self.text_id_list:
                # filter out nodes that have 0 score
                if pr[node] > 0.0:
                    text_dict_score[node] = pr[node]
                
            # return topN passages
            sorted_passages = sorted(text_dict_score.items(), key=lambda x: x[1], reverse=True)
            sorted_passages = sorted_passages[:self.config.topN_passages]
            sorted_passages = [passage_value_pairs[0] for passage_value_pairs in sorted_passages]
        
        precision = self.calculate_precision_reward(sorted_passages, self.supporting_context)

        # generate answer
        context = "\n".join(sorted_passages)
        prompt = ANSWER_GENERATION_PROMPT
        messages = [
            {"role": "system", "content": prompt},
        ]
        self.sampling_params["temperature"] = self.config.temperature_reasoning
        messages.append({"role": "user", "content": f"{context}\n\n{question}"})
        generated_text = await self.llm_generator.generate_response(messages, **self.sampling_params)
        if "Answer:" in generated_text:
            generated_text = generated_text.split("Answer:")[-1]
        elif "answer:" in generated_text:
            generated_text = generated_text.split("answer:")[-1]
        # if answer is none
        if not generated_text:
            return "none"
        return json.dumps({
            "answer": generated_text,
            "precision": precision,
        })
            
    def calculate_precision_reward(self, retrieved_passages: List[str], ground_truth: str):
        # calculate precision reward
        retrieved_set = set(retrieved_passages)
        if not ground_truth:
            self.precision_reward = 1.0
            return
        ground_truth_set = set([ground_truth])
        intersection = retrieved_set.intersection(ground_truth_set)
        self.precision_reward = len(intersection) / len(retrieved_set) if len(retrieved_set) > 0 else 0.0
        return self.precision_reward