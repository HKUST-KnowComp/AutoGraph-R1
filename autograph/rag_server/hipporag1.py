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
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional

import networkx as nx
import json
from tqdm import tqdm
import json
from tqdm import tqdm
from typing import Dict, List, Tuple
import networkx as nx
import numpy as np
import json_repair
from logging import Logger
from dataclasses import dataclass
from typing import Optional

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

class HippoRAGRetriever(BaseRetriever):
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
        self.node_to_text_neighbors = defaultdict(list)
        # get the text neighbors for each entity node
        for u, v, attrs in self.KG.edges(data=True):
            if self.KG.nodes[u].get("node_type")=="entity":
                if self.KG.nodes[v].get("node_type")=="text":
                    self.node_to_text_neighbors[u].append(v)
                    if v not in self.text_list:
                        self.text_list.append(v)
            if self.KG.nodes[v].get("node_type")=="entity":
                if self.KG.nodes[u].get("node_type")=="text":
                    self.node_to_text_neighbors[v].append(u)
                    if u not in self.text_list:
                        self.text_list.append(u)


    async def ner(self, text):
        """Extract topic entities from the query using LLM."""
        messages = [
            {
                "role": "system",
                "content": "Extract the named entities from the provided question and output them as a JSON object in the format: {\"entities\": [\"entity1\", \"entity2\", ...]}"
            },
            {
                "role": "user",
                "content": f"Extract all the named entities from: {text}"
            }
        ]
        response = await self.llm_generator.generate_response(messages, **self.sampling_params)
        entities_json = json_repair.loads(response)
        if "entities" not in entities_json or not isinstance(entities_json["entities"], list):
            return {}
        return entities_json
    async def retrieve_topk_nodes(self, query):
        """Retrieve top-k nodes relevant to the query, with fallback to similar nodes."""
        entities_json = await self.ner(query)
        entities = entities_json.get("entities", [])
        if not entities:
            entities = [query]
        topk_nodes = []
        entities_not_in_kg = []
        entities = [str(e) for e in entities]
        entities = list(set(entities))  # deduplicate
        for entity in entities:
            if entity in self.KG.nodes:
                topk_nodes.append(entity)
            else:
                entities_not_in_kg.append(entity)
        if entities_not_in_kg:
            kg_nodes = list(self.entity_KG.nodes)
            sim_scores = await self.reranker.compute_similarity(entities_not_in_kg, kg_nodes)
            indices = np.argsort(sim_scores, axis=1)[:, -1:]  # Get the last k indices after sorting
    
            for i in range(indices.shape[0]):
                for j in indices[i]:
                    top_node = kg_nodes[j]
                    topk_nodes.append(top_node)
        topk_nodes = list(set(topk_nodes))
        return topk_nodes
    async def query2node(self, query, topN = 10):
        topk_nodes = await self.retrieve_topk_nodes(query)
        
        freq_dict_for_nodes = {}
        for node in topk_nodes:
            if node in self.node_to_text_neighbors:
                freq_dict_for_nodes[node] = len(self.node_to_text_neighbors[node])
            else:
                freq_dict_for_nodes[node] = 1
            
        personalization_dict = {node: 1 / freq_dict_for_nodes[node]  for node in topk_nodes}
        
        return personalization_dict
    
    async def retrieve_personalization_dict(self, query, topN=30, weight_adjust=0.05):
        node_dict = await self.query2node(query, topN=topN)
        # text_dict = await self.query2passage(query, weight_adjust=weight_adjust)
  
        # return node_dict, text_dict
        return node_dict

    async def retrieve(self, question, kg: DiGraph, sampling_params: dict, **kwargs):
        topN_edges = self.config.topN_edges
        weight_adjust = self.config.weight_adjust
        self.text_list = []
        self.node_list = []
        self.KG = kg
        top_n_passages = kwargs.get("top_n_passages", self.config.topN_passages)

        for node, attrs in kg.nodes(data=True):
            if attrs.get("node_type") == "text":
                self.text_list.append(node)
            else:
                self.node_list.append(node)

        self.entity_KG = kg.subgraph(self.node_list)
        self.edge_list = list(self.entity_KG.edges(data="relation"))
        self.sampling_params = sampling_params

        self.title_triple_dict = kwargs.get("title_triple_dict", {})
        self.supporting_context = kwargs.get("supporting_context", [])
        self.full_context = kwargs.get("full_context", [])

        await self.index_kg()

        node_dict = await self.retrieve_personalization_dict(question, topN=topN_edges, weight_adjust=weight_adjust)

        personalization_dict = {}

        personalization_dict.update(node_dict)
        # personalization_dict.update(text_dict)
        # retrieve the top N passages

        pr = nx.pagerank(self.entity_KG, personalization=personalization_dict)
        # print("Personalization dict after pagerank:", pr)
        # get the top N passages based on the text_id list and pagerank score
        text_dict_score = {}
        for node in pr:
            try:
                if node in list(self.entity_KG.nodes) and self.KG.nodes[node].get("node_type") == "entity":
                    # Find text nodes connected to this entity node with "source" relation
                    text_neighbors = self.node_to_text_neighbors.get(node, [])
                    
                    # If we found text neighbors, add them to the text_dict_score
                    for text_node in text_neighbors:
                        if text_node not in text_dict_score:
                            text_dict_score[text_node] = pr[node]
                        else:
                            # If a text node is connected to multiple entities, take the max score
                            text_dict_score[text_node] += pr[node]
            except Exception as e:
                print(f"Error processing node {node}: {str(e)}")

        # return topN passages
        sorted_passages = sorted(text_dict_score.items(), key=lambda x: x[1], reverse=True)
        sorted_passages = sorted_passages[:top_n_passages]
        sorted_passages = [passage_value_pairs[0] for passage_value_pairs in sorted_passages]
        
        precision = await self.calculate_precision_reward(sorted_passages, self.supporting_context)
        recall = await self.calculate_recall_reward(sorted_passages, self.supporting_context)

        # # generate answer
        # context = "\n".join(sorted_passages)
        # prompt = ANSWER_GENERATION_PROMPT
        # messages = [
        #     {"role": "system", "content": prompt},
        # ]
        # self.sampling_params["temperature"] = self.config.temperature_reasoning
        # messages.append({"role": "user", "content": f"{context}\n\n{question}"})

        # generated_text = await self.llm_generator.generate_response(messages, **self.sampling_params)
        # if "Answer:" in generated_text:
        #     generated_text = generated_text.split("Answer:")[-1]
        # elif "answer:" in generated_text:
        #     generated_text = generated_text.split("answer:")[-1]
        # if answer is none
        # if not generated_text:
        #     return json.dumps({
        #         "answer": "none",
        #         "precision": precision,
        #         "recall": recall
        #     })
        return json.dumps({
            "answer": "recall only",
            "precision": precision,
            "recall": recall
        })
            
    async def calculate_precision_reward(self, retrieved_passages: List[str], ground_truth: List[str]):
        # calculate precision reward
        retrieved_set = set(retrieved_passages)
        if not ground_truth:
            self.precision_reward = 1.0
            return 1.0
        ground_truth_set = set(ground_truth)
        # instead of using intersection, since the retrieve test is in "Doc i: content" and ground truth is only "content", we use substring match
        match_count = 0
        for retrieved in retrieved_set:
            for gt in ground_truth_set:
                if gt in retrieved:
                    match_count += 1
                    break
        self.precision_reward = match_count / len(retrieved_set) if len(retrieved_set) > 0 else 0.0
        return self.precision_reward

    async def calculate_recall_reward(self, retrieved_passages: List[str], ground_truths: List[str]):
        # calculate recall reward
        retrieved_set = set(retrieved_passages)
        if not ground_truths:
            self.recall_reward = 1.0
            return 1.0
        ground_truth_set = set(ground_truths)
        # instead of using intersection, since the retrieve test is in "Doc i: content" and ground truth is only "content", we use substring match
        match_count = 0
        for gt in ground_truth_set:
            for retrieved in retrieved_set:
                if gt in retrieved:
                    match_count += 1
                    break
        self.recall_reward = match_count / len(ground_truth_set) if len(ground_truth_set) > 0 else 0.0
        return self.recall_reward