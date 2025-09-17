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

class HippoRAG2Retriever(BaseRetriever):
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

    async def query2edge(self, query, topN = 10):
        def get_query_instruct(sub_query: str) -> str:
            task = "Given a question, retrieve the most relevant knowledge graph triple that can help answer the question."
            return f"Instruct: {task}\nQuery: {sub_query}"
        query_emb = await self.reranker.embed([get_query_instruct(query)])
        query_emb = query_emb[0].cpu().numpy()
        scores = min_max_normalize(self.triple_embeddings@query_emb.T)
        index_matrix = np.argsort(scores)[-topN:][::-1]            

        similarity_matrix = [scores[i] for i in index_matrix]
        # construct the edge list
        node_scores = {}
        before_filter_edge_json = {}
        before_filter_edge_json['fact'] = []
        for idx, (index, sim_score) in enumerate(zip(index_matrix, similarity_matrix)):
            edge = self.edge_list[index]
            edge_str = [edge[0], edge[2], edge[1]]
            before_filter_edge_json['fact'].append(edge_str)
            # Store scores for later use
            node_scores[idx] = sim_score
        # use filtered facts to get the edge id and check if it exists in the original candidate list.
        node_score_dict = {}

        for i, edge in enumerate(before_filter_edge_json['fact']):
            # "China national day" -> is on -> 10-1
            head, relation, tail = edge[0], edge[1], edge[2]
            curr_score = node_scores[i]
            
            if head not in node_score_dict:
                node_score_dict[head] = [curr_score]
            else:
                node_score_dict[head].append(curr_score)
            if tail not in node_score_dict:
                node_score_dict[tail] = [curr_score]
            else:
                node_score_dict[tail].append(curr_score)

        # take average of the scores
        for node in node_score_dict:
            node_score_dict[node] = sum(node_score_dict[node]) / len(node_score_dict[node])
        
        return node_score_dict
    
    async def retrieve_personalization_dict(self, query, topN=30, weight_adjust=0.05):
        node_dict = await self.query2edge(query, topN=topN)
        # text_dict = await self.query2passage(query, weight_adjust=weight_adjust)
  
        # return node_dict, text_dict
        return node_dict

    async def retrieve(self, question, kg: DiGraph, sampling_params: dict, **kwargs):
        topN_edges = self.config.topN_edges
        weight_adjust = self.config.weight_adjust
        self.text_list = []
        self.node_list = []
        self.KG = kg

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
        for node in self.entity_KG.nodes():
            try:
                if node in pr:
                    # Find text nodes connected to this entity node with "source" relation
                    text_neighbors = []
                    for neighbor in self.KG.neighbors(node):
                        # Check if this is a text node connected by "source" relation
                        edge_data = self.KG.get_edge_data(node, neighbor)
                        if (edge_data and edge_data.get('relation') == 'source' and 
                            self.KG.nodes[neighbor].get('node_type') == 'text'):
                            text_neighbors.append(neighbor)
                    
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
        sorted_passages = sorted_passages[:self.config.topN_passages]
        sorted_passages = [passage_value_pairs[0] for passage_value_pairs in sorted_passages]
        
        precision = await self.calculate_precision_reward(sorted_passages, self.supporting_context)
        recall = await self.calculate_recall_reward(sorted_passages, self.supporting_context)

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
            return json.dumps({
                "answer": "none",
                "precision": precision,
                "recall": recall
            })
        return json.dumps({
            "answer": generated_text,
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