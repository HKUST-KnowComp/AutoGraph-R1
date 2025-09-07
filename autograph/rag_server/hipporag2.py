import asyncio
import json_repair
import json
import re
import numpy as np
import networkx as nx
from networkx import DiGraph
from collections import defaultdict
from .llm_api import LLMGenerator
from .reranker_api import Reranker
from .base_retriever import RetrieverConfig, BaseRetriever
from .tog_prompt import ANSWER_GENERATION_PROMPT
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

class HippoRAGRetriever(BaseRetriever):
    def __init__(self, config: RetrieverConfig, llm_generator: LLMGenerator, reranker: Reranker, data: dict):
        self.config = config
        self.llm_generator = llm_generator
        self.reranker = reranker
        self.KG = data.get("KG")
        self.passage_dict = data.get("text_dict", {})
        self.text_id_list = list(self.passage_dict.keys())
        self.node_list = data.get("node_list", [])
        self.edge_list = data.get("edge_list", [])
        self.node_embeddings = None
        self.edge_embeddings = None
        self.text_embeddings = None
        self.num_hop = self.config.num_hop

        self.retrieve_node_fn = self.query2edge

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
        try:
            entities_json = json_repair.loads(response)
        except:
            entities_json = {}
        if "entities" not in entities_json or not isinstance(entities_json["entities"], list):
            return []
        return entities_json["entities"]

    async def filter_triples(self, query, facts_json_str):
        """Filter relevant triples using LLM, simulating filter_triples_with_entity_event."""
        # Assume a prompt for filtering
        prompt = f"Given the query: {query}\nFilter the following facts to keep only those relevant to the query. Output as JSON: {{\"fact\": [[subject, relation, object], ...]}}\nFacts: {facts_json_str}"
        messages = [
            {"role": "system", "content": "You are a helpful assistant that filters knowledge triples based on relevance."},
            {"role": "user", "content": prompt}
        ]
        response = await self.llm_generator.generate_response(messages, **self.sampling_params)
        try:
            filtered_json = json_repair.loads(response)
        except:
            filtered_json = {"fact": []}
        return filtered_json.get("fact", [])

    async def index_kg(self, kg: DiGraph):
        """Compute embeddings for nodes, edges, and texts in the KG."""
        self.node_list = [node for node in kg.nodes if kg.nodes[node].get('type') != 'passage']
        node_texts = [kg.nodes[node].get('id', str(node)) for node in self.node_list]
        self.node_embeddings = await self.reranker.embed(node_texts)
        
        self.edge_list = list(kg.edges)
        edge_texts = [f"{kg.nodes[src].get('id', str(src))} {kg.edges[src, dst].get('relation', '')} {kg.nodes[dst].get('id', str(dst))}" for src, dst in self.edge_list]
        self.edge_embeddings = await self.reranker.embed(edge_texts)
        
        self.text_id_list = [node for node in kg.nodes if kg.nodes[node].get('type') == 'passage']
        text_contents = [self.passage_dict.get(node, '') for node in self.text_id_list]
        self.text_embeddings = await self.reranker.embed(text_contents)
        
        if self.sub_queries:
            sub_query_texts = [f"Given a question with its golden answer, retrieve the most relevant knowledge: {sub_query}" for sub_query in self.sub_queries]
            self.sub_queries_embeddings = await self.reranker.embed(sub_query_texts)

    async def retrieve_topk_nodes(self, query):
        topN = self.config.get('topk_nodes', 10)
        node_score_dict = await self.retrieve_node_fn(query, topN=topN)
        return list(node_score_dict.keys())

    async def query2edge(self, query, topN=10):
        query_emb = await self.reranker.embed([query])
        sim_scores = min_max_normalize(np.dot(self.edge_embeddings, query_emb[0]))
        indices = np.argsort(sim_scores)[-topN:][::-1]
        before_filter_edge_json = {'fact': []}
        for idx in indices:
            src, dst = self.edge_list[idx]
            edge_str = [self.KG.nodes[src].get('id', str(src)), self.KG.edges[src, dst].get('relation', ''), self.KG.nodes[dst].get('id', str(dst))]
            before_filter_edge_json['fact'].append(edge_str)
        filtered_facts = await self.filter_triples(query, json.dumps(before_filter_edge_json))
        topk_nodes = []
        for fact in filtered_facts:
            # Assume fact = [sub, rel, obj]
            for entity in [fact[0], fact[2]]:
                if entity in self.node_list:
                    topk_nodes.append(entity)
                else:
                    # Find similar
                    entity_emb = await self.reranker.embed([entity])
                    sims = np.dot(self.node_embeddings, entity_emb[0])
                    top_idx = np.argmax(sims)
                    topk_nodes.append(self.node_list[top_idx])
        return list(set(topk_nodes))

    async def construct_subgraph(self, query, initial_nodes):
        subgraph = DiGraph()
        visited = set()
        queue = [(node, 0) for node in initial_nodes if node in self.KG.nodes]
        for node, _ in queue:
            subgraph.add_node(node)
            visited.add(node)
        while queue:
            current_node, hop_count = queue.pop(0)
            if hop_count >= self.num_hop:
                continue
            for neighbor in self.KG.successors(current_node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    subgraph.add_node(neighbor)
                    queue.append((neighbor, hop_count + 1))
                relation = self.KG.edges[(current_node, neighbor)].get("relation", '')
                subgraph.add_edge(current_node, neighbor, relation=relation)
            for neighbor in self.KG.predecessors(current_node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    subgraph.add_node(neighbor)
                    queue.append((neighbor, hop_count + 1))
                relation = self.KG.edges[(neighbor, current_node)].get("relation", '')
                subgraph.add_edge(neighbor, current_node, relation=relation)
        return subgraph

    async def retrieve(self, question, kg: DiGraph, sampling_params: dict, **kwargs) -> str:
        self.sub_queries = kwargs.get("sub_queries", [])
        self.KG = kg
        self.sampling_params = sampling_params
        await self.index_kg(kg)

        if self.config.use_full_kg:
            subgraph = kg
        else:
            initial_nodes = await self.retrieve_topk_nodes(question)
            subgraph = await self.construct_subgraph(question, initial_nodes)

        # For Hippo style, compute personalization and PR on subgraph
        node_dict = await self.retrieve_node_fn(question, topN=len(initial_nodes))
        text_dict = {}
        query_emb = await self.reranker.embed([question])
        sim_scores = min_max_normalize(np.dot(self.text_embeddings, query_emb[0])) * 0.05
        text_dict = dict(zip(self.text_id_list, sim_scores))
        personalization_dict = {**node_dict, **text_dict}
        pr = nx.pagerank(subgraph, personalization=personalization_dict, alpha=0.85, max_iter=100, tol=1e-06)

        # Get top passages from PR
        topN = 5
        text_pr = {k: v for k, v in pr.items() if k in self.text_id_list and v > 0}
        sorted_passages = sorted(text_pr.items(), key=lambda x: x[1], reverse=True)[:topN]
        passages_contents = [self.passage_dict.get(pid, '') for pid, _ in sorted_passages]

        answer = await self.generate_answer(question, passages_contents)

        total_edges = len(self.KG.edges)
        subgraph_edges = len(subgraph.edges)
        edge_coverage = (subgraph_edges / total_edges) if total_edges > 0 else 0.0
        semantic_reward = await self.compute_semantic_reward()

        return json.dumps({
            "answer": answer,
            "edge_coverage": edge_coverage,
            "semantic_reward": semantic_reward
        })

    async def generate_answer(self, query, passages_contents):
        passages_string = ". ".join(passages_contents)
        if not passages_string:
            passages_string = "No relevant passages found."
        prompt = ANSWER_GENERATION_PROMPT
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"{passages_string}\n\n{query}"}
        ]
        self.sampling_params["temperature"] = self.config.temperature_reasoning
        generated_text = await self.llm_generator.generate_response(messages, **self.sampling_params)
        if "Answer:" in generated_text:
            generated_text = generated_text.split("Answer:")[-1]
        elif "answer:" in generated_text:
            generated_text = generated_text.split("answer:")[-1]
        if not generated_text:
            return "none"
        return generated_text

    async def compute_semantic_reward(self, fuzzy_threshold=0.85):
        def normalize_text(text: str) -> str:
            text = text.lower().strip()
            text = re.sub(r"[^\w\s]", "", text)
            text = re.sub(r"\s+", " ", text)
            return text
        if not self.sub_queries:
            return 0.0

        rewards = []
        sim_scores = np.dot(self.sub_queries_embeddings, self.text_embeddings.T)
        for sim_score, sub_query in zip(sim_scores, self.sub_queries):
            if "Answer:" not in sub_query:
                continue
            golden_answer = normalize_text(sub_query.split("Answer:")[-1])
            text_index = np.argmax(sim_score)
            top_text = normalize_text(self.passage_dict.get(self.text_id_list[text_index], ""))
            max_score = sim_score[text_index]
            if golden_answer == top_text or golden_answer in top_text:
                reward = 1.0
            else:
                sim = jellyfish.jaro_winkler_similarity(golden_answer, top_text)
                if sim >= fuzzy_threshold:
                    reward = float(max_score)
                else:
                    reward = 0.0
            rewards.append(reward)
        return float(np.mean(rewards)) if rewards else 0.0