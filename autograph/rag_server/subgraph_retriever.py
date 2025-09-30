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
from autograph.rag_server.tog_prompt import REASONING_PROMPT, ANSWER_GENERATION_PROMPT, FEW_SHOT_EXAMPLE, VERIFY_ANSWER_PROMPT
import jellyfish
import logging 

def batch(iterable, n=100):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]
class SubgraphRetriever(BaseRetriever):
    def __init__(self, config: RetrieverConfig, llm_generator: LLMGenerator, reranker: Reranker):
        self.config = config
        self.llm_generator = llm_generator
        self.reranker = reranker
        self.KG = None
        self.node_embeddings = None
        self.num_hop = self.config.num_hop

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
        except Exception as e:
            return {}
        if "entities" not in entities_json or not isinstance(entities_json["entities"], list):
            return {}
        return entities_json
    
    async def index_kg(self, kg: DiGraph, batch_size:int = 100):
        nodes = list(kg.nodes)
        node_embeddings = []
        for node_batch in batch(nodes, batch_size):
            node_embeddings.extend(await self.reranker.embed(node_batch))
        self.node_embeddings = np.array(node_embeddings)

        # Batched triple embeddings
        triples = [f"{src} {rel} {dst}" for src, dst, rel in kg.edges(data="relation")]
        triple_embeddings = []
        for triple_batch in batch(triples, batch_size):
            triple_embeddings.extend(await self.reranker.embed(triple_batch))
        self.triple_embeddings = np.array(triple_embeddings)
        
        assert len(self.triple_embeddings) == len(kg.edges), f"len(triple_embeddings): {len(self.triple_embeddings)}, len(kg.edges): {len(kg.edges)}"
        def get_subquery_instruct(sub_query: str) -> str:
            task = "Given a question with its golden answer, retrieve the most relevant knowledge graph triple."
            return f"Instruct: {task}\nQuery: {sub_query}"
        # sub_query_texts = []
        # for sub_query in self.sub_queries:
        #     sub_query_text = get_subquery_instruct(sub_query)
        #     sub_query_texts.append(sub_query_text)
        # self.sub_queries_embeddings = await self.reranker.embed(sub_query_texts)
        # assert len(self.sub_queries_embeddings) == len(self.sub_queries), f"len(self.sub_queries_embeddings): {len(self.sub_queries_embeddings)}, len(self.sub_queries): {len(self.sub_queries)}"
        

    async def retrieve_topk_nodes(self, query):
        """Retrieve top-k nodes relevant to the query, with fallback to similar nodes."""
        is_query_only = False
        entities_json = await self.ner(query)
        entities = entities_json.get("entities", [])
        if not entities:
            entities = [query]
            is_query_only = True
        topk_nodes = []
        entities_not_in_kg = []
        if is_query_only:
            def search_entity_in_kg_instruct(entity: str) -> str:
                task = "Given a question, retrieve the most relevant knowledge graph nodes."
                return f"Instruct: {task}\nQuestion: {entity}"
            entity_texts = [search_entity_in_kg_instruct(e) for e in entities]
            entity_embeddings = await self.reranker.embed(entity_texts)
            sim_scores = entity_embeddings @ self.node_embeddings.T
            indices = np.argsort(sim_scores, axis=1)[:, -self.num_hop:]  # Get the last k indices after sorting
            entities = []
            for i in range(indices.shape[0]):
                for j in indices[i]:
                    top_node = list(self.KG.nodes)[j]
                    entities.append(top_node)
            entities = list(set(entities))  # deduplicate
        else:
            entities = [str(e) for e in entities]
            entities = list(set(entities))  # deduplicate
        for entity in entities:
            if entity in self.KG.nodes:
                topk_nodes.append(entity)
            else:
                entities_not_in_kg.append(entity)
        if entities_not_in_kg:
            kg_nodes = list(self.KG.nodes)
            sim_scores = await self.reranker.compute_similarity(entities_not_in_kg, kg_nodes)
            indices = np.argsort(sim_scores, axis=1)[:, -1:]  # Get the last k indices after sorting
    
            for i in range(indices.shape[0]):
                for j in indices[i]:
                    top_node = kg_nodes[j]
                    topk_nodes.append(top_node)
        topk_nodes = list(set(topk_nodes))
        return topk_nodes

    async def construct_subgraph(self, query, initial_nodes):
        """Construct a multi-hop subgraph around initial nodes up to self.num_hop."""
        subgraph = DiGraph()
        visited = set()  # Track visited nodes to avoid cycles
        queue = [(node, 0) for node in initial_nodes if node in self.KG.nodes]  # (node, hop_count)
        
        # Add initial nodes
        for node, _ in queue:
            subgraph.add_node(node)
            visited.add(node)

        # Breadth-first search to collect neighbors up to num_hop
        while queue:
            current_node, hop_count = queue.pop(0)
            if hop_count >= self.num_hop:
                continue

            # Add successors (outgoing edges)
            for neighbor in self.KG.successors(current_node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    subgraph.add_node(neighbor)
                    queue.append((neighbor, hop_count + 1))
                relation = self.KG.edges[(current_node, neighbor)]["relation"]
                subgraph.add_edge(current_node, neighbor, relation=relation)

            # Add predecessors (incoming edges)
            for neighbor in self.KG.predecessors(current_node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    subgraph.add_node(neighbor)
                    queue.append((neighbor, hop_count + 1))
                relation = self.KG.edges[(neighbor, current_node)]["relation"]
                subgraph.add_edge(neighbor, current_node, relation=relation)

        return subgraph

    async def retrieve(self, question, kg: DiGraph, sampling_params: dict, **kwargs) -> str:
        """Retrieve a subgraph (or full KG) and generate an answer."""
        self.sub_queries = kwargs.get("sub_queries", [])
        self.answer = kwargs.get("answer", "unknown")
        self.KG = kg
        self.sampling_params = sampling_params
        await self.index_kg(kg)

        if self.config.use_full_kg:
            # Use the entire KG
            subgraph = kg
        else:
            # Construct a 1-hop subgraph
            initial_nodes = await self.retrieve_topk_nodes(question)
            subgraph = await self.construct_subgraph(question, initial_nodes)

        # Generate answer using the subgraph (or full KG)
        # answer = await self.deduce_answer(question, subgraph, self.answer)
        answer = await self.generate_answer(question, subgraph)

        total_edges = len(self.KG.edges)
        subgraph_edges = len(subgraph.edges)
        edge_coverage = (subgraph_edges / total_edges) if total_edges > 0 else 0.0
        # semantic_reward = await self.compute_semantic_reward()
        # Return answer and coverage metrics
        return json.dumps({
            "answer": answer,
            "edge_coverage": edge_coverage,
            "semantic_reward": 0.0
        })

    async def generate_answer(self, query, subgraph: DiGraph):
        """Generate an answer using the subgraph (or full KG) with a single LLM call."""
        triples = [(u, d["relation"], v) for u, v, d in subgraph.edges(data=True)]
        triples_string = ". ".join([f"({s}-{r}->{o})" for s, r, o in triples])
        if not triples_string:
            triples_string = "No relevant triples found."
        prompt = ANSWER_GENERATION_PROMPT
        messages = [
            {"role": "system", "content": prompt},
        ]
        self.sampling_params["temperature"] = self.config.temperature_reasoning
        messages.append({"role": "user", "content": f"{triples_string}\n\n{query}"})
        generated_text = await self.llm_generator.generate_response(messages, **self.sampling_params)
        if "Answer:" in generated_text:
            generated_text = generated_text.split("Answer:")[-1]
        elif "answer:" in generated_text:
            generated_text = generated_text.split("answer:")[-1]
        # if answer is none
        if not generated_text:
            return "none"
        return generated_text
    
    async def deduce_answer(self, query, subgraph: DiGraph, answer):
        """Generate an answer using the subgraph (or full KG) with a single LLM call."""
        triples = [(u, d["relation"], v) for u, v, d in subgraph.edges(data=True)]
        triples_string = ". ".join([f"({s}-{r}->{o})" for s, r, o in triples])
        if not triples_string:
            triples_string = "No relevant triples found."
        prompt = VERIFY_ANSWER_PROMPT
        messages = [
            {"role": "system", "content": prompt},
        ]
        self.sampling_params["temperature"] = self.config.temperature_reasoning
        messages.append({"role": "user", "content": f"Knowledge graph (KG) context:{triples_string}\nQuestion:{query}\nTrue Answer:{answer}\nCan the true answer be deduced from the KG context? Answer 'Yes' or 'No' only."})
        generated_text = await self.llm_generator.generate_response(messages, **self.sampling_params)
        generated_text = generated_text.strip().lower()
        if "yes" in generated_text:
            generated_text = 'yes'
        elif "no" in generated_text:
            generated_text = 'no'
        # if answer is none
        if not generated_text:
            return "no"
        return generated_text

    async def compute_semantic_reward(self, fuzzy_threshold=0.85):
        """
        Compute semantic (local fidelity) reward:
        For each decomposed atomic query with its golden answer,
        check whether any triple in the subgraph can answer it.
        """
        def normalize_text(text: str) -> str:
            # Lowercase
            text = text.lower().strip()
            # Remove punctuation
            text = re.sub(r"[^\w\s]", "", text)
            # Collapse multiple spaces
            text = re.sub(r"\s+", " ", text)
            return text
        if not hasattr(self, "sub_queries") or not self.sub_queries:
            return 0.0

        triples = [(u, data["relation"], v) for u, v, data in self.KG.edges(data=True)]
        
        if not triples:
            return 0.0

        rewards = []
        sim_scores = self.sub_queries_embeddings @ self.triple_embeddings.T
        assert sim_scores.shape == (len(self.sub_queries), len(triples)), f"sim_scores shape {sim_scores.shape} does not match expected {(triples, self.kg.edges(data="relation") )}"
        for sim_score, sub_query in zip(sim_scores, self.sub_queries):
            # Parse golden answer string (assumes format "... Answer: X")
            if "Answer:" not in sub_query:
                continue
            golden_answer = normalize_text(sub_query.split("Answer:")[-1])

            triple_index = np.argmax(sim_score)
            top_triple = triples[triple_index]
            top_head, _, top_tail = map(normalize_text, top_triple)
            max_score = sim_score[triple_index]
            # --- Strict exact match ---
            if golden_answer == top_head or golden_answer == top_tail:
                reward = 1.0

            else:
                # compute fuzzy similarity with head & tail
                head_sim = jellyfish.jaro_winkler_similarity(golden_answer, top_head)
                tail_sim = jellyfish.jaro_winkler_similarity(golden_answer, top_tail)

                if (
                    golden_answer in top_head
                    or golden_answer in top_tail
                    or head_sim >= fuzzy_threshold
                    or tail_sim >= fuzzy_threshold
                ):
                    reward = float(max_score)
                else:
                    reward = 0.0

            rewards.append(reward)

        # Final reward = average over sub-queries
        return float(np.mean(rewards)) if rewards else 0.0