from llm_api import LLMGenerator
from .reranker_api import Reranker
import numpy as np
from .base_retriever import RetrieverConfig, BaseRetriever
import json_repair
import networkx as nx
from networkx import DiGraph
from .tog_prompt import *
from collections import defaultdict
import json
class TogRetriever(BaseRetriever):
    def __init__(self, config:RetrieverConfig, llm_generator:LLMGenerator, reranker:Reranker):
        self.config = config
        self.llm_generator = llm_generator
        self.reranker = reranker

    async def ner(self, text):
        messages = [
            {
                "role": "system",
                "content": "Extract the topic entities from the following question and output them as a JSON object in the format: {\"entities\": [\"entity1\", \"entity2\", ...]}"
            },
            {
                "role": "user",
                "content": f"Identify starting points for reasoning within a knowledge graph to find relevant information and clues for answering the question. Extract the topic entities from: {text}"
            }
        ]
        response = await self.llm_generator.generate_response(messages)
        
        # Parse the response to ensure it is a valid JSON object
        entities_json = json_repair.loads(response)
        if "entities" not in entities_json or not isinstance(entities_json["entities"], list):
            return {}
        return entities_json

    async def topic_prune(self, query, entities):        
        # Step 2: If no entities are retrieved, return an empty dictionary
        if not entities:
            return {}
        
        if len(entities) < self.config.width:
            return {"entities": entities}
        
        # Step 3: Prepare a prompt for the LLM to analyze the suitability of each entity
        analysis_prompt = TOPIC_PRUNE_PROMPT % (query, entities)
        
        # Step 4: Generate the LLM response to analyze and prune the entities
        messages = [
            {
                "role": "system",
                "content": "You are a reasoning assistant tasked with pruning topic entities for a knowledge graph query. Provide a JSON-formatted output containing only the entities suitable as starting points for reasoning, based on the provided question and topic entities."
            },
            {
                "role": "user",
                "content": analysis_prompt
            }
        ]
        
        response = await self.llm_generator.generate_response(messages, temperature=self.config.temperature_exploration)
        
        # Step 5: Parse the response to ensure it is a valid JSON dictionary
        pruned_entities = json_repair.loads(response)
        if not isinstance(pruned_entities, dict):
            return {}
        return pruned_entities

    async def relation_prune_combination(self, query, entities_relation: dict):
        # only prompt with entities with number of relations larger than self.config.width
        # For those that are less than self.config.width, we keep all relations
        entities_relation_less = {entity: relations for entity, relations in entities_relation.items() if len(relations) <= self.config.width}
        entities_relation_more = {entity: relations for entity, relations in entities_relation.items() if len(relations) > self.config.width}

        # Step 1: Prepare the prompt for the LLM
        entities_prompt = "\n".join(
            [
                f'Entity: "{entity}"\nAvailable Relations: {relations}'
                for entity, relations in entities_relation_more.items()
            ]
        )
        analysis_prompt = RELATION_PRUNE_PROMPT % (self.config.width, self.config.width) + f"\nQuestion: {query}\n{entities_prompt}"

        # Step 2: Generate the LLM response to analyze and prune the relations
        messages = [
            {
                "role": "system",
                "content": "You are a reasoning assistant tasked with pruning relations for entities in a knowledge graph query. Provide a JSON-formatted output containing the most relevant relations for each entity."
            },
            {
                "role": "user",
                "content": analysis_prompt
            }
        ]

        response = await self.llm_generator.generate_response(messages, temperature=self.config.temperature_exploration)

        # Step 3: Parse the response to ensure it is a valid JSON dictionary
        try:
            pruned_relations = json_repair.loads(response)
            if not isinstance(pruned_relations, list):
                return entities_relation
            # combine the pruned relations with those that were less than self.config.width
            if not all("entity" in item and "relations" in item for item in pruned_relations):
                return entities_relation
            return {item["entity"]: item["relations"] for item in pruned_relations if "entity" in item and "relations" in item} | entities_relation_less
        except Exception as e:
            print(f"Error parsing relation prune response: {e}")
            return entities_relation

    async def retrieve_topk_nodes(self, query):
        # Step 1: Extract topic entities using the ner method
        entities_json = await self.ner(query)
        entities = entities_json.get("entities", [])
        if not entities:
            entities = [query]
        else:
            if self.config.topic_prune:
                entities_json = await self.topic_prune(query, entities)
                entities = entities_json.get("entities", [entities])
        topk_nodes = []
        entities_not_in_kg = []
        # map entities into KG's nodes
        entities = [str(e) for e in entities]
        for entity in entities:
            if entity in self.KG.nodes:
                topk_nodes.append(entity)
            else:
                entities_not_in_kg.append(entity)

        if entities_not_in_kg:
            kg_nodes = list(self.KG.nodes)
            sim_scores = await self.reranker.compute_similarity(entities_not_in_kg, kg_nodes)
            # get the top-1 similar node
            index = np.argmax(sim_scores, axis=1)
            assert len(index) == len(entities_not_in_kg), "Index length does not match entities_not_in_kg length"
            for i in index:
                top_node = kg_nodes[i]
                topk_nodes.append(top_node)

        return topk_nodes

    async def index_kg(self, kg: DiGraph):
        # compute embedding for all nodes using self.reranker
        self.node_embeddings = await self.reranker.embed(kg.nodes)
        relations = list(nx.get_edge_attributes(kg, "relation").values())
        self.relation_embeddings = await self.reranker.embed(relations)

    async def retrieve(self, question, kg:DiGraph, sampling_params:dict) -> str:
        self.KG = kg
        self.sampling_params = sampling_params
        await self.index_kg(kg)
        topN = self.config.num_sents_for_reasoning

        initial_nodes = await self.retrieve_topk_nodes(question)
        E = initial_nodes
        P = [[e] for e in E]
        D = 0
        answerable = False
        while D <= self.config.depth:
            P = await self.search(question, P)
            answerable, answer = await self.reasoning(question, P)
            if answerable:
                break
            D += 1
            if D < self.config.depth:
                P = await self.prune(question, P, topN)
                
        if not answerable:
            answer = await self.generate(question, P)
        return json.dumps({"answer": answer})

    async def search(self, query, P):
        new_paths = []
        entity_relation_dict = defaultdict(list)

        # Step 1: Collect relations for each tail entity
        for path in P:
            tail_entity = path[-1]
            # Collect successors and predecessors
            for neighbor, direction in [(n, "successor") for n in self.KG.successors(tail_entity)] + \
                                        [(n, "predecessor") for n in self.KG.predecessors(tail_entity)]:
                relation = self.KG.edges[(tail_entity, neighbor)]["relation"] if direction == "successor" else \
                           self.KG.edges[(neighbor, tail_entity)]["relation"]
                if relation not in entity_relation_dict[tail_entity]:
                    entity_relation_dict[tail_entity].append(relation)

        # Step 2: Perform relation pruning if enabled
        if self.config.remove_unnecessary_rel:
            # currently we only implemented relation_prune_combination
            entity_relation_dict = await self.relation_prune_combination(query, entity_relation_dict)

        # Step 3: Expand paths using pruned relations
        for path in P:
            tail_entity = path[-1]
            if tail_entity not in entity_relation_dict:
                continue  # Skip if no relations are available for the tail entity

            for relation in entity_relation_dict[tail_entity]:
                # Expand paths with successors
                for neighbor in self.KG.successors(tail_entity):
                    if neighbor not in path:  # Avoid cycles
                        new_paths.append(path + [relation, neighbor])
                # Expand paths with predecessors
                for neighbor in self.KG.predecessors(tail_entity):
                    if neighbor not in path:  # Avoid cycles
                        new_paths.append(path + [relation, neighbor])

        return new_paths

    async def prune(self, query, P, topN):
        # construct KG path string
        all_paths = []
        for path in P:
            triples = []
            for i in range(0, len(path)-2, 2):
                s = path[i]
                r = path[i+1]
                o = path[i+2]
                triples.append((s, r, o))
            triples_string = ". ".join([f"({s}, {r}, {o})" for s, r, o in triples])
            # consturct the query prompt
            all_paths.append(triples_string)
        # use reranker to score the paths
        query = f"Instruct: {RERANKER_RANK_PATH_PROMPT}\nQuery: {query}"
        ratings = await self.reranker.compute_similarity([query], all_paths)
        ratings = ratings.flatten().tolist()
        sorted_paths = [path for _, path in sorted(zip(ratings, P), reverse=True)]
        return sorted_paths[:topN]

    async def reasoning(self, query, P):
        all_paths = []
        for path in P:
            triples = []
            for i in range(0, len(path)-2, 2):
                s = path[i]
                r = path[i+1]
                o = path[i+2]
                triples.append((s, r, o))
            triples_string = ". ".join([f"({s}, {r}, {o})" for s, r, o in triples])
            # consturct the query prompt
            all_paths.append(triples_string)
        all_paths_string = "\n".join(all_paths)
        prompt = REASONING_PROMPT % (query, all_paths_string)
        messages = [
            {"role": "system", "content": "You are a helpful assistant that reasons and answers questions based on the provided knowledge graph context."},
            {"role": "user", "content": prompt}
        ]
        response = await self.llm_generator.generate_response(messages, temperature=self.config.temperature_reasoning)
        result_json = json_repair.loads(response)
        if not isinstance(result_json, dict) or "answer" not in result_json or "is_answerable" not in result_json:
            return False, ""
        is_answerable = result_json.get("is_answerable", False)
        answer = result_json.get("answer", "")
        if is_answerable:
            return True, answer
        return False, ""

    async def generate(self, query, P):
        all_paths = []
        for path in P:
            triples = []
            for i in range(0, len(path)-2, 2):
                s = path[i]
                r = path[i+1]
                o = path[i+2]
                triples.append((s, r, o))
            triples_string = ". ".join([f"({s}, {r}, {o})" for s, r, o in triples])
            # consturct the query prompt
            all_paths.append(triples_string)
        all_paths_string = "\n".join(all_paths)
        prompt = ANSWER_GENERATION_PROMPT % (query, all_paths_string)
        messages = [
            {"role": "system", "content": "You are a helpful assistant that reasons and answers questions based on the provided knowledge graph context."},
            {"role": "user", "content": prompt}
        ]
        response = await self.llm_generator.generate_response(messages, temperature=self.config.temperature_reasoning)
        result_json = json_repair.loads(response)
        if not isinstance(result_json, dict) or "answer" not in result_json:
            return response
        answer = result_json.get("answer", "")
        return answer