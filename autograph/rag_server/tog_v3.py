from autograph.rag_server.llm_api import LLMGenerator
from autograph.rag_server.reranker_api import Reranker
import numpy as np
from autograph.rag_server.base_retriever import RetrieverConfig, BaseRetriever
import json_repair
import networkx as nx
from networkx import DiGraph
from autograph.rag_server.tog_prompt import *
from collections import defaultdict
import json

def batch(iterable, n=100):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]
class TogV3Retriever(BaseRetriever):
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
        response = await self.llm_generator.generate_response(messages, **self.sampling_params)
        
        # Parse the response to ensure it is a valid JSON object
        entities_json = json_repair.loads(response)
        if "entities" not in entities_json or not isinstance(entities_json["entities"], list):
            return {}
        return entities_json

    async def retrieve_topk_nodes(self, query, topN=5):
        # Step 1: Extract topic entities using the ner method
        entities_json = await self.ner(query)
        entities = entities_json.get("entities", [])
        if not entities:
            def get_node_query_instruct(query):
                return f"Find the most relevant entity in the knowledge graph for the following query: {query}"
            query_emb = await self.reranker.embed([get_node_query_instruct(query)]) 
            sim_score = query_emb @ self.node_embeddings.T
            assert sim_score.shape[0] == 1, "sim_score shape is incorrect"
            top_indices = np.argsort(sim_score[0])[::-1][:topN]
            topk_nodes = [list(self.KG.nodes)[i] for i in top_indices]
            return topk_nodes
        topk_nodes = []
        entities_not_in_kg = []
        # map entities into KG's nodes
        entities = [str(e) for e in entities]
        entities = list(set(entities))  # remove duplicates
        for entity in entities:
            if entity in self.node_list:
                topk_nodes.append(entity)
            else:
                entities_not_in_kg.append(entity)

        if entities_not_in_kg:
            kg_nodes = self.node_list
            sim_scores = await self.reranker.compute_similarity(entities_not_in_kg, kg_nodes)
            # get the top-1 similar node
            index = np.argmax(sim_scores, axis=1)
            assert len(index) == len(entities_not_in_kg), "Index length does not match entities_not_in_kg length"
            for i in index:
                top_node = kg_nodes[i]
                topk_nodes.append(top_node)

        return topk_nodes

    async def index_kg(self, kg: DiGraph, query, batch_size:int = 100):
        nodes = list(kg.nodes)
        self.node_list = nodes
        node_embeddings = []
        for node_batch in batch(nodes, batch_size):
            node_embeddings.extend(await self.reranker.embed(node_batch))
        self.node_embeddings = np.array(node_embeddings)
        def get_path_query_instruct(query):
            return f"Find the most relevant knowledge graph paths for the following query: {query}"
        self.path_query_emb = np.array(await self.reranker.embed([get_path_query_instruct(query)]))

    async def retrieve(self, query, kg: DiGraph, sampling_params: dict, **kwargs):
        """ 
        Retrieve the top N paths that connect the entities in the query.
        Dmax is the maximum depth of the search.
        """
        topN = kwargs.get("topN", 5)
        Dmax = kwargs.get("Dmax", 3)
        self.answer = kwargs.get("answer", "N/A")
        self.sampling_params = sampling_params
        self.KG = kg
        await self.index_kg(self.KG, query)
        # in the first step, we retrieve the top k nodes
        initial_nodes = await self.retrieve_topk_nodes(query, topN=topN)
        E = initial_nodes
        P = [ [e] for e in E]
        D = 0

        while D <= Dmax:
            P = self.search(query, P)
            P = await self.prune(query, P, topN)
            if await self.reasoning(query, P):
                generated_text = await self.generate(query, P, self.answer)
                break
            D += 1
        
        if D > Dmax:    
            generated_text = await self.generate(query, P, self.answer)
        
        # print(generated_text)
        return json.dumps({
            "answer": generated_text
        })

    def search(self, query, P):
        new_paths = []
        for path in P:
            tail_entity = path[-1]
            sucessors = list(self.KG.successors(tail_entity))

            # print(f"tail_entity: {tail_entity}")
            # print(f"sucessors: {sucessors}")
            # print(f"predecessors: {predecessors}")

            # # print the attributes of the tail_entity
            # print(f"attributes of the tail_entity: {self.KG.nodes[tail_entity]}")
           
            # remove the entity that is already in the path
            sucessors = [neighbour for neighbour in sucessors if neighbour not in path]
            # predecessors = [neighbour for neighbour in predecessors if neighbour not in path]

            if len(sucessors) == 0:
                new_paths.append(path)
                continue
            for neighbour in sucessors:
                relation = self.KG.edges[(tail_entity, neighbour)]["relation"]
                new_path = path + [relation, neighbour]
                new_paths.append(new_path)
            
            # for neighbour in predecessors:
            #     relation = self.KG.edges[(neighbour, tail_entity)]["relation"]
            #     new_path = path + [relation, neighbour]
            #     new_paths.append(new_path)
        
        return new_paths
    
    async def prune(self, query, P, topN=3):
        # Create path strings for embedding comparison
        path_strings = []
        
        for path in P:
            path_string = "->".join(path)
            path_strings.append(path_string)
        
        # Encode query and paths
        # Pass query_type='edge' for appropriate prefixing in embedding models that support it
        
        query_embedding = self.path_query_emb[0]
        path_embeddings = await self.reranker.embed(path_strings)
        path_embeddings = np.array(path_embeddings)
        
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        path_embeddings = path_embeddings / np.linalg.norm(path_embeddings, axis=1, keepdims=True)

        # Compute similarity scores
        scores = path_embeddings @ query_embedding
        
        # Sort paths by similarity scores (higher is better)
        sorted_indices = np.argsort(scores)[::-1]  # Descending order
        sorted_paths = [P[i] for i in sorted_indices[:topN]]
        
        return sorted_paths

    async def reasoning(self, query, P):
        triples_string = []
        for path in P:
            for i in range(0, len(path)-2, 2):
                triples_string.append(f'{path[i]}, {path[i+1]}, {path[i+2]}')
        
        triples_string = ". ".join(triples_string)

        prompt = f"Given a question and the associated retrieved knowledge graph triples (entity, relation, entity), you are asked to answer whether it's sufficient for you to answer the question with these triples and your knowledge (Yes or No). Query: {query} \n Knowledge triples: {triples_string}"
        
        messages = [{"role": "system", "content": "Answer the question following the prompt."},
        {"role": "user", "content": f"{prompt}"}]

        response = await self.llm_generator.generate_response(messages, **self.sampling_params)
        return "yes" in response.lower()

    async def generate(self, query, P, answer):
        triples = []
        for path in P:
            for i in range(0, len(path)-2, 2):
                triples.append((path[i], path[i+1], path[i+2]))
        
        triples_string = [f"({triple[0]}, {triple[1]}, {triple[2]})" for triple in triples]
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

        