from configparser import ConfigParser
from openai import OpenAI
from atlas_rag.retriever import *
from atlas_rag.vectorstore.embedding_model import Qwen3Emb
from atlas_rag.vectorstore.create_graph_index import create_embeddings_and_index
from atlas_rag.logging import setup_logger
from atlas_rag.llm_generator import LLMGenerator
from atlas_rag.evaluation import BenchMarkConfig, RAGBenchmark
from transformers import AutoModel
from sentence_transformers import SentenceTransformer
from atlas_rag.retriever.inference_config import InferenceConfig
from atlas_rag.evaluation.evaluation import QAJudger
import torch
import argparse
import time
import os
import json
import pandas as pd
import tqdm
from collections import defaultdict

argparser = argparse.ArgumentParser(description="Run Atlas Multi-hop QA Benchmark")
# set store true if using upperbound retrieval
argparser.add_argument("--regenerate_dir",type=str, help="Regenerate JSON files")
argparser.add_argument("--regen_num", type=int, default=0, help="Number of times to regenerate each question")
args = argparser.parse_args()
kg_names = ["musique"]
def main():
    for kg_name in kg_names:
        reader_model_name = "Qwen/Qwen2.5-7B-Instruct"
        client = OpenAI(
            base_url="http://0.0.0.0:8137/v1",
            api_key="EMPTY KEY",
        )
        llm_generator = LLMGenerator(client=client, model_name=reader_model_name)
        # regenerate_dir will be something like /data/autograph/checkpoints/20250920_053621_qwen2.5-7b-autograph-distract_easy-docsize15-textlinkingTrue-hipporag2-tight

        # Configure benchmarking
        retriever_names = ["HippoRAGRetriever", "HippoRAG2Retriever"]
        qa_judge = QAJudger()
        if kg_name == "2021wiki":
            qa_names = ["nq", "popqa"]
        else:
            qa_names = [kg_name]
        for qa_name in qa_names:
            # get qa_name dir and get the results json files for regenerate
            qa_dir = f"{args.regenerate_dir}/{qa_name}"
            # get the json files in the qa_dir
            json_files = [f for f in os.listdir(qa_dir) if f.endswith(".json") and f.startswith("result_")]
            json_file = json_files[0]
            # it is jsonl in .json file
            print(f"Regenerating {json_file} in {qa_dir}")
            with open(f"{qa_dir}/{json_file}", "r") as f:
                results = [json.loads(line) for line in f.readlines()]
            em_dict = defaultdict(list)
            f1_dict = defaultdict(list)
            for i, result in tqdm.tqdm(enumerate(results), total=len(results)):
                question = result["question"]
                ground_truth = result["answer"]
                for retriever_name in retriever_names:
                    retrieved_passage_key = f"{retriever_name}_passages"
                    retrieved_context = result[retrieved_passage_key]
                    retrieved_context = "\n".join(retrieved_context)
                    llm_generated_answer = llm_generator.generate_with_context_kg(question, retrieved_context, max_new_tokens=2048, temperature=0.0)
                    em, f1 = qa_judge.judge(llm_generated_answer, ground_truth)
                    em_dict[retriever_name].append(em)
                    f1_dict[retriever_name].append(f1)
            
            # save the avg em and avg f1 to another json files:
            output_dir = f"{args.regenerate_dir}/{qa_name}/regen_{args.regen_num}_results"
            os.makedirs(output_dir, exist_ok=True)
            with open(f"{output_dir}/regen_results.json", "w") as f:
                regen_results = {}
                for retriever_name in retriever_names:
                    avg_em = sum(em_dict[retriever_name]) / len(em_dict[retriever_name])
                    avg_f1 = sum(f1_dict[retriever_name]) / len(f1_dict[retriever_name])
                    regen_results[retriever_name] = {
                        "avg_em": avg_em,
                        "avg_f1": avg_f1
                    }
                json.dump(regen_results, f, indent=4)


if __name__ == "__main__":
    main()