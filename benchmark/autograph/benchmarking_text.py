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
import torch
import argparse
import time

argparser = argparse.ArgumentParser(description="Run Atlas Multi-hop QA Benchmark")
argparser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="Keyword for extraction")
argparser.add_argument("--port", type=int, default=8110, help="Port number for LLM server")
# set store true if using upperbound retrieval
argparser.add_argument("--use_upperbound", action="store_true", help="Use upperbound retrieval")
# set store true if using dense retrieval only
argparser.add_argument("--use_dense_only", action="store_true", help="Use dense retrieval only")
args = argparser.parse_args()
kg_names = ["2wikimultihopqa","musique", 'hotpotqa', '2021wiki']
# kg_names = ['2021wiki']
# kg_names = ['hotpotqa']
# kg_names = ['2wikimultihopqa']
# kg_names = ['musique']
def main():
    for kg_name in kg_names:
        # Load SentenceTransformer model
        encoder_model_name = "Qwen/Qwen3-Embedding-0.6B"
        sentence_model = OpenAI(
            base_url="http://0.0.0.0:8130/v1",
            api_key="EMPTY KEY",
        )
        sentence_encoder = Qwen3Emb(sentence_model)

        reader_model_name = "Qwen/Qwen2.5-7B-Instruct"
        client = OpenAI(
            base_url="http://0.0.0.0:8131/v1",
            api_key="EMPTY KEY",
        )
        llm_generator = LLMGenerator(client=client, model_name=reader_model_name)
        
        checkpoint_path = args.model_name
        if checkpoint_path == "Qwen/Qwen2.5-3B-Instruct" or checkpoint_path == "Qwen/Qwen2.5-7B-Instruct" or checkpoint_path == 'meta-llama/Llama-3.2-3B-Instruct' or checkpoint_path == 'meta-llama/Llama-3.2-1B-Instruct':
        # get the name after '/'
            output_directory = f'/data/autograph/checkpoints/{checkpoint_path.split("/")[-1]}/constructed_kg/{kg_name}_output'
        else:
            output_directory = f'{checkpoint_path}/constructed_kg/{kg_name}_output'
        if not args.use_upperbound:
            data = create_embeddings_and_index(
                sentence_encoder=sentence_encoder,
                model_name=encoder_model_name,
                working_directory=output_directory,
                keyword=kg_name,
                include_concept=False,
                include_events=False,
                normalize_embeddings=False,
                text_batch_size=512,
                node_and_edge_batch_size=512,
            )

        # Configure benchmarking
        if kg_name == "2021wiki":
            qa_names = ["nq", "popqa"]
        else:
            qa_names = [kg_name]
        for qa_name in qa_names:
            inference_config = InferenceConfig(keyword=qa_name, ppr_max_iter=10000, weight_adjust=0.01, is_filter_edges=False)
            # get the parent directory of output_directory
            base_dir = '/'.join(output_directory.split('/')[:-2])
            if args.use_upperbound:
                base_dir = base_dir + "_upperbound"
            if args.use_dense_only:
                base_dir = base_dir + "_dense"
            benchmark_config = BenchMarkConfig(
                dataset_name=qa_name,
                question_file=f"/home/knowcomp/projects/autograph-r1/benchmark/{qa_name}.json",
                result_dir=f"{base_dir}/benchmark/text_retrieval",
                include_concept=False,
                include_events=False,
                reader_model_name=reader_model_name,
                encoder_model_name=encoder_model_name,
                number_of_samples=-1,  # -1 for all samples
                upper_bound_mode=args.use_upperbound,
            )
            # Set up logger
            logger = setup_logger(benchmark_config, 
                                  log_path = f"{base_dir}/benchmark/text_retrieval/{qa_name}_{time.time()}_benchmark.log")
            if args.use_upperbound:
                from atlas_rag.retriever.upper_bound_retriever import UpperBoundRetriever
                upperbound_retriever = UpperBoundRetriever()
                benchmark = RAGBenchmark(config=benchmark_config, logger=logger)
                benchmark.run([upperbound_retriever], llm_generator=llm_generator)
            if args.use_dense_only:
                # Initialize DenseRetriever
                dense_retriever = SimpleTextRetriever(
                    passage_dict=data["text_dict"],
                    sentence_encoder=sentence_encoder,
                    data=data,
                    inference_config=inference_config,
                )
                benchmark = RAGBenchmark(config=benchmark_config, logger=logger)
                benchmark.run([dense_retriever], llm_generator=llm_generator)
            elif not args.use_upperbound and not args.use_dense_only:
                # Initialize HippoRAG2Retriever
                hipporag2_retriever = HippoRAG2Retriever(
                    llm_generator=llm_generator,
                    sentence_encoder=sentence_encoder,
                    data=data,
                    inference_config=inference_config,
                    logger=logger
                )
                hipporag_retriever = HippoRAGRetriever(
                    llm_generator=llm_generator,
                    sentence_encoder=sentence_encoder,
                    data=data,
                    logger=logger,
                    inference_config=inference_config,
                )

                # Start benchmarking
                benchmark = RAGBenchmark(config=benchmark_config, logger=logger)
                benchmark.run([hipporag_retriever, hipporag2_retriever], 
                            llm_generator=llm_generator)

if __name__ == "__main__":
    main()