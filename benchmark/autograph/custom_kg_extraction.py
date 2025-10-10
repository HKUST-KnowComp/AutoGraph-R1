from atlas_rag.kg_construction.triple_extraction import KnowledgeGraphExtractor
from atlas_rag.kg_construction.triple_config import ProcessingConfig
from atlas_rag.llm_generator import LLMGenerator
from openai import OpenAI
from configparser import ConfigParser
import argparse
parser = argparse.ArgumentParser(description="Custom KG Extraction")
parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="Keyword for extraction")
args = parser.parse_args()
keywords = ["2wikimultihopqa","musique", "hotpotqa", "2021wiki"]
# keywords = ["2021wiki"]
for keyword in keywords:
      model_name = args.model_name
      client = OpenAI(base_url="http://0.0.0.0:8111/v1", api_key="EMPTY")
      triple_generator = LLMGenerator(client=client, model_name=model_name)

      filename_pattern = keyword
      checkpoint_path = args.model_name
      input_directory = f'/home/knowcomp/projects/autograph-r1/benchmark/autograph/{keyword}'
      output_directory = f'{checkpoint_path}/constructed_kg/{keyword}_output'
      if checkpoint_path == "Qwen/Qwen2.5-3B-Instruct" or checkpoint_path == "Qwen/Qwen2.5-7B-Instruct" or checkpoint_path == 'meta-llama/Llama-3.2-3B-Instruct' or checkpoint_path == 'meta-llama/Llama-3.1-8B-Instruct':
      # get the name after '/'
            output_directory = f'/data/autograph/checkpoints/{checkpoint_path.split("/")[-1]}/constructed_kg/{keyword}_output'
      print(f"Output directory: {output_directory}")
      # triple_generator = LLMGenerator(client, model_name=model_name)
      kg_extraction_config = ProcessingConfig(
            model_path=model_name,
            data_directory=f'{input_directory}',
            filename_pattern=filename_pattern,
            batch_size_triple=16,
            batch_size_concept=16,
            output_directory=f"{output_directory}",
            max_new_tokens=8192,
            max_workers=5,
            remove_doc_spaces=True, # For removing duplicated spaces in the document text
            include_concept=False, # Whether to include concept nodes and edges in the knowledge graph
            triple_extraction_prompt_path='/home/knowcomp/projects/autograph-r1/benchmark/autograph/custom_prompt.json',
            triple_extraction_schema_path='/home/knowcomp/projects/autograph-r1/benchmark/autograph/custom_schema.json',
            record=False, # Whether to record the results in a JSON file
      )
      kg_extractor = KnowledgeGraphExtractor(model=triple_generator, config=kg_extraction_config)
      # construct entity&event graph
      kg_extractor.run_extraction()
      # # Convert Triples Json to CSV
      kg_extractor.convert_json_to_csv()
      # convert csv to graphml for networkx
      kg_extractor.convert_to_graphml()