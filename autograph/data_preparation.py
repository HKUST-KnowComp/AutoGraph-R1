import argparse
import logging
import os
import tempfile
import random
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm 
import os
from huggingface_hub import hf_hub_download
import json_repair
import asyncio
from rag_server.llm_api import LLMGenerator
from tqdm.asyncio import tqdm_asyncio
import json
import re
# import library for hashing the tuple
from hashlib import sha256

DEFAULT_SYSTEM_CONTENT = """You are an expert knowledge graph constructor.  
Your task is to extract factual information from the provided text and represent it strictly as a JSON array of knowledge graph triples.  

### Output Format
- The output must be a **JSON array**.
- Each element in the array must be a **JSON object** with exactly three non-empty keys:
  - "subject": the main entity, concept, event, or attribute.  
  - "relation": a concise, descriptive phrase or verb that describes the relationship (e.g., "founded by", "started on", "is a", "has circulation of").  
  - "object": the entity, concept, value, event, or attribute that the subject has a relationship with.  

### Constraints
- **Do not include any text other than the JSON output.**
- Do not add explanations, comments, or formatting outside of the JSON array.  
- Extract **all possible and relevant triples**.  
- All keys must exist and all values must be non-empty strings.  
- The "subject" and "object" can be specific entities (e.g., "Radio City", "Football in Albania", "Echosmith") or specific values (e.g., "3 July 2001", "1,310,696").  
- If no triples can be extracted, return exactly: `[]`."""

MINIMAL_SYSTEM_PROMPT = """"You are an expert knowledge graph constructor. Your task is to extract list of knowledge graph triples in the form of list of json object.
Each triple should be a JSON object with three keys:\n1. subject, relation, object. Do not include any text other than the list of JSON output."""

MCQ_PROMPT = """
You are an expert in generating multiple-choice questions (MCQs) from scientific texts.
Your task is to generate 5 multiple-choice questions based on the following passage.

Each question should:
- Focus on factual claims, numerical data, definitions, or relational knowledge from the passage.
- Have 4 options (one correct answer and three plausible distractors).
- Clearly indicate the correct answer.

The output should be in JSON format, with each question as a dictionary containing:
- "question": The MCQ question.
- "options": A list of 4 options (e.g., ["A: ..", "B: ..", "C: ..", "D: .."]).
- "answer": The correct answer (e.g., "A").

Output Example:
```
[
  {
    "question": "What is the primary role of a catalyst in a chemical reaction?",
    "options": [
      "A: To make a thermodynamically unfavorable reaction proceed",
      "B: To provide a lower energy pathway between reactants and products",
      "C: To decrease the rate of a chemical reaction",
      "D: To change the overall reaction itself"
    ],
    "answer": "B"
  },
  {
    "question": "By how much can catalysis speed up a chemical reaction compared to its rate without a catalyst?",
    "options": [
      "A: By a factor of several hundred times",
      "B: By a factor of several thousand times",
      "C: By a factor of several million times",
      "D: By a factor of several billion times"
    ],
    "answer": "C"
  },
  ...
]
```

Passage:
{passage}

Output:
"""


parser = argparse.ArgumentParser(description="Download QA dataset from HuggingFace, process, and save to Parquet.")
parser.add_argument("--local_dir", default="/data/autograph/data", help="Local directory to save the processed Parquet files.")
parser.add_argument("--include_distractor", action="store_true", help="Create 'distractor' version of the dataset.")
parser.add_argument("--doc_size", default=10, type=int, help="Number of documents to be included per sample")
parser.add_argument("--first_10_instances", action="store_true", help="Process only the first 10 instances for testing.")
parser.add_argument("--minimum_difficulty", default="medium", choices=["easy", "medium", "hard"], help="Minimum difficulty level of questions to include in the dataset.")
parser.add_argument("--dataset", default="dgslibisey/MuSiQue" , choices=["hotpotqa/hotpot_qa", "dgslibisey/MuSiQue", "xanhho/2WikiMultihopQA"], help="Dataset to process.")
parser.add_argument("--generate_mcq", action="store_true", help="Whether to generate multiple-choice questions (MCQs) from the dataset.")
parser.add_argument("--mcq_path", default="/data/autograph/data/mcq", help="Path to save the generated MCQs.")


args = parser.parse_args()
os.makedirs(args.local_dir, exist_ok=True)  # Create directory if it doesn't exist

max_concurrent = 4  # You can adjust this value as needed
semaphore = asyncio.Semaphore(max_concurrent)

async def run_api(payload, **kwargs):
    async with semaphore:
        return await llm_generator.generate_response(payload, **kwargs)

async def process_hotpotqa_single_row(row, split_name, row_index, mcq_dict = None):
    """
    Process a single row of HotpotQA data for SearchR1-like format.
    """
    question = row.get("question", "")
    answer = row.get("answer", "")
    context = row.get("context", [])
    supporting_facts = row.get("supporting_facts", [])
    supporting_fact_titles = supporting_facts.get("title", [])
    # Flatten context into a string
    titles = context.get("title", [])
    sentences = context.get("sentences", [])

    document_list = []

    # Add supporting facts first
    for title, sentence in zip(titles, sentences):
        if title in supporting_fact_titles:
            document_list.append(f"{title}: {''.join(sentence)}")

    # Add distractors if include_distractor is True
    if args.include_distractor:
        for title, sentence in zip(titles, sentences):
            if title not in supporting_fact_titles:
                document_list.append(f"{title}: {''.join(sentence)}")
    if len(document_list) > args.doc_size:
        document_list = document_list[:args.doc_size]
    # shuffle document list order
    random.shuffle(document_list)
    if args.generate_mcq:
        mcq_list = []
        for i in range(len(document_list)):
            passage = document_list[i]
            response = await run_api({
                "role": "user",
                "content": MCQ_PROMPT.format(passage=passage)
            }, max_tokens=8192, temperature=0.0)
            # validate
            mcq_dict = json_repair.loads(response)
            assert len(mcq_dict) == 5, "must generate 5 mcq for each passage"
            for mcq in mcq_dict:
                assert "question" in mcq, "mcq must have question field"
                assert "options" in mcq, "mcq must have options field"
                assert "answer" in mcq, "mcq must have answer field"
                assert len(mcq["options"]) == 4, "mcq options must be a list of 4 items"
            mcq_list.append(mcq_dict)

    context_str = "\n".join(document_list)
    context_str = context_str.rstrip("\n")  # Remove trailing newline

    prompt = [
        {"role": "system", "content": DEFAULT_SYSTEM_CONTENT},
        {"role": "user", "content": context_str}
    ]
    
    reward_model_data = {
        "ground_truth": {
            "target": [answer],
        },
        "style": "rule"
    }
    if args.generate_mcq:
        reward_model_data["ground_truth"]["mcq"] = mcq_list
    # For HotpotQA, the ground truth is the answer
    ground_truth = [answer] if answer else []


    interaction_kwargs = {
        "name": "graph_construction",
        "question": question,
        "ground_truth": ground_truth,
    }
    
    extra_info = {
        "index": str(row_index),
        "need_tools_kwargs": False,
        "question": question,
        "split": split_name,
        "interaction_kwargs": interaction_kwargs,
    }

    return pd.Series({
        "data_source": f"hotpotqa_{split_name}",
        "prompt": prompt,
        "ability": "graph_construction",
        "reward_model": reward_model_data,
        "extra_info": extra_info,
        "metadata": None,
    })

def get_all_musique_docs(row, split_name, row_index, all_doc_set):
    all_context = row.get("paragraphs", [])
    for context in all_context:
        title = context.get("title", "")
        paragraph = context.get("paragraph_text", "")
        all_doc_set.add((title, paragraph))

async def process_musique_single_row(row, split_name, row_index, mcq_dict = None):
    """
    Process a single row of MuSiQue data for SearchR1-like format.
    """
    question = row.get("question", "")
    answers = row.get("answer_alias", [])
    answer = row.get("answer", None)
    if answer:
        answers.append(answer)
    all_context = row.get("paragraphs", [])

    document_list = []
    document_key = []
    # Add supporting paragraphs first
    for context in all_context:
        if context.get("is_supporting"):
            document_list.append(f"{context.get('title', '')}: {context.get('paragraph_text', '')}")
            document_key.append((context.get("title", ""), context.get("paragraph_text", "")))

    # Add distractors if include_distractor is True
    if args.include_distractor:
        for context in all_context:
            if not context.get("is_supporting"):
                document_list.append(f"{context.get('title', '')}: {context.get('paragraph_text', '')}")
                document_key.append((context.get("title", ""), context.get("paragraph_text", "")))
    if len(document_list) > args.doc_size:
        document_list = document_list[:args.doc_size]
    # shuffle document list order
    random.shuffle(document_list)
    if args.generate_mcq:
        mcq_list = []
        for key in document_key:
            hash_key = sha256(f"{key[0]}: {key[1]}".encode()).hexdigest()
            mcq_list.append(mcq_dict[hash_key])

    context_str = "\n".join(document_list)
    context_str = context_str.rstrip("\n")

    prompt = [
        {"role": "system", "content": DEFAULT_SYSTEM_CONTENT},
        {"role": "user", "content": context_str}
    ]
    
    reward_model_data = {
        "ground_truth": {
            "target": answers
        },
        "style": "rule"
    }
    if args.generate_mcq:
        reward_model_data["ground_truth"]["mcq"] = mcq_list
    # generate initial node and relation and intermediate answer
    sub_queries = row['question_decomposition']
    atomic_sub_query_with_answer = []
    for i, sub_query in enumerate(sub_queries):
        if "#" in sub_query['question']:
            sub_question = sub_query['question']
            match = re.search(r"#(\d+)", sub_question)
            if match:
                index = int(match.group(1))
            else:
                raise ValueError("Invalid question format")
            sub_query_answer = sub_queries[int(index)-1]["answer"]
            # replace #<index> with the actual answer
            sub_question = sub_question.replace(f"#{index}", sub_query_answer)
            atomic_sub_query_with_answer.append(f"{sub_question} Answer: {sub_query['answer']}")
        else:
            atomic_sub_query_with_answer.append(f"{sub_query['question']} Answer: {sub_query['answer']}")
    

    interaction_kwargs = {
        "name": "graph_construction",
        "question": question,
        "ground_truth": answers,
        "sub_queries": atomic_sub_query_with_answer
    }
    
    extra_info = {
        "index": str(row_index),
        "need_tools_kwargs": False,
        "question": question,
        "split": split_name,
        "interaction_kwargs": interaction_kwargs,
    }

    return pd.Series({
        "data_source": f"musique_{split_name}",
        "prompt": prompt,
        "ability": "graph_construction",
        "reward_model": reward_model_data,
        "extra_info": extra_info,
        "metadata": None,
    })

async def process_2wikimultihopqa_single_row(row, split_name, row_index):
    """
    Process a single row of HotpotQA data for SearchR1-like format.
    """
    question = row.get("question", "")
    answer = row.get("answer", "")
    context = row.get("context", [])
    supporting_facts = row.get("supporting_facts", [])
    supporting_fact_titles = supporting_facts.get("title", [])
    # Flatten context into a string
    titles = context.get("title", [])
    sentences = context.get("sentences", [])

    context_str = ""

    for title, sentence in zip(titles, sentences):
        if not args.include_distractor:
            if title not in supporting_fact_titles:
                continue
        context_str += f"{title}: {''.join(sentence)}\n"
    context_str = context_str.rstrip("\n")  # Remove trailing newline

    prompt = [
        {"role": "system", "content": DEFAULT_SYSTEM_CONTENT},
        {"role": "user", "content": context_str}
    ]
    
    reward_model_data = {
        "ground_truth": {
            "target": [answer]
        },
        "style": "rule"
    }
    # For HotpotQA, the ground truth is the answer
    ground_truth = [answer] if answer else []

    interaction_kwargs = {
        "name": "graph_construction",
        "question": question,
        "ground_truth": ground_truth,
    }
    
    extra_info = {
        "index": str(row_index),
        "need_tools_kwargs": False,
        "question": question,
        "split": split_name,
        "interaction_kwargs": interaction_kwargs,
    }

    return pd.Series({
        "data_source": f"hotpotqa_{split_name}",
        "prompt": prompt,
        "ability": "graph_construction",
        "reward_model": reward_model_data,
        "extra_info": extra_info,
        "metadata": None,
    })

# Load the HotpotQA dataset (default configuration: 'distractor')
if args.dataset == "hotpotqa/hotpot_qa":
    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor")
    process_single_row = process_hotpotqa_single_row
    dataset_name = "hotpotqa"
elif args.dataset == "dgslibisey/MuSiQue":
    dataset = load_dataset("dgslibisey/MuSiQue")
    process_single_row = process_musique_single_row
    dataset_name = "musique"
elif args.dataset == "xanhho/2WikiMultihopQA":
    # dataset = load_dataset("xanhho/2WikiMultihopQA")
    # process_single_row = process_2wikimultihopqa_single_row
    # dataset_name = "2wikimultihopqa"
    raise NotImplementedError("2WikiMultihopQA dataset processing is not implemented yet.")
else:
    raise ValueError(f"Unsupported dataset: {args.dataset}")


async def main():
    local_save_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_save_dir, exist_ok=True)
    
    # get all doc to prevent duplicated computation
    # check if mcq dict for musique exist in mcq path
    if args.generate_mcq:
        os.makedirs(args.mcq_path, exist_ok=True)
        mcq_path = os.path.join(args.mcq_path, f"musique_mcq.json")
        if os.path.exists(mcq_path):
            with open(mcq_path, "r") as f:
                mcq_dict = json.load(f)
        else:
            mcq_dict = {}
            all_doc_set = set()
            for split in dataset.keys():
                df = pd.DataFrame(dataset[split])
                for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Collecting all documents for {split}"):
                    get_all_musique_docs(row, split, idx, all_doc_set=all_doc_set)
            for doc in tqdm(list(all_doc_set), total=len(all_doc_set), desc="Generating MCQs for all documents"):
                (title, paragraph) = doc
                response = await run_api([{
                    "role": "user",
                    "content": MCQ_PROMPT.replace("{passage}",f'{title}: {paragraph}')
                }], max_tokens=8192, temperature=0.0)
                # validate
                mcq = json_repair.loads(response)
                for mc in mcq:
                    assert "question" in mc, "mcq must have question field"
                    assert "options" in mc, "mcq must have options field"
                    assert "answer" in mc, "mcq must have answer field"
                    assert len(mc["options"]) == 4, "mcq options must be a list of 4 items"
                hash_key = sha256(f"{title}: {paragraph}".encode()).hexdigest()
                mcq_dict[hash_key] = mcq

            json.dump(mcq_dict, open(mcq_path, "w"), indent=4)
 
    for split in dataset.keys():
        df_raw = pd.DataFrame(dataset[split])
        if args.first_10_instances:
            df_raw = df_raw.head(10)
        tqdm.pandas(desc=f"Processing {split}") 
        # filter according to minimum difficulty
        if args.minimum_difficulty and 'difficulty' in df_raw.columns:
            df_raw = df_raw[df_raw['difficulty'] == args.minimum_difficulty]
        elif args.dataset == "dgslibisey/MuSiQue":
            # since we don't have difficulty field in MuSiQue, we use number of hop indicate in "id" as the difficulty
            # 3 hop as medium, 4 hop as difficult. Id in the format of "2hop__{doc1}_{doc2}"
            if args.minimum_difficulty == "easy":
                df_raw = df_raw[df_raw['id'].str.contains('2hop')]
            elif args.minimum_difficulty == "medium":
                df_raw = df_raw[df_raw['id'].str.contains('3hop')]
            elif args.minimum_difficulty == "hard":
                df_raw = df_raw[df_raw['id'].str.contains('4hop')]
        processed_rows = []
        print(f"Processing {split} with {len(df_raw)} rows")
        for idx, row in tqdm(df_raw.iterrows(), total=len(df_raw), desc=f"Processing {split}"):
            processed = await process_single_row(row, split, idx)
            processed_rows.append(processed)
        df_processed = pd.DataFrame(processed_rows)
        output_file_path = os.path.join(local_save_dir, f"{dataset_name}_{split}_doc_size_{args.doc_size}_distract_{args.include_distractor}_with_mcq_{args.generate_mcq}_difficulty_{args.minimum_difficulty}.parquet")
        if args.first_10_instances:
            output_file_path = os.path.join(local_save_dir, f"{dataset_name}_{split}_distract_{args.include_distractor}_first_10.parquet")
        df_processed.to_parquet(output_file_path, index=False)
        print(f"Saved {len(df_processed)} processed rows to {output_file_path}")

if __name__ == "__main__":
    from openai import AsyncOpenAI
    openai_client = AsyncOpenAI(base_url="http://0.0.0.0:8129/v1", api_key="EMPTY")
    llm_generator = LLMGenerator(openai_client, model_name="Qwen/Qwen2.5-7B-Instruct")
    asyncio.run(main())
