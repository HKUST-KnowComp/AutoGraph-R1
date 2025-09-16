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
from openai import OpenAI
import configparser
import os
import pickle
import numpy as np
import hashlib

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
parser.add_argument("--mix_dataset", action="store_true", help="Whether to mix HotpotQA and MuSiQue datasets together for training")

args = parser.parse_args()
os.makedirs(args.local_dir, exist_ok=True)  # Create directory if it doesn't exist

max_concurrent = 4  # You can adjust this value as needed
semaphore = asyncio.Semaphore(max_concurrent)

def get_query_instruction(text: str):
    return f"Instruct: Find the document most similar to the following text.\nText: {text}"

def get_answer_instruction(text: str):
    return f"Instruct: Based on the provided context, answer the question.\nContext: {text}"

async def run_api(payload, **kwargs):
    async with semaphore:
        return await llm_generator.generate_response(payload, **kwargs)

async def process_hotpotqa_single_row(row, split_name, row_index, mcq_dict = None, **kwargs):
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
    supporting_context = []
    # Add supporting facts first
    for title, sentence in zip(titles, sentences):
        if title in supporting_fact_titles:
            document_list.append(f"{title}: {''.join(sentence)}")
            supporting_context.append(f"{title}: {''.join(sentence)}")

    # Add distractors if include_distractor is True
    if args.include_distractor and not args.mix_dataset:
        for title, sentence in zip(titles, sentences):
            if title not in supporting_fact_titles:
                document_list.append(f"{title}: {''.join(sentence)}")
    elif args.include_distractor and args.mix_dataset:
        doc_embeddings = kwargs.get("doc_embeddings", None)
        all_doc_list = kwargs.get("doc_list", None)
        doc_hash_to_query_answer_embedding = kwargs.get("doc_hash_to_query_answer_embedding", None)
        assert doc_embeddings is not None, "doc_embeddings must be provided for distractor retrieval"
        assert all_doc_list is not None, "all_doc_list must be provided for distractor retrieval"
        targeted_doc_size = args.doc_size
        supported_doc_size = len(supporting_context)
        each_supporting_distract_doc_size = args.doc_size - supported_doc_size
        each_supporting_distract_doc_size = each_supporting_distract_doc_size // supported_doc_size # to ensure each supporting doc has same number of distractors
        supporting_doc_size_dict = {}
        for doc in supporting_context:
            supporting_doc_size_dict[doc] = each_supporting_distract_doc_size
        remain_doc_size = args.doc_size - supported_doc_size - (each_supporting_distract_doc_size * supported_doc_size)
        if remain_doc_size > 0:
            initial_index = random.randint(0, len(supporting_context)-1)
            for i in range(remain_doc_size):
                supporting_doc_size_dict[supporting_context[(initial_index + i) % len(supporting_context)]] += 1
        assert sum(supporting_doc_size_dict.values()) + supported_doc_size == args.doc_size, "Total document size mismatch"
        for supporting_doc, sup_doc_size in supporting_doc_size_dict.items():
            # find the hard negative 
            query_hash = hashlib.sha256(supporting_doc.encode()).hexdigest()
            query_emb = doc_hash_to_query_answer_embedding[query_hash]
            scores = query_emb @ doc_embeddings.T
            top_indices = np.argsort(scores)[::-1]  # Exclude the first one (itself)
            add_count = 0
            buffer_size = sup_doc_size + 50  # Increased buffer for robustness
            candidate_indices = top_indices[:buffer_size]
            for idx in candidate_indices:
                candidate_doc = all_doc_list[idx]
                # Verify document format
                if ":" not in candidate_doc:
                    continue  # Skip malformed documents
                if candidate_doc == supporting_doc:
                    continue  # Skip self or duplicates
                if  candidate_doc in document_list:
                    continue  # Skip duplicates
                document_list.append(candidate_doc)
                title, paragraph = candidate_doc.split(":", 1)
                add_count += 1
                if add_count >= sup_doc_size:
                    break
            if add_count < sup_doc_size:
                raise ValueError(f"Could not find enough unique distractors for {supporting_doc}. Found {add_count}, needed {sup_doc_size}")
        assert len(document_list) == args.doc_size, "Too much after distractor retrieval"

    if len(document_list) > args.doc_size:
        document_list = document_list[:args.doc_size]
    # shuffle document list order
    random.shuffle(document_list)
    # add Document 1 ... Document N prefix
    document_list = [f"Document {i+1}: {doc}" for i, doc in enumerate(document_list)]
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

    atomic_sub_query_with_answer = []
    interaction_kwargs = {
        "name": "graph_construction",
        "question": question,
        "ground_truth": ground_truth,
        "sub_queries": atomic_sub_query_with_answer,
        "supporting_context": supporting_context,
        "full_context": document_list
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

def get_all_musique_docs(row, all_doc_set):
    all_context = row.get("paragraphs", [])
    for context in all_context:
        title = context.get("title", "")
        paragraph = context.get("paragraph_text", "")
        all_doc_set.add(f'{title}: {paragraph}')

def get_all_hotpotqa_docs(row, all_doc_set):
    context = row.get("context", [])
    titles = context.get("title", [])
    sentences = context.get("sentences", [])
    for title, sentence in zip(titles, sentences):
        paragraph = ''.join(sentence)
        all_doc_set.add(f'{title}: {paragraph}')

async def process_musique_single_row(row, split_name, row_index, mcq_dict = None, **kwargs):
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
    supporting_doc_list = []
    supporting_doc_key = []
    for context in all_context:
        if context.get("is_supporting"):
            supporting_doc_list.append(f"{context.get('title', '')}: {context.get('paragraph_text', '')}")
            supporting_doc_key.append((context.get("title", ""), context.get("paragraph_text", "")))
    document_list.extend(supporting_doc_list)
    document_key.extend(supporting_doc_key)
    # Add distractors if include_distractor is True
    if args.include_distractor and not args.mix_dataset:
        for context in all_context:
            if not context.get("is_supporting"):
                document_list.append(f"{context.get('title', '')}: {context.get('paragraph_text', '')}")
                document_key.append((context.get("title", ""), context.get("paragraph_text", "")))
    elif args.include_distractor and args.mix_dataset:
        doc_embeddings = kwargs.get("doc_embeddings", None)
        all_doc_list = kwargs.get("doc_list", None)
        doc_hash_to_query_answer_embedding = kwargs.get("doc_hash_to_query_answer_embedding", None)
        assert doc_embeddings is not None, "doc_embeddings must be provided for distractor retrieval"
        assert all_doc_list is not None, "all_doc_list must be provided for distractor retrieval"
        targeted_doc_size = args.doc_size
        supported_doc_size = len(supporting_doc_list)
        each_supporting_distract_doc_size = args.doc_size - supported_doc_size
        each_supporting_distract_doc_size = each_supporting_distract_doc_size // supported_doc_size # to ensure each supporting doc has same number of distractors
        supporting_doc_size_dict = {}
        for doc in supporting_doc_list:
            supporting_doc_size_dict[doc] = each_supporting_distract_doc_size
        remain_doc_size = args.doc_size - supported_doc_size - (each_supporting_distract_doc_size * supported_doc_size)
        if remain_doc_size > 0:
            initial_index = random.randint(0, len(supporting_doc_list)-1)
            for i in range(remain_doc_size):
                supporting_doc_size_dict[supporting_doc_list[(initial_index + i) % len(supporting_doc_list)]] += 1
        assert sum(supporting_doc_size_dict.values()) + supported_doc_size == args.doc_size, "Total document size mismatch"
        for supporting_doc, sup_doc_size in supporting_doc_size_dict.items():
            # find the hard negative 
            query_hash = hashlib.sha256(supporting_doc.encode()).hexdigest()
            query_emb = doc_hash_to_query_answer_embedding[query_hash]
            scores = query_emb @ doc_embeddings.T
            top_indices = np.argsort(scores)[::-1]  # Exclude the first one (itself)
            add_count = 0
            buffer_size = sup_doc_size + 50  # Increased buffer for robustness
            candidate_indices = top_indices[:buffer_size]
            for idx in candidate_indices:
                candidate_doc = all_doc_list[idx]
                # Verify document format
                if ":" not in candidate_doc:
                    continue  # Skip malformed documents
                if candidate_doc == supporting_doc:
                    continue  # Skip self or duplicates
                if  candidate_doc in document_list:
                    continue  # Skip duplicates
                document_list.append(candidate_doc)
                title, paragraph = candidate_doc.split(":", 1)
                document_key.append((title.strip(), paragraph.strip()))
                add_count += 1
                if add_count >= sup_doc_size:
                    break
            if add_count < sup_doc_size:
                raise ValueError(f"Could not find enough unique distractors for {supporting_doc}. Found {add_count}, needed {sup_doc_size}")
        assert len(document_list) == args.doc_size, "Too much after distractor retrieval"
    if len(document_list) > args.doc_size:
        document_list = document_list[:args.doc_size]
    # shuffle document list order
    random.shuffle(document_list)
    # add Document 1: ... Document 2: ...
    document_list = [f"Document {i+1}: {doc}" for i, doc in enumerate(document_list)]
    if args.generate_mcq:
        mcq_list = []
        for key in document_key:
            hash_key = hashlib.sha256(f"{key[0]}: {key[1]}".encode()).hexdigest()
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
            if index > len(sub_queries):
                atomic_sub_query_with_answer.append(f"{sub_query['question']} Answer: {sub_query['answer']}")
            else:
                sub_query_answer = sub_queries[int(index)-1]["answer"]
                # replace #<index> with the actual answer
                sub_question = sub_question.replace(f"#{index}", sub_query_answer)
                atomic_sub_query_with_answer.append(f"{sub_question} Answer: {sub_query['answer']}")
        else:
            atomic_sub_query_with_answer.append(f"{sub_query['question']} Answer: {sub_query['answer']}")
    
    supporting_context = []
    for context in all_context:
        if context.get("is_supporting"):
            supporting_context.append(f"{context.get('title', '')}: {context.get('paragraph_text', '')}")

    interaction_kwargs = {
        "name": "graph_construction",
        "question": question,
        "ground_truth": answers,
        "sub_queries": atomic_sub_query_with_answer,
        "supporting_context": supporting_context,
        "full_context": document_list
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
                    get_all_musique_docs(row, all_doc_set)
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
                hash_key = hashlib.sha256(f"{title}: {paragraph}".encode()).hexdigest()
                mcq_dict[hash_key] = mcq

            json.dump(mcq_dict, open(mcq_path, "w"), indent=4)
    if not args.mix_dataset:
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
    else:
        
        processed_rows = []
        # combine hotpotqa and musique train dataset
        musique_train = load_dataset("dgslibisey/MuSiQue", split="train")
        hotpotqa_train = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train")
        sample_size = min(len(musique_train), len(hotpotqa_train))
        # sample hotpotqa
        hotpotqa_train = hotpotqa_train.select(range(sample_size))
        # generate document embedding and document list for each data
        # for ds, ds_name in zip([musique_train, hotpotqa_train], ["musique", "hotpotqa"]):
        #     # check if "/data/autograph/data/train_{ds_name}_doc_list.pickle" and "/data/autograph/data/train_{ds_name}_doc_embs.pickle" exist or not
        #     doc_list_path = f'/data/autograph/data/train_{ds_name}_doc_list.pickle'
        #     doc_embs_path = f'/data/autograph/data/train_{ds_name}_doc_embs.pickle'
        #     doc_qa_emb_dict_path = f'/data/autograph/data/train_{ds_name}_doc_qa_emb_dict.pickle'
        #     if os.path.exists(doc_list_path) and os.path.exists(doc_embs_path) and os.path.exists(doc_qa_emb_dict_path):
        #         with open(doc_list_path, "rb") as f:
        #             all_doc_list = pickle.load(f)
        #         with open(doc_embs_path, "rb") as f:
        #             doc_embeddings = pickle.load(f)
        #         with open(doc_qa_emb_dict_path, "rb") as f:
        #             doc_hash_to_query_answer_embedding = pickle.load(f)
        #         print(f"Loaded {len(all_doc_list)} documents and embeddings from cache for {ds_name}")
        #     else:
        #         df = pd.DataFrame(ds)
        #         all_doc_set = set()
        #         for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Collecting all documents"):
        #             if ds == musique_train:
        #                 get_all_musique_docs(row, all_doc_set)
        #             elif ds == hotpotqa_train:
        #                 get_all_hotpotqa_docs(row, all_doc_set)
        #         print(f"Total unique documents: {len(all_doc_set)}")
        #         all_doc_list = list(all_doc_set)
        #         # generate embedding for each document
        #         batch_size = 128
        #         doc_embeddings = []
        #         doc_hash_to_query_answer_embedding = {}
        #         for i in tqdm(range(0, len(all_doc_list), batch_size), desc="Generating document embeddings"):
        #             batch_texts = all_doc_list[i:i+batch_size]
        #             batch_query_texts = [get_query_instruction(text) for text in batch_texts]
        #             response = emb_client.embeddings.create(input=batch_texts, model="Qwen/Qwen3-Embedding-8B")
        #             batch_embeddings = [data.embedding for data in response.data]

        #             response = emb_client.embeddings.create(input=batch_query_texts, model="Qwen/Qwen3-Embedding-8B")
        #             batch_query_embeddings = [data.embedding for data in response.data]

        #             doc_embeddings.extend(batch_embeddings)
        #             for text, query_emb in zip(batch_texts, batch_query_embeddings):
        #                 doc_hash = hashlib.sha256(text.encode()).hexdigest()
        #                 doc_hash_to_query_answer_embedding[doc_hash] = (query_emb)

        #         # save it as np array
        #         doc_embeddings = np.array(doc_embeddings)
        #         assert len(doc_embeddings) == len(all_doc_list), "Number of embeddings must match number of documents"
        #         assert len(doc_hash_to_query_answer_embedding) == len(all_doc_list), "Number of doc_hash_to_query_answer_embedding must match number of documents"
        #         # save to local file
        #         with open(doc_list_path, "wb") as f:
        #             pickle.dump(all_doc_list, f)
        #         with open(doc_embs_path, "wb") as f:
        #             pickle.dump(doc_embeddings, f)
        #         with open(doc_qa_emb_dict_path, "wb") as f:
        #             pickle.dump(doc_hash_to_query_answer_embedding, f)
        #         print(f"Saved {len(all_doc_list)} documents and embeddings to cache for {ds_name}")
        #     df = pd.DataFrame(ds)
        #     if ds_name == "hotpotqa":
        #         process_single_row = process_hotpotqa_single_row
        #     elif ds_name == "musique":
        #         process_single_row = process_musique_single_row
        #     for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {ds_name} for mixed dataset"):
        #         processed = await process_single_row(row, "train", idx, mcq_dict if args.generate_mcq else None, 
        #                                              doc_list=all_doc_list, doc_embeddings=doc_embeddings, doc_hash_to_query_answer_embedding=doc_hash_to_query_answer_embedding)
        #         processed_rows.append(processed)
       
        # output_file_path = os.path.join(local_save_dir, f"mixed_hotpot_musique_train_doc_size_{args.doc_size}_distract_{args.include_distractor}.parquet")
        # df_processed = pd.DataFrame(processed_rows)
        # df_processed.to_parquet(output_file_path, index=False)
        # print(f"Saved {len(df_processed)} processed rows to {output_file_path}")
        
        # loop  through valid dataset
        processed_rows = []
        musique_valid = load_dataset("dgslibisey/MuSiQue", split="validation")
        hotpotqa_valid = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
        sample_size = min(len(musique_valid), len(hotpotqa_valid))
        # sample hotpotqa
        hotpotqa_valid = hotpotqa_valid.select(range(sample_size))
        for ds, ds_name in zip([musique_valid, hotpotqa_valid], ["musique", "hotpotqa"]):
            # check if "/data/autograph/data/train_{ds_name}_doc_list.pickle" and "/data/autograph/data/train_{ds_name}_doc_embs.pickle" exist or not
            doc_list_path = f'/data/autograph/data/valid_{ds_name}_doc_list.pickle'
            doc_embs_path = f'/data/autograph/data/valid_{ds_name}_doc_embs.pickle'
            doc_qa_emb_dict_path = f'/data/autograph/data/valid_{ds_name}_doc_qa_emb_dict.pickle'
            if os.path.exists(doc_list_path) and os.path.exists(doc_embs_path) and os.path.exists(doc_qa_emb_dict_path):
                with open(doc_list_path, "rb") as f:
                    all_doc_list = pickle.load(f)
                with open(doc_embs_path, "rb") as f:
                    doc_embeddings = pickle.load(f)
                with open(doc_qa_emb_dict_path, "rb") as f:
                    doc_hash_to_query_answer_embedding = pickle.load(f)
                print(f"Loaded {len(all_doc_list)} documents and embeddings from cache for {ds_name}")
            else:
                df = pd.DataFrame(ds)
                all_doc_set = set()
                for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Collecting all documents"):
                    if ds == musique_valid:
                        get_all_musique_docs(row, all_doc_set)
                    elif ds == hotpotqa_valid:
                        get_all_hotpotqa_docs(row, all_doc_set)
                print(f"Total unique documents: {len(all_doc_set)}")
                all_doc_list = list(all_doc_set)
                # generate embedding for each document
                batch_size = 128
                doc_embeddings = []
                doc_hash_to_query_answer_embedding = {}
                for i in tqdm(range(0, len(all_doc_list), batch_size), desc="Generating document embeddings"):
                    batch_texts = all_doc_list[i:i+batch_size]
                    batch_query_texts = [get_query_instruction(text) for text in batch_texts]
                    response = emb_client.embeddings.create(input=batch_texts, model="Qwen/Qwen3-Embedding-8B")
                    batch_embeddings = [data.embedding for data in response.data]

                    response = emb_client.embeddings.create(input=batch_query_texts, model="Qwen/Qwen3-Embedding-8B")
                    batch_query_embeddings = [data.embedding for data in response.data]

                    doc_embeddings.extend(batch_embeddings)
                    for text, query_emb in zip(batch_texts, batch_query_embeddings):
                        doc_hash = hashlib.sha256(text.encode()).hexdigest()
                        doc_hash_to_query_answer_embedding[doc_hash] = (query_emb)
                doc_embeddings = np.array(doc_embeddings)
                with open(doc_list_path, "wb") as f:
                    pickle.dump(all_doc_list, f)
                with open(doc_embs_path, "wb") as f:
                    pickle.dump(doc_embeddings, f)
                with open(doc_qa_emb_dict_path, "wb") as f:
                    pickle.dump(doc_hash_to_query_answer_embedding, f)
            df = pd.DataFrame(ds)
            if ds_name == "hotpotqa":
                process_single_row = process_hotpotqa_single_row
            elif ds_name == "musique":
                process_single_row = process_musique_single_row
            for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {ds_name} for mixed dataset"):
                processed = await process_single_row(row, "valid", idx, mcq_dict if args.generate_mcq else None, doc_list=all_doc_list, doc_embeddings=doc_embeddings, doc_hash_to_query_answer_embedding=doc_hash_to_query_answer_embedding)
                processed_rows.append(processed)
        output_file_path = os.path.join(local_save_dir, f"mixed_hotpot_musique_valid_doc_size_{args.doc_size}_distract_{args.include_distractor}.parquet")
        df_processed = pd.DataFrame(processed_rows)
        df_processed.to_parquet(output_file_path, index=False)
        print(f"Saved {len(df_processed)} processed rows to {output_file_path}")
if __name__ == "__main__":
    from openai import AsyncOpenAI
    openai_client = AsyncOpenAI(base_url="http://0.0.0.0:8129/v1", api_key="EMPTY")
    llm_generator = LLMGenerator(openai_client, model_name="Qwen/Qwen2.5-7B-Instruct")

    emb_client = OpenAI(base_url="http://0.0.0.0:8128/v1", api_key="EMPTY")
    asyncio.run(main())
