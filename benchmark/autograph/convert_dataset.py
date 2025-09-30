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
from tqdm.asyncio import tqdm_asyncio
import json
import re
# import library for hashing the tuple
from hashlib import sha256
import json_repair


parser = argparse.ArgumentParser(description="Download QA dataset from HuggingFace, process, and save to Parquet.")
parser.add_argument("--local_dir", default="/home/httsangaj/projects/AutoSchemaKG/benchmark_data/autograph", help="Local directory to save the processed Parquet files.")
parser.add_argument("--include_distractor", action="store_true", help="Create 'distractor' version of the dataset.")
parser.add_argument("--doc_size", default=10, type=int, help="Number of documents to be included per sample")
parser.add_argument("--dataset", default="akariasai/PopQA" , choices=["sentence-transformers/natural-questions", "akariasai/PopQA"], help="Dataset to process.")

args = parser.parse_args()
in_domain_data_path = os.path.join(args.local_dir, "2wikimultihopqa")
# # Get original wiki size
# original_wiki_titles = set()
# with open(os.path.join(in_domain_data_path, "2wikimultihopqa_corpus_kg_input.json"), "r") as f:
#     for line in f:
#         item = json.loads(line)
#         original_wiki_titles.add(item.get("id", ""))
# print(f"Loaded {len(original_wiki_titles)} wiki titles from original data.") # 6119

# # Get wiki title dataset
# wiki_titles = set()
# wiki_dataset = load_dataset("framolfese/2WikiMultihopQA")
# def get_title_wiki_dataset_single_row(row, row_index):
#     title = row['context']['title']
#     return title
# df_wiki = pd.DataFrame(wiki_dataset['validation'])
# for idx, row in tqdm(df_wiki.iterrows(), total=len(df_wiki), desc=f"Processing {'validation'}"):
#     processed = get_title_wiki_dataset_single_row(row, idx)
#     wiki_titles.update(processed)
# print(f"Loaded {len(wiki_titles)} wiki titles from in-domain data.")

# # Get PopQA wiki titles
# popqa_titles = set()
# def get_title_pop_qa_single_row(row, row_index):
#     title_list = []
#     s_title = row.get("s_wiki_title", [])
#     o_title = row.get("o_wiki_title", [])
#     title_list.extend([s_title, o_title])
#     return title_list
# pop_qa_dataset = load_dataset("akariasai/PopQA")
# df_raw = pd.DataFrame(pop_qa_dataset['test'])
# for idx, row in tqdm(df_raw.iterrows(), total=len(df_raw), desc=f"Processing {'test'}"):
#     processed = get_title_pop_qa_single_row(row, idx)
#     popqa_titles.update(processed)
# print(f"Loaded {len(popqa_titles)} wiki titles from in-domain data.")
# print(f"Pop QA intersection size: {len(original_wiki_titles.intersection(popqa_titles))}")

# dataset_name = "popqa"
# local_save_dir = os.path.join(args.local_dir, f"{dataset_name}")
# os.makedirs(local_save_dir, exist_ok=True)  # Create directory if it doesn't exist

# # Get NQ dataset
# nq_titles = set()
# nq_dataset = load_dataset("google-research-datasets/natural_questions", split="validation")
# def get_title_nq_single_row(row, row_index):
#     answer = row['annotations']['short_answers'][0]['text'] if row['annotations']['short_answers'] else ""
#     if answer == "":
#         return False, {}
#     return True, {
#     "title": row["document"]["title"],
#     "question": row["question"]["text"],
#     "answer": answer,
#     }
# df_nq = pd.DataFrame(nq_dataset)
# short_answer_count = 0
# for idx, row in tqdm(df_nq.iterrows(), total=len(df_nq), desc=f"Processing {'validation'}"):
#     has_answer, processed = get_title_nq_single_row(row, idx)
#     if has_answer:
#         print(processed['title'])
#         short_answer_count += 1
#         nq_titles.update(processed['title'])
# print(f"Loaded {len(nq_titles)} wiki titles from in-domain data.")
# print(f"NQ short answer count: {short_answer_count}") # 1000 / 3610
# print(f"Pop QA and NQ intersection size: {len(popqa_titles.intersection(nq_titles))}")
import pickle
wiki_2021_dataset_titles = pickle.load(open("/data/httsangaj/autograph/corpora/wiki/enwiki-dec2021/title-set-100-sec.pkl", "rb"))
# print(f"Wiki 2021 titles: {len(wiki_2021_dataset_titles)}")
# print(f"Pop QA and Wiki 2021 intersection size: {len(popqa_titles.intersection(wiki_2021_dataset_titles))}") # 19134
# print(f"NQ and Wiki 2021 intersection size: {len(nq_titles.intersection(wiki_2021_dataset_titles))}") # 38

'''
Required fields for benchmark dataset:
question: The question text
answer: The answer text
supporting_facts: List of supporting facts (if available)
    - [0] : title of the document
    - [1] : sentence index in the document
context: The list of all documents (title and text)
    each object in the list is:
    - title: title of the document
    - List of sentences in the document
'''
# Use the NQ and Wiki 2021 intersection and PopQA and Wiki 2021 intersection as union set of titles for generating the corpus
# first generate the union set, then generate qa pairs 1000 for NQ and 1000 for PopQA, then get the title set from the sampled qa pairs and finally use the title set to filter the corpus
# final_wiki_titles = set()
# filtered_nq_titles = nq_titles.intersection(wiki_2021_dataset_titles)
# filtered_popqa_titles = popqa_titles.intersection(wiki_2021_dataset_titles)

# Get NQ dataset
# nq_dataset = load_dataset("google-research-datasets/natural_questions", split="validation")
# def process_title_nq_single_row(row, row_index):
#     answer = row['annotations']['short_answers'][0]['text'] if row['annotations']['short_answers'] else ""
#     title = row["document"]["title"]
#     if title not in wiki_2021_dataset_titles:
#         return False, {}
#     if answer == "" or answer is None or len(answer) == 0:
#         return False, {}
#     print(title)
#     return True, {
#     "title": row["document"]["title"],
#     "question": row["question"]["text"],
#     "answer": answer,
#     }
# df_nq = pd.DataFrame(nq_dataset)
# nq_qa_pairs = []
# for idx, row in tqdm(df_nq.iterrows(), total=len(df_nq), desc=f"Processing {'validation'}"):
#     has_answer, processed = process_title_nq_single_row(row, idx)
#     if has_answer:
#         nq_qa_pairs.append(processed) 
# print(f"NQ short answer count after filtering with wiki 2021 titles: {len(nq_qa_pairs)}") 

# # Get PopQA dataset
# pop_qa_dataset = load_dataset("akariasai/PopQA")
# def process_title_pop_qa_single_row(row, row_index):
#     title_list = []
#     s_title = row.get("s_wiki_title", "")
#     o_title = row.get("o_wiki_title", "")
#     if s_title in wiki_2021_dataset_titles and o_title in wiki_2021_dataset_titles:
#         answers = row.get("possible_answers", [])
#         answers = json.loads(answers) if isinstance(answers, str) else answers
#         answer = answers[0] if len(answers) > 0 else ""
#         return True, {
#             "title": [s_title, o_title],
#             "question": row.get("question", ""),
#             "answer": answer,
#         }
#     return False, {}
# df_raw = pd.DataFrame(pop_qa_dataset['test'])
# popqa_qa_pairs = []
# for idx, row in tqdm(df_raw.iterrows(), total=len(df_raw), desc=f"Processing {'test'}"):
#     has_answer, processed = process_title_pop_qa_single_row(row, idx)
#     if has_answer:
#         popqa_qa_pairs.append(processed) 
# print(f"PopQA count after filtering with wiki 2021 titles: {len(popqa_qa_pairs)}")

# export the filtered qa pairs
dataset_name = "nq"
# local_save_dir = os.path.join(args.local_dir, f"{dataset_name}")
# os.makedirs(local_save_dir, exist_ok=True)  # Create directory if it doesn't exist
# df_nq_filtered = pd.DataFrame(nq_qa_pairs)
# # sample 1000 from nq_qa_pairs and save it as json
# nq_qa_pairs_sampled = random.sample(nq_qa_pairs, min(1000, len(nq_qa_pairs)))
# with open(os.path.join(local_save_dir, f"{dataset_name}_validation_1000.json"), "w") as f:
#     for item in nq_qa_pairs_sampled:
#         f.write(json.dumps(item, ensure_ascii=False) + "\n")
nq_qa_pairs_sampled = []
with open("/home/httsangaj/projects/AutoSchemaKG/benchmark_data/autograph/nq/nq_validation_1000.json", "r") as f:
    for item in f:
        nq_qa_pairs_sampled.append(json.loads(item))
print(f"NQ sampled count: {len(nq_qa_pairs_sampled)}")

# sample 1000 from popqa_qa_pairs and save it as json
dataset_name = "popqa"
# local_save_dir = os.path.join(args.local_dir, f"{dataset_name}")
# os.makedirs(local_save_dir, exist_ok=True)  # Create directory if it doesn't exist
# popqa_qa_pairs_sampled = random.sample(popqa_qa_pairs, min(1000, len(popqa_qa_pairs)))
# with open(os.path.join(local_save_dir, f"{dataset_name}_test_1000.json"), "w") as f:
#     for item in popqa_qa_pairs_sampled:
#         f.write(json.dumps(item, ensure_ascii=False) + "\n")
popqa_qa_pairs_sampled = []
with open("/home/httsangaj/projects/AutoSchemaKG/benchmark_data/autograph/popqa/popqa_test_1000.json", "r") as f:
    for item in f:
        popqa_qa_pairs_sampled.append(json.loads(item))
print(f"PopQA sampled count: {len(popqa_qa_pairs_sampled)}")


# get the final title set
final_wiki_titles = set()
for item in nq_qa_pairs_sampled:
    final_wiki_titles.add(item['title'])
for item in popqa_qa_pairs_sampled:
    for t in item['title']:
        final_wiki_titles.add(t)
print(f"Final wiki titles size before adding more: {len(final_wiki_titles)}")

# then fill the titles set up to 6000 with random titles from wiki_2021_dataset_titles
# shuffle the order of wiki_2021_dataset_titles then use for loop to add
wiki_2021_dataset_titles = list(wiki_2021_dataset_titles)
random.shuffle(wiki_2021_dataset_titles)
for title in tqdm(wiki_2021_dataset_titles):
    if len(final_wiki_titles) >= 7000:
        break
    final_wiki_titles.add(title)
print(f"Final wiki titles size: {len(final_wiki_titles)}")
# save it as pickle
pickle.dump(final_wiki_titles, open(os.path.join(args.local_dir, "final_wiki_titles.pkl"), "wb"))

