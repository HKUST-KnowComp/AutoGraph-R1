import pickle
import json
import os
import tqdm

# read wiki title set from pickle
with open("final_wiki_titles.pkl", "rb") as f:
    final_wiki_titles = pickle.load(f)

print(f"Final wiki titles size: {len(final_wiki_titles)}")

# ensure all titles include the qa titles
# read nq_qa_pairs from nq_validation_1000.json
nq_qa_pairs = []
with open("nq/nq_validation_1000.json", "r") as f:
    for line in f:
        nq_qa_pairs.append(json.loads(line))
print(f"NQ qa pairs size: {len(nq_qa_pairs)}")
# read popqa_qa_pairs from popqa_test_1000.json
popqa_qa_pairs = []
with open("popqa/popqa_test_1000.json", "r") as f:
    for line in f:
        popqa_qa_pairs.append(json.loads(line))
print(f"PopQA qa pairs size: {len(popqa_qa_pairs)}")

qa_titles = set()
for item in nq_qa_pairs:
    qa_titles.add(item['title'])
for item in popqa_qa_pairs:
    for t in item['title']:
        qa_titles.add(t)
print(f"QA titles size: {len(qa_titles)}")
for title in qa_titles:
    if title not in final_wiki_titles:
        print(f"Title {title} from QA pairs not in final wiki titles")
    
print(f"All QA titles are included in the final wiki titles set. Docs size: {len(final_wiki_titles)}")

# read from /data/httsangaj/autograph/corpora/wiki/enwiki-dec2021/text-list-100-sec-merged.jsonl to get the text 
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
# and write to wiki_3000_corpus_kg_input.jsonl
original_docs_path = "/data/httsangaj/autograph/corpora/wiki/enwiki-dec2021/text-list-100-sec-merged.jsonl"
output_path = "2021wiki_corpus_kg_input.jsonl"
# first create index for original docs
original_title_docs = {}
with open(original_docs_path, "r") as f:
    for i, line in tqdm.tqdm(enumerate(f)):
        doc = json.loads(line)
        title = doc['title']
        if title in final_wiki_titles:
            original_title_docs[title] = doc['text']

print(f"Original docs index size: {len(original_title_docs)}")
# print the set of titles that are in final_wiki_titles but not in original_title_docs
missing_titles = final_wiki_titles - set(original_title_docs.keys())
for t in missing_titles:
    print(f"Missing title: {t}")
assert len(original_title_docs) == len(final_wiki_titles), f"Original docs index size {len(original_title_docs)} does not match final wiki titles size {len(final_wiki_titles)}"

# loop through final_wiki_titles and get the docs
with open(output_path, "w") as out_f:
    for title, sentences in original_title_docs.items():
        out_f.write(json.dumps({
            "metadata": {"lang": "en"},
            "id": title,
            "text": sentences
        }, ensure_ascii=False) + "\n")

# loop through final_wiki_titles to generate popqa.json and nq.json for evaluation

# nq_output_path = "nq.json"
# popqa_output_path = "popqa.json"

# # output as list of json objects
# nq_list = []
# for item in nq_qa_pairs:
#     nq_list.append({
#         "question": item['question'],
#         "answer": item['answer'][0] if isinstance(item['answer'], list) else item['answer'],
#         "supporting_facts": [[item['title'], 0]], # sentence index is 0 as we do not have sentence level info
#         "context": [
#             [item['title'], [original_title_docs[item['title']]]]
#         ]
#     })
# with open(nq_output_path, "w") as f:
#     json.dump(nq_list, f, ensure_ascii=False, indent=2)
# print(f"NQ output size: {len(nq_list)}")

# popqa_list = []
# for item in popqa_qa_pairs:
#     context = []
#     for t in item['title'][:1]:
#         context.append([t, [original_title_docs[t]]])
#     popqa_list.append({
#         "question": item['question'],
#         "answer": item['answer'],
#         "supporting_facts": [[t, 0] for t in item['title']], # sentence index is 0 as we do not have sentence level info
#         "context": context
#     })
# with open(popqa_output_path, "w") as f:
#     json.dump(popqa_list, f, ensure_ascii=False, indent=2)
# print(f"PopQA output size: {len(popqa_list)}")
