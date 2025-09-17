# read parquet files and check data quality
import pandas as pd
import os
import re

def extract_docs(content: str):
    """
    Extract docs in the format:
    Document 1: ...
    Document 2: ...
    Document N: ...
    """
    # Split by "Document X:" but keep the doc number and text
    matches = re.split(r"Document\s+(\d+):", content)
    docs = {}
    # re.split gives ['', '1', ' text...', '2', ' text...', ...]
    for i in range(1, len(matches), 2):
        doc_num = int(matches[i])
        doc_text = matches[i + 1].strip()
        docs[doc_num] = doc_text
    return docs
data_dir = "/data/autograph/data"
# get all files end with .parquet in data/autograph/data
data_files = [f for f in os.listdir(data_dir) if f.endswith(".parquet")]
data_files = [f for f in data_files if 'mixed' in f and "_iterate" not in f]

print(f"Found {len(data_files)} parquet files.")
# get targeted doc size x in the file name it contains 'doc_size_x'
doc_size_list = [int(f.split('doc_size_')[1].split('_')[0]) for f in data_files if 'doc_size_' in f]
print(f"Found {len(doc_size_list)} doc size x values.")
print(f"All doc name: {data_files}")
print(f"Doc size x values: {doc_size_list}")
max_prompt_length = 0
for targeted_doc_size, doc_file_name in zip(doc_size_list, data_files):
    df = pd.read_parquet(os.path.join(data_dir, doc_file_name))
    print(f"Checking file: {doc_file_name} with targeted doc size: {targeted_doc_size}")

    new_rows = []
    for idx, row in df.iterrows():
        prompts = row["prompt"]
        assert prompts[1]["role"] == "user", f"prompts[1]['role'] is {prompts[1]['role']}, expected 'user'"

        docs = extract_docs(prompts[1]["content"])
        # Only keep Document 1
        if 1 in docs:
            prompts[1]["content"] = f"Extracts for {docs[1]}"
        else:
            print(f"Warning: Document 1 not found in row {idx}")
            prompts[1]["content"] = "Document 1: "

        # assign the prompt
        row["prompt"] = prompts
        # Append modified row
        new_rows.append(row)

    # Create new DataFrame
    new_df = pd.DataFrame(new_rows)

    # Save as new parquet with `_iterate.parquet` suffix
    out_file = os.path.join(data_dir, doc_file_name.replace(".parquet", "_iterate.parquet"))
    new_df.to_parquet(out_file, index=False)
    print(f"Saved transformed file: {out_file}")

print(f"Max prompt length across all files: {max_prompt_length}")