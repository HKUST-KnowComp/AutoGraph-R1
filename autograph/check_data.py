# read parquet files and check data quality
import pandas as pd
import os

# get all files end with .parquet in data/autograph/data
data_files = [f for f in os.listdir("/data/autograph/data") if f.endswith(".parquet")]
data_files = [f for f in data_files if 'text_linking' not in f]

print(f"Found {len(data_files)} parquet files.")
# get targeted doc size x in the file name it contains 'doc_size_x'
doc_size_list = [int(f.split('doc_size_')[1].split('_')[0]) for f in data_files if 'doc_size_' in f]
print(f"Found {len(doc_size_list)} doc size x values.")
print(f"All doc name: {data_files}")
print(f"Doc size x values: {doc_size_list}")
max_prompt_length = 0
for targeted_doc_size, doc_file_name in zip(doc_size_list, data_files):
    df = pd.read_parquet(os.path.join("/data/autograph/data", doc_file_name))
    print(f"Checking file: {doc_file_name} with targeted doc size: {targeted_doc_size}")
    for idx, row in df.iterrows():
        
        prompts = row['prompt']
        prompt_length = len(prompts[0]['content']) + len(prompts[1]['content'])
        if prompt_length > max_prompt_length:
            max_prompt_length = prompt_length
print(f"Max prompt length across all files: {max_prompt_length}")