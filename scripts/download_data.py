# scripts/download_dataset.py
import os
import argparse
from datasets import load_dataset

def download_and_save_dataset(repo_id: str, output_path: str):
    """
    Downloads a dataset from the Hugging Face Hub and saves its splits
    as Parquet files to a specified local directory.

    Args:
        repo_id (str): The ID of the repository on the Hugging Face Hub 
                       (e.g., 'username/dataset_name').
        output_path (str): The local directory where the Parquet files will be saved.
    """
    print(f"Downloading dataset '{repo_id}' from the Hugging Face Hub...")
    
    # Load the dataset from the Hub
    try:
        dataset_dict = load_dataset(repo_id)
    except Exception as e:
        print(f"Error: Failed to load dataset '{repo_id}'.")
        print(f"Please ensure the repository exists and you have the necessary permissions.")
        print(f"Original error: {e}")
        return

    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)
    print(f"Output directory '{output_path}' is ready.")

    # Iterate through the splits (e.g., 'train', 'validation') and save them
    for split_name, dataset in dataset_dict.items():
        file_path = os.path.join(output_path, f"{split_name}.parquet")
        print(f"Saving '{split_name}' split to '{file_path}'...")
        try:
            dataset.to_parquet(file_path)
            print(f"Successfully saved {split_name} split.")
        except Exception as e:
            print(f"Error: Failed to save '{split_name}' split to Parquet.")
            print(f"Original error: {e}")
    
    print("\nDataset download and preparation complete.")
    print(f"Files saved in: {os.path.abspath(output_path)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download a dataset from Hugging Face and save as Parquet files."
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="gzone0111/musique_hotpotqa_graph_retriever",
        help="The repository ID of the dataset on the Hugging Face Hub."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./data",
        help="The local directory to save the Parquet files."
    )
    
    args = parser.parse_args()
    download_and_save_dataset(args.repo_id, args.output_path)