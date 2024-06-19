"""
This script performs deduplication on a dataset of Verilog files. The initial dataset contains a large number of duplicates, 
which need to be removed to improve the quality and usability of the dataset. The script loads the dataset, removes rows with 
duplicate content, calculates the hash of the content to further ensure uniqueness, and then pushes the deduplicated dataset 
back to the Hugging Face Hub.

Steps:
1. Load the dataset from the Hugging Face Hub.
2. Remove rows with None content.
3. Remove duplicate rows based on the content.
4. Create a hash of the content and remove any remaining duplicates.
5. Push the deduplicated dataset back to the Hugging Face Hub.
"""

import hashlib
from datasets import load_dataset, Dataset
from huggingface_hub import HfApi

# Read the Hugging Face API read token from file
read_token_file = "read_token.txt"
with open(read_token_file, "r") as file:
    hf_token = file.read().strip()

# Initialize the Hugging Face API and set the token
hf_api = HfApi()
hf_api.token = hf_token

# Specify the cache directory path for the dataset
cache_dir = "./cache"

# Load the dataset to be deduplicated from the Hugging Face Hub
dataset = load_dataset("Von-R/verilog_unprocessed", token=hf_token, cache_dir=cache_dir)

def deduplicate(dataset):
    df = dataset['train'].to_pandas()

    # Drop rows with None content
    b4_len = df.shape[0]
    df = df.dropna(subset=['content'])
    after_len = df.shape[0]
    print("Number of None content rows dropped: ", b4_len - after_len)

    print("Rows before deduplication: ", df.shape[0])
    # Remove duplicate rows based on the 'content' column
    dedup_dataset_df = df.drop_duplicates(subset=['content'])
    print("Rows after deduplication: ", dedup_dataset_df.shape[0])

    # Create a new column 'content_hash' with the hash value of the 'content' column
    dedup_dataset_df = dedup_dataset_df.copy()
    print("Rows before hash deduplication: ", dedup_dataset_df.shape[0])
    dedup_dataset_df.loc[:, 'content_hash'] = dedup_dataset_df['content'].apply(
        lambda x: hashlib.sha256(x.encode()).hexdigest())

    # Drop duplicate rows based on the 'content_hash' column
    dedup_dataset_df = dedup_dataset_df.drop_duplicates(subset='content_hash', keep='first')
    print("Rows after hash deduplication: ", dedup_dataset_df.shape[0])

    # Drop the 'content_hash' column, as it is no longer needed
    dedup_dataset_df.drop('content_hash', axis=1, inplace=True)
    # Reset the index of the DataFrame to remove the old index
    dedup_dataset_df.reset_index(drop=True, inplace=True)

    # Convert the deduplicated DataFrame back to a Dataset object
    deduplicated_dataset = Dataset.from_pandas(dedup_dataset_df)

    # Print EDA (Exploratory Data Analysis) information
    print("Original dataset_with_splits shape:", dataset['train'].num_rows)
    print("Deduplicated dataset shape:", deduplicated_dataset.num_rows)
    print("Reduction of {:.2f}%".format(100 * (1 - (deduplicated_dataset.num_rows / dataset['train'].num_rows))))

    return deduplicated_dataset

# Perform deduplication on the dataset
deduplicated_dataset = deduplicate(dataset)

# Print size-related EDA information
dedup_size = sum(deduplicated_dataset['size'])
print("Size of deduplicated dataset: ", dedup_size)
print("Size of original dataset: ", sum(dataset['train']['size']))
print("Size reduction: ", sum(dataset['train']['size']) - dedup_size)
print("Size reduction percentage: ", (sum(dataset['train']['size']) - dedup_size) / sum(dataset['train']['size']) * 100)

# Read the Hugging Face API write token from file
write_token_file = "write_token.txt"
with open(write_token_file, "r") as file:
    hf_token = file.read().strip()

# Initialize the Hugging Face API and set the token for writing
hf_api.token = hf_token

# Push the deduplicated dataset back to the Hugging Face Hub
# Change the dataset name and token as required
deduplicated_dataset.push_to_hub('Von-R/test', token=hf_api.token)
