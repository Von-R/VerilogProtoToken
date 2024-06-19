"""
EDA and Cleanup on Processed Verilog Dataset

This script performs exploratory data analysis (EDA) and cleanup on a dataset of processed Verilog files. The main
objective is to identify and remove files with an excessive number of distinct variables, which can bloat the vocabulary
and complicate model training. Specifically, files with more than 100 distinct variables are flagged for removal. The 
script performs the following steps:

1. Loads the dataset from the Hugging Face Hub.
2. Analyzes the distribution of variables to identify files with an excessive number of distinct variables.
3. Flags files for removal based on a variable-to-content ratio and the presence of large variable names.
4. Removes flagged files from the dataset.
5. Uploads the cleaned dataset back to the Hugging Face Hub.
"""

from typing import List, Tuple, Any
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict
from huggingface_hub import HfApi

# Initialize the Hugging Face API and load the dataset
hf_api = HfApi()
read_token_file = "read_token.txt"
with open(read_token_file, "r") as file:
    hf_read_token = file.read().strip()
    
dataset = load_dataset("Von-R/preprocessed_anonymized", token=hf_read_token)
initial_number_of_examples = len(dataset['train']) + len(dataset['test']) + len(dataset['validation'])

import re
from collections import Counter

# Combine all content from the dataset for analysis
all_texts_list = []
for split in ['train', 'test', 'validation']:
    all_texts_list.extend(dataset[split]['content'])

# Define a regex pattern to identify variables
var_pattern = re.compile(r'\bVAR[0-9]+\b')

# Find all variable matches in the combined content
matches = var_pattern.findall(' '.join(dataset['train']['content']))

# Print the total and distinct number of variables
print("Total VAR's: ", len(matches))
print("Distinct VAR's: ", len(set(matches)))

# Count the occurrences of each variable
var_counter = Counter(matches)

# Define a threshold for maximum occurrences of a variable
max_occurrences = 100
less_frequent_items = [item for item, count in var_counter.items() if count < max_occurrences]
print("Less frequent items: ", less_frequent_items[-10:])

# Print the most and least common variables
print(var_counter.most_common(10))
least_common = var_counter.most_common()[-100:]
print("Least common: ", least_common)
print("test: ", least_common[0][0])

# Initialize counters for large variable patterns
big_var_count = 0
big_var_pattern = re.compile(r'\bVAR1[0-9]{2,}\b')

_90_count = 0
_80_count = 0
_70_count = 0
_60_count = 0
_50_count = 0

# Function to initialize the 'remove' flag in each example
def set_remove_flag(example):
    example['remove'] = False
    return example

# Apply the 'remove' flag initialization to all splits
for split in ['train', 'test', 'validation']:
    dataset[split] = dataset[split].map(set_remove_flag, batched=False)

# Function to compute the variable-to-content ratio and flag examples for removal based on thresholds
def var_to_content_ratio(dataset):
    global _90_count, _80_count, _70_count, _60_count, _50_count
    for split in ['train', 'test', 'validation']:
        for example in dataset[split]:
            content = example.get('content', '')  # Default to empty string if 'content' key is missing
            if example['content'] is None or example['content'] == '':
                example['remove'] = True
            var_matches = var_pattern.findall(content)
            accumulator = 0
            for var_match in var_matches:
                accumulator += len(var_match)
            content_len = len(content)

            if content_len > 0:  # Ensure you don't divide by zero
                ratio = accumulator / content_len
                if ratio > 0.9:
                    example['remove'] = True
                    _90_count += 1
                elif ratio > 0.8:
                    example['remove'] = True
                    _80_count += 1
                elif ratio > 0.7:
                    example['remove'] = True
                    _70_count += 1
                elif ratio > 0.6:
                    example['remove'] = True
                    _60_count += 1
                elif ratio > 0.49:
                    example['remove'] = True
                    _50_count += 1

    print("Percentile counts: ", _90_count, _80_count, _70_count, _60_count, _50_count)

    return dataset

# Function to flag examples containing large variable names for removal
def big_var(dataset, big_var_pattern, big_var_count=0):
    modified_dataset = {}
    for split in ['train', 'test', 'validation']:
        modified_dataset[split] = []
        for example in dataset[split]:
            content = example.get('content', '')
            if big_var_pattern.findall(content):
                big_var_count += 1
                example['remove'] = True
            modified_dataset[split].append(example)
    print("Big VAR count: ", big_var_count)
    return modified_dataset

# Function to remove flagged examples from the dataset
def remove_examples(dataset):
    remove_counter = 0
    removed_list = []
    for split in ['train', 'test', 'validation']:
        print(f"Initial number of examples in split \"{split}\": ", len(dataset[split]))
        flag_list = [example['remove'] for example in dataset[split]]
        if True not in flag_list:
            print(f"No examples to remove in split \"{split}\"")
            continue
        new_dataset = []
        for example in dataset[split]:
            if error_match := check_and_print_example(example) and type(error_match) == KeyError:
                print("Error occurred in example: ", error_match)
                exit(-1)
            if not check_and_print_example(example) and example['remove'] is False:
                new_dataset.append(example)
            elif check_and_print_example(example) and example['remove'] is True:
                remove_counter += 1
                removed_list.append(example['path'])

        dataset[split] = new_dataset
        print(f"Remaining examples in split \"{split}\": ", len(dataset[split]))
    print("Removed examples: ", remove_counter)
    if remove_counter > 0:
        print("Removed examples: ", removed_list)

    print("Remaining examples: ", len(dataset['train']) + len(dataset['test']) + len(dataset['validation']))
    return dataset

# Function to check if an example contains the 'remove' key and handle KeyErrors
def check_and_print_example(example):
    try:
        return example.get('remove')
    except KeyError as e:
        print(f"Example \"{example['path']}\" does not contain 'remove' key")
        print(f"Example: {example['content']}")
        return e

# Function to upload the dataset to the Hugging Face Hub
def upload_to_hub(dataset: Any, dataset_name: str, token: str) -> None:
    dataset.push_to_hub(dataset_name, token=token)

# Apply the variable-to-content ratio check
dataset = var_to_content_ratio(dataset)

# Apply the large variable check
dataset = big_var(dataset, big_var_pattern, big_var_count)

# Print the number of examples flagged for removal
num_remove_flags = sum([example['remove'] for example in dataset['train']])
print("Examples to be removed: ", num_remove_flags)

# Remove flagged examples from the dataset
dataset = remove_examples(dataset)

final_number_of_examples = len(dataset['train']) + len(dataset['test']) + len(dataset['validation'])
print("Examples in dataset: ", final_number_of_examples)
print("Examples removed: ", initial_number_of_examples - final_number_of_examples)
for key, value in dataset.items():
    print(f"Length of column {key}: {len(value)}")

# Convert the list of dictionaries to a dictionary of lists (expected format for Dataset)
def list_of_dicts_to_dict_of_lists(data):
    reformatted_data = {}
    for item in data:
        for key, value in item.items():
            if key in reformatted_data:
                reformatted_data[key].append(value)
            else:
                reformatted_data[key] = [value]
    return reformatted_data

# Convert each split to the appropriate format
train_data = list_of_dicts_to_dict_of_lists(dataset['train'])
test_data = list_of_dicts_to_dict_of_lists(dataset['test'])
validation_data = list_of_dicts_to_dict_of_lists(dataset['validation'])

# Create Dataset objects from the converted data
train_dataset = Dataset.from_dict(train_data)
test_dataset = Dataset.from_dict(test_data)
validation_dataset = Dataset.from_dict(validation_data)

# Combine into a DatasetDict
dataset_dict = DatasetDict({
    'train': train_dataset,
    'test': test_dataset,
    'validation': validation_dataset
})

# Remove the 'remove' column from each split
for split in dataset_dict:
    dataset_dict[split] = dataset_dict[split].remove_columns('remove')

# Set the token for authentication and push to the Hugging Face Hub
write_token_file = "write_token.txt"
with open(write_token_file, "r") as file:
    hf_write_token = file.read().strip()
from huggingface_hub import HfFolder
HfFolder.save_token(hf_write_token)
dataset_dict.push_to_hub('Von-R/test')
