"""
Stage 2: Anonymizing Identifiers in Verilog Code

This script processes a dataset of Verilog files to anonymize the identifiers. The goal is to reduce the size of the
vocabulary and simplify the code for training, while preserving semantically meaningful information. Exceptions are
made for meaningful variables and port identifiers. The script performs the following steps:

1. Loads the dataset from the Hugging Face Hub.
2. Anonymizes identifiers while preserving semantically meaningful identifiers.
3. Applies the anonymization function to the dataset.
4. Optionally saves a sample of the anonymized dataset for comparison.
5. Splits the anonymized dataset into train, test, and validation sets.
6. Saves and pushes the anonymized dataset back to the Hugging Face Hub.
"""

import math
import random
import re
import sys

# Ensure stdout and stderr are set correctly
if not sys.__stdout__.closed:
    sys.stdout = sys.__stdout__
else:
    print("sys.__stdout__ is closed.", file=sys.__stderr__)
sys.stderr = sys.__stderr__

import ssl
print(ssl.OPENSSL_VERSION)

from datasets import load_dataset, DatasetDict
from huggingface_hub import HfApi
from verilog_lists import verilog_keywords, meaningful_identifiers, reserved_words, non_synthesizable_verilog_strings, \
    non_synth_path_keywords

# Initialize the Hugging Face API
hf_api = HfApi()

# Read the Hugging Face API read token from file
read_token_file = "read_token.txt"
with open(read_token_file, "r") as file:
    hf_read_token = file.read().strip()

print("Loading dataset")
dataset = load_dataset("Von-R/interim1", token=hf_read_token)

VAR = "VAR"
MODULE = "MODULE"
meaningful_identifiers_pattern = re.compile('|'.join(meaningful_identifiers))

TEST_DATA = False
# Function to create a smaller test dataset for faster testing and verification
def test_ds(dataset):
    shuffled = dataset['train'].shuffle(seed=42)
    dataset['train'] = shuffled.select(range(100))
    return dataset

if TEST_DATA:
    dataset = test_ds(dataset)

original_content = dataset['train']['content']
original_dataset_rows = dataset['train'].num_rows
print("Original dataset rows: ", original_dataset_rows)
original_dataset_size = sum(dataset['train']['size'])
print(f"Original dataset size: {original_dataset_size / (2 ** 30):.2f} GB")

# Function for safe removal of underscores: skips protected words that must be exact match
def custom_replacement(match):
    word = match.group(0)
    if meaningful_identifiers_pattern.search(word):
        return word  # Return the protected word unchanged
    else:
        return word.replace('_', '')  # Remove underscores for other words

# Function to get a unique identifier
def get_unique_identifier(available, mode, identifier, example, counter):
    if not available:
        if mode == 'ID':
            counter += 1
            print(f"Ran out of available identifiers, generating new one: {VAR}{counter} for ID \"{identifier}\" in \"{example['path']}\"")
            return f"{VAR}{counter}"
        elif mode == 'module':
            counter += 1
            print(f"Ran out of available modules, generating new one: {MODULE}{counter} for module \"{identifier}\" in  \"{example['path']}\"")
            return f"{MODULE}{counter}"
    else:
        # Randomly select and remove an identifier from the set
        name = random.sample(available, 1)[0]
        available.remove(name)
        return name

# Function to anonymize identifiers in Verilog code
def anonymize_identifiers(example):
    local_ID_counter, local_module_counter = 0, 0

    code = example['content']

    # Define regex patterns for different types of identifiers and literals
    identifier_pattern = re.compile(r'\b(?!module MODULE\d\b)[_a-zA-Z][_a-zA-Z0-9]*\b')
    VAR_pattern = re.compile(r'\bVAR\d+\b')
    NUM_pattern = re.compile(r'(?:\d+\'d\d+|d\d+|\d+\'b\d+|b\d+)')
    module_pattern = re.compile(r'\bmodule\s+([_a-zA-Z\$\\][_a-zA-Z0-9\$\\]*)\b', re.MULTILINE)
    underscore_pattern = re.compile(r'_+')
    dollar_pattern = re.compile(r'\$+')
    underscore_var_pattern = re.compile(r'(?<![\S$#])_+(?![\S$#])')
    mutant_pattern = re.compile(r'^[\$\#]+$')

    # Remove underscores and dollar signs
    code = re.sub(underscore_pattern, '', code)
    code = re.sub(dollar_pattern, '', code)

    # Find all module names
    module_list = module_pattern.findall(code)
    if len(module_list) == 0:
        code = f"module dummymodule(\n{code}"
        module_list.append('dummymodule')

    # Convert underscore variables to a recognizable pattern
    if matches := underscore_var_pattern.findall(code):
        ctr = 1
        for match in matches:
            code = re.sub(match.group(0), f'underscore{ctr}', code)

    # Find all identifiers
    id_list = identifier_pattern.findall(code)
    number_of_IDs = len(id_list)
    id_list = list(set(filter(lambda x: x not in reserved_words and x not in module_list, id_list)))

    if number_of_IDs == 0:
        print("ERROR: ID's failed to generate")
        print("Path: ", example['path'])
        print("Content: ", code[:1000])
        print("Exiting...")
        exit(-1)

    # Find all numeric literals
    numerical_literal_pattern = re.compile(r"'(d\d+)|(b[01]+)|(o[0-7]+)|(h[\da-fA-F]+)", re.MULTILINE)
    numeric_literals = set(numerical_literal_pattern.findall(code))
    numeric_literals = [group for match in numeric_literals for group in match if group != '']

    # Filter out meaningful identifiers and numeric literals
    id_list = list(set(filter(lambda x: not meaningful_identifiers_pattern.search(x) and \
                              x not in numeric_literals, id_list)))

    local_ID_counter = len(id_list)
    available_IDs = {f"{VAR}{i}" for i in range(1, local_ID_counter + 1)}

    if len(module_list) == 0:
        print("\nERROR: Modules failed to generate")
        print("File path: ", example['repo_name'] + '/' + example['path'])
        exit(-3)

    # Replace meaningful identifiers in the module list
    for match in module_list:
        if word := meaningful_identifiers_pattern.search(match):
            word = word.group(0)
            if word == match:
                continue
            elif word not in module_list:
                module_list.append(word)
                code = re.sub(re.escape(match), word, code)
            else:
                ctr = 1
                while f"{word}{ctr}" in module_list:
                    ctr += 1
                module_list.append(f"{word}{ctr}")
                code = re.sub(re.escape(match), f"{word}{ctr}", code)
            module_list.remove(match)

    # Create a regex pattern for module names
    module_list_pattern = '|'.join(re.escape(module_name) for module_name in module_list)
    submodule_pattern = re.compile(r'\b(?:' + module_list_pattern + r')\s+([_a-zA-Z\$\\][_a-zA-Z0-9\$\\]*)\b', re.MULTILINE)

    # Find all submodule names
    submodule_list = [
        name.lower() for name in set(submodule_pattern.findall(code))
        if name not in reserved_words
    ]
    if len(submodule_list) > 0:
        meaningful_submodule_list = []
        for match in submodule_list:
            if word := meaningful_identifiers_pattern.search(match):
                word = word.group(0)
                if word == match:
                    continue
                elif word not in meaningful_submodule_list:
                    meaningful_submodule_list.append(word)
                    code = re.sub(match, word, code)
                else:
                    ctr = 1
                    while f"{word}{ctr}" in meaningful_submodule_list:
                        ctr += 1
                    meaningful_submodule_list.append(f"{word}{ctr}")
                    code = re.sub(match, f"{word}{ctr}", code)
                submodule_list.remove(match)

    local_module_counter = len(module_list)
    available_modules = {f"{MODULE}{i}" for i in range(1, local_module_counter + 1)}

    local_submodule_counter = len(submodule_list)
    available_submodules = {f"{MODULE}{i}" for i in range(1, local_submodule_counter + 1)}

    # Replace non-protected module names
    for module_name in module_list:
        new_module_name = get_unique_identifier(available_modules, 'module', module_name, example, local_module_counter)
        code = re.sub(re.compile(r'\b{}\b'.format(re.escape(module_name))), new_module_name, code)

    # Replace non-protected submodule names
    for submodule_name in submodule_list:
        new_submodule_name = get_unique_identifier(available_submodules, 'module', submodule_name, example, local_submodule_counter)
        code = re.sub(re.compile(r'\b{}\b'.format(re.escape(submodule_name))), new_submodule_name, code)

    # Replace non-protected identifiers
    for identifier in id_list:
        new_id = get_unique_identifier(available_IDs, 'ID', identifier, example, local_ID_counter)
        code = re.sub(re.compile(r'(?<!\bmodule\s)\b{}\b'.format(re.escape(identifier))), new_id, code)

    meaningful_identifiers_list = [re.escape(word.replace('\b', '')) for word in meaningful_identifiers_pattern.split('|')]
    protected_words = meaningful_identifiers_list + verilog_keywords
    protected_pattern = '|'.join(map(re.escape, protected_words))

    # Remove mutants now that underscores have been culled
    if mutants_matches := re.match(mutant_pattern, code):
        for match in mutants_matches:
            code = re.sub(match.group(0), '', code)

    # Check for aberrant identifiers
    ID_matches = re.findall(identifier_pattern, code)
    for match in ID_matches:
        if numerical_literal_pattern.search(match) or meaningful_identifiers_pattern.search(match):
            continue
        if not re.match(r'\bVAR\d+\b', match) and match not in verilog_keywords and not re.match(r'\bMODULE\d+\b', match) and match not in meaningful_identifiers and not re.match(NUM_pattern, match):
            if match in non_synthesizable_verilog_strings or match in non_synth_path_keywords:
                code = re.sub(re.compile(f'^.*{re.escape(match)}.*$', re.MULTILINE), '', code)
                if match in code:
                    print("Failed to remove aberrant identifier: ", match)
                    exit(-6)
            else:
                print("File path: ", example['repo_name'] + '/' + example['path'])
                print("Aberrant identifier: ", match)
                print("Original code: ", example['content'])
                print("Print out: ", code)
                exit(-7)

    example['content'] = code
    return example

# Batch processing function for parallel processing
def batch_process(batch):
    processed_examples = []
    for i in range(len(batch['content'])):
        example = {key: batch[key][i] for key in batch}
        processed_example = anonymize_identifiers(example)
        processed_examples.append(processed_example)

    return {key: [example[key] for example in processed_examples] for key in processed_examples[0]}

# Apply the anonymize_identifiers function to the dataset
anonymized_dataset = dataset.map(batch_process, batched=True, batch_size=1000, num_proc=4)

# Optionally save a sample of the anonymized dataset for comparison
SAVE_SAMPLE = False
if SAVE_SAMPLE:
    from datasets import Dataset
    csv_dataset = anonymized_dataset['train']
    def add_original_content(example, index):
        example['original_content'] = original_content[index]
        return example
    csv_dataset = csv_dataset.map(add_original_content, with_indices=True, load_from_cache_file=False)
    df = csv_dataset.to_pandas()
    df = df[['repo_name', 'path', 'size', 'original_content', 'content', 'license']]
    reordered_dataset = Dataset.from_pandas(df)
    reordered_dataset.to_csv('anonymized_dataset_test.csv', index=False)

# Function to create train, test, and validation splits
def create_splits(dataset, test_size=0.1, validation_size=0.1):
    shuffled_dataset = dataset.shuffle(seed=42)
    shuffled_dataset = shuffled_dataset.remove_columns(['COQ_keywords'])
    split = shuffled_dataset.train_test_split(test_size=test_size)
    adjusted_validation_size = validation_size / (1 - test_size)
    valid_split = split['train'].train_test_split(test_size=adjusted_validation_size)
    return DatasetDict({
        'train': valid_split['train'],
        'test': split['test'],
        'validation': valid_split['test']
    })

print("Rows before splitting: ", anonymized_dataset['train'].num_rows)
anonymized_dataset_with_splits = create_splits(anonymized_dataset['train'])
for split in anonymized_dataset_with_splits:
    print(f"Rows in split {split}: ", anonymized_dataset_with_splits[split].num_rows)

new_dataset_rows = sum(anonymized_dataset_with_splits[split].num_rows for split in anonymized_dataset_with_splits)
new_dataset_size = sum(sum(anonymized_dataset_with_splits[split]['size']) if isinstance(anonymized_dataset_with_splits[split]['size'], list) \
                           else anonymized_dataset_with_splits[split]['size'] for split in anonymized_dataset_with_splits)

print("New dataset rows: ", new_dataset_rows)
print(f"New dataset size: {new_dataset_size / (2 ** 30):.2f} GB")
print("Row reduction of ", math.ceil((original_dataset_rows - new_dataset_rows) / original_dataset_rows * 100), "% in dataset rows")
print(f"Size reduction of {((original_dataset_size - new_dataset_size) / original_dataset_size * 100):.2f}% in dataset size")

from datasets import concatenate_datasets, DatasetDict

# Assuming 'anonymized_dataset_with_splits' is a DatasetDict containing 'train', 'test', 'validation' splits
all_datasets = [split_dataset for split_dataset in anonymized_dataset_with_splits.values()]
combined_dataset = concatenate_datasets(all_datasets)

# Read the Hugging Face API write token from file
write_token_file = "write_token.txt"
with open(write_token_file, "r") as file:
    hf_write_token = file.read().strip()

# Push the anonymized dataset with splits to the Hugging Face Hub
anonymized_dataset_with_splits.push_to_hub('Von-R/test', token=hf_write_token)
