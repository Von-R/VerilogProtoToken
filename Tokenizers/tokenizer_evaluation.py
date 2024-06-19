"""
Tokenizer Evaluation Script for Verilog Dataset

This script evaluates a tokenizer trained on a dataset of Verilog files. The main objectives are to assess the 
vocabulary coverage, token count distribution, out-of-vocabulary (OOV) rate, and subword fragmentation of the tokenizer.
The script performs the following steps:

1. Loads the dataset and tokenizer.
2. Evaluates the tokenizer based on several metrics:
   - Vocabulary coverage
   - Mean and median token count per example
   - OOV rate
   - Subword fragmentation
3. Saves the evaluation results and token count distribution to files.
"""

from datasets import load_dataset
from tokenizers import Tokenizer
import numpy as np
import collections
import pandas as pd
import json

# Define the model and paths for dataset and tokenizer
model = "GPT2"
DATASET_NAME = "Von-R/sub500var"
TOKENIZER_PATH = f"custom_tokenizer_{model}.json"

read_token_file = "read_token.txt"
with open(read_token_file, "r") as file:
    hf_read_token = file.read().strip()

# Load the dataset and tokenizer
dataset = load_dataset(DATASET_NAME, split="train", use_auth_token=hf_read_token)
tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

def evaluate_tokenizer(dataset, tokenizer):
    """
    Evaluates the tokenizer on the given dataset and calculates various metrics.

    Parameters:
    dataset (Dataset): The dataset to evaluate on.
    tokenizer (Tokenizer): The tokenizer to evaluate.

    Returns:
    dict: A dictionary containing evaluation metrics.
    """
    total_words = 0
    total_tokens = 0
    oov_count = 0
    subword_splits = 0
    word_freq = collections.Counter()
    token_lengths = []

    for example in dataset['content']:
        words = example.split()
        total_words += len(words)
        word_freq.update(words)

        tokens = tokenizer.encode(example).tokens
        total_tokens += len(tokens)
        token_lengths.append(len(tokens))

        for word in words:
            tokenized_word = tokenizer.encode(word).tokens
            if len(tokenized_word) > 1:
                subword_splits += 1
            if '[UNK]' in tokenized_word:
                oov_count += 1

    # Vocabulary Coverage
    unique_words = len(word_freq)
    covered_words = unique_words - oov_count
    vocab_coverage = covered_words / unique_words

    # Token Count Distribution
    mean_token_count = np.mean(token_lengths)
    median_token_count = np.median(token_lengths)
    token_count_distribution = {
        "mean_token_count": mean_token_count,
        "median_token_count": median_token_count,
        "token_lengths": token_lengths
    }

    # OOV Rate
    oov_rate = oov_count / total_words

    # Subword Fragmentation
    subword_fragmentation = subword_splits / total_words

    # Prepare results for output
    results = {
        "vocab_coverage": vocab_coverage,
        "mean_token_count": mean_token_count,
        "median_token_count": median_token_count,
        "oov_rate": oov_rate,
        "subword_fragmentation": subword_fragmentation
    }

    # Save results
    with open("tokenizer_evaluation.json", "w") as json_file:
        json.dump(results, json_file, indent=4)

    token_count_df = pd.DataFrame(token_count_distribution)
    token_count_df.to_csv(f"token_count_distribution_{model}.csv", index=False)

    print("Evaluation results saved to 'tokenizer_evaluation.json' and 'token_count_distribution.csv'")

# Evaluate the tokenizer
evaluate_tokenizer(dataset, tokenizer)
