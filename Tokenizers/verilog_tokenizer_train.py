"""
Tokenizer Training Script for Verilog Dataset

This script trains custom tokenizers for a dataset of Verilog files using different tokenization models. The main 
objective is to create tokenizers suitable for various NLP models like BERT, GPT-2, Mistral, Gemma, and LLaMA. The 
script performs the following steps:

1. Reads the Hugging Face token from a file for authentication.
2. Loads the dataset from the Hugging Face Hub.
3. Combines all splits (train, test, validation) into one dataset.
4. Defines a function to create and train a tokenizer based on the specified model type.
5. Trains and saves tokenizers for each model type.
6. Tests the trained tokenizers with a sample input to ensure they work correctly.
"""

from datasets import load_dataset
import os
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# Define the path to the token file
read_token_file_path = "./hf_read_token.txt"
hf_read_token = None

try:
    # Attempt to read the token from the file if it exists
    if os.path.exists(read_token_file_path):
        with open(read_token_file_path, "r") as read_token_file:
            hf_read_token = read_token_file.read().strip()
    else:
        raise FileNotFoundError(f"No token file found at '{read_token_file_path}'.")
        exit(-1)
except Exception as e:
    print(f"Failed to read the Hugging Face token: {e}")
    exit(-1)

# Proceed with token, if available
if hf_read_token:
    try:
        # Load the dataset from the Hugging Face Hub using the read token
        dataset = load_dataset("Von-R/sub500var", token=hf_read_token)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        exit(-1)
else:
    print("Read token missing. Exiting.")
    exit(-1)

# Combine the splits into one dataset
all_texts = []
for split in ['train', 'test', 'validation']:
    all_texts.extend(dataset[split]['content'])  # Assuming 'content' is the field name

# Function to create and train a tokenizer based on the model type
def train_tokenizer(model_name, texts, file_name):
    # Configure the tokenizer and trainer based on the model type
    if model_name == "BERT":
        trainer = trainers.WordPieceTrainer(special_tokens=["[CLS]", "[SEP]", "[PAD]", "[MASK]", "[UNK]"], vocab_size=30522)
        model = models.WordPiece(unk_token="[UNK]")

    elif model_name == "GPT2":
        trainer = trainers.BpeTrainer(
            vocab_size=50257,  # Adjust based on the model
            min_frequency=2,   # Tune based on dataset
            special_tokens=["[UNK]", "[PAD]", "[MASK]", "[BOS]", "[EOS]"],  # Adjust based on the model
            limit_alphabet=1000,  # Adjust based on dataset
            continuing_subword_prefix="##",  # Adjust based on subword strategy
            end_of_word_suffix="</w>",  # Adjust based on subword strategy
        )
        model = models.BPE(unk_token="[UNK]")

    elif model_name == "Mistral":
        trainer = trainers.BpeTrainer(special_tokens=["[CLS]", "[SEP]", "[PAD]", "[UNK]", "[MASK]"], vocab_size=32000)
        model = models.BPE(unk_token="[UNK]")

    elif model_name == "Gemma":
        trainer = trainers.BpeTrainer(special_tokens=["[CLS]", "[SEP]", "[PAD]", "[UNK]", "[MASK]"], vocab_size=30522)  # Adjust as needed
        model = models.BPE(unk_token="[UNK]")

    elif model_name == "LLaMA":
        trainer = trainers.BpeTrainer(special_tokens=["[BOS]", "[EOS]", "[UNK]", "[PAD]"], vocab_size=50257)
        model = models.BPE(unk_token="[UNK]")

    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    # Create the tokenizer and train it
    tokenizer = Tokenizer(model)
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer.save(file_name)
    return tokenizer

# Train and save tokenizers for each model type
bert_tokenizer = train_tokenizer("BERT", all_texts, "custom_tokenizer_bert.json")
gpt2_tokenizer = train_tokenizer("GPT2", all_texts, "custom_tokenizer_gpt2.json")
mistral_tokenizer = train_tokenizer("Mistral", all_texts, "custom_tokenizer_mistral.json")
gemma_tokenizer = train_tokenizer("Gemma", all_texts, "custom_tokenizer_gemma.json")
llama_tokenizer = train_tokenizer("LLaMA", all_texts, "custom_tokenizer_llama.json")

# Testing the tokenizers with a sample input
# BERT Tokenizer
bert_output = bert_tokenizer.encode("assign out = a & b;")
print("BERT Tokenized:", bert_output.tokens)

# GPT2 Tokenizer
gpt2_output = gpt2_tokenizer.encode("assign out = a & b;")
print("GPT2 Tokenized:", gpt2_output.tokens)

# Mistral Tokenizer
mistral_tokenizer_output = mistral_tokenizer.encode("assign out = a & b;")
print("Mistral Tokenized:", mistral_tokenizer_output.tokens)

# Gemma Tokenizer
gemma_tokenizer_output = gemma_tokenizer.encode("assign out = a & b;")
print("Gemma Tokenized:", gemma_tokenizer_output.tokens)

# LLaMA Tokenizer
llama_tokenizer_output = llama_tokenizer.encode("assign out = a & b;")
print("LLaMA Tokenized:", llama_tokenizer_output.tokens)
