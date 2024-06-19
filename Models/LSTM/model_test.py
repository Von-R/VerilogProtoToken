"""
LSTM Model Evaluation Script

This script evaluates a trained LSTM model on a Verilog test dataset. The main objectives are:
1. Loading the trained model and tokenizer.
2. Tokenizing the test dataset.
3. Evaluating the model's performance using various metrics.
4. Saving and plotting the evaluation results.

Key components include:
- Argument parsing for specifying the model directory.
- Tokenization of the test dataset using a custom tokenizer.
- Definition of the LSTM model.
- Evaluation loop for calculating metrics and plotting results.

Dependencies: 
- transformers, datasets, tokenizers, torch, sklearn, matplotlib, tqdm
"""

import argparse
import os
import re
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset
from tokenizers import Tokenizer
from torch.nn.utils.rnn import pad_sequence
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

# Set device to GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, lstm_units):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # Embedding layer
        self.lstm1 = nn.LSTM(embedding_dim, lstm_units, batch_first=True)  # First LSTM layer
        self.lstm2 = nn.LSTM(lstm_units, lstm_units, batch_first=True)  # Second LSTM layer
        self.fc = nn.Linear(lstm_units, vocab_size)  # Fully connected layer

    def forward(self, x):
        x = self.embedding(x)  # Apply embedding
        x, _ = self.lstm1(x)  # Apply first LSTM layer
        x, _ = self.lstm2(x)  # Apply second LSTM layer
        x = self.fc(x)  # Apply fully connected layer to predict the next token
        return x

# Function to print GPU memory usage
def print_memory_usage(stage):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        print(f"[{stage}] Memory allocated: {allocated:.2f} GB, Memory reserved: {reserved:.2f} GB")
        print(torch.cuda.memory_summary())

# Function to calculate accuracy
def calculate_accuracy(predictions, labels):
    preds = predictions.argmax(dim=-1)  # Get the index of the max log-probability
    correct = (preds == labels).float()  # Check which predictions are correct
    acc = correct.sum() / correct.numel()  # Calculate accuracy
    return acc

# Function to calculate precision, recall, and F1 score
def calculate_metrics(predictions, labels, mask):
    preds = predictions.argmax(dim=-1)  # Get the index of the max log-probability
    masked_preds = preds[mask]  # Apply mask to predictions
    masked_labels = labels[mask]  # Apply mask to labels

    print(f"Masked Predictions: {masked_preds[:100]}")  # Print first 100 masked predictions for debugging
    print(f"Masked Labels: {masked_labels[:100]}")  # Print first 100 masked labels for debugging

    accuracy = (masked_preds == masked_labels).sum().float() / len(masked_labels)  # Calculate accuracy
    precision = precision_score(masked_labels.cpu(), masked_preds.cpu(), average='macro', zero_division=0)  # Calculate precision
    recall = recall_score(masked_labels.cpu(), masked_preds.cpu(), average='macro', zero_division=0)  # Calculate recall
    f1 = f1_score(masked_labels.cpu(), masked_preds.cpu(), average='macro', zero_division=0)  # Calculate F1 score
    
    return accuracy.item(), precision, recall, f1

# Function to create a mask for labels that match a specific pattern
def mask_labels(labels, pattern, wrapped_tokenizer):
    mask = torch.full(labels.shape, True, dtype=torch.bool)  # Create a mask with all True values
    for i in range(labels.size(0)):
        for j in range(labels.size(1)):
            token_str = wrapped_tokenizer.decode([labels[i, j].item()])  # Decode label to token string
            if re.match(pattern, token_str):  # Check if token matches the pattern
                mask[i, j] = False  # Set mask to False for matching tokens
    return mask

# Function to evaluate the model on the test dataset
def evaluate_model(model_dir, model_file):
    tokenizer_path = os.path.join(model_dir, "tokenizer.json")
    model_path = os.path.join(model_dir, model_file)
    
    # Load the tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_path)
    wrapped_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    wrapped_tokenizer.pad_token = "[PAD]"
    wrapped_tokenizer.padding_side = "left"

    # Get the vocabulary size from the tokenizer
    vocab_size = len(wrapped_tokenizer.get_vocab())

    # Load the model with the correct vocabulary size
    embedding_dim = 50
    lstm_units = 100

    model = LSTMModel(vocab_size, embedding_dim, lstm_units).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Load the test dataset
    token = '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'
    dataset_test = load_dataset("Von-R/sub500var", split='test', use_auth_token=token)

    # Tokenize the test dataset
    def tokenize_function(examples):
        tokenized_inputs = wrapped_tokenizer(examples["content"], padding="max_length", truncation=True, max_length=512)
        tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
        return tokenized_inputs

    columns_to_remove = ['repo_name', 'path', 'size', 'content', 'license']
    tokenized_test_dataset = dataset_test.map(tokenize_function, batched=True, remove_columns=columns_to_remove)

    # Custom collate function for padding sequences
    def custom_collate_fn(batch):
        input_ids = [torch.tensor(item['input_ids'], dtype=torch.long) for item in batch]
        labels = [torch.tensor(item['labels'], dtype=torch.long) for item in batch]
        input_ids_padded = pad_sequence(input_ids, batch_first=True)
        labels_padded = pad_sequence(labels, batch_first=True)
        return {
            'input_ids': input_ids_padded,
            'labels': labels_padded
        }

    # Create DataLoader for the evaluation dataset
    eval_dataloader = DataLoader(tokenized_test_dataset, batch_size=1, collate_fn=custom_collate_fn, pin_memory=True)

    model.eval()  # Set model to evaluation mode
    eval_loss = 0
    total_accuracy = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0

    criterion = nn.CrossEntropyLoss()  # Define the loss function

    # Iterate over the evaluation DataLoader
    for step, batch in enumerate(eval_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}  # Move batch to the appropriate device (CPU/GPU)
        with torch.no_grad():  # Disable gradient calculation
            outputs = model(batch['input_ids'])  # Forward pass
            outputs_reshaped = outputs.view(-1, vocab_size)  # Reshape outputs for loss calculation
            labels_reshaped = batch['labels'].view(-1)  # Reshape labels for loss calculation

            loss = criterion(outputs_reshaped, labels_reshaped)  # Calculate loss
            eval_loss += loss.item()

            # Calculate metrics for the batch and update totals
            accuracy, precision, recall, f1 = calculate_metrics(outputs, batch['labels'], batch['labels'] != -100)
            total_accuracy += accuracy
            total_precision += precision
            total_recall += recall
            total_f1 += f1

            # Clear CUDA cache to avoid memory overflow
            torch.cuda.empty_cache()

        if step % 10 == 0:
            print(f"Step {step}, Loss: {loss.item()}")

    eval_loss /= len(eval_dataloader)  # Average the evaluation loss
    total_accuracy /= len(eval_dataloader)  # Average the accuracy
    total_precision /= len(eval_dataloader)  # Average the precision
    total_recall /= len(eval_dataloader)  # Average the recall
    total_f1 /= len(eval_dataloader)  # Average the F1 score

    # Store results in a dictionary
    results = {
        "Next Token Prediction Loss": eval_loss,
        "Perplexity": np.exp(eval_loss),
        "Accuracy": total_accuracy,
        "Precision": total_precision,
        "Recall": total_recall,
        "F1 Score": total_f1
    }

    # Ensure all values are converted to native Python types
    results = {k: (v.item() if isinstance(v, torch.Tensor) else v) for k, v in results.items()}

    results_path = os.path.join(model_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    # Plot and save the evaluation metrics
    metrics = ["Next Token Prediction Loss", "Perplexity", "Accuracy", "Precision", "Recall", "F1 Score"]
    values = [eval_loss, np.exp(eval_loss), total_accuracy, total_precision, total_recall, total_f1]
    plt.figure(figsize=(10, 6))
    plt.bar(metrics, values, color=['blue', 'green', 'red', 'purple', 'orange', 'brown'])
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.title('Model Evaluation Metrics')
    plt.savefig(os.path.join(model_dir, 'evaluation_metrics.png'))

    print(f"Results saved for model at {model_dir}")

# Main function to handle argument parsing and directory traversal
def main():
    parser = argparse.ArgumentParser(description='Model Evaluation')
    parser.add_argument('--dir', type=str, help='Specific directory to evaluate')
    args = parser.parse_args()

    base_path = 'final_model'
    if args.dir:
        model_path = os.path.join(base_path, args.dir)
        if os.path.exists(model_path):
            pt_files = [f for f in os.listdir(model_path) if f.endswith('.pt')]
            if pt_files:
                print(f"Evaluating model in directory: {model_path}")
                evaluate_model(model_path, pt_files[0])
            else:
                print(f"No .pt file found in directory: {model_path}")
        else:
            print(f"Directory {model_path} does not exist.")
    else:
        # Limit directory traversal to one level deep
        for root, dirs, files in os.walk(base_path):
            if root == base_path:
                for dir in dirs:
                    model_path = os.path.join(root, dir)
                    pt_files = [f for f in os.listdir(model_path) if f.endswith('.pt')]
                    if os.path.exists(model_path) and pt_files:
                        print(f"Evaluating model in directory: {model_path}")
                        evaluate_model(model_path, pt_files[0])
                    else:
                        print(f"Skipping directory: {model_path}")
                break  # Break after the first level

if __name__ == "__main__":
    main()
