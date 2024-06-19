"""
LSTM Training Script for Verilog Next Token Prediction

This script trains an LSTM model on a Verilog dataset to predict the next token. The main objectives are:
1. Loading the dataset and tokenizer.
2. Setting up the training parameters and dataloaders.
3. Defining and training the LSTM model.
4. Evaluating the model on the validation and test datasets.
5. Saving the best model based on validation loss.

Key components include:
- Argument parsing for specifying training parameters.
- Tokenization of the dataset using a custom tokenizer.
- Definition of the LSTM model.
- Training loop with gradient accumulation and evaluation.
- Functions for calculating metrics and plotting results.

Dependencies: 
- transformers, datasets, tokenizers, torch, accelerate, sklearn, matplotlib, tqdm
"""

import argparse
import os
import shutil
import torch
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset
from tokenizers import Tokenizer
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import json
import matplotlib.pyplot as plt

# Set up argument parser to handle command-line arguments
parser = argparse.ArgumentParser(description='Train LSTM Model for Verilog Next Token Prediction')
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train the model')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='Number of steps to accumulate gradients before updating weights')
parser.add_argument('--save_best_only', action='store_true', help='Save only the best model based on validation loss')
args = parser.parse_args()

# Setup Accelerator for mixed precision training
accelerator = Accelerator(mixed_precision="fp16")
device = accelerator.device

# Token and tokenizer path
token = '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'
tokenizer_path = "custom_tokenizer_gpt2.json"

# Load the training and validation datasets
print("Loading dataset...")
dataset_train = load_dataset("Von-R/sub500var", split='train', use_auth_token=token)
dataset_validation = load_dataset("Von-R/sub500var", split='validation', use_auth_token=token)
dataset_test = load_dataset("Von-R/sub500var", split='test', use_auth_token=token)

# Load and wrap the custom tokenizer using PreTrainedTokenizerFast
tokenizer = Tokenizer.from_file(tokenizer_path)
wrapped_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
wrapped_tokenizer.pad_token = "[PAD]"  # Set padding token

# Verify tokenizer vocabulary size
vocab_size = tokenizer.get_vocab_size()  # Should be 50000
print("Vocabulary size:", vocab_size)

# Tokenize the training data
def tokenize_function(examples):
    tokenized_inputs = wrapped_tokenizer(examples["content"], padding="max_length", truncation=True, max_length=512)
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
    return tokenized_inputs

print("Tokenizing training data...")
tokenized_train_dataset = dataset_train.map(tokenize_function, batched=True, remove_columns=['repo_name', 'path', 'size', 'content', 'license'])

print("Tokenizing validation data...")
tokenized_val_dataset = dataset_validation.map(tokenize_function, batched=True, remove_columns=['repo_name', 'path', 'size', 'content', 'license'])

print("Tokenizing test data...")
tokenized_test_dataset = dataset_test.map(tokenize_function, batched=True, remove_columns=['repo_name', 'path', 'size', 'content', 'license'])

# Create the dataloaders
def custom_collate_fn(batch):
    input_ids = [torch.tensor(item['input_ids'], dtype=torch.long) for item in batch]
    labels = [torch.tensor(item['labels'], dtype=torch.long) for item in batch]
    input_ids_padded = pad_sequence(input_ids, batch_first=True)
    labels_padded = pad_sequence(labels, batch_first=True)
    return {
        'input_ids': input_ids_padded,
        'labels': labels_padded
    }

# Create DataLoader instances for training, validation, and test datasets
train_dataloader = DataLoader(tokenized_train_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=custom_collate_fn)
eval_dataloader = DataLoader(tokenized_val_dataset, batch_size=args.batch_size * 2, collate_fn=custom_collate_fn)
test_dataloader = DataLoader(tokenized_test_dataset, batch_size=args.batch_size * 2, collate_fn=custom_collate_fn)

# Prepare DataLoaders using the accelerator for distributed training
train_dataloader, eval_dataloader, test_dataloader = accelerator.prepare(train_dataloader, eval_dataloader, test_dataloader)

# Define the LSTM model in PyTorch
class LSTMModel(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, lstm_units):
        super(LSTMModel, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)  # Embedding layer to learn vector representations of tokens
        self.lstm1 = torch.nn.LSTM(embedding_dim, lstm_units, batch_first=True)  # First LSTM layer for sequence modeling
        self.lstm2 = torch.nn.LSTM(lstm_units, lstm_units, batch_first=True)  # Second LSTM layer for deeper sequence modeling
        self.fc = torch.nn.Linear(lstm_units, vocab_size)  # Fully connected layer to predict the next token

    def forward(self, x):
        x = self.embedding(x)  # Apply embedding to input tokens
        x, _ = self.lstm1(x)  # Apply first LSTM layer
        x, _ = self.lstm2(x)  # Apply second LSTM layer
        x = self.fc(x[:, -1, :])  # Apply fully connected layer to the last LSTM output to predict the next token
        return x

# Function to calculate accuracy, precision, recall, and F1 score from model predictions and labels
def calculate_metrics(predictions, labels):
    preds = predictions.argmax(dim=-1).cpu().numpy()
    labels = labels.cpu().numpy()
    mask = labels != -100

    print(f"preds shape: {preds.shape}")
    print(f"labels shape: {labels.shape}")
    print(f"mask shape: {mask.shape}")

    preds = preds[mask]
    labels = labels[mask]

    print(f"masked preds shape: {preds.shape}")
    print(f"masked labels shape: {labels.shape}")

    accuracy = (preds == labels).sum() / len(labels)
    precision = precision_score(labels, preds, average='macro', zero_division=0)
    recall = recall_score(labels, preds, average='macro', zero_division=0)
    f1 = f1_score(labels, preds, average='macro', zero_division=0)
    
    return accuracy, precision, recall, f1

# Function to calculate a composite metric from accuracy, precision, recall, and F1 score
def composite_metric(accuracy, precision, recall, f1):
    # Example composite metric that weights F1 score highest
    return (0.1 * accuracy) + (0.2 * precision) + (0.2 * recall) + (0.5 * f1)

# Function to plot and save batch-wise metrics
def plot_batch_metrics(batch_metrics, model_path):
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
    plt.figure(figsize=(12, 6))

    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i + 1)
        plt.plot(batch_metrics[metric], label=metric, color='blue')
        plt.title(f'Batch-wise {metric}')
        plt.xlabel('Batch')
        plt.ylabel(metric)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_path, 'batch_metrics.png'))
    plt.show()

# Function to plot and save final evaluation metrics
def plot_final_metrics(results, model_path):
    metrics = ["Next Token Prediction Loss", "Perplexity", "Accuracy", "Precision", "Recall", "F1 Score"]
    values = [results[m] for m in metrics]
    
    plt.figure(figsize=(10, 5))
    plt.bar(metrics, values, color='blue', alpha=0.7)
    plt.title('Final Evaluation Metrics')
    plt.xlabel('Metrics')
    plt.ylabel('Value')
    plt.savefig(os.path.join(model_path, 'final_metrics.png'))
    plt.show()

# Function to evaluate the model on a given dataloader
def evaluate_model(model, dataloader, model_dir):
    model.eval()  # Set model to evaluation mode
    eval_loss = 0
    total_accuracy = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0

    batch_metrics = {"Accuracy": [], "Precision": [], "Recall": [], "F1 Score": []}

    # Iterate over the evaluation DataLoader
    for step, batch in enumerate(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}  # Move batch to device
        with torch.no_grad():  # Disable gradient calculation
            outputs = model(batch['input_ids'])  # Get model outputs
            batch_loss = torch.nn.functional.cross_entropy(outputs, batch['labels'].argmax(dim=1))  # Compute loss
            eval_loss += batch_loss.item()  # Accumulate evaluation loss
            print(f"Batch {step} | Loss = {batch_loss}")
                
            accuracy, precision, recall, f1 = calculate_metrics(outputs, batch['labels'])

            total_accuracy += accuracy
            total_precision += precision
            total_recall += recall
            total_f1 += f1

            # Store batch-wise metrics
            batch_metrics["Accuracy"].append(accuracy)
            batch_metrics["Precision"].append(precision)
            batch_metrics["Recall"].append(recall)
            batch_metrics["F1 Score"].append(f1)

    # Calculate average metrics
    eval_loss /= len(dataloader)
    perplexity = np.exp(eval_loss)
    total_accuracy /= len(dataloader)
    total_precision /= len(dataloader)
    total_recall /= len(dataloader)
    total_f1 /= len(dataloader)

    # Store results in a dictionary
    results = {
        "Next Token Prediction Loss": eval_loss,
        "Perplexity": perplexity,
        "Accuracy": total_accuracy,
        "Precision": total_precision,
        "Recall": total_recall,
        "F1 Score": total_f1
    }

    # Save results to a JSON file
    results_path = os.path.join(model_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {results_path}")

    # Plot and save batch-wise metrics
    plot_batch_metrics(batch_metrics, model_dir)

    # Plot and save final evaluation metrics
    plot_final_metrics(results, model_dir)

# Function to train and evaluate the model
def train_and_evaluate_model(config):
    model = LSTMModel(vocab_size, config['embedding_dim'], config['lstm_units']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    model, optimizer = accelerator.prepare(model, optimizer)

    best_eval_loss = float('inf')
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for step, batch in enumerate(progress_bar):
            with accelerator.accumulate(model):
                outputs = model(batch['input_ids'].to(device))  # Forward pass
                loss = torch.nn.functional.cross_entropy(outputs, batch['labels'].argmax(dim=1).to(device))  # Compute loss
                accelerator.backward(loss)  # Backward pass

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                total_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})

        print(f"Epoch {epoch + 1} | Training Loss: {total_loss / len(train_dataloader)}")

        # Evaluation
        model.eval()
        eval_loss = 0
        progress_bar = tqdm(eval_dataloader, desc="Evaluating")
        with torch.no_grad():
            for batch in progress_bar:
                outputs = model(batch['input_ids'].to(device))
                loss = torch.nn.functional.cross_entropy(outputs, batch['labels'].argmax(dim=1).to(device))
                eval_loss += loss.item()
                progress_bar.set_postfix({'eval_loss': loss.item()})
        avg_eval_loss = eval_loss / len(eval_dataloader)
        print(f"Epoch {epoch + 1} | Validation Loss: {avg_eval_loss}")

        # Save the best model based on validation loss
        if args.save_best_only:
            if avg_eval_loss < best_eval_loss:
                best_eval_loss = avg_eval_loss
                print("Saving best model...")
                torch.save(model.state_dict(), os.path.join(model_dir, 'best_model.pt'))
                wrapped_tokenizer.save_pretrained(model_dir)
        else:
            torch.save(model.state_dict(), os.path.join(model_dir, f'model_epoch_{epoch + 1}.pt'))
            wrapped_tokenizer.save_pretrained(model_dir)

    # Save the final model if save_best_only is not set
    if not args.save_best_only:
        torch.save(model.state_dict(), os.path.join(model_dir, 'final_model.pt'))
        wrapped_tokenizer.save_pretrained(model_dir)

    # Evaluate the model on the test dataset
    evaluate_model(model, test_dataloader, model_dir)

if __name__ == "__main__":
    # List of model configurations to train and evaluate
    model_configs = [
        {'embedding_dim': 50, 'lstm_units': 100, 'learning_rate': 5e-5},
        {'embedding_dim': 100, 'lstm_units': 200, 'learning_rate': 3e-5}
    ]

    # Train and evaluate each model configuration
    for config in model_configs:
        model_dir = f"./final_model/epochs_{args.epochs}_batch_size_{args.batch_size}_embedding_dim_{config['embedding_dim']}_lstm_units_{config['lstm_units']}"
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)  # Remove existing model directory if it exists
        os.makedirs(model_dir)  # Create new directory for model
        train_and_evaluate_model(config)  # Train and evaluate the model with the current configuration
