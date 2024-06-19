"""
Model Evaluation Script for GPT-2

This script evaluates a pre-trained GPT-2 model on a Verilog dataset. The main objectives are:
1. Loading the dataset and tokenizer.
2. Setting up the evaluation parameters.
3. Calculating various evaluation metrics including loss, perplexity, accuracy, precision, recall, F1 score, top-5 accuracy, entropy, and prediction confidence.
4. Plotting and saving evaluation results.

Key components include:
- Argument parsing for specifying the model directory.
- Function definitions for metric calculations and plotting.
- Model and tokenizer loading.
- Evaluation loop with batch-wise metric calculations and logging.
"""

import argparse
import os
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, AutoConfig, PreTrainedTokenizerFast
from datasets import load_dataset
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import precision_score, recall_score, f1_score

# Print versions of torch, transformers, and datasets to ensure compatibility and debug potential issues
print("Torch version:", torch.__version__)

try:
    import transformers
    print("Transformers version:", transformers.__version__)
except ImportError as e:
    print("Transformers import error:", e)

try:
    import datasets
    print("Datasets version:", datasets.__version__)
except ImportError as e:
    print("Datasets import error:", e)

# Check if CUDA is available and select the device accordingly
print("CUDA is available:", torch.cuda.is_available())
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Using device:", device)

# Function to print memory usage at different stages of the script
def print_memory_usage(stage):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        print(f"[{stage}] Memory allocated: {allocated:.2f} GB, Memory reserved: {reserved:.2f} GB")
        print(torch.cuda.memory_summary())

# Function to calculate accuracy, precision, recall, and F1 score from model predictions and labels
def calculate_metrics(predictions, labels):
    preds = predictions.argmax(dim=-1).cpu().numpy()  # Get the index of the max log-probability
    labels = labels.cpu().numpy()  # Move labels to CPU and convert to numpy array
    mask = labels != -100  # Mask to ignore padding tokens in the labels

    preds = preds[mask]  # Apply mask to predictions
    labels = labels[mask]  # Apply mask to labels

    # Calculate accuracy, precision, recall, and F1 score
    accuracy = (preds == labels).sum() / len(labels)
    precision = precision_score(labels, preds, average='macro', zero_division=0)
    recall = recall_score(labels, preds, average='macro', zero_division=0)
    f1 = f1_score(labels, preds, average='macro', zero_division=0)
    
    return accuracy, precision, recall, f1

# Function to calculate top-k accuracy
def top_k_accuracy(predictions, labels, k=5):
    top_k_preds = torch.topk(predictions, k, dim=-1).indices
    labels = labels.unsqueeze(-1).expand_as(top_k_preds)
    correct = (top_k_preds == labels).any(dim=-1).float()
    return correct.sum().item() / correct.numel()

# Function to calculate entropy
def calculate_entropy(predictions):
    probs = torch.softmax(predictions, dim=-1)
    log_probs = torch.log_softmax(predictions, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1).mean().item()
    return entropy

# Function to calculate prediction confidence
def prediction_confidence(predictions):
    confidences = torch.softmax(predictions, dim=-1).max(dim=-1).values
    return confidences.mean().item()

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
    metrics = ["Next Token Prediction Loss", "Perplexity", "Accuracy", "Precision", "Recall", "F1 Score", "Top-5 Accuracy", "Entropy", "Prediction Confidence"]
    values = [results[m] for m in metrics]
    
    plt.figure(figsize=(10, 5))
    plt.bar(metrics, values, color='blue', alpha=0.7)
    plt.title('Final Evaluation Metrics')
    plt.xlabel('Metrics')
    plt.ylabel('Value')
    plt.savefig(os.path.join(model_path, 'final_metrics.png'))
    plt.show()

# Function to evaluate the model on the test dataset
def evaluate_model(model_path):
    config_path = os.path.join(model_path, 'config.json')
    config = AutoConfig.from_pretrained(config_path)  # Load model configuration

    print("Loading model and tokenizer...")
    model = GPT2LMHeadModel.from_pretrained(model_path, config=config)  # Load model
    tokenizer = Tokenizer.from_file(f"{model_path}/tokenizer.json")  # Load tokenizer
    wrapped_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)  # Wrap tokenizer with PreTrainedTokenizerFast
    wrapped_tokenizer.pad_token = "[PAD]"  # Set pad token
    wrapped_tokenizer.padding_side = "left"  # Set padding side

    model.resize_token_embeddings(len(wrapped_tokenizer))  # Resize token embeddings to match tokenizer

    print_memory_usage("After model and tokenizer initialization")

    # Load the test dataset
    token = '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'
    dataset_test = load_dataset("Von-R/sub500var", split='test', use_auth_token=token)
    dataset_test = dataset_test.select(range(500))  # Select a subset for faster evaluation
    print("Dataset Structure:", dataset_test)

    # Tokenize the dataset
    def tokenize_function(examples):
        tokenized_inputs = wrapped_tokenizer(examples["content"], padding="max_length", truncation=True, max_length=512)
        tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
        return tokenized_inputs

    # Remove unnecessary columns
    columns_to_remove = ['repo_name', 'path', 'size', 'content', 'license']
    tokenized_test_dataset = dataset_test.map(tokenize_function, batched=True, remove_columns=columns_to_remove)

    print_memory_usage("After tokenizing test dataset")

    # Custom collate function to pad sequences
    def custom_collate_fn(batch):
        input_ids = [torch.tensor(item['input_ids'], dtype=torch.long) for item in batch]
        attention_mask = [torch.tensor(item['attention_mask'], dtype=torch.long) for item in batch]
        labels = [torch.tensor(item['labels'], dtype=torch.long) for item in batch]
        input_ids_padded = pad_sequence(input_ids, batch_first=True)
        attention_mask_padded = pad_sequence(attention_mask, batch_first=True)
        labels_padded = pad_sequence(labels, batch_first=True)
        return {
            'input_ids': input_ids_padded,
            'attention_mask': attention_mask_padded,
            'labels': labels_padded
        }

    # Create DataLoader for the evaluation dataset
    eval_dataloader = DataLoader(tokenized_test_dataset, batch_size=1, collate_fn=custom_collate_fn)

    print_memory_usage("After creating DataLoader")

    # Move model to the selected device
    model.to(device)
    print_memory_usage("After moving model to device")

    model.eval()  # Set model to evaluation mode
    eval_loss = 0
    total_accuracy = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_top_5_accuracy = 0
    total_entropy = 0
    total_confidence = 0

    batch_metrics = {"Accuracy": [], "Precision": [], "Recall": [], "F1 Score": []}

    # Iterate over the evaluation DataLoader
    for step, batch in enumerate(eval_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}  # Move batch to device
        with torch.no_grad():  # Disable gradient calculation
            outputs = model(**batch)  # Get model outputs
            if outputs.loss is not None:
                batch_loss = outputs.loss.item()
                eval_loss += batch_loss  # Accumulate evaluation loss
                print(f"Batch {step}: Loss = {batch_loss}")
                
                accuracy, precision, recall, f1 = calculate_metrics(outputs.logits, batch['labels'])
                top_5_acc = top_k_accuracy(outputs.logits, batch['labels'], k=5)
                entropy = calculate_entropy(outputs.logits)
                confidence = prediction_confidence(outputs.logits)

                total_accuracy += accuracy
                total_precision += precision
                total_recall += recall
                total_f1 += f1
                total_top_5_accuracy += top_5_acc
                total_entropy += entropy
                total_confidence += confidence

                # Store batch-wise metrics
                batch_metrics["Accuracy"].append(accuracy)
                batch_metrics["Precision"].append(precision)
                batch_metrics["Recall"].append(recall)
                batch_metrics["F1 Score"].append(f1)

    # Calculate average metrics
    eval_loss /= len(eval_dataloader)
    perplexity = np.exp(eval_loss)
    total_accuracy /= len(eval_dataloader)
    total_precision /= len(eval_dataloader)
    total_recall /= len(eval_dataloader)
    total_f1 /= len(eval_dataloader)
    total_top_5_accuracy /= len(eval_dataloader)
    total_entropy /= len(eval_dataloader)
    total_confidence /= len(eval_dataloader)

    # Store results in a dictionary
    results = {
        "Next Token Prediction Loss": eval_loss,
        "Perplexity": perplexity,
        "Accuracy": total_accuracy,
        "Precision": total_precision,
        "Recall": total_recall,
        "F1 Score": total_f1,
        "Top-5 Accuracy": total_top_5_accuracy,
        "Entropy": total_entropy,
        "Prediction Confidence": total_confidence
    }

    # Save results to a JSON file
    results_path = os.path.join(model_path, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {results_path}")

    # Plot and save batch-wise metrics
    plot_batch_metrics(batch_metrics, model_path)

    # Plot and save final evaluation metrics
    plot_final_metrics(results, model_path)

def main():
    parser = argparse.ArgumentParser(description='Model Evaluation')
    parser.add_argument('--dir', type=str, help='Specific directory to evaluate')
    args = parser.parse_args()

    base_path = 'final_model'
    if args.dir:
        model_path = os.path.join(base_path, args.dir)
        if os.path.exists(model_path):
            print(f"Evaluating model in directory: {model_path}")
            evaluate_model(model_path)
        else:
            print(f"Directory {model_path} does not exist.")
    else:
        # Limit directory traversal to one level deep
        for root, dirs, files in os.walk(base_path):
            # Check the depth
            if root == base_path:
                for dir in dirs:
                    model_path = os.path.join(root, dir)
                    if os.path.exists(model_path) and "config.json" in os.listdir(model_path):
                        print(f"Evaluating model in directory: {model_path}")
                        evaluate_model(model_path)
                    else:
                        print(f"Skipping directory: {model_path}")
                break  # Break after the first level

if __name__ == "__main__":
    main()
