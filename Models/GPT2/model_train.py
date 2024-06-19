"""
GPT-2 Training Script for Verilog Dataset

This script trains a GPT-2 model on a Verilog dataset using the Hugging Face Transformers library. The main objectives are:
1. Loading the dataset and tokenizer.
2. Setting up the model and training parameters.
3. Training the model with optional grid search for hyperparameters.
4. Implementing early stopping based on a composite metric.
5. Evaluating the model on a validation set.
6. Saving the trained model and tokenizer.

Key components include:
- Argument parsing for training setup.
- Grid search for hyperparameter optimization.
- Early stopping mechanism.
- Training and evaluation loops with logging.
"""

import argparse
import os
import torch
import itertools
import json
import numpy as np
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast, get_scheduler, AutoConfig
from datasets import load_dataset
from tokenizers import Tokenizer
from tqdm import tqdm
from accelerate import Accelerator
from sklearn.metrics import precision_score, recall_score, f1_score

# Argument parsing for distributed training setup
parser = argparse.ArgumentParser(description='Distributed Training Setup')
parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
parser.add_argument('--job_name', type=str, default='model', help='Name of the job. Used for directory naming')
parser.add_argument('--early_stopping_patience', type=int, default=3, help='Number of epochs to wait for improvement before stopping')
parser.add_argument('--grid_search', type=bool, default=False, help='Enable grid search for hyperparameters')
args = parser.parse_args()
num_epochs = args.num_epochs
early_stopping_patience = args.early_stopping_patience
grid_search = args.grid_search

# Define the grid of hyperparameters for grid search
param_grid = {
    'batch_size': [4, 8],
    'learning_rate': [5e-5, 3e-5],
    'weight_decay': [0.01, 0.001],
    'warmup_steps': [500, 1000]
}

# Dictionary to store the best hyperparameters and the corresponding composite metric
best_params = {
    'batch_size': None,
    'learning_rate': None,
    'weight_decay': None,
    'warmup_steps': None,
    'validation_composite_metric': float('inf')
}

# Early stopping class to stop training if no improvement is seen for a specified number of epochs
class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_composite_max = -np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_composite, model):
        score = val_composite

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_composite, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_composite, model)
            self.counter = 0

    def save_checkpoint(self, val_composite, model):
        '''Saves model when composite metric increases.'''
        if self.verbose:
            self.trace_func(f'Composite metric increased ({self.val_composite_max:.6f} --> {val_composite:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_composite_max = val_composite

# Function to calculate evaluation metrics
def calculate_metrics(predictions, labels):
    preds = predictions.argmax(dim=-1).cpu().numpy()
    labels = labels.cpu().numpy()
    mask = labels != -100  # Mask to ignore padded tokens

    preds = preds[mask]
    labels = labels[mask]

    accuracy = (preds == labels).sum() / len(labels)
    precision = precision_score(labels, preds, average='macro', zero_division=0)
    recall = recall_score(labels, preds, average='macro', zero_division=0)
    f1 = f1_score(labels, preds, average='macro', zero_division=0)
    
    return accuracy, precision, recall, f1

# Function to calculate a composite metric based on evaluation metrics
def composite_metric(accuracy, precision, recall, f1):
    # Example composite metric that weights F1 score highest
    return (0.1 * accuracy) + (0.2 * precision) + (0.2 * recall) + (0.5 * f1)

# Function to train the model with specified hyperparameters
def train_model(num_epochs, batch_size, learning_rate, weight_decay, warmup_steps, early_stopping_patience, job_name):
    accelerator = Accelerator(mixed_precision="fp16")  # Use mixed precision for faster training
    device = accelerator.device

    model_dir = f"./final_model/{args.job_name}"
    best_model_dir = f"{model_dir}_best"
    final_model_dir = f"{model_dir}_final"
    logs_dir = f"./logs/{job_name}"
    token = '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'
    tokenizer_path = "custom_tokenizer_gpt2.json"

    # Load datasets
    dataset_train = load_dataset("Von-R/sub500var", split='train', use_auth_token=token)
    dataset_validation = load_dataset("Von-R/sub500var", split='validation', use_auth_token=token)

    # Load and wrap the custom tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_path)
    wrapped_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    wrapped_tokenizer.pad_token = "[PAD]"
    wrapped_tokenizer.padding_side = "left"

    # Load the pre-trained GPT-2 configuration
    config = AutoConfig.from_pretrained("gpt2", cache_dir=model_dir)

    # Load the pre-trained GPT-2 model
    model = GPT2LMHeadModel.from_pretrained("gpt2", config=config)
    model.to(device)
    # Resize token embeddings to match the custom tokenizer
    model.resize_token_embeddings(len(wrapped_tokenizer))

    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()

    # Prepare model and optimizer with accelerator
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model, optimizer = accelerator.prepare(model, optimizer)

    # Function to tokenize the input content
    def tokenize_function(examples):
        tokenized_inputs = wrapped_tokenizer(examples["content"], padding="max_length", truncation=True, max_length=512)
        tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
        return tokenized_inputs

    columns_to_remove = ['repo_name', 'path', 'size', 'content', 'license']
    tokenized_datasets = dataset_train.map(tokenize_function, batched=True, remove_columns=columns_to_remove)
    tokenized_eval_dataset = dataset_validation.map(tokenize_function, batched=True, remove_columns=columns_to_remove)

    # Create the dataloaders
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

    train_dataloader = DataLoader(tokenized_datasets, shuffle=True, batch_size=batch_size, collate_fn=custom_collate_fn)
    eval_dataloader = DataLoader(tokenized_eval_dataset, batch_size=batch_size, collate_fn=custom_collate_fn)

    # Prepare the dataloaders with accelerator
    train_dataloader, eval_dataloader = accelerator.prepare(train_dataloader, eval_dataloader)

    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )

    # Early stopping object
    early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True, path=f"{best_model_dir}/checkpoint.pt")

    # Training loop with progress bar and logging
    print(f"Training model with batch_size={batch_size}, learning_rate={learning_rate}, weight_decay={weight_decay}, warmup_steps={warmup_steps}...")
    for epoch in range(num_epochs):
        model.train()
        train_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        total_loss = 0
        accumulation_steps = 4  # Accumulate gradients over 4 steps
        for step, batch in enumerate(train_iterator):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss / accumulation_steps  # Scale loss
                accelerator.backward(loss)

            if (step + 1) % accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps  # Unscale the loss

            if step % max(1, len(train_dataloader) // 100) == 0 and step > 0:
                avg_loss = total_loss / (len(train_dataloader) // 100)
                print(f"Step {step} | Loss: {avg_loss:.4f}")
                total_loss = 0

            train_iterator.set_postfix({'loss': loss.item()})

        # Run evaluation at the end of each epoch
        model.eval()
        torch.cuda.empty_cache()  # Clear cache before evaluation
        eval_loss = 0
        eval_accuracy = 0
        eval_precision = 0
        eval_recall = 0
        eval_f1 = 0
        eval_iterator = tqdm(eval_dataloader, desc="Evaluating")
        all_preds = []
        all_labels = []
        for step, batch in enumerate(eval_iterator):
            with torch.no_grad():
                outputs = model(**batch)
                eval_loss += outputs.loss.item()
                all_preds.append(outputs.logits.cpu())
                all_labels.append(batch['labels'].cpu())
                torch.cuda.empty_cache()  # Clear cache to avoid memory overflow

        eval_loss /= len(eval_dataloader)
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        eval_accuracy, eval_precision, eval_recall, eval_f1 = calculate_metrics(all_preds, all_labels)
        eval_composite_metric = composite_metric(eval_accuracy, eval_precision, eval_recall, eval_f1)
        
        print(f"Epoch {epoch + 1} | Validation Loss: {eval_loss} | Composite Metric: {eval_composite_metric}")

        # Early stopping logic and save the best model state
        early_stopping(eval_composite_metric, model)

        if early_stopping.early_stop:
            print(f"Stopping early after {epoch + 1} epochs due to no improvement in composite metric.")
            break

    print("Training complete!")
    # Save the final model and tokenizer
    model.save_pretrained(final_model_dir)
    wrapped_tokenizer.save_pretrained(final_model_dir)
    # Save the config.json file
    config.save_pretrained(final_model_dir)
    print("Final model saved!")

if __name__ == "__main__":
    # If grid search is enabled, iterate over all combinations of hyperparameters
    if grid_search:
        for batch_size, learning_rate, weight_decay, warmup_steps in itertools.product(
            param_grid['batch_size'],
            param_grid['learning_rate'],
            param_grid['weight_decay'],
            param_grid['warmup_steps']
        ):
            job_name_combination = f"{args.job_name}_bs{batch_size}_lr{learning_rate}_wd{weight_decay}_ws{warmup_steps}"
            train_model(
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                warmup_steps=warmup_steps,
                early_stopping_patience=early_stopping_patience,
                job_name=job_name_combination
            )
    else:
        # Use default hyperparameters if grid search is not enabled
        default_params = {
            'batch_size': 8,
            'learning_rate': 5e-5,
            'weight_decay': 0.01,
            'warmup_steps': 500
        }
        train_model(
            num_epochs=num_epochs,
            batch_size=default_params['batch_size'],
            learning_rate=default_params['learning_rate'],
            weight_decay=default_params['weight_decay'],
            warmup_steps=default_params['warmup_steps'],
            early_stopping_patience=early_stopping_patience,
            job_name=args.job_name
        )

    # Save best hyperparameters to JSON file
    if grid_search:
        with open(f'best_hyperparameters_{args.job_name}.json', 'w') as f:
            json.dump(best_params, f, indent=4)
