**Verilog Next Token Prediction Model**
Author: Von Davis
Huggingface repository: Von-R/VerilogProtoToken
Huggingface dataset: Von-R/verilog_preprocessed_anonymized
Contact: Von.Roth.1991@gmail.com
https://huggingface.co/Von-R/VerilogProtoToken/blob/main/README.md

**Project Overview**

Verilog is a Hardware Description Language (HDL) used for designing electronic systems such as Field Programmable Gate Arrays (FPGAs) and Application-Specific Integrated Circuits (ASICs). Programming in Verilog can be tedious and time-consuming. This project aims to create a foundation for a future Verilog coding copilot—a tool designed to streamline the coding process, ensure best practices, and reduce errors.

**Model Description**

The model is trained for next-token prediction using cleaned and open-source Verilog code from GitHub. It leverages transformer models, which are effective at understanding long-range dependencies in sequences, making them ideal for the dynamic nature of Verilog code. By fine-tuning pre-trained transformer models, this project aims to predict the next token in a sequence, setting the stage for a future Verilog code copilot.

Additionally, a Long Short-Term Memory (LSTM) model was developed to provide a comparative baseline for performance evaluation.

**Data Collection and Preprocessing**

**Scraping**

Verilog code with permissive licenses was scraped from GitHub using Google BigQuery. This process targeted repositories containing Verilog code, filtering based on file extensions (.v and .sv) and checking for permissive licenses (MIT, BSD, Apache 2.0). The data was hosted on Huggingface Datasets.

**Preprocessing Steps**

Remove Duplicates: Reduced the dataset size from 2.78 GB to 1.54 GB.
Remove Non-Synthesizable/Non-Verilog Code: Filtered out simulation and verification code to focus on synthesizable Verilog.
Removing Comments and Whitespace: Used regular expressions to strip out comments and non-essential elements.
Identifier Anonymization: Replaced unique identifiers with generic placeholders to reduce vocabulary size and improve model performance.
Dropping Short Files: Removed files with fewer than 30 lines of code to maintain high-quality training data.

**Tokenizer Training**

Tokenizers were trained using the Byte Pair Encoding (BPE) algorithm. The following models were considered:

GPT-2: Vocabulary size of 42000~ token
Evaluations showed that GPT-2 tokenizer achieved 100% vocabulary coverage of dataset tokens and 0% token out-of-vocabulary.

**Model Training**

The training script included setting up the environment, defining the transformer model, initializing the loss function and optimizer, and implementing early stopping to avoid overfitting. Grid search was used to determine the best hyperparameters for optimal model performance.

**Training Hyperparameters**

Training regime: fp32
Learning rate: 5e-5
Batch size: 16
Epochs: 3

**Evaluation Metrics**

Next Token Prediction Loss: 0.8175709573030472
Perplexity: 2.2649913893200004
Accuracy (Top-1): 0.52189453125
Precision: 0.023324851765829015
Recall: 0.023883036472085516
F1 Score: 0.02345157579189002
Top-5 Accuracy: 0.50113671875
Entropy: 0.8339920132160187
Prediction Confidence: 0.8293982080221176

**Comparison with LSTM Model**

**LSTM Training Results**

**1 Epoch:**

Next Token Prediction Loss: 8.31
Perplexity: 4056.09
Accuracy: 0.0
Precision, Recall, F1 Score: 0.0

**10 Epochs:**

Next Token Prediction Loss: 6.13
Perplexity: 461.57
Accuracy: 0.45
Precision: 0.016
Recall: 0.021
F1 Score: 0.018

**50 Epochs:**

Next Token Prediction Loss: 6.29
Perplexity: 539.42
Accuracy: 0.44
Precision: 0.016
Recall: 0.020
F1 Score: 0.017

**GPT-2 Training Results**

**1 Epoch:**

Next Token Prediction Loss: 0.82
Perplexity: 2.27
Accuracy: 0.52
Precision: 0.023
Recall: 0.024
F1 Score: 0.023

**10 Epochs:**

Next Token Prediction Loss: 0.78
Perplexity: 2.17
Accuracy: 0.33
Precision: 0.024
Recall: 0.016
F1 Score: 0.018

**50 Epochs:**

Next Token Prediction Loss: 1.18
Perplexity: 3.26
Accuracy: 0.18
Precision: 0.024
Recall: 0.009
F1 Score: 0.011

**Analysis**

**Learning Speed:**

GPT-2 shows a significantly lower loss and perplexity after just one epoch compared to the LSTM model, indicating quicker learning.
LSTM shows improvement over multiple epochs but learns more slowly overall.

**Accuracy and F1 Scores:**

GPT-2 has higher initial accuracy and F1 scores after the first epoch compared to LSTM.
LSTM shows better accuracy at 10 epochs compared to GPT-2 at the same number of epochs but does not surpass GPT-2's one-epoch performance.

**Perplexity:**

GPT-2's lower perplexity in early epochs suggests higher confidence in predictions.
LSTM's higher perplexity indicates less confidence initially but decreases significantly with more epochs.

**Overfitting:**

GPT-2 shows signs of rapid overfitting with additional epochs.
LSTM does not overfit as quickly, suggesting better robustness for longer training periods.

**Future Improvements**

Identifier and Tokenization Enhancements: Implement deep metric learning and k-means clustering for more meaningful identifier separation. Refine tokenization to handle underscores, common prefixes, and suffixes better.
Regularization Techniques: Introduce noise during training and apply various regularization methods to improve generalization.
Model Evaluation and Validation: Use k-fold validation and assess model performance at a granular level through manual inspection.
Model Optimization Techniques: Explore model pruning, quantization, and distillation to enhance efficiency and performance.

**Conclusion**

This project lays the groundwork for a Verilog code copilot, aiming to make Verilog coding more efficient and accurate. By leveraging advanced machine learning techniques, this project aspires to improve the Verilog coding process, making it more accessible and less error-prone.
