import json
import torch
import os
from transformers import PreTrainedTokenizer

class TokenizerOutput:
    def __init__(self, tokens, ids):
        self.tokens = tokens
        self.ids = ids

class CustomBPEWrapper(PreTrainedTokenizer):
    def __init__(self, tokenizer, pad_token="[PAD]", eos_token="<|endoftext|>", unk_token="[UNK]", *args, **kwargs):
        self.tokenizer = tokenizer
        # Initialize with pad_token, eos_token, and unk_token. Add more as needed.
        super().__init__(pad_token=pad_token, eos_token=eos_token, unk_token=unk_token, *args, **kwargs)
        self.padding_side = "left"


    def _tokenize(self, text):
        # Use the 'encode' method from your custom tokenizer
        return self.tokenizer.encode(text).tokens

    def encode(self, text, add_special_tokens=True):
        # Directly use the 'encode' method from your custom tokenizer, but ensure the return value is structured correctly
        encoded = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
        # You might need to adjust the following line if your original tokenizer uses different attribute names
        return TokenizerOutput(tokens=encoded.tokens, ids=encoded.ids)

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the tokenizer"""
        return self.tokenizer.token_to_id(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the tokenizer"""
        return self.tokenizer.id_to_token(index)

    def __call__(self, text_or_texts, padding=True, padding_side = 'left', truncation=True, max_length=512, return_tensors=None, *args,
                 **kwargs):
        if isinstance(text_or_texts, str):
            text_or_texts = [text_or_texts]  # Convert single text to a list for uniform processing

        # Encode the texts and directly work with token IDs for truncation
        tokenized_outputs = [self.tokenizer.encode(text, add_special_tokens=True).ids for text in text_or_texts]

        # Apply truncation
        if truncation:
            tokenized_outputs = [output[:max_length] for output in tokenized_outputs]

        # Find the length of the longest sequence after truncation
        max_len = max(len(output) for output in tokenized_outputs)

        # Initialize a padded tensor with the pad token ID
        pad_token_id = self._convert_token_to_id(self.pad_token)
        input_ids = torch.full((len(tokenized_outputs), max_len), pad_token_id, dtype=torch.long)

        # Copy the tokenized output IDs into the tensor, effectively applying left-padding
        for i, output in enumerate(tokenized_outputs):
            input_ids[i, -len(output):] = torch.tensor(output, dtype=torch.long)  # Adjust for left-padding

        # Prepare the output dictionary
        output_data = {"input_ids": input_ids}
        if padding and return_tensors == "pt":
            # Assume that attention_mask is needed if padding is applied
            attention_mask = torch.zeros_like(input_ids)
            attention_mask[input_ids != pad_token_id] = 1
            output_data["attention_mask"] = attention_mask

        return output_data

    def token_to_id(self, token):
        """Converts a token string to its integer ID using the wrapped tokenizer."""
        if hasattr(self.tokenizer, 'token_to_id'):
            return self.tokenizer.token_to_id(token)
        else:
            raise NotImplementedError("The wrapped tokenizer does not support token_to_id method.")

    def get_vocab(self):
        """
        Returns the vocabulary of the tokenizer as a dictionary of tokens to integer IDs.
        """
        # This is a placeholder implementation.
        # Replace `get_vocab()` with the actual method call or attribute access
        # that returns the vocabulary from your custom tokenizer.
        if hasattr(self.tokenizer, 'get_vocab'):
            return self.tokenizer.get_vocab()
        elif hasattr(self.tokenizer, 'vocab'):
            return self.tokenizer.vocab
        else:
            raise NotImplementedError(
                "The custom tokenizer does not have a method or attribute to retrieve the vocabulary.")

    def save_vocabulary(self, save_directory, filename_prefix="tokenizer"):
        """
        Saves the tokenizer vocabulary to a directory.

        Args:
            save_directory (str): The directory where the vocabulary files will be saved.
            filename_prefix (str): Prefix for the filenames under which the vocabulary will be saved.
        """
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory, exist_ok=True)

        # Assuming the custom tokenizer has a method to retrieve the vocabulary
        vocab = self.tokenizer.get_vocab()

        # Save vocabulary in a file
        vocab_file = os.path.join(save_directory, f"{filename_prefix}-vocab.json")
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(vocab, f, ensure_ascii=False)

        return vocab_file



def tokenize_function(examples, tokenizer):
    # Tokenize the examples
    tokenized_outputs = tokenizer(examples["content"],
                                          padding="max_length",
                                          truncation=True,
                                          padding_side="left",
                                          max_length=512)

    input_ids = tokenized_outputs["input_ids"].numpy().tolist()

    # Check if 'attention_mask' is generated, otherwise initialize it
    if "attention_mask" in tokenized_outputs:
        attention_mask = tokenized_outputs["attention_mask"].numpy().tolist()
    else:
        # If not present, generate a mask where every token ID is attended to (1)
        attention_mask = [[1] * len(input_id) for input_id in input_ids]

    labels = input_ids.copy()

    print("tokenize_function::: Attention_mask: ", attention_mask[0])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }