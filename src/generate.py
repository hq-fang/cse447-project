# the completion of this file includes the assistance from ChatGPT

import argparse
import torch
import random
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
import pandas as pd

# Define your character-level tokenizer mappings (ensure they match training)

# dataset = load_dataset('zeroshot/twitter-financial-news-topic')
# aya_dataset = dataset["train"]
# num_data = len(aya_dataset["text"])
# train_texts = aya_dataset[:]["text"]

# Define the character set and tokenizer mapping
# chars = set([c for s in train_texts for c in s])
# chars.update(list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 \n!.,?;:'\"()-"))
chars = set(list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 \n!.,?;:'\"()-"))
# print(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

def encode(text):
    """Encode text into indices, skipping characters not in dictionary."""
    return [char_to_idx[ch] for ch in text if ch in char_to_idx]

def decode(indices):
    """Decode indices into text."""
    return ''.join([idx_to_char[i] for i in indices if i in idx_to_char])

def generate_best_chars(model, start_string, device, num_candidates=3):
    """Generate `num_candidates` best possible next alphabetic characters."""

    try:
        if not isinstance(start_string, str) or not start_string or any(c not in char_to_idx for c in start_string):
            return ''.join(random.choices(list(char_to_idx.keys()), k=num_candidates))
        
        model.eval()
        input_ids = encode(start_string)
        
        with torch.no_grad():
            # Ensure tensor is on the correct device
            input_tensor = torch.tensor([input_ids]).to(device)

            # Get the logits for the next token
            outputs = model(input_tensor)
            logits = outputs.logits[:, -1]  # Get logits for the last token

            # Get the top `num_candidates` tokens
            topk_values, topk_indices = torch.topk(logits, k=num_candidates * 2, dim=-1)
            
            # Generate a list of possible next alphabetic characters, filtering out non-alphabetic ones
            possible_next_chars = [idx_to_char[topk_indices[0][i].item()] for i in range(len(topk_indices[0])) if idx_to_char[topk_indices[0][i].item()].isalpha()][:num_candidates]
            
            # If not enough valid characters, fill randomly
            while len(possible_next_chars) < num_candidates:
                possible_next_chars.append(random.choice(list(chars)))

        return possible_next_chars  # Return the top alphabetic characters as a list

    except Exception as e:
        return ''.join(random.choices(list(char_to_idx.keys()), k=num_candidates))


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(args.model_dir)
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_dir)
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)

    # Read input file
    with open(args.test_data, 'r') as f:
        input_lines = f.readlines()

    # Generate text for each line in the input file
    generated_texts = []
    for line in input_lines:
        line = line.strip()
        best_chars = generate_best_chars(model, line, device, args.num_candidates)
        generated_texts.append(''.join(best_chars))

    # Write only the generated text to output file
    with open(args.test_output, 'w') as f:
        for text in generated_texts:
            f.write(text + '\n')

    print(f"Generated text has been saved to {args.test_output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text using a fine-tuned GPT-2 model.")

    # Add command-line arguments
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the fine-tuned model directory.")
    parser.add_argument("--test_data", type=str, required=True, help="Path to the input .txt file with many lines of text.")
    parser.add_argument("--num_candidates", type=int, default=3, help="Number of possible next characters to generate.")
    parser.add_argument("--test_output", type=str, required=True, help="Path to save the generated text.")

    # Parse arguments
    args = parser.parse_args()

    # Run the main function
    main(args)