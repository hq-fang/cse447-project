# the completion of this file includes the assistance from ChatGPT

import argparse
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
import pandas as pd

# Define your character-level tokenizer mappings (ensure they match training)

dataset = load_dataset('zeroshot/twitter-financial-news-topic')
aya_dataset = dataset["train"]
num_data = len(aya_dataset["text"])
train_texts = aya_dataset[:]["text"]

# Define the character set and tokenizer mapping
chars = set([c for s in train_texts for c in s])
chars.update(list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 \n!.,?;:'\"()-"))
# print(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

def encode(text):
    """Encode text into indices."""
    return [char_to_idx[ch] for ch in text]

def decode(indices):
    """Decode indices into text."""
    return ''.join([idx_to_char[i] for i in indices])

def generate_text(model, start_string, num_candidates=3):
    """Generate `num_candidates` possible next characters for each step."""
    model.eval()
    input_ids = encode(start_string)
    result = ""  # Start with the initial input string

    while True:
        with torch.no_grad():
            # Get the logits for the next token
            outputs = model(torch.tensor([input_ids]).cuda())
            logits = outputs.logits[:, -1]  # Get logits for the last token

            # Get the top `num_candidates` tokens
            topk_values, topk_indices = torch.topk(logits, k=num_candidates, dim=-1)

            # Generate a list of possible next characters
            possible_next_chars = [idx_to_char[topk_indices[0][i].item()] for i in range(num_candidates)]

            # Append the possible characters to the result
            result += ''.join(possible_next_chars)  # Separate candidates with "|"

            # Just for illustration, you can stop or continue generating as per your needs
            # If you want to limit the generation, we can stop after adding the first batch
            return result  # Stop here for now, or you can continue if needed.

def main(args):
    # Load model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(args.model_dir)
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_dir)
    model.resize_token_embeddings(len(tokenizer))
    model = model.cuda()

    # Read input file
    with open(args.test_data, 'r') as f:
        input_lines = f.readlines()

    # Generate text for each line in the input file
    generated_texts = []
    for line in input_lines:
        line = line.strip()  # Remove any extra whitespace or newline characters
        generated_text = generate_text(model, line, args.num_candidates)
        generated_texts.append(generated_text)

    # Write only the generated text to output file (no input text)
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