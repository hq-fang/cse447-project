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
print(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

def encode(text):
    """Encode text into indices."""
    return [char_to_idx[ch] for ch in text]

def decode(indices):
    """Decode indices into text."""
    return ''.join([idx_to_char[i] for i in indices])

def generate_text(model, start_string, length=100):
    """Generate text using the trained model."""
    model.eval()
    input_ids = encode(start_string)
    result = start_string
    for _ in range(length):
        with torch.no_grad():
            outputs = model(torch.tensor([input_ids]).cuda())
            topk_values, topk_indices = torch.topk(outputs.logits[:, -1], k=3, dim=-1)
            best_index = topk_indices[0][1].item()  # Pick the second-best token for variety
            result += idx_to_char[best_index]
            input_ids.append(best_index)
    return result

def main(args):
    # Load model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(args.model_dir)
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_dir)
    model.resize_token_embeddings(len(tokenizer))
    model = model.cuda()

    # Generate text
    generated_text = generate_text(model, args.start_string, args.length)
    print("\nGenerated Text:")
    print("-" * 50)
    print(generated_text)
    print("-" * 50)

    # Save generated text to file (optional)
    if args.output_file:
        with open(args.output_file, "w") as f:
            f.write(generated_text)
        print(f"\nGenerated text saved to {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text using a fine-tuned GPT-2 model.")
    
    # Add command-line arguments
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the fine-tuned model directory.")
    parser.add_argument("--start_string", type=str, required=True, help="Starting string for text generation.")
    parser.add_argument("--length", type=int, default=100, help="Length of the generated text.")
    parser.add_argument("--output_file", type=str, default=None, help="File to save the generated text.")
    
    args = parser.parse_args()
    main(args)
