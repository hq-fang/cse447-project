# the completion of this file includes the assistance from ChatGPT

import argparse
import torch
import random
from transformers import GPT2LMHeadModel, GPT2Tokenizer

import time

tokenizer = None
model = None
device = None

def get_next_token_logits(context_ids):
    """
    Given the tokenized context_ids (without the last word prefix),
    return the model's logits for the *next token* in sequence.
    """
    with torch.no_grad():
        outputs = model(context_ids)
        # outputs.logits.shape = [batch_size, sequence_length, vocab_size]
        next_token_logits = outputs.logits[0, -1, :]
    return next_token_logits

def decode_token(token_id):
    """
    Helper to decode a single token ID into text.
    """
    return tokenizer.decode([token_id], clean_up_tokenization_spaces=False)

def get_top_k_tokens(logits, k=50):
    """
    Return top-k token IDs and their logits (or probabilities).
    """
    top_k_values, top_k_indices = torch.topk(logits, k)
    top_k_probs = torch.softmax(top_k_values, dim=-1).tolist()
    top_k_indices = top_k_indices.tolist()
    
    # Pair them up for convenience: [(token_id, probability), ...]
    token_id_probs = list(zip(top_k_indices, top_k_probs))
    return token_id_probs

def fill_random_chars(n=1):
    """
    Returns a string of length n from random English letters/digits.
    (If you want strictly letters, remove digits from 'alphabet'.)
    """
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    return "".join(random.choice(alphabet) for _ in range(n))

def get_gpt2_suggestions(user_input, top_k=50):
    """
    - Split user_input into context (all but last word) and prefix (last word).
    - Use GPT-2 to get top K next-token predictions given the context.
    - Filter for tokens that start with prefix.
    - Take the top 3 (by probability). If fewer than 3, fill the rest randomly.
    - Extract the "next characters" from each word, ensuring minimal collisions.
    - Return a list of up to 3 substrings, typically 1 char each (but can be longer).
    """

    try:
        words = user_input.strip().split()
        if len(words) < 1:
            # If there's only one word or none, short-circuit
            # We'll just fill 3 random single-chars
            results = []
            while len(results) < 3:
                char_rand = fill_random_chars(1)
                if char_rand not in results:
                    results.append(char_rand)

            return results

        elif len(words) == 1:
            prefix = words[-1]
            context_text = "."
        else:
            prefix = words[-1]
            context_text = " ".join(words[:-1])

        # Encode the context text
        context_ids = tokenizer.encode(context_text, return_tensors="pt")
        if torch.cuda.is_available():
            context_ids = context_ids.to(device)

        # Get next-token logits from the model
        logits = get_next_token_logits(context_ids)
        
        # Get top-k token IDs (and their probabilities) for the *very next* token
        token_id_probs = get_top_k_tokens(logits, k=top_k)

        # Filter to those that start with the prefix (case-insensitive)
        candidate_words = []
        for t_id, prob in token_id_probs:
            decoded = decode_token(t_id).strip()  # remove leading spaces
            if len(decoded) > len(prefix) and decoded.lower().startswith(prefix.lower()):
                candidate_words.append((decoded, prob))

        # Sort by probability descending
        candidate_words.sort(key=lambda x: x[1], reverse=True)
        # print(candidate_words)
        # Take top 3
        results = []
        for word, prob in candidate_words:
            if len(results) == 3:
                break
            char = word[len(prefix)].lower()
            if char not in results:
                results.append(char)

        # If none match, fill all with random single chars
        if len(results) == 0:

            while len(results) < 3:
                char_rand = fill_random_chars(1)
                if char_rand not in results:
                    results.append(char_rand)

            return results

        elif len(results) < 3:
            # If we only have 1 or 2, fill the rest randomly
            needed = 3 - len(results)
            
            placeholders = []
            while len(placeholders) < needed:
                char_rand = fill_random_chars(1)
                if char_rand not in placeholders and char_rand not in results:
                    placeholders.append(char_rand)

            results = results + placeholders

        return results
    
    except Exception as e:

        results = []
        while len(results) < 3:
            char_rand = fill_random_chars(1)
            if char_rand not in results:
                results.append(char_rand)

        return results


def process_file(input_path, output_path, top_k):
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            # Get up to 3 distinct next-substrings from GPT-2
            suggestions = get_gpt2_suggestions(line, top_k=top_k)
            
            # Usually each element in `suggestions` is a short substring like "l", "t", "o"
            # We join them into one string => e.g. "lto"
            out_str = "".join(suggestions)

            # Write each line’s result to output
            fout.write(out_str + "\n")


def main(args):
    global tokenizer, model, device
    
    model_name = args.model_dir
    tokenizer = GPT2Tokenizer.from_pretrained(model_name, local_files_only=True)
    model = GPT2LMHeadModel.from_pretrained(model_name, local_files_only=True)

    # model_name = "gpt2-medium"
    # tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    # model = GPT2LMHeadModel.from_pretrained(model_name)
    # local_model_path = "./checkpoints"
    # tokenizer.save_pretrained(local_model_path)
    # model.save_pretrained(local_model_path)

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        model.to(device)

    # start_time = time.time()

    process_file(args.test_input, args.test_output, args.top_k)

    # end_time = time.time()

    # elapsed_time = end_time - start_time
    # print("Time elapsed:", elapsed_time, "seconds")

    print(f"Done! Results written to {args.test_output}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text using a fine-tuned GPT-2 model.")

    # Add command-line arguments
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the fine-tuned model directory.")
    parser.add_argument("--top_k", type=int, default=100, help="Number of possible next words.")
    parser.add_argument("--test_input", type=str, required=True, help="Path to the input .txt file with many lines of text.")
    parser.add_argument("--test_output", type=str, required=True, help="Path to save the generated text.")


    # Parse arguments
    args = parser.parse_args()

    # Run the main function
    main(args)
