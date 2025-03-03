# the completion of this file includes the assistance from ChatGPT

import argparse
import torch
import random
from transformers import AutoTokenizer, AutoModelForCausalLM

import time
import re

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


def remove_special_chars(text):
    # This regex removes characters that are not word characters or whitespace.
    # In Unicode mode, \w includes letters (from many languages), digits, and the underscore.
    # If you don't want the underscore, you can remove it afterward.
    text = text.replace('\n', '')
    cleaned = re.sub(r'[^\w\s]', '', text, flags=re.UNICODE)
    # cleaned = re.sub(r'\d', '', cleaned)  # Remove numbers
    # Optionally, remove underscore if you consider it a special character:
    cleaned = cleaned.replace('_', '')
    return cleaned


def get_candidate_words_sub(context_text, prefix, top_k, device):
    """
    Given a context text, returns candidate words that start with the specified prefix 
    from the top-k next-token predictions.

    Parameters:
        context_text (str): The input context text to encode.
        prefix (str): The prefix to filter candidate tokens.
        top_k (int): Number of top tokens to retrieve.
        device (torch.device): The device to run the computation on.

    Returns:
        list of tuples: Each tuple contains a candidate word and its corresponding probability.
    """
    # Encode the context text
    context_ids = tokenizer.encode(context_text, return_tensors="pt")
    if torch.cuda.is_available():
        context_ids = context_ids.to(device)

    # Get next-token logits from the model
    logits = get_next_token_logits(context_ids)
    
    # Get top-k token IDs (and their probabilities) for the very next token
    token_id_probs = get_top_k_tokens(logits, k=top_k)

    # Filter to those tokens that start with the given prefix (case-insensitive)
    candidate_words = []
    for t_id, prob in token_id_probs:
        decoded = decode_token(t_id)  # This returns the raw token string
        if not decoded.startswith(" "):
            decoded = remove_special_chars(decoded.strip()) 
            if len(decoded) > 0:
                candidate_words.append((decoded, prob))

    return candidate_words


def get_candidate_words_full(context_text, prefix, top_k, device):
    """
    Given a context text, returns candidate words that start with the specified prefix 
    from the top-k next-token predictions.

    Parameters:
        context_text_sub (str): The input context text to encode.
        prefix (str): The prefix to filter candidate tokens.
        top_k (int): Number of top tokens to retrieve.
        device (torch.device): The device to run the computation on.

    Returns:
        list of tuples: Each tuple contains a candidate word and its corresponding probability.
    """
    # Encode the context text
    context_ids = tokenizer.encode(context_text, return_tensors="pt")
    if torch.cuda.is_available():
        context_ids = context_ids.to(device)

    # Get next-token logits from the model
    logits = get_next_token_logits(context_ids)
    
    # Get top-k token IDs (and their probabilities) for the very next token
    token_id_probs = get_top_k_tokens(logits, k=top_k)

    # Filter to those tokens that start with the given prefix (case-insensitive)
    candidate_words = []
    for t_id, prob in token_id_probs:
        decoded = decode_token(t_id).strip()  # remove leading spaces
        if len(decoded) > len(prefix) and decoded.lower().startswith(prefix.lower()):
            candidate_words.append((decoded, prob))

    return candidate_words


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
        words_raw = user_input.strip()
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
            context_text_sub = words_raw
            context_text_full = None
        else:
            prefix = words[-1]
            context_text_full = " ".join(words[:-1])
            context_text_sub = words_raw

        # print(context_text_full)

        candidate_words_sub = get_candidate_words_sub(context_text_sub, prefix, top_k, device)
        candidate_words_sub = [(prefix + wd, prob) for wd, prob in candidate_words_sub]
        candidate_words_sub.sort(key=lambda x: x[1], reverse=True)

        if context_text_full:
            candidate_words_full = get_candidate_words_full(context_text_full, prefix, top_k, device)
        else:
            candidate_words_full = []
        candidate_words_full.sort(key=lambda x: x[1], reverse=True)

        # print(candidate_words_full)
        # print(candidate_words_sub[:10])

        # candidate_words = candidate_words_sub + candidate_words_full
        # Sort by probability descending
        # candidate_words.sort(key=lambda x: x[1], reverse=True)
        
        # print(candidate_words)

        # word-based
        results = []
        for word, prob in candidate_words_full:
            if len(results) == 3:
                break
            char = word[len(prefix)].lower()
            if char not in results:
                results.append(char)

        # subword-based
        if len(results) < 3:
            for word, prob in candidate_words_sub:
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

        # print(e)

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
    global tokenizer, model, device #, single_char_token_ids
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_name = args.model_dir
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True, torch_dtype=torch.bfloat16)
    
    # model_name = "Qwen/Qwen2.5-3B"
    # model_name = "meta-llama/Llama-3.2-3B"
    # model_name = "work/checkpoints-llama-3"
    # model_name = "work/checkpoints-llama-3"
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # # model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device)
    # model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True, device_map=device)

    # local_model_path = "work/checkpoints-qwen-3"
    # tokenizer.save_pretrained(local_model_path)
    # model.save_pretrained(local_model_path)

    model.eval()
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
