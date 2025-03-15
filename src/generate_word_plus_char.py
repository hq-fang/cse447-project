# the completion of this file includes the assistance from ChatGPT

import argparse
import torch
import random
from transformers import AutoTokenizer, AutoModelForCausalLM

import time
import re

from collections import defaultdict
import os
from typing import List, Dict
import numpy as np
import pickle
import gzip

class WordNGramLMWithInterpolation:
    """
    Remember you can use the inheritance from WordNGramLM in your implementation!
    """

    def __init__(self, N: int, lambdas: List[float]):

        """
        Constructor for WordNGramLMWithInterpolation class.
        Inputs:
            - N: int, the N in N-gram
            - lambdas: List[float], the list of lambdas for interpolation between 1-gram, 2-gram, 3-gram, ..., N-gram models
                Note: The length of lambdas should be N. The sum of lambdas should be 1. lambdas[0] corresponds to 1-gram model, lambdas[1] corresponds to 2-gram model and so on.
        """

        # YOUR CODE HERE
        # raise NotImplementedError()
        self.N = N
        self.ngram_counts = defaultdict(int)  # Count of N-grams
        self.context_counts = defaultdict(int)  # Count of (N-1)-gram contexts
        self.vocab = set()
        self.ngram_probs = {}  # Cache of N-gram probabilities
        self.next_word_probs = {}  # Cache of next-word probabilities for sampling
        self.context_ngram_counts = {}
        self.lambdas = lambdas

    
    def save(self, work_dir: str):
        model_filename = os.path.join(work_dir, 'checkpoint.pickle.gz')
        model_data = {
            "N": self.N,
            "ngram_counts": dict(self.ngram_counts),  # Tuples remain tuples
            "context_counts": dict(self.context_counts),  # Tuples remain tuples
            "vocab": list(self.vocab),
            "ngram_probs": self.ngram_probs,
            "next_word_probs": self.next_word_probs,
            "context_ngram_counts": self.context_ngram_counts,
            "lambdas": self.lambdas
        }
        with gzip.open(model_filename, 'wb') as f:
            pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Model saved to {model_filename} (using pickle)")

    def load(self, work_dir: str):
        model_filename = os.path.join(work_dir, 'checkpoint.pickle.gz')
        if not os.path.exists(model_filename):
            raise FileNotFoundError(f"Checkpoint file {model_filename} not found.")
        with gzip.open(model_filename, 'rb') as f:
            model_data = pickle.load(f)
        self.N = model_data["N"]
        self.lambdas = model_data["lambdas"]
        self.ngram_counts = defaultdict(int, model_data["ngram_counts"])
        self.context_counts = defaultdict(int, model_data["context_counts"])
        self.vocab = set(model_data["vocab"])
        self.ngram_probs = model_data["ngram_probs"]
        self.next_word_probs = model_data["next_word_probs"]
        self.context_ngram_counts = model_data["context_ngram_counts"]
        print(f"Model loaded from {model_filename} (using pickle)")



    def fit(self, train_data: List[str]):

        """
        Trains an N-gram language model with interpolation.

        Inputs:
            - train_data: str, sentences in the training data

        """

        # YOUR CODE HERE
        for sentence in train_data:
            chars = ["<sos>"] * (self.N - 1) + list(sentence)
            self.vocab.update(chars)

            for i in range(len(chars) - self.N + 1):
                for n in range(1, self.N + 1):
                    ngram = tuple(chars[i + self.N - n:i + self.N])
                    context = ngram[:-1] if n > 1 else ()
                    self.ngram_counts[ngram] += 1
                    self.context_counts[context] += 1

        total_ngram_count = sum(self.ngram_counts.values())
        vocab_size = len(self.vocab)

        # ** Precompute probabilities to avoid recomputation in eval_perplexity & sample_text **
        for ngram, count in self.ngram_counts.items():
            token = ngram[-1]
            interpolated_prob = 0

            for n in range(1, self.N + 1, 1):  # Interpolate unigram, bigram, trigram, etc.
                sub_context = ngram[-n:-1]  if n > 1 else ()
                sub_ngram = sub_context + (token,)

                count_sub_ngram = self.ngram_counts.get(sub_ngram, 0)
                count_sub_context = self.context_counts.get(sub_context, total_ngram_count)

                prob = count_sub_ngram / count_sub_context if count_sub_context != 0 else 0
                interpolated_prob += self.lambdas[n - 1] * prob

            self.ngram_probs[ngram] = interpolated_prob

        # ** Precompute possible next words and probabilities for each context **
        temp_context_map = defaultdict(list)
        for ngram, prob in self.ngram_probs.items():
            context = ngram[:-1]
            word = ngram[-1]
            temp_context_map[context].append((word, prob))

        for context, word_prob_list in temp_context_map.items():
            chars, probs = zip(*word_prob_list)
            probs = np.array(probs, dtype=float)
            sum_probs = probs.sum()
            if sum_probs == 0.0:
                self.next_word_probs[context] = (["<unk>"], [1.0])
            else:
                probs /= sum_probs
                self.next_word_probs[context] = (list(chars), probs.tolist())

        # Initialize a dictionary for each order (1 to N)
        self.context_ngram_counts = {n: defaultdict(dict) for n in range(1, self.N + 1)}
        
        # Loop through each n-gram in your counts
        for ngram, count in self.ngram_counts.items():
            n = len(ngram)  # n can be 1 to N
            context = ngram[:-1]  # context is tuple of length n-1; for unigrams, context is ()
            word = ngram[-1]
            
            # Use the context as key in the corresponding order mapping.
            # If the word is already present for that context, add the count.
            if word in self.context_ngram_counts[n][context]:
                self.context_ngram_counts[n][context][word] += count
            else:
                self.context_ngram_counts[n][context][word] = count


    def eval_perplexity(self, eval_data: List[str]) -> float:
        """
        Evaluates the perplexity of the N-gram language model with interpolation on the eval set.

        Input:
            - eval_data: List[str], the evaluation text

        Output:
            - float, the perplexity of the model on the evaluation set

        Note : For tokens that are not in the vocabulary, replace them with the <unk> token.

        """

        # YOUR CODE HERE
        # raise NotImplementedError()
        total_log_prob = 0
        token_count = 0
        total_ngram_count = sum(self.ngram_counts.values())
        for sentence in eval_data:
            chars = ["<sos>"] * (self.N - 1) + list(sentence)
            chars = [char if char in self.vocab else "<unk>" for char in chars]

            for i in range(len(chars) - self.N + 1):
                context = tuple(chars[i:i + self.N - 1])
                token = chars[i + self.N - 1]

                interpolated_prob = 0
                for n in range(1, self.N + 1):
                    sub_context = tuple(chars[i + self.N - n: i + self.N - 1]) if n > 1 else ()
                    ngram = sub_context + (token,)

                    count_ngram = self.ngram_counts.get(ngram, 0)
                    count_context = self.context_counts.get(sub_context, 0)

                    if count_context == 0:
                        continue
                    prob = (count_ngram / count_context)
                    interpolated_prob += self.lambdas[n - 1] * prob

                total_log_prob += np.log(interpolated_prob)
                token_count += 1

        return np.exp(-total_log_prob / token_count)

    def sample_text(self, prefix: str = "<sos>", top_k=10) -> str:

        """
        Samples text from the N-gram language model with interpolation.

        Inputs:
            - prefix: str, the prefix to start the sampling from. Can also be multiple words separated by spaces.
            - max_words: int, the maximum number of words to sample

        Outputs:
            - str, the sampled text

        Note: Please use np.random.choice for sampling next words
        """

        # YOUR CODE HERE
        # raise NotImplementedError()
        words = list(prefix)
        if len(words) < self.N - 1:
            words = ["<sos>"] * (self.N - 1 - len(words)) + words
        # generated_text = words.copy()

        # for _ in range(max_words):
        context = tuple(words[-(self.N - 1):]) if self.N > 1 else ()
        # word_probs = {}

        # for n in range(1, self.N + 1, 1):
        #     sub_context = tuple(words[-(n - 1):]) if n > 1 else ()
        #     for ngram, count in self.ngram_counts.items():
        #         if ngram[:-1] == sub_context:
        #             word_probs[ngram[-1]] = word_probs.get(ngram[-1], 0) + self.lambdas[n - 1] * (count / self.context_counts.get(sub_context, 1))

        word_probs = {}
        # Loop over orders 1 to N (n=1 for unigram, up to N-gram)
        for n in range(1, self.N + 1):
            # For order n, the context length is n-1.
            sub_context = tuple(words[-(n - 1):]) if n > 1 else ()
            
            # Look up the candidate next words for this sub_context.
            # This dictionary was precomputed and holds only the relevant n-grams.
            candidate_dict = self.context_ngram_counts[n].get(sub_context, {})
            
            # Compute and accumulate the weighted probabilities.
            for word, count in candidate_dict.items():
                # Use the total count for sub_context (or 1 if not found, to avoid division by zero)
                context_count = self.context_counts.get(sub_context, 1)
                p = count / context_count
                if word in eng_chars and word != " ":
                    word_probs[word] = word_probs.get(word, 0) + self.lambdas[n - 1] * p


        # if not word_probs:
        #     break

        chars, probs = zip(*word_probs.items())
        # sampled_word = np.random.choice(words, p=np.array(probs) / sum(probs))
        # Get the top 3 indices by sorting the probabilities
        top_k_indices = np.argsort(probs)[-top_k:][::-1]  # Sort in descending order

        # Get the corresponding top 3 words using the sorted indices
        top_k_chars = [chars[i] for i in top_k_indices]

        top_k_pairs = [(chars[i], probs[i]) for i in top_k_indices]


        # return top_k_pairs

        return top_k_pairs, "".join(top_k_chars)


tokenizer = None
model = None
device = None
ngram_lm = None

eng_chars = set(list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ "))

lambdas = [0.21559011436417874, 0.22613303857691858, 0.2070162756071627, 0.13131311521748157, 0.08609488206866313, 0.05620449593829095, 0.0339299193961247, 0.021772810756838616, 0.012385328857100947, 0.009560019217240064]


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
        # print(decoded)
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
        # print(decoded)
        if len(decoded) > len(prefix) and decoded.lower().startswith(prefix.lower()):
            candidate_words.append((decoded, prob))

    return candidate_words


def get_suggestions(user_input, top_k=50, threshold=0):
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

        # print(candidate_words_full)
        # print(candidate_words_sub[:10])

        # candidate_words = candidate_words_sub + candidate_words_full
        # Sort by probability descending
        # candidate_words.sort(key=lambda x: x[1], reverse=True)
        
        # print(candidate_words)

        # context based
        # predict word
        results = []

        if set(remove_special_chars(words_raw)).issubset(eng_chars):
            # print('use ngram')
            pairs, _ = ngram_lm.sample_text(words_raw, top_k=20)

            tre = threshold

            for c, prob in pairs:
                if len(results) >= 3:
                    break
                char = c.lower()
                if char not in results and prob >= tre:
                    results.append(char)

        results_next = []
        need = 3 - len(results)

        # composition based
        # ngram
        if need > 0: # and set(remove_special_chars(words_raw)).issubset(eng_chars):
            # print('use ngram')
            # pairs, _ = ngram_lm.sample_text(words_raw, top_k=20)

            # tre = threshold

            # for c, prob in pairs:
            #     if len(results_next) >= need:
            #         break
            #     char = c.lower()
            #     if char not in results and char not in results_next and prob >= tre:
            #         results_next.append(char)

            if context_text_full:
                candidate_words_full = get_candidate_words_full(context_text_full, prefix, top_k, device)
            else:
                candidate_words_full = []
            candidate_words_full.sort(key=lambda x: x[1], reverse=True)

            for word, prob in candidate_words_full:
                if len(results_next) >= need:
                    break
                char = word[len(prefix)].lower()
                if char not in results and char not in results_next:
                    results_next.append(char)

        # subword based
        # predict subword
        # print('length of results:', len(results_next) + len(results))
        if len(results_next) < need:

            candidate_words_sub = get_candidate_words_sub(context_text_sub, prefix, top_k, device)
            candidate_words_sub = [(prefix + wd, prob) for wd, prob in candidate_words_sub]
            candidate_words_sub.sort(key=lambda x: x[1], reverse=True)

            for word, prob in candidate_words_sub:
                if len(results_next) >= need:
                    break
                char = word[len(prefix)].lower()
                if char not in results_next and char not in results:
                    results_next.append(char)

        results += results_next

        # If none match, fill all with random single chars
        if len(results) == 0:

            # print('###############')
            # print(words_raw)
            # print(results)

            while len(results) < 3:
                char_rand = fill_random_chars(1)
                if char_rand not in results:
                    results.append(char_rand)

            return results

        elif len(results) < 3:
            # print('###############')
            # print(words_raw)
            # print(results)
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


def process_file(input_path, output_path, top_k, threshold):
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            # Get up to 3 distinct next-substrings from GPT-2
            suggestions = get_suggestions(line, top_k=top_k, threshold=threshold)
            
            # Usually each element in `suggestions` is a short substring like "l", "t", "o"
            # We join them into one string => e.g. "lto"
            out_str = "".join(suggestions)

            # Write each line’s result to output
            fout.write(out_str + "\n")


def main(args):
    global tokenizer, model, device, ngram_lm #, single_char_token_ids
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("loading llm")
    model_name = args.model_dir
    # model_name = 'Qwen/Qwen2.5-0.5B'
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

    print("loading ngram")
    ngram_lm = WordNGramLMWithInterpolation(10, lambdas)
    ngram_lm.load('work/checkpoints-ngram')

    # start_time = time.time()

    process_file(args.test_input, args.test_output, args.top_k, args.threshold)

    # end_time = time.time()

    # elapsed_time = end_time - start_time
    # print("Time elapsed:", elapsed_time, "seconds")

    print(f"Done! Results written to {args.test_output}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text using a fine-tuned GPT-2 model.")

    # Add command-line arguments
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the fine-tuned model directory.")
    parser.add_argument("--top_k", type=int, default=100, help="Number of possible next words.")
    parser.add_argument("--threshold", type=float, default=0.0, help="Threshold to take ngram preds.")
    parser.add_argument("--test_input", type=str, required=True, help="Path to the input .txt file with many lines of text.")
    parser.add_argument("--test_output", type=str, required=True, help="Path to save the generated text.")


    # Parse arguments
    args = parser.parse_args()

    # Run the main function
    main(args)
