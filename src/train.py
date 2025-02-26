# the completion of this file includes the assistance from ChatGPT

import argparse
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from torch.utils.data import Dataset
from datasets import load_dataset
import torch
import pandas as pd

# Define the character-level dataset
class CharDataset(Dataset):
    def __init__(self, data, seq_length, char_to_idx):
        self.text = "".join(data)
        self.seq_length = seq_length
        self.char_to_idx = char_to_idx
        self.data = self.encode(self.text)

    def encode(self, text):
        return [self.char_to_idx[ch] for ch in text]

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length]
        y = self.data[idx + 1: idx + self.seq_length + 1]
        return {
            "input_ids": torch.tensor(x, dtype=torch.long),
            "labels": torch.tensor(y, dtype=torch.long)
        }

def preprocess_text(sentence, char):
    """
    Removes non-English words while keeping numbers and punctuation.
    """
    # Filter non-English words, keep numbers & punctuation
    filtered_tokens = [word for word in sentence if word in char]

    return "".join(filtered_tokens)

# Define the main training function
def main(args):
    # Load the dataset
    # dataset = load_dataset("CohereForAI/aya_dataset")
    dataset = load_dataset("sentence-transformers/wikipedia-en-sentences")
    aya_dataset = dataset["train"]
    # num_data = len(aya_dataset["text"])
    # train_texts = aya_dataset[:]["text"]

    # df = pd.DataFrame(dataset["train"])  # Convert to pandas DataFrame

    # # Filter to keep only English language entries
    # english_data = df[df['language'] == 'English']
    # train_texts = english_data['inputs'].tolist()

    # Define the character set and tokenizer mapping
    # chars = set([c for s in train_texts for c in s])
    chars = set(list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 \n!.,?;:'\"()-"))

    train_texts = []
    # dataset too large, choose first 100k
    for sentence in aya_dataset['sentence'][:100000]:
        train_texts.append(preprocess_text(sentence, chars))

    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}

    # Prepare datasets
    seq_length = args.seq_length
    full_dataset = CharDataset(train_texts, seq_length, char_to_idx)

    # Split into train/validation sets
    train_size = int(0.99 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    # Load pre-trained GPT-2 model and tokenizer
    model_name = "gpt2-medium"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Add new tokens for character-level handling
    new_tokens = set(chars) - set(tokenizer.get_vocab().keys())
    tokenizer.add_tokens(list(new_tokens))
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        logging_dir=args.logging_dir,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        fp16=args.fp16,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_strategy="steps"
    )

    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer
    )

    # Train the model
    trainer.train()

    # Save the model checkpoint
    # torch.save({
    #     'epoch': training_args.num_train_epochs,
    #     'model_state_dict': model.state_dict(),
    #     'optimizer_state_dict': trainer.optimizer.state_dict(),
    #     'loss': trainer.state.global_step,
    # }, args.checkpoint_path)

    print(f"Training completed. Model saved to {args.output_dir}")

if __name__ == "__main__":
    # Define argument parser
    parser = argparse.ArgumentParser(description="Train GPT-2 on a character-level dataset")

    # Add arguments
    parser.add_argument("--output_dir", type=str, default="./char_gpt2_finetune", help="Output directory for the model")
    parser.add_argument("--checkpoint_path", type=str, default="model.pt", help="Path to save the model checkpoint")
    parser.add_argument("--seq_length", type=int, default=64, help="Sequence length for the dataset")
    parser.add_argument("--eval_steps", type=int, default=1000, help="Steps between evaluations")
    parser.add_argument("--logging_dir", type=str, default="./logs", help="Directory for logs")
    parser.add_argument("--logging_steps", type=int, default=200, help="Steps between logging")
    parser.add_argument("--save_steps", type=int, default=1000, help="Steps between model checkpoints")
    parser.add_argument("--save_total_limit", type=int, default=3, help="Total number of checkpoints to keep")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=32, help="Training batch size per device")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32, help="Evaluation batch size per device")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate for training")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for the optimizer")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision training")

    # Parse arguments
    args = parser.parse_args()

    # Run the training pipeline
    main(args)
