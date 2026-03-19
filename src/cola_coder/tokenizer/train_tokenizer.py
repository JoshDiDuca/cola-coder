"""Train a BPE tokenizer on code data.

A tokenizer converts text into numbers (token IDs) that the model can process.
BPE (Byte Pair Encoding) is the standard algorithm used by GPT, LLaMA, etc.

How BPE works (simplified):
1. Start with individual characters as tokens
2. Find the most common pair of adjacent tokens (e.g., "t" + "h")
3. Merge that pair into a new token ("th")
4. Repeat 32,768 times (or however many tokens you want)

The result: common words/patterns become single tokens, rare text is split
into smaller pieces. This balances vocabulary size vs sequence length.

For code specifically, the tokenizer needs to handle:
- Indentation (spaces/tabs matter in Python!)
- Common keywords (def, function, class, etc.)
- Operators (==, !=, +=, etc.)
- String delimiters (' " ` ''')
- Numbers
"""

from pathlib import Path
from typing import Iterator

from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders


# Special tokens that have specific meanings in our model
SPECIAL_TOKENS = [
    "<|pad|>",      # Padding token (fills unused positions in a batch)
    "<|bos|>",      # Beginning of sequence
    "<|eos|>",      # End of sequence
    "<|unk|>",      # Unknown token (fallback for characters not in vocab)
    "<|fim_prefix|>",  # Fill-in-the-middle: text before the gap
    "<|fim_middle|>",  # Fill-in-the-middle: text in the gap (what model generates)
    "<|fim_suffix|>",  # Fill-in-the-middle: text after the gap
]


def create_tokenizer(vocab_size: int = 32768) -> tuple[Tokenizer, trainers.BpeTrainer]:
    """Create a BPE tokenizer configured for code.

    Returns:
        (tokenizer, trainer) tuple. The tokenizer is the engine,
        the trainer is the algorithm that learns the vocabulary.
    """
    # BPE model: the core algorithm
    tokenizer = Tokenizer(models.BPE())

    # Pre-tokenizer: how to split text BEFORE BPE learning
    # ByteLevel means:
    # 1. Convert all bytes to visible characters (handles any encoding)
    # 2. Split on whitespace boundaries
    # add_prefix_space=False: don't add a space at the start of text
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # Decoder: converts token IDs back to text
    tokenizer.decoder = decoders.ByteLevel()

    # Trainer: configures the BPE learning process
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        # These control how the trainer finds merge pairs:
        min_frequency=2,  # A pair must appear at least 2x to be merged
        show_progress=True,
        # Initial alphabet: all byte values (ensures we can encode anything)
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    return tokenizer, trainer


def train_from_iterator(
    iterator: Iterator[str],
    vocab_size: int = 32768,
    output_path: str = "tokenizer.json",
) -> Tokenizer:
    """Train a tokenizer from an iterator of text strings.

    Args:
        iterator: Yields strings of code to learn vocabulary from.
        vocab_size: How many tokens to learn (32K is standard for code models).
        output_path: Where to save the trained tokenizer.

    Returns:
        The trained tokenizer, ready to encode/decode.
    """
    tokenizer, trainer = create_tokenizer(vocab_size)

    # This is where the actual learning happens:
    # The trainer iterates through all the text, counts byte pair frequencies,
    # and merges the most common pairs until we have vocab_size tokens
    tokenizer.train_from_iterator(iterator, trainer)

    # Save to disk (a JSON file with the full vocabulary and merge rules)
    tokenizer.save(output_path)
    print(f"Tokenizer saved to {output_path}")
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")

    return tokenizer


def train_from_files(
    file_paths: list[str],
    vocab_size: int = 32768,
    output_path: str = "tokenizer.json",
) -> Tokenizer:
    """Train a tokenizer from a list of text files.

    Args:
        file_paths: Paths to text files containing code.
        vocab_size: Target vocabulary size.
        output_path: Where to save the trained tokenizer.

    Returns:
        The trained tokenizer.
    """
    tokenizer, trainer = create_tokenizer(vocab_size)
    tokenizer.train(file_paths, trainer)
    tokenizer.save(output_path)
    print(f"Tokenizer saved to {output_path}")
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    return tokenizer


def load_tokenizer(path: str = "tokenizer.json") -> Tokenizer:
    """Load a previously trained tokenizer from disk."""
    return Tokenizer.from_file(path)
