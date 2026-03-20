# Feature 28: Byte-Level Fallback

**Status:** Optional | **CLI Flag:** `--byte-fallback` | **Complexity:** Medium

---

## Overview

Extend the Cola-Coder tokenizer with 256 byte-level tokens to handle any Unicode character, unusual code patterns, and characters that would otherwise produce `<unk>`. The primary tokenizer remains BPE (Byte Pair Encoding), but when BPE cannot encode a character, it falls back to byte-level encoding. A small percentage of training data is processed through the byte path to teach the model byte-level decoding. Handles all Unicode, binary literals, exotic identifiers, and non-ASCII comments in code.

Reference: ByT5 (Xue et al., 2022); byte fallback in SentencePiece (`byte_fallback=True`).

---

## Motivation

TypeScript/JavaScript codebases contain:
- Unicode identifiers: `const café = "coffee";`, `const π = Math.PI;`
- Template literals with emoji: `` const msg = `Hello 🌍`; ``
- Non-ASCII string literals in i18n code
- Comments in Chinese, Japanese, Arabic (common in open-source projects)
- Binary/hex literals: `0xFF`, `0b10101010`
- Regex patterns with unusual characters

Without byte fallback, these cases produce `<unk>` tokens which:
- Break the model's ability to regenerate the exact character
- Create training noise (the model learns to ignore `<unk>`)
- Prevent the model from completing code that includes non-ASCII

SentencePiece's `byte_fallback=True` mode adds 256 byte tokens (`<0x00>` through `<0xFF>`) and routes unknown characters through byte-level encoding. Cola-Coder should replicate this behavior.

---

## Architecture / Design

### Vocabulary Extension

```
Base BPE vocabulary:  tokens 0 to V-1    (e.g., V=32000)
Byte tokens added:    tokens V to V+255  (256 new tokens)
  - <0x00> = V
  - <0x01> = V+1
  - ...
  - <0xFF> = V+255

New vocab_size = V + 256 = 32256
```

### Encoding Algorithm

```python
def encode_with_fallback(text: str) -> list[int]:
    tokens = bpe.encode(text)  # Standard BPE
    result = []
    for token in tokens:
        if token == unk_id:
            # Find the problematic character and byte-encode it
            ...  # See implementation
        else:
            result.append(token)
    return result
```

### Training Data Mix

```
95% of training data: standard BPE tokenization (no byte fallback needed)
5% of training data:  byte-heavy text (non-ASCII heavy code, or forced byte encoding)
```

The 5% ensures the model learns to decode byte sequences back into characters.

---

## Implementation Steps

### Step 1: Byte Token Definitions

```python
# cola_coder/tokenizer/byte_tokens.py

BYTE_TOKEN_PREFIX = "<0x"
BYTE_TOKEN_FORMAT = "<0x{:02X}>"

def byte_token_id(byte_val: int, base_vocab_size: int) -> int:
    """Return the token ID for a given byte value (0-255)."""
    return base_vocab_size + byte_val

def id_to_byte(token_id: int, base_vocab_size: int) -> int | None:
    """Return the byte value if token_id is a byte token, else None."""
    idx = token_id - base_vocab_size
    if 0 <= idx <= 255:
        return idx
    return None

def make_byte_token_list() -> list[str]:
    """Return the 256 byte token strings."""
    return [BYTE_TOKEN_FORMAT.format(i) for i in range(256)]

# Example output:
# ["<0x00>", "<0x01>", ..., "<0xFF>"]
```

### Step 2: HybridTokenizer

```python
# cola_coder/tokenizer/hybrid_tokenizer.py
import unicodedata
from typing import Union
from .byte_tokens import byte_token_id, id_to_byte, make_byte_token_list

class HybridTokenizer:
    """
    BPE tokenizer with byte-level fallback for unknown characters.
    Compatible with SentencePiece byte_fallback=True behavior.
    """
    def __init__(
        self,
        base_tokenizer,        # Existing Cola-Coder BPE tokenizer
        base_vocab_size: int,  # Original vocab size before byte tokens
        unk_id: int,
    ):
        self.base = base_tokenizer
        self.base_vocab_size = base_vocab_size
        self.unk_id = unk_id
        self.byte_vocab_offset = base_vocab_size
        self.vocab_size = base_vocab_size + 256

        # Extend vocabulary with byte tokens
        self._byte_tokens = make_byte_token_list()
        self._byte_to_id = {
            tok: byte_token_id(i, base_vocab_size)
            for i, tok in enumerate(self._byte_tokens)
        }

    @property
    def extended_vocab_size(self) -> int:
        return self.vocab_size

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """
        Encode text with byte fallback for unknown characters.
        """
        # First pass: try BPE
        initial_tokens = self.base.encode(text, add_special_tokens=add_special_tokens)

        # Check if any <unk> tokens exist
        if self.unk_id not in initial_tokens:
            return initial_tokens

        # Second pass: character-by-character with byte fallback
        return self._encode_with_fallback(text, add_special_tokens)

    def _encode_with_fallback(
        self,
        text: str,
        add_special_tokens: bool,
    ) -> list[int]:
        """Character-level encoding with byte fallback for unknown chars."""
        result = []

        # Try to encode each unicode character or known subword
        # Use a greedy approach: try to match longest known BPE unit first
        i = 0
        while i < len(text):
            # Try progressively longer substrings
            matched = False
            for length in range(min(20, len(text) - i), 0, -1):
                substr = text[i:i+length]
                tokens = self.base.encode(substr, add_special_tokens=False)
                if self.unk_id not in tokens:
                    result.extend(tokens)
                    i += length
                    matched = True
                    break

            if not matched:
                # Single character could not be BPE-encoded: use bytes
                char = text[i]
                char_bytes = char.encode("utf-8")
                for byte_val in char_bytes:
                    result.append(byte_token_id(byte_val, self.base_vocab_size))
                i += 1

        if add_special_tokens and hasattr(self.base, "bos_token_id"):
            result = [self.base.bos_token_id] + result + [self.base.eos_token_id]

        return result

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to text.
        Handles both BPE tokens and byte tokens.
        """
        # Split into runs of BPE tokens and byte tokens
        bpe_tokens = []
        byte_buffer = []
        result_parts = []

        for tid in token_ids:
            byte_val = id_to_byte(tid, self.base_vocab_size)
            if byte_val is not None:
                # Flush any pending BPE tokens first
                if bpe_tokens:
                    result_parts.append(
                        self.base.decode(bpe_tokens, skip_special_tokens=skip_special_tokens)
                    )
                    bpe_tokens = []
                byte_buffer.append(byte_val)
            else:
                # Flush any pending byte buffer
                if byte_buffer:
                    result_parts.append(
                        bytes(byte_buffer).decode("utf-8", errors="replace")
                    )
                    byte_buffer = []
                bpe_tokens.append(tid)

        # Flush remaining
        if bpe_tokens:
            result_parts.append(
                self.base.decode(bpe_tokens, skip_special_tokens=skip_special_tokens)
            )
        if byte_buffer:
            result_parts.append(bytes(byte_buffer).decode("utf-8", errors="replace"))

        return "".join(result_parts)

    def encode_batch(self, texts: list[str], **kwargs) -> list[list[int]]:
        return [self.encode(t, **kwargs) for t in texts]

    def get_vocab(self) -> dict[str, int]:
        base_vocab = self.base.get_vocab()
        byte_vocab = {
            tok: byte_token_id(i, self.base_vocab_size)
            for i, tok in enumerate(self._byte_tokens)
        }
        return {**base_vocab, **byte_vocab}
```

### Step 3: Model Embedding Extension

```python
# cola_coder/model/embedding_extension.py
import torch
import torch.nn as nn

def extend_embedding_for_bytes(
    model,
    old_vocab_size: int,
    new_vocab_size: int,
    init_std: float = 0.02,
) -> None:
    """
    Extend the model's embedding and LM head for 256 additional byte tokens.
    Mutates the model in-place.
    """
    old_embed = model.embedding
    old_lm_head = model.lm_head

    # New embedding
    new_embed = nn.Embedding(new_vocab_size, old_embed.embedding_dim)
    new_embed.weight.data[:old_vocab_size] = old_embed.weight.data
    nn.init.normal_(new_embed.weight.data[old_vocab_size:], std=init_std)
    model.embedding = new_embed

    # New LM head
    new_head = nn.Linear(old_lm_head.in_features, new_vocab_size, bias=False)
    new_head.weight.data[:old_vocab_size] = old_lm_head.weight.data
    nn.init.normal_(new_head.weight.data[old_vocab_size:], std=init_std)
    model.lm_head = new_head

    print(f"Extended embedding: {old_vocab_size} → {new_vocab_size} tokens")
```

### Step 4: Training Data Preprocessing

```python
# cola_coder/data/byte_fallback_processor.py
import random
from .byte_tokens import make_byte_token_list

def add_byte_training_examples(
    dataset_path: str,
    output_path: str,
    tokenizer,
    byte_fraction: float = 0.05,
    seed: int = 42,
):
    """
    Add byte-heavy examples to training data.
    Takes byte_fraction% of the dataset and forces byte-level encoding.
    Also adds examples with non-ASCII content (Unicode identifiers, etc.).
    """
    import json, random
    random.seed(seed)

    # Examples with deliberate non-ASCII content for training
    BYTE_TRAINING_EXAMPLES = [
        # Unicode identifiers
        "const café = 'espresso';\nconst π = 3.14159;\nconst λ = (x: number) => x * 2;",
        # Emoji in template literals
        "const greeting = `Hello 🌍! Welcome to ${city} 🎉`;",
        # Chinese comments
        "// 用户认证模块\nasync function authenticate(userId: string): Promise<User> {}",
        # Hex/binary literals
        "const mask: number = 0xFF00FF;\nconst flags = 0b10101010;\nconst bigNum = 0xDEAD_BEEF;",
        # Regex with special chars
        "const emailRegex = /^[\\w.±]+@[\\w-]+\\.[\\w.]+$/;",
    ]

    with open(dataset_path) as fin, open(output_path, "w") as fout:
        lines = fin.readlines()
        for line in lines:
            fout.write(line)
        # Add byte training examples
        n_byte = int(len(lines) * byte_fraction)
        byte_lines = random.choices(BYTE_TRAINING_EXAMPLES, k=n_byte)
        for text in byte_lines:
            fout.write(json.dumps({"text": text, "byte_mode": True}) + "\n")

    print(f"Added {n_byte} byte-training examples ({byte_fraction:.1%} of dataset)")
```

### Step 5: CLI Support

```python
@app.command()
def extend_tokenizer_bytes(
    base_checkpoint: str = typer.Argument(...),
    output_dir: str = typer.Option("checkpoints/byte-extended/"),
):
    """Extend an existing model checkpoint with 256 byte tokens."""
    from cola_coder.model.embedding_extension import extend_embedding_for_bytes
    from safetensors.torch import save_file
    # Load model, extend, save
    model = load_model(base_checkpoint)
    old_vocab = model.cfg.vocab_size
    extend_embedding_for_bytes(model, old_vocab, old_vocab + 256)
    # Save extended checkpoint
    save_file(model.state_dict(), f"{output_dir}/model.safetensors")
    console.print(f"[green]Extended to {old_vocab + 256} vocab size[/green]")


# Example: check for unk in encoding
@app.command()
def check_encoding(text: str = typer.Argument(...)):
    """Check how text is encoded (show byte fallbacks)."""
    tokens = tokenizer.encode(text)
    for tid in tokens:
        bv = id_to_byte(tid, tokenizer.base_vocab_size)
        if bv is not None:
            console.print(f"  [yellow]BYTE[/yellow] 0x{bv:02X} = {chr(bv) if 32 <= bv < 128 else '?'}")
        else:
            piece = tokenizer.decode([tid])
            console.print(f"  [green]BPE[/green]  {repr(piece)}")
```

---

## Key Files to Modify

- `cola_coder/tokenizer/byte_tokens.py` — new file
- `cola_coder/tokenizer/hybrid_tokenizer.py` — new file (HybridTokenizer)
- `cola_coder/model/embedding_extension.py` — new file (extend embeddings)
- `cola_coder/model/config.py` — add `use_byte_fallback`, `base_vocab_size`
- `cola_coder/data/byte_fallback_processor.py` — training data augmentation
- `cola_coder/cli.py` — `extend-tokenizer-bytes`, `check-encoding` commands
- `configs/tokenizer.yaml` — `byte_fallback: true` flag

---

## Testing Strategy

```python
def test_byte_token_roundtrip():
    """Encode then decode a string with non-ASCII characters."""
    text = "const π = 3.14; // café au lait 🎉"
    tokens = hybrid_tokenizer.encode(text, add_special_tokens=False)
    decoded = hybrid_tokenizer.decode(tokens)
    assert decoded == text, f"Roundtrip failed: {repr(decoded)} != {repr(text)}"

def test_no_unk_tokens():
    text = "const 日本語 = 'hello';"
    tokens = hybrid_tokenizer.encode(text, add_special_tokens=False)
    assert hybrid_tokenizer.base.unk_id not in tokens

def test_byte_token_ids_in_range():
    base = 32000
    for i in range(256):
        assert byte_token_id(i, base) == base + i
    assert id_to_byte(base + 65, base) == 65
    assert id_to_byte(base - 1, base) is None

def test_ascii_code_unchanged():
    """Standard ASCII code should not use byte fallback."""
    text = "const x: number = 42;\nfunction add(a: number, b: number) { return a + b; }"
    standard_tokens = base_tokenizer.encode(text)
    hybrid_tokens = hybrid_tokenizer.encode(text, add_special_tokens=False)
    # Should be identical (no byte tokens used)
    assert hybrid_tokenizer.base.unk_id not in hybrid_tokens

def test_embedding_extension_preserves_existing():
    old_vocab = 100
    model = MockModel(vocab_size=old_vocab, d_model=32)
    old_weights = model.embedding.weight.data[:old_vocab].clone()
    extend_embedding_for_bytes(model, old_vocab, old_vocab + 256)
    assert torch.allclose(model.embedding.weight.data[:old_vocab], old_weights)
    assert model.embedding.num_embeddings == old_vocab + 256
```

---

## Performance Considerations

- **Encoding speed:** The fallback path (character-by-character) is ~10x slower than standard BPE. However, it only activates for non-ASCII characters. For typical TypeScript code that is 99%+ ASCII, overhead is negligible.
- **Vocabulary size impact:** Adding 256 tokens increases embedding table by 256 × d_model params. For d_model=512: 256 × 512 × 2 bytes (fp16) = 256KB — completely negligible.
- **Byte token frequency:** Byte tokens will be rare in the training set. Use a higher LM loss weight for byte token positions (e.g., 2.0) to ensure the model learns to reconstruct them despite low frequency.
- **Decoding:** Collecting byte tokens into a buffer before decoding (as implemented above) is critical for correct multi-byte UTF-8 character reconstruction.

---

## Dependencies

- Python standard library (`unicodedata`, `struct`)
- Existing Cola-Coder BPE tokenizer
- `safetensors` for checkpoint extension

---

## Estimated Complexity

| Task                              | Effort  |
|-----------------------------------|---------|
| Byte token definitions            | 0.5h    |
| HybridTokenizer encode            | 3h      |
| HybridTokenizer decode            | 2h      |
| Embedding extension               | 1h      |
| Training data processor           | 1h      |
| CLI commands                      | 1h      |
| Tests (roundtrip, edge cases)     | 2h      |
| **Total**                         | **~10.5h** |

Overall complexity: **Medium** (Unicode byte handling is subtle, roundtrip correctness is critical)

---

## 2026 Best Practices

- **SentencePiece byte_fallback reference:** The SentencePiece library's `byte_fallback=True` option implements exactly this behavior. Read its source (`sentencepiece/src/unigram_model.cc`) for edge cases in multi-byte UTF-8 handling.
- **UTF-8 boundary awareness:** When byte-encoding a multi-byte character (e.g., 3-byte kanji), always encode all bytes as a group — never split a UTF-8 character across BPE and byte tokens.
- **Normalize unicode first:** Apply `unicodedata.normalize('NFC', text)` before encoding to reduce variability in how composed characters are represented.
- **Test with the full Unicode test suite:** Use the Unicode Consortium's official test files (available at unicode.org) to verify roundtrip correctness for all character categories.
- **Byte token loss weighting:** Consider upweighting byte token positions in the training loss (similar to how rare tokens benefit from higher loss weight) to compensate for their infrequency.
- **ByT5 as reference implementation:** ByT5 uses purely byte-level tokens (no BPE). If byte fallback proves insufficient for a specific use case, consider a full byte-level model for that domain.
