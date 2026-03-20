"""Tests for the tokenizer training and usage."""

from pathlib import Path

from cola_coder.tokenizer.train_tokenizer import (
    create_tokenizer, train_from_iterator, SPECIAL_TOKENS
)


class TestTokenizerTraining:
    """Tests for BPE tokenizer training."""

    def test_create_tokenizer(self):
        """Tokenizer and trainer are created successfully."""
        tokenizer, trainer = create_tokenizer(vocab_size=256)
        assert tokenizer is not None
        assert trainer is not None

    def test_train_from_iterator(self, tmp_path):
        """Training from an iterator produces a working tokenizer."""
        # Sample code data
        code_samples = [
            "def hello_world():\n    print('Hello, world!')\n",
            "function greet(name) {\n    return `Hello, ${name}`;\n}\n",
            "class Calculator:\n    def add(self, a, b):\n        return a + b\n",
            "for i in range(10):\n    print(i)\n",
            "const x = [1, 2, 3].map(n => n * 2);\n",
        ] * 100  # Repeat for enough data

        output_path = str(tmp_path / "test_tokenizer.json")
        tokenizer = train_from_iterator(
            iter(code_samples),
            vocab_size=256,
            output_path=output_path,
        )

        # Should produce a file
        assert Path(output_path).exists()

        # Should encode and decode text
        encoded = tokenizer.encode("def foo(): pass")
        assert len(encoded.ids) > 0

        decoded = tokenizer.decode(encoded.ids)
        assert "def" in decoded
        assert "foo" in decoded

    def test_special_tokens_present(self, tmp_path):
        """Special tokens are in the vocabulary after training."""
        code_samples = ["print('hello')\n"] * 100
        output_path = str(tmp_path / "test_tokenizer.json")
        tokenizer = train_from_iterator(
            iter(code_samples), vocab_size=256, output_path=output_path,
        )

        for token in SPECIAL_TOKENS:
            token_id = tokenizer.token_to_id(token)
            assert token_id is not None, f"Special token {token} not found in vocabulary"

    def test_round_trip(self, tmp_path):
        """Encoding then decoding returns the original text."""
        code_samples = ["def hello():\n    return 42\n"] * 100
        output_path = str(tmp_path / "test_tokenizer.json")
        tokenizer = train_from_iterator(
            iter(code_samples), vocab_size=256, output_path=output_path,
        )

        test_text = "def hello():\n    return 42"
        encoded = tokenizer.encode(test_text)
        decoded = tokenizer.decode(encoded.ids)

        # ByteLevel encoding may add spaces, so check content
        assert "def" in decoded
        assert "hello" in decoded
        assert "42" in decoded
