"""
Unit tests for data utilities.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.data_utils import Tokenizer, Dataset


def test_tokenizer_word_level():
    """Test word-level tokenizer."""
    tokenizer = Tokenizer(level='word')
    
    texts = [
        "show network interfaces",
        "list all files",
        "show disk usage"
    ]
    
    tokenizer.fit(texts, max_vocab_size=100)
    
    # Test vocabulary size
    assert len(tokenizer) > 4  # At least special tokens + some words
    
    # Test encoding
    encoded = tokenizer.encode("show network", add_special_tokens=True)
    assert encoded[0] == tokenizer.vocab[tokenizer.START_TOKEN]
    assert encoded[-1] == tokenizer.vocab[tokenizer.END_TOKEN]
    
    # Test decoding
    decoded = tokenizer.decode(encoded, skip_special_tokens=True)
    assert "show" in decoded
    assert "network" in decoded


def test_tokenizer_char_level():
    """Test character-level tokenizer."""
    tokenizer = Tokenizer(level='char')
    
    texts = ["hello", "world"]
    tokenizer.fit(texts, max_vocab_size=100)
    
    # Test encoding/decoding
    encoded = tokenizer.encode("hello", add_special_tokens=False)
    decoded = tokenizer.decode(encoded, skip_special_tokens=True)
    assert decoded == "hello"


def test_tokenizer_unknown_token():
    """Test handling of unknown tokens."""
    tokenizer = Tokenizer(level='word')
    
    texts = ["hello world"]
    tokenizer.fit(texts, max_vocab_size=100)
    
    # Encode text with unknown word
    encoded = tokenizer.encode("hello unknown", add_special_tokens=False)
    
    # Should contain UNK token
    assert tokenizer.vocab[tokenizer.UNK_TOKEN] in encoded


def test_dataset():
    """Test Dataset class."""
    input_tokenizer = Tokenizer(level='word')
    output_tokenizer = Tokenizer(level='word')
    
    inputs = ["show files", "list users"]
    outputs = ["ls", "who"]
    
    input_tokenizer.fit(inputs, max_vocab_size=100)
    output_tokenizer.fit(outputs, max_vocab_size=100)
    
    dataset = Dataset(inputs, outputs, input_tokenizer, output_tokenizer)
    
    # Test length
    assert len(dataset) == 2
    
    # Test getting item
    input_indices, output_indices = dataset[0]
    assert len(input_indices) > 0
    assert len(output_indices) > 0


def test_tokenizer_save_load(tmp_path):
    """Test saving and loading tokenizer."""
    tokenizer = Tokenizer(level='word')
    
    texts = ["hello world", "test data"]
    tokenizer.fit(texts, max_vocab_size=100)
    
    # Save tokenizer
    save_path = tmp_path / "tokenizer.pkl"
    tokenizer.save(str(save_path))
    
    # Load tokenizer
    loaded_tokenizer = Tokenizer(level='word')
    loaded_tokenizer.load(str(save_path))
    
    # Test that loaded tokenizer works the same
    assert len(tokenizer) == len(loaded_tokenizer)
    assert tokenizer.vocab == loaded_tokenizer.vocab
    
    # Test encoding
    text = "hello world"
    assert tokenizer.encode(text) == loaded_tokenizer.encode(text)
