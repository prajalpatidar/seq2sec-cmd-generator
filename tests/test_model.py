"""
Unit tests for seq2seq model.
"""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.seq2seq import Encoder, Decoder, Seq2SeqModel


def test_encoder():
    """Test encoder forward pass."""
    vocab_size = 100
    embedding_dim = 32
    hidden_dim = 64
    batch_size = 4
    seq_len = 10

    encoder = Encoder(vocab_size, embedding_dim, hidden_dim, num_layers=1)

    # Create dummy input
    x = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Forward pass
    outputs, hidden = encoder(x)

    # Check output shapes
    assert outputs.shape == (batch_size, seq_len, hidden_dim)
    assert hidden.shape == (1, batch_size, hidden_dim)


def test_decoder():
    """Test decoder forward pass."""
    vocab_size = 100
    embedding_dim = 32
    hidden_dim = 64
    batch_size = 4
    seq_len = 10

    decoder = Decoder(vocab_size, embedding_dim, hidden_dim, num_layers=1)

    # Create dummy inputs
    x = torch.randint(0, vocab_size, (batch_size, 1))
    hidden = torch.randn(1, batch_size, hidden_dim)
    encoder_outputs = torch.randn(batch_size, seq_len, hidden_dim)

    # Forward pass
    output, new_hidden = decoder(x, hidden, encoder_outputs)

    # Check output shapes
    assert output.shape == (batch_size, vocab_size)
    assert new_hidden.shape == (1, batch_size, hidden_dim)


def test_seq2seq_model():
    """Test complete seq2seq model."""
    input_vocab_size = 100
    output_vocab_size = 100
    embedding_dim = 32
    hidden_dim = 64
    batch_size = 4
    src_len = 10
    tgt_len = 8

    model = Seq2SeqModel(
        input_vocab_size, output_vocab_size, embedding_dim, hidden_dim, num_layers=1
    )

    # Create dummy inputs
    src = torch.randint(0, input_vocab_size, (batch_size, src_len))
    tgt = torch.randint(0, output_vocab_size, (batch_size, tgt_len))

    # Forward pass
    outputs = model(src, tgt, teacher_forcing_ratio=0.5)

    # Check output shape
    assert outputs.shape == (batch_size, tgt_len, output_vocab_size)


def test_seq2seq_predict():
    """Test seq2seq prediction."""
    input_vocab_size = 100
    output_vocab_size = 100
    embedding_dim = 32
    hidden_dim = 64

    model = Seq2SeqModel(
        input_vocab_size, output_vocab_size, embedding_dim, hidden_dim, num_layers=1
    )

    # Create dummy input (single sample)
    src = torch.randint(0, input_vocab_size, (1, 10))

    # Predict
    predictions = model.predict(src, max_len=20, start_token=1, end_token=2)

    # Check that predictions is a list
    assert isinstance(predictions, list)
    assert len(predictions) > 0


def test_model_parameters():
    """Test model parameter count for embedded systems."""
    input_vocab_size = 500
    output_vocab_size = 500
    embedding_dim = 64
    hidden_dim = 128

    model = Seq2SeqModel(
        input_vocab_size, output_vocab_size, embedding_dim, hidden_dim, num_layers=1
    )

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())

    # Should be less than 500K parameters for embedded systems
    assert num_params < 500000
    print(f"Model has {num_params} parameters")


def test_model_with_different_sequence_lengths():
    """Test model with varying sequence lengths."""
    input_vocab_size = 100
    output_vocab_size = 100
    embedding_dim = 32
    hidden_dim = 64

    model = Seq2SeqModel(
        input_vocab_size, output_vocab_size, embedding_dim, hidden_dim, num_layers=1
    )

    # Test with different sequence lengths
    for src_len in [5, 10, 20]:
        for tgt_len in [5, 10, 15]:
            src = torch.randint(0, input_vocab_size, (2, src_len))
            tgt = torch.randint(0, output_vocab_size, (2, tgt_len))

            outputs = model(src, tgt, teacher_forcing_ratio=0.5)
            assert outputs.shape == (2, tgt_len, output_vocab_size)
