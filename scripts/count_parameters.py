#!/usr/bin/env python3
"""
Script to calculate and print the number of parameters in the Seq2Seq model.
"""

import os
import sys
import torch
import yaml

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.seq2seq import Seq2SeqModel
from scripts.data_utils import Tokenizer

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "train_config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Load tokenizers to get vocab sizes
    input_tokenizer_path = os.path.join(
        os.path.dirname(__file__), "..", "models", "checkpoints", "input_tokenizer.pkl"
    )
    output_tokenizer_path = os.path.join(
        os.path.dirname(__file__), "..", "models", "checkpoints", "output_tokenizer.pkl"
    )
    
    input_tokenizer = Tokenizer(level="word")
    input_tokenizer.load(input_tokenizer_path)
    
    output_tokenizer = Tokenizer(level="word")
    output_tokenizer.load(output_tokenizer_path)
    
    input_vocab_size = len(input_tokenizer)
    output_vocab_size = len(output_tokenizer)

    print(f"Input Vocabulary Size: {input_vocab_size}")
    print(f"Output Vocabulary Size: {output_vocab_size}")
    print("-" * 30)

    # Initialize model
    model = Seq2SeqModel(
        input_vocab_size=input_vocab_size,
        output_vocab_size=output_vocab_size,
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
    )

    # Count parameters
    total_params = count_parameters(model)
    encoder_params = count_parameters(model.encoder)
    decoder_params = count_parameters(model.decoder)
    
    print(f"Model Configuration:")
    print(f"  Embedding Dim: {config['embedding_dim']}")
    print(f"  Hidden Dim:    {config['hidden_dim']}")
    print(f"  Num Layers:    {config['num_layers']}")
    print("-" * 30)
    print(f"Total Trainable Parameters: {total_params:,}")
    print(f"  Encoder Parameters:       {encoder_params:,}")
    print(f"  Decoder Parameters:       {decoder_params:,}")
    
    # Calculate estimated size in MB (assuming float32 = 4 bytes)
    size_mb = (total_params * 4) / (1024 * 1024)
    print(f"Estimated Model Size (FP32): {size_mb:.2f} MB")

if __name__ == "__main__":
    main()
