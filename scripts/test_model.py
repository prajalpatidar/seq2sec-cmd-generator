#!/usr/bin/env python3
"""Test the trained model with repetition penalty comparison."""

import os
import sys
import torch

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.seq2seq import Seq2SeqModel
from scripts.data_utils import Tokenizer


def main():
    """Test model with and without repetition penalty."""
    
    # Paths
    checkpoint_path = os.path.join(
        os.path.dirname(__file__), "..", "models", "checkpoints", "best_model.pth"
    )
    input_tokenizer_path = os.path.join(
        os.path.dirname(__file__), "..", "models", "checkpoints", "input_tokenizer.pkl"
    )
    output_tokenizer_path = os.path.join(
        os.path.dirname(__file__), "..", "models", "checkpoints", "output_tokenizer.pkl"
    )
    
    # Load tokenizers
    print("Loading tokenizers...")
    input_tokenizer = Tokenizer(level="word")
    input_tokenizer.load(input_tokenizer_path)
    
    output_tokenizer = Tokenizer(level="word")
    output_tokenizer.load(output_tokenizer_path)
    
    print(f"Input vocabulary size: {len(input_tokenizer)}")
    print(f"Output vocabulary size: {len(output_tokenizer)}")
    
    # Load checkpoint
    print("\nLoading model checkpoint...")
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location='cpu')
    
    # Create model with config from checkpoint or use defaults
    if isinstance(checkpoint, dict) and 'config' in checkpoint:
        config = checkpoint['config']
        print(f"Using config from checkpoint: {config}")
    else:
        # Use config from train_config.yaml
        import yaml
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "config", "train_config.yaml"
        )
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
        config = {
            'embedding_dim': yaml_config['embedding_dim'],
            'hidden_dim': yaml_config['hidden_dim'],
            'num_layers': yaml_config['num_layers']
        }
        print(f"Using config from train_config.yaml: {config}")
    
    model = Seq2SeqModel(
        input_vocab_size=len(input_tokenizer),
        output_vocab_size=len(output_tokenizer),
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers']
    )
    
    # Load state dict (handle both old and new checkpoint formats)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    print(f"Model loaded with config: {config}")
    
    # Test cases
    test_inputs = [
        'show network interface configuration',
        'list all iptables rules',
        'show routing table',
        'display memory information',
        'add iptables rule to drop packets',
        'show tcp connections'
    ]
    
    print("\n" + "="*80)
    print("Testing WITHOUT repetition penalty (default):")
    print("="*80 + "\n")
    
    for text in test_inputs:
        # Tokenize input
        tokens = input_tokenizer.encode(text)
        src_tensor = torch.tensor([tokens]).long()
        
        # Predict
        predictions = model.predict(
            src_tensor,
            max_len=50,
            start_token=output_tokenizer.vocab['<START>'],
            end_token=output_tokenizer.vocab['<END>'],
            repetition_penalty=1.0  # No penalty
        )
        
        # Decode output
        output_text = output_tokenizer.decode(predictions)
        
        print(f"Input:  {text}")
        print(f"Output: {output_text}")
        print()
    
    print("\n" + "="*80)
    print("Testing WITH repetition penalty (1.5):")
    print("="*80 + "\n")
    
    for text in test_inputs:
        # Tokenize input
        tokens = input_tokenizer.encode(text)
        src_tensor = torch.tensor([tokens]).long()
        
        # Predict
        predictions = model.predict(
            src_tensor,
            max_len=50,
            start_token=output_tokenizer.vocab['<START>'],
            end_token=output_tokenizer.vocab['<END>'],
            repetition_penalty=1.5  # With penalty
        )
        
        # Decode output
        output_text = output_tokenizer.decode(predictions)
        
        print(f"Input:  {text}")
        print(f"Output: {output_text}")
        print()


if __name__ == "__main__":
    main()
