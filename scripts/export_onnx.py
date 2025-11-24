"""
Export trained PyTorch model to ONNX format for efficient inference.
"""

import os
import argparse
import torch
import onnx
from onnx import helper, checker

from src.model import Seq2SeqWithAttention, Encoder, Decoder
from src.dataset import Vocabulary


class EncoderONNX(torch.nn.Module):
    """Wrapper for encoder to make it ONNX-compatible."""
    def __init__(self, encoder):
        super(EncoderONNX, self).__init__()
        self.encoder = encoder
    
    def forward(self, input_seq, input_lengths):
        return self.encoder(input_seq, input_lengths)


class DecoderONNX(torch.nn.Module):
    """Wrapper for decoder to make it ONNX-compatible."""
    def __init__(self, decoder):
        super(DecoderONNX, self).__init__()
        self.decoder = decoder
    
    def forward(self, input_token, hidden, encoder_outputs, mask):
        output, hidden, attention_weights = self.decoder(input_token, hidden, encoder_outputs, mask)
        return output, hidden


def export_encoder_to_onnx(model, output_path, device='cpu', max_seq_len=50):
    """
    Export encoder to ONNX format.
    
    Args:
        model: Seq2SeqWithAttention model
        output_path: path to save ONNX model
        device: device to use
        max_seq_len: maximum sequence length for dummy input
    """
    encoder = model.encoder
    encoder_wrapper = EncoderONNX(encoder).to(device)
    encoder_wrapper.eval()
    
    # Create dummy inputs
    dummy_input = torch.randint(0, 100, (1, max_seq_len), dtype=torch.long).to(device)
    dummy_lengths = torch.tensor([max_seq_len], dtype=torch.long)
    
    # Export to ONNX
    torch.onnx.export(
        encoder_wrapper,
        (dummy_input, dummy_lengths),
        output_path,
        input_names=['input_seq', 'input_lengths'],
        output_names=['encoder_outputs', 'hidden'],
        dynamic_axes={
            'input_seq': {0: 'batch_size', 1: 'seq_len'},
            'input_lengths': {0: 'batch_size'},
            'encoder_outputs': {0: 'batch_size', 1: 'seq_len'},
            'hidden': {1: 'batch_size'}
        },
        opset_version=14,
        do_constant_folding=True
    )
    
    print(f"Encoder exported to {output_path}")


def export_decoder_to_onnx(model, output_path, device='cpu', max_seq_len=50):
    """
    Export decoder to ONNX format.
    
    Args:
        model: Seq2SeqWithAttention model
        output_path: path to save ONNX model
        device: device to use
        max_seq_len: maximum sequence length for dummy input
    """
    decoder = model.decoder
    decoder_wrapper = DecoderONNX(decoder).to(device)
    decoder_wrapper.eval()
    
    hidden_dim = decoder.hidden_dim
    num_layers = decoder.num_layers
    
    # Create dummy inputs
    dummy_token = torch.randint(0, 100, (1, 1), dtype=torch.long).to(device)
    dummy_hidden = torch.randn(num_layers, 1, hidden_dim).to(device)
    dummy_encoder_outputs = torch.randn(1, max_seq_len, hidden_dim).to(device)
    dummy_mask = torch.ones(1, max_seq_len).to(device)
    
    # Export to ONNX
    torch.onnx.export(
        decoder_wrapper,
        (dummy_token, dummy_hidden, dummy_encoder_outputs, dummy_mask),
        output_path,
        input_names=['input_token', 'hidden', 'encoder_outputs', 'mask'],
        output_names=['output', 'new_hidden'],
        dynamic_axes={
            'input_token': {0: 'batch_size'},
            'hidden': {1: 'batch_size'},
            'encoder_outputs': {0: 'batch_size', 1: 'seq_len'},
            'mask': {0: 'batch_size', 1: 'seq_len'},
            'output': {0: 'batch_size'},
            'new_hidden': {1: 'batch_size'}
        },
        opset_version=14,
        do_constant_folding=True
    )
    
    print(f"Decoder exported to {output_path}")


def verify_onnx_model(onnx_path):
    """Verify that the ONNX model is valid."""
    model = onnx.load(onnx_path)
    try:
        checker.check_model(model)
        print(f"✓ ONNX model {onnx_path} is valid!")
        return True
    except Exception as e:
        print(f"✗ ONNX model {onnx_path} is invalid: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Export trained model to ONNX format")
    parser.add_argument('--model_path', type=str, default='models/best_model.pt',
                      help='Path to trained PyTorch model')
    parser.add_argument('--output_dir', type=str, default='models',
                      help='Directory to save ONNX models')
    parser.add_argument('--max_seq_len', type=int, default=50,
                      help='Maximum sequence length for export')
    parser.add_argument('--device', type=str, default='cpu',
                      help='Device to use for export (use cpu for best compatibility)')
    
    args = parser.parse_args()
    
    print("="*50)
    print("Exporting Model to ONNX Format")
    print("="*50)
    
    # Load model checkpoint
    print(f"\nLoading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=args.device)
    config = checkpoint['config']
    
    # Initialize model
    model = Seq2SeqWithAttention(
        encoder_vocab_size=config['encoder_vocab_size'],
        decoder_vocab_size=config['decoder_vocab_size'],
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(args.device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("Model loaded successfully!")
    print(f"Model configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Export encoder
    print("\nExporting encoder...")
    encoder_path = os.path.join(args.output_dir, 'encoder.onnx')
    export_encoder_to_onnx(model, encoder_path, args.device, args.max_seq_len)
    verify_onnx_model(encoder_path)
    
    # Export decoder
    print("\nExporting decoder...")
    decoder_path = os.path.join(args.output_dir, 'decoder.onnx')
    export_decoder_to_onnx(model, decoder_path, args.device, args.max_seq_len)
    verify_onnx_model(decoder_path)
    
    # Save model config
    config_path = os.path.join(args.output_dir, 'model_config.json')
    import json
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\nModel configuration saved to {config_path}")
    
    print("\n" + "="*50)
    print("Export Complete!")
    print("="*50)
    print(f"\nONNX models saved:")
    print(f"  Encoder: {encoder_path}")
    print(f"  Decoder: {decoder_path}")
    print(f"  Config: {config_path}")
    print("\nNext steps:")
    print("  1. Quantize models using: python scripts/quantize_onnx.py")
    print("  2. Run inference using: python scripts/inference.py")


if __name__ == '__main__':
    main()
