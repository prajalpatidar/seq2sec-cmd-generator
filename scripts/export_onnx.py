"""
Export PyTorch model to ONNX format with quantization for embedded deployment.
"""

import os
import sys
import torch
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import yaml

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.seq2seq import Seq2SeqModel
from scripts.data_utils import Tokenizer


class ONNXWrapper(torch.nn.Module):
    """Wrapper for ONNX export that handles encoder-decoder separately."""
    
    def __init__(self, model):
        super(ONNXWrapper, self).__init__()
        self.encoder = model.encoder
        self.decoder = model.decoder
    
    def forward(self, src):
        """
        Forward pass for ONNX export.
        
        Args:
            src: Source tensor of shape (batch_size, src_seq_len)
        Returns:
            encoder_outputs, hidden
        """
        encoder_outputs, hidden = self.encoder(src)
        return encoder_outputs, hidden


class DecoderWrapper(torch.nn.Module):
    """Wrapper for decoder ONNX export."""
    
    def __init__(self, decoder):
        super(DecoderWrapper, self).__init__()
        self.decoder = decoder
    
    def forward(self, x, hidden, encoder_outputs):
        """
        Forward pass for decoder ONNX export.
        
        Args:
            x: Input tensor of shape (batch_size, 1)
            hidden: Hidden state
            encoder_outputs: Encoder outputs
        Returns:
            output, hidden
        """
        output, hidden = self.decoder(x, hidden, encoder_outputs)
        return output, hidden


def export_to_onnx(model, input_tokenizer, output_tokenizer, output_dir):
    """
    Export model to ONNX format.
    
    Args:
        model: Trained Seq2SeqModel
        input_tokenizer: Input tokenizer
        output_tokenizer: Output tokenizer
        output_dir: Directory to save ONNX models
    """
    model.eval()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Export encoder
    encoder_wrapper = ONNXWrapper(model)
    dummy_input = torch.randint(0, len(input_tokenizer), (1, 10))
    
    encoder_path = os.path.join(output_dir, 'encoder.onnx')
    torch.onnx.export(
        encoder_wrapper,
        dummy_input,
        encoder_path,
        input_names=['input'],
        output_names=['encoder_outputs', 'hidden'],
        dynamic_axes={
            'input': {0: 'batch_size', 1: 'seq_len'},
            'encoder_outputs': {0: 'batch_size', 1: 'seq_len'},
            'hidden': {1: 'batch_size'}
        },
        opset_version=13
    )
    print(f"Encoder exported to {encoder_path}")
    
    # Export decoder
    decoder_wrapper = DecoderWrapper(model.decoder)
    dummy_decoder_input = torch.randint(0, len(output_tokenizer), (1, 1))
    dummy_hidden = torch.randn(1, 1, model.hidden_dim)
    dummy_encoder_outputs = torch.randn(1, 10, model.hidden_dim)
    
    decoder_path = os.path.join(output_dir, 'decoder.onnx')
    torch.onnx.export(
        decoder_wrapper,
        (dummy_decoder_input, dummy_hidden, dummy_encoder_outputs),
        decoder_path,
        input_names=['input', 'hidden', 'encoder_outputs'],
        output_names=['output', 'new_hidden'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'hidden': {1: 'batch_size'},
            'encoder_outputs': {0: 'batch_size', 1: 'seq_len'},
            'output': {0: 'batch_size'},
            'new_hidden': {1: 'batch_size'}
        },
        opset_version=13
    )
    print(f"Decoder exported to {decoder_path}")
    
    # Verify ONNX models
    onnx_encoder = onnx.load(encoder_path)
    onnx.checker.check_model(onnx_encoder)
    print("Encoder ONNX model verified!")
    
    onnx_decoder = onnx.load(decoder_path)
    onnx.checker.check_model(onnx_decoder)
    print("Decoder ONNX model verified!")


def quantize_model(onnx_path, quantized_path):
    """
    Quantize ONNX model for embedded deployment.
    
    Args:
        onnx_path: Path to original ONNX model
        quantized_path: Path to save quantized model
    """
    quantize_dynamic(
        onnx_path,
        quantized_path,
        weight_type=QuantType.QUInt8
    )
    print(f"Quantized model saved to {quantized_path}")


def main():
    """Main export function."""
    
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'train_config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load tokenizers
    input_tokenizer = Tokenizer(level='word')
    output_tokenizer = Tokenizer(level='word')
    
    input_tokenizer.load(os.path.join(os.path.dirname(__file__), '..', 'models', 'checkpoints', 'input_tokenizer.pkl'))
    output_tokenizer.load(os.path.join(os.path.dirname(__file__), '..', 'models', 'checkpoints', 'output_tokenizer.pkl'))
    
    print(f"Input vocabulary size: {len(input_tokenizer)}")
    print(f"Output vocabulary size: {len(output_tokenizer)}")
    
    # Load model
    model = Seq2SeqModel(
        input_vocab_size=len(input_tokenizer),
        output_vocab_size=len(output_tokenizer),
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers']
    )
    
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'checkpoints', 'best_model.pth')
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    print(f"Model loaded from {model_path}")
    
    # Export to ONNX
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'models', 'onnx')
    export_to_onnx(model, input_tokenizer, output_tokenizer, output_dir)
    
    # Quantize models for embedded deployment
    print("\nQuantizing models for embedded deployment...")
    quantize_model(
        os.path.join(output_dir, 'encoder.onnx'),
        os.path.join(output_dir, 'encoder_quantized.onnx')
    )
    quantize_model(
        os.path.join(output_dir, 'decoder.onnx'),
        os.path.join(output_dir, 'decoder_quantized.onnx')
    )
    
    print("\nExport completed successfully!")
    print(f"Models saved in: {output_dir}")


if __name__ == '__main__':
    main()
