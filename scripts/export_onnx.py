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
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.seq2seq import Seq2SeqModel
from scripts.data_utils import Tokenizer


class ONNXWrapper(torch.nn.Module):
    """
    Wrapper for ONNX export that handles encoder-decoder separately.
    
    ONNX (Open Neural Network Exchange) is a standard format for representing machine learning models.
    It allows models trained in PyTorch to be run in other environments (like C++ on embedded devices).
    
    We split the Seq2Seq model into two separate ONNX models:
    1. Encoder: Processes the input sentence once.
    2. Decoder: Generates the output command one word at a time.
    """

    def __init__(self, model):
        super(ONNXWrapper, self).__init__()
        self.encoder = model.encoder
        self.decoder = model.decoder

    def forward(self, src):
        """
        Forward pass for ONNX export.
        
        This defines exactly what inputs the Encoder expects and what it outputs.
        
        Args:
            src: Source tensor of shape (batch_size, src_seq_len)
                 Contains the token IDs of the input sentence.
        Returns:
            encoder_outputs: Context vectors for every word in the input.
            hidden: The final internal state of the encoder (summary of the whole sentence).
        """
        encoder_outputs, hidden = self.encoder(src)
        return encoder_outputs, hidden


class DecoderWrapper(torch.nn.Module):
    """
    Wrapper for decoder ONNX export.
    
    The decoder is more complex because it runs in a loop.
    For ONNX, we export a single step of that loop.
    The C++ application will call this ONNX model repeatedly.
    """

    def __init__(self, decoder):
        super(DecoderWrapper, self).__init__()
        self.decoder = decoder

    def forward(self, x, hidden, encoder_outputs):
        """
        Forward pass for decoder ONNX export (Single Step).

        Args:
            x: Input token (the word generated in the previous step).
               Shape: (batch_size, 1)
            hidden: The internal state from the previous step.
            encoder_outputs: The context from the encoder (remains constant).
        Returns:
            output: Prediction scores for the next word.
            hidden: Updated internal state for the next step.
        """
        output, hidden = self.decoder(x, hidden, encoder_outputs)
        return output, hidden


def export_to_onnx(model, input_tokenizer, output_tokenizer, output_dir, num_layers=1):
    """
    Export model to ONNX format.
    
    This function performs the actual conversion from PyTorch (.pth) to ONNX (.onnx).

    Args:
        model: Trained Seq2SeqModel
        input_tokenizer: Input tokenizer (needed to know vocab size for dummy inputs)
        output_tokenizer: Output tokenizer
        output_dir: Directory to save ONNX models
        num_layers: Number of GRU layers (needed to shape the hidden state correctly)
    """
    # Set model to evaluation mode (disables dropout, etc.)
    model.eval()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # ==========================================
    # 1. Export Encoder
    # ==========================================
    encoder_wrapper = ONNXWrapper(model)
    
    # Create dummy input (random numbers) to trace the execution path
    # The values don't matter, only the shape and type (Long/Int64)
    dummy_input = torch.randint(0, len(input_tokenizer), (1, 10))

    encoder_path = os.path.join(output_dir, "encoder.onnx")
    torch.onnx.export(
        encoder_wrapper,
        dummy_input,
        encoder_path,
        input_names=["input"],
        output_names=["encoder_outputs", "hidden"],
        # Dynamic axes allow the model to accept different batch sizes and sentence lengths
        dynamic_axes={
            "input": {0: "batch_size", 1: "seq_len"},
            "encoder_outputs": {0: "batch_size", 1: "seq_len"},
            "hidden": {1: "batch_size"},
        },
        opset_version=13,
        dynamo=False,  # Use legacy exporter for better compatibility
    )
    print(f"Encoder exported to {encoder_path}")

    # ==========================================
    # 2. Export Decoder
    # ==========================================
    decoder_wrapper = DecoderWrapper(model.decoder)
    
    # Dummy inputs for the decoder
    dummy_decoder_input = torch.randint(0, len(output_tokenizer), (1, 1))
    # Hidden state shape: (num_layers, batch_size, hidden_dim)
    dummy_hidden = torch.randn(num_layers, 1, model.hidden_dim)
    # Encoder outputs shape: (batch_size, seq_len, hidden_dim)
    dummy_encoder_outputs = torch.randn(1, 10, model.hidden_dim)

    decoder_path = os.path.join(output_dir, "decoder.onnx")
    torch.onnx.export(
        decoder_wrapper,
        (dummy_decoder_input, dummy_hidden, dummy_encoder_outputs),
        decoder_path,
        input_names=["input", "hidden", "encoder_outputs"],
        output_names=["output", "new_hidden"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "hidden": {1: "batch_size"},
            "encoder_outputs": {0: "batch_size", 1: "seq_len"},
            "output": {0: "batch_size"},
            "new_hidden": {1: "batch_size"},
        },
        opset_version=13,
        dynamo=False,  # Use legacy exporter
    )
    print(f"Decoder exported to {decoder_path}")

    # Verify ONNX models (check for structural errors)
    onnx_encoder = onnx.load(encoder_path)
    onnx.checker.check_model(onnx_encoder)
    print("Encoder ONNX model verified!")

    onnx_decoder = onnx.load(decoder_path)
    onnx.checker.check_model(onnx_decoder)
    print("Decoder ONNX model verified!")


def quantize_model(onnx_path, quantized_path):
    """
    Quantize ONNX model for embedded deployment.
    
    Current setting: INT8 (8-bit Integer) via QuantType.QUInt8
    
    ACCURACY VS SIZE GUIDE:
    ----------------------
    1. 32-bit (Float32): 
       - The original exported 'encoder.onnx' / 'decoder.onnx' files.
       - Accuracy: Highest (Exact match to PyTorch).
       - Size: Largest (e.g., 100 MB).
       - Use case: When accuracy is critical and memory is abundant.
       
    2. 16-bit (Float16):
       - Requires 'onnxconverter_common' library.
       - Accuracy: Very close to 32-bit.
       - Size: Half of 32-bit (e.g., 50 MB).
       - Use case: GPUs or modern CPUs with FP16 support.
       
    3. 8-bit (INT8) - CURRENT:
       - Accuracy: Good, but slight drop possible.
       - Size: Smallest (e.g., 25 MB).
       - Use case: Embedded devices, mobile, older CPUs.

    Args:
        onnx_path: Path to original ONNX model
        quantized_path: Path to save quantized model
    """
    print(f"Quantizing {onnx_path} to INT8 (8-bit)...")
    quantize_dynamic(onnx_path, quantized_path, weight_type=QuantType.QUInt8)
    print(f"Quantized model saved to {quantized_path}")


def convert_to_float16(onnx_path, fp16_path):
    """
    Convert ONNX model to Float16 (Half Precision).
    
    Args:
        onnx_path: Path to original Float32 ONNX model
        fp16_path: Path to save Float16 model
    """
    try:
        from onnxconverter_common import float16
        import onnx
        
        print(f"Converting {onnx_path} to Float16 (16-bit)...")
        model = onnx.load(onnx_path)
        model_fp16 = float16.convert_float_to_float16(model)
        onnx.save(model_fp16, fp16_path)
        print(f"Float16 model saved to {fp16_path}")
        return True
    except ImportError:
        print("Skipping Float16 conversion: 'onnxconverter_common' not installed.")
        print("To enable 16-bit export, run: pip install onnxconverter-common")
        return False


def main():
    """Main export function."""

    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "train_config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Load tokenizers
    input_tokenizer = Tokenizer(level="word")
    output_tokenizer = Tokenizer(level="word")

    input_tokenizer.load(
        os.path.join(
            os.path.dirname(__file__), "..", "models", "checkpoints", "input_tokenizer.pkl"
        )
    )
    output_tokenizer.load(
        os.path.join(
            os.path.dirname(__file__), "..", "models", "checkpoints", "output_tokenizer.pkl"
        )
    )

    print(f"Input vocabulary size: {len(input_tokenizer)}")
    print(f"Output vocabulary size: {len(output_tokenizer)}")

    # Load model
    model = Seq2SeqModel(
        input_vocab_size=len(input_tokenizer),
        output_vocab_size=len(output_tokenizer),
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
    )

    model_path = os.path.join(
        os.path.dirname(__file__), "..", "models", "checkpoints", "best_model.pth"
    )
    
    # Load checkpoint (handle new format with metadata)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        # Get num_layers from checkpoint config if available
        if 'config' in checkpoint:
            num_layers = checkpoint['config'].get('num_layers', config['num_layers'])
        else:
            num_layers = config['num_layers']
        print(f"Model loaded from {model_path} (new format with config)")
    else:
        model.load_state_dict(checkpoint)
        num_layers = config['num_layers']
        print(f"Model loaded from {model_path} (legacy format)")


    # Export to ONNX
    output_dir = os.path.join(os.path.dirname(__file__), "..", "models", "onnx")
    export_to_onnx(model, input_tokenizer, output_tokenizer, output_dir, num_layers=num_layers)

    # Quantize models for embedded deployment
    print("\nQuantizing models for embedded deployment...")
    quantize_model(
        os.path.join(output_dir, "encoder.onnx"), os.path.join(output_dir, "encoder_quantized.onnx")
    )
    quantize_model(
        os.path.join(output_dir, "decoder.onnx"), os.path.join(output_dir, "decoder_quantized.onnx")
    )
    
    # Attempt Float16 conversion (Optional)
    print("\nAttempting Float16 conversion (for higher accuracy)...")
    convert_to_float16(
        os.path.join(output_dir, "encoder.onnx"), os.path.join(output_dir, "encoder_fp16.onnx")
    )
    convert_to_float16(
        os.path.join(output_dir, "decoder.onnx"), os.path.join(output_dir, "decoder_fp16.onnx")
    )

    print("\nExport completed successfully!")
    print(f"Models saved in: {output_dir}")
    print("\nSummary of generated models:")
    print("1. *.onnx           -> 32-bit Float (Best Accuracy, Largest Size)")
    print("2. *_fp16.onnx      -> 16-bit Float (High Accuracy, Medium Size) [If generated]")
    print("3. *_quantized.onnx -> 8-bit INT    (Good Accuracy, Smallest Size)")


if __name__ == "__main__":
    main()
