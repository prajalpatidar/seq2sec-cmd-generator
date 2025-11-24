"""
Quantization script for ONNX models using dynamic INT8 quantization.
Reduces model size and improves inference speed for embedded deployment.
"""

import os
import argparse
from onnxruntime.quantization import quantize_dynamic, QuantType


def quantize_model(model_path, output_path):
    """
    Apply dynamic INT8 quantization to ONNX model.
    
    Args:
        model_path: path to ONNX model
        output_path: path to save quantized model
    """
    print(f"Quantizing {model_path}...")
    
    quantize_dynamic(
        model_input=model_path,
        model_output=output_path,
        weight_type=QuantType.QInt8,  # Quantize weights to INT8
        optimize_model=True,           # Apply ONNX optimization
        extra_options={
            'EnableSubgraph': True,     # Enable subgraph optimization
            'ForceQuantizeNoInputCheck': False
        }
    )
    
    # Get file sizes
    original_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
    quantized_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    compression_ratio = (1 - quantized_size / original_size) * 100
    
    print(f"  Original size: {original_size:.2f} MB")
    print(f"  Quantized size: {quantized_size:.2f} MB")
    print(f"  Compression: {compression_ratio:.1f}%")
    print(f"  Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Quantize ONNX models for efficient inference")
    parser.add_argument('--encoder_path', type=str, default='models/encoder.onnx',
                      help='Path to encoder ONNX model')
    parser.add_argument('--decoder_path', type=str, default='models/decoder.onnx',
                      help='Path to decoder ONNX model')
    parser.add_argument('--output_dir', type=str, default='models',
                      help='Directory to save quantized models')
    
    args = parser.parse_args()
    
    print("="*50)
    print("ONNX Model Quantization (Dynamic INT8)")
    print("="*50)
    print("\nThis process will:")
    print("  - Quantize model weights to INT8")
    print("  - Optimize model for inference")
    print("  - Reduce model size by ~75%")
    print("  - Improve inference speed")
    print()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Quantize encoder
    if os.path.exists(args.encoder_path):
        encoder_output = os.path.join(args.output_dir, 'encoder_quantized.onnx')
        quantize_model(args.encoder_path, encoder_output)
    else:
        print(f"Warning: Encoder model not found at {args.encoder_path}")
    
    print()
    
    # Quantize decoder
    if os.path.exists(args.decoder_path):
        decoder_output = os.path.join(args.output_dir, 'decoder_quantized.onnx')
        quantize_model(args.decoder_path, decoder_output)
    else:
        print(f"Warning: Decoder model not found at {args.decoder_path}")
    
    print("\n" + "="*50)
    print("Quantization Complete!")
    print("="*50)
    print("\nQuantized models are ready for deployment.")
    print("Use the inference script to test them:")
    print("  python scripts/inference.py --use_quantized")
    print("\nBenefits of quantization:")
    print("  ✓ 70-75% smaller model size")
    print("  ✓ 2-4x faster inference on CPU")
    print("  ✓ Lower memory footprint")
    print("  ✓ Better for embedded systems")


if __name__ == '__main__':
    main()
