"""
Example usage demonstration for seq2sec-cmd-generator.

This script shows how to use the model for command generation.
Note: Requires trained models to be available.
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.data_utils import Tokenizer
from models.seq2seq import Seq2SeqModel
import torch


def demonstrate_training_data():
    """Show examples from the training dataset."""
    print("=" * 60)
    print("TRAINING DATA EXAMPLES")
    print("=" * 60)

    import pandas as pd

    df = pd.read_csv("../data/commands_dataset.csv")

    print(f"\nDataset size: {len(df)} command pairs\n")
    print("Sample natural language → Linux command mappings:")
    print("-" * 60)

    # Show first 10 examples
    for idx, row in df.head(10).iterrows():
        print(f"{row['input']:40s} → {row['output']}")

    print("\n... and many more!\n")


def demonstrate_model_architecture():
    """Show model architecture details."""
    print("=" * 60)
    print("MODEL ARCHITECTURE")
    print("=" * 60)

    # Create a sample model
    input_vocab_size = 500
    output_vocab_size = 500
    embedding_dim = 64
    hidden_dim = 128

    model = Seq2SeqModel(
        input_vocab_size=input_vocab_size,
        output_vocab_size=output_vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=1,
    )

    print("\nModel Configuration:")
    print(f"  - Input vocabulary size: {input_vocab_size}")
    print(f"  - Output vocabulary size: {output_vocab_size}")
    print(f"  - Embedding dimension: {embedding_dim}")
    print(f"  - Hidden dimension: {hidden_dim}")
    print(f"  - Number of layers: 1")
    print(f"  - Architecture: GRU-based Encoder-Decoder with Attention")

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {num_params:,}")
    print(f"Estimated model size: ~{num_params * 4 / (1024*1024):.2f} MB (float32)")
    print(f"Quantized model size: ~{num_params / (1024*1024):.2f} MB (int8)")

    print("\nOptimizations for Embedded Systems:")
    print("  ✓ Single-layer GRU (vs multi-layer LSTM)")
    print("  ✓ Small embedding & hidden dimensions")
    print("  ✓ Limited vocabulary size")
    print("  ✓ Unidirectional encoding")
    print("  ✓ ONNX export support")
    print("  ✓ INT8 quantization")
    print()


def demonstrate_tokenization():
    """Show tokenization examples."""
    print("=" * 60)
    print("TOKENIZATION EXAMPLE")
    print("=" * 60)

    tokenizer = Tokenizer(level="word")

    # Sample texts
    texts = ["show network interfaces", "list all files", "show disk usage"]

    tokenizer.fit(texts, max_vocab_size=100)

    print(f"\nVocabulary size: {len(tokenizer)}")
    print(f"Special tokens: {tokenizer.special_tokens}")

    print("\nTokenization examples:")
    print("-" * 60)

    for text in texts:
        encoded = tokenizer.encode(text, add_special_tokens=True)
        decoded = tokenizer.decode(encoded, skip_special_tokens=False)

        print(f"\nOriginal: {text}")
        print(f"Encoded:  {encoded}")
        print(f"Decoded:  {decoded}")

    print()


def demonstrate_inference_workflow():
    """Show the inference workflow."""
    print("=" * 60)
    print("INFERENCE WORKFLOW")
    print("=" * 60)

    print("\n1. Load trained ONNX models:")
    print("   - encoder_quantized.onnx (~250KB)")
    print("   - decoder_quantized.onnx (~250KB)")

    print("\n2. Load tokenizers:")
    print("   - input_tokenizer.pkl")
    print("   - output_tokenizer.pkl")

    print("\n3. Process input:")
    print("   Input: 'show network interfaces'")
    print("   Tokenize: ['show', 'network', 'interfaces']")
    print("   Encode: [1, 45, 67, 89, 2]  # <START>, tokens..., <END>")

    print("\n4. Run encoder:")
    print("   Input tokens → Encoder → Context vectors")

    print("\n5. Run decoder iteratively:")
    print("   Context + <START> → 'ifconfig'")
    print("   - Step 1: predict 'ifconfig'")
    print("   - Step 2: predict <END>")

    print("\n6. Decode output:")
    print("   Token IDs → Words → 'ifconfig'")

    print("\n7. Return result:")
    print("   Output: 'ifconfig'")

    print("\nInference time: ~20-50ms on Intel Atom")
    print("Memory usage: ~50-100MB RAM")
    print()


def main():
    """Run all demonstrations."""
    print("\n")
    print("*" * 60)
    print("SEQ2SEC-CMD-GENERATOR DEMONSTRATION")
    print("*" * 60)
    print()

    demonstrate_training_data()
    demonstrate_model_architecture()
    demonstrate_tokenization()
    demonstrate_inference_workflow()

    print("=" * 60)
    print("USAGE EXAMPLES")
    print("=" * 60)
    print()
    print("Training:")
    print("  python scripts/train.py")
    print()
    print("Export to ONNX:")
    print("  python scripts/export_onnx.py")
    print()
    print("Generate command (standard):")
    print("  python cli/cmd_generator.py generate 'show network interfaces'")
    print()
    print("Generate command (quantized for embedded):")
    print("  python cli/cmd_generator.py generate 'list all files' --quantized")
    print()
    print("Interactive mode:")
    print("  python cli/cmd_generator.py interactive")
    print()
    print("Run tests:")
    print("  pytest tests/ -v")
    print()
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
