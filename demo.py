"""
Quick demonstration of the model architecture and dataset without training.
This script shows how the components work together.
"""

import json
import torch
from src.model import Seq2SeqWithAttention
from src.dataset import Vocabulary, CommandDataset, collate_fn
from torch.utils.data import DataLoader

def main():
    print("="*60)
    print("Seq2Seq Command Generator - Architecture Demo")
    print("="*60)
    print()
    
    # 1. Load and display dataset
    print("1. Dataset Information")
    print("-" * 60)
    with open('data/commands_dataset.json', 'r') as f:
        data = json.load(f)
    
    print(f"Total examples: {len(data)}")
    print("\nSample examples:")
    for i, item in enumerate(data[:5], 1):
        print(f"  {i}. '{item['input']}' → '{item['output']}'")
    print()
    
    # 2. Build vocabularies
    print("2. Building Vocabularies")
    print("-" * 60)
    dataset = CommandDataset('data/commands_dataset.json', build_vocab=True)
    
    input_vocab = dataset.input_vocab
    output_vocab = dataset.output_vocab
    
    print(f"Input vocabulary size: {len(input_vocab)} tokens")
    print(f"Output vocabulary size: {len(output_vocab)} tokens")
    print(f"\nSpecial tokens: {list(input_vocab.token2idx.keys())[:4]}")
    print()
    
    # 3. Show encoding/decoding
    print("3. Token Encoding/Decoding")
    print("-" * 60)
    sample_text = "show network interfaces"
    encoded = input_vocab.encode(sample_text, add_sos=True, add_eos=True)
    decoded = input_vocab.decode(encoded)
    
    print(f"Original text: '{sample_text}'")
    print(f"Encoded: {encoded}")
    print(f"Decoded: '{decoded}'")
    print()
    
    # 4. Show dataset item
    print("4. Dataset Item Structure")
    print("-" * 60)
    item = dataset[0]
    print(f"Input text: {item['input_text']}")
    print(f"Output text: {item['output_text']}")
    print(f"Input sequence: {item['input_seq'].tolist()}")
    print(f"Output sequence: {item['output_seq'].tolist()}")
    print(f"Input length: {item['input_len']}")
    print(f"Output length: {item['output_len']}")
    print()
    
    # 5. Create dataloader
    print("5. DataLoader with Batching")
    print("-" * 60)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    batch = next(iter(dataloader))
    
    print(f"Batch size: {batch['input_seq'].size(0)}")
    print(f"Input sequence shape: {batch['input_seq'].shape}")
    print(f"Output sequence shape: {batch['output_seq'].shape}")
    print(f"Input lengths: {batch['input_len'].tolist()}")
    print()
    
    # 6. Initialize model
    print("6. Model Architecture")
    print("-" * 60)
    model = Seq2SeqWithAttention(
        encoder_vocab_size=len(input_vocab),
        decoder_vocab_size=len(output_vocab),
        embedding_dim=128,
        hidden_dim=256,
        num_layers=1,
        dropout=0.1
    )
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: Seq2Seq with Bahdanau Attention")
    print(f"Trainable parameters: {num_params:,}")
    print(f"Embedding dimension: 128")
    print(f"Hidden dimension: 256")
    print(f"Number of layers: 1")
    print()
    
    # 7. Show model components
    print("7. Model Components")
    print("-" * 60)
    print("Encoder:")
    print("  - Embedding layer")
    print("  - GRU (Gated Recurrent Unit)")
    print("  - Dropout")
    print()
    print("Attention:")
    print("  - Bahdanau attention mechanism")
    print("  - Computes context vectors")
    print()
    print("Decoder:")
    print("  - Embedding layer")
    print("  - Attention layer")
    print("  - GRU with context")
    print("  - Output projection")
    print()
    
    # 8. Test forward pass
    print("8. Forward Pass Test")
    print("-" * 60)
    model.eval()
    
    with torch.no_grad():
        src = batch['input_seq'][:2]  # Take 2 samples
        src_len = batch['input_len'][:2]
        trg = batch['output_seq'][:2]
        
        output = model(src, src_len, trg, teacher_forcing_ratio=0)
        
        print(f"Input shape: {src.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Output interpretation: (batch_size, sequence_length, vocab_size)")
        print()
        
        # Show generation
        generated = model.generate(src, src_len, max_length=20)
        print(f"Generated shape: {generated.shape}")
        
        for i in range(generated.size(0)):
            input_text = batch['input_text'][i]
            generated_text = output_vocab.decode(generated[i])
            expected_text = batch['output_text'][i]
            
            print(f"\nExample {i+1}:")
            print(f"  Input: {input_text}")
            print(f"  Generated: {generated_text}")
            print(f"  Expected: {expected_text}")
    print()
    
    # 9. Model size estimation
    print("9. Model Size Estimation")
    print("-" * 60)
    
    # Save temporary model
    torch.save(model.state_dict(), '/tmp/temp_model.pt')
    import os
    model_size = os.path.getsize('/tmp/temp_model.pt') / (1024 * 1024)  # MB
    
    print(f"PyTorch model size: {model_size:.2f} MB")
    print(f"Estimated ONNX size: {model_size * 0.5:.2f} MB")
    print(f"Estimated quantized size: {model_size * 0.125:.2f} MB")
    print()
    
    # 10. Summary
    print("="*60)
    print("Summary")
    print("="*60)
    print(f"✓ Dataset loaded: {len(dataset)} examples")
    print(f"✓ Vocabularies built: {len(input_vocab)} + {len(output_vocab)} tokens")
    print(f"✓ Model initialized: {num_params:,} parameters")
    print(f"✓ Forward pass tested: Working correctly")
    print()
    print("Next steps:")
    print("  1. Train the model: python scripts/train.py")
    print("  2. Export to ONNX: python scripts/export_onnx.py")
    print("  3. Quantize models: python scripts/quantize_onnx.py")
    print("  4. Run inference: python scripts/inference.py")
    print()
    print("Or run the complete workflow: bash run_workflow.sh")
    print("="*60)


if __name__ == '__main__':
    main()
