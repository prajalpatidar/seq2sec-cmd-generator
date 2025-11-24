# Testing Guide

## Overview

This guide provides instructions for testing the seq2seq command generator at various stages of the pipeline.

## Prerequisites

Install dependencies:
```bash
pip install -r requirements.txt
```

## Testing Components

### 1. Dataset Validation

Test that the dataset is properly formatted:

```python
import json

# Load dataset
with open('data/commands_dataset.json', 'r') as f:
    data = json.load(f)

# Validate structure
print(f"Total examples: {len(data)}")
assert len(data) > 0, "Dataset is empty"

for i, item in enumerate(data[:5]):
    assert 'input' in item, f"Missing 'input' in item {i}"
    assert 'output' in item, f"Missing 'output' in item {i}"
    assert isinstance(item['input'], str), f"'input' must be string in item {i}"
    assert isinstance(item['output'], str), f"'output' must be string in item {i}"
    print(f"✓ Example {i+1}: {item['input']} → {item['output']}")

print("\n✓ Dataset validation passed!")
```

### 2. Vocabulary Building

Test vocabulary creation:

```python
from src.dataset import Vocabulary

# Create vocabulary
vocab = Vocabulary()

# Add sample sentences
vocab.add_sentence("show network interfaces")
vocab.add_sentence("list all files")

print(f"Vocabulary size: {len(vocab)}")
print(f"Token mapping: {vocab.token2idx}")

# Test encoding
encoded = vocab.encode("show network", add_sos=True, add_eos=True)
print(f"Encoded: {encoded}")

# Test decoding
decoded = vocab.decode(encoded)
print(f"Decoded: {decoded}")

print("\n✓ Vocabulary test passed!")
```

### 3. Model Architecture

Test model initialization:

```python
import torch
from src.model import Seq2SeqWithAttention

# Initialize model
model = Seq2SeqWithAttention(
    encoder_vocab_size=100,
    decoder_vocab_size=100,
    embedding_dim=64,
    hidden_dim=128,
    num_layers=1,
    dropout=0.1
)

# Count parameters
num_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {num_params:,}")

# Test forward pass
batch_size = 2
src = torch.randint(0, 100, (batch_size, 10))
src_lengths = torch.tensor([10, 8])
trg = torch.randint(0, 100, (batch_size, 15))

output = model(src, src_lengths, trg, teacher_forcing_ratio=0.5)
print(f"Output shape: {output.shape}")
assert output.shape == (batch_size, 15, 100), "Output shape mismatch"

print("\n✓ Model test passed!")
```

### 4. Dataset Loading

Test dataset class:

```python
from src.dataset import CommandDataset, collate_fn
from torch.utils.data import DataLoader

# Create dataset
dataset = CommandDataset('data/commands_dataset.json', build_vocab=True)

print(f"Dataset size: {len(dataset)}")
print(f"Input vocab size: {len(dataset.input_vocab)}")
print(f"Output vocab size: {len(dataset.output_vocab)}")

# Test single item
item = dataset[0]
print(f"\nSample item:")
print(f"  Input text: {item['input_text']}")
print(f"  Output text: {item['output_text']}")
print(f"  Input seq: {item['input_seq']}")
print(f"  Output seq: {item['output_seq']}")

# Test dataloader
dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
batch = next(iter(dataloader))
print(f"\nBatch shapes:")
print(f"  Input: {batch['input_seq'].shape}")
print(f"  Output: {batch['output_seq'].shape}")

print("\n✓ Dataset test passed!")
```

### 5. Training (Quick Test)

Run a quick training test with minimal epochs:

```bash
python scripts/train.py \
  --data_path data/commands_dataset.json \
  --embedding_dim 64 \
  --hidden_dim 128 \
  --num_layers 1 \
  --batch_size 8 \
  --epochs 2 \
  --output_dir models_test
```

Expected behavior:
- Training should complete without errors
- Validation loss should decrease
- Model files should be created in `models_test/`
- Example inferences should be shown

### 6. ONNX Export

Test ONNX export:

```bash
python scripts/export_onnx.py \
  --model_path models_test/best_model.pt \
  --output_dir models_test \
  --device cpu
```

Verify:
- `encoder.onnx` is created
- `decoder.onnx` is created
- Both models pass ONNX validation
- No export errors

### 7. Quantization

Test quantization:

```bash
python scripts/quantize_onnx.py \
  --encoder_path models_test/encoder.onnx \
  --decoder_path models_test/decoder.onnx \
  --output_dir models_test
```

Verify:
- Quantized models are created
- File sizes are ~75% smaller
- Compression ratio is reported

### 8. Inference

Test inference with quantized models:

```bash
# Single query
python scripts/inference.py \
  --use_quantized \
  --encoder_path models_test/encoder_quantized.onnx \
  --decoder_path models_test/decoder_quantized.onnx \
  --input_vocab models_test/input_vocab.json \
  --output_vocab models_test/output_vocab.json \
  --input "show network interfaces"
```

Expected output:
- Should print a Linux command
- Inference should complete in <100ms

## Integration Tests

### Full Pipeline Test

Test the complete workflow:

```bash
# Run complete workflow
bash run_workflow.sh
```

This tests:
1. Training
2. ONNX export
3. Quantization
4. Inference on sample queries

### Accuracy Test

Evaluate model accuracy on validation set:

```python
import torch
from torch.utils.data import DataLoader
from src.model import Seq2SeqWithAttention
from src.dataset import CommandDataset, collate_fn

# Load model
checkpoint = torch.load('models/best_model.pt', map_location='cpu')
config = checkpoint['config']

model = Seq2SeqWithAttention(
    encoder_vocab_size=config['encoder_vocab_size'],
    decoder_vocab_size=config['decoder_vocab_size'],
    embedding_dim=config['embedding_dim'],
    hidden_dim=config['hidden_dim'],
    num_layers=config['num_layers'],
    dropout=0.0  # No dropout for inference
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load dataset
dataset = CommandDataset('data/commands_dataset.json', build_vocab=True)

# Test accuracy
correct = 0
total = 0

with torch.no_grad():
    for i in range(len(dataset)):
        item = dataset[i]
        src = item['input_seq'].unsqueeze(0)
        src_len = torch.tensor([item['input_len']])
        
        # Generate
        output = model.generate(src, src_len, max_length=30)
        predicted = dataset.output_vocab.decode(output[0])
        expected = item['output_text']
        
        total += 1
        if predicted == expected:
            correct += 1
        else:
            if total <= 10:  # Show first 10 mismatches
                print(f"Input: {item['input_text']}")
                print(f"Expected: {expected}")
                print(f"Predicted: {predicted}")
                print()

accuracy = correct / total * 100
print(f"\nAccuracy: {accuracy:.2f}% ({correct}/{total})")
```

### Performance Benchmark

Measure inference performance:

```python
import time
import numpy as np
from scripts.inference import ONNXInferenceEngine

# Initialize engine
engine = ONNXInferenceEngine(
    'models/encoder_quantized.onnx',
    'models/decoder_quantized.onnx',
    'models/input_vocab.json',
    'models/output_vocab.json'
)

# Test queries
queries = [
    "show network interfaces",
    "check disk usage",
    "list running processes",
    "display memory usage",
    "show system uptime"
] * 20  # 100 queries total

# Warm up
for _ in range(5):
    engine.translate(queries[0])

# Benchmark
latencies = []
for query in queries:
    start = time.time()
    result = engine.translate(query)
    latency = (time.time() - start) * 1000  # ms
    latencies.append(latency)

# Report
print("Performance Metrics:")
print(f"  Mean: {np.mean(latencies):.2f}ms")
print(f"  Median: {np.median(latencies):.2f}ms")
print(f"  P95: {np.percentile(latencies, 95):.2f}ms")
print(f"  P99: {np.percentile(latencies, 99):.2f}ms")
print(f"  Min: {np.min(latencies):.2f}ms")
print(f"  Max: {np.max(latencies):.2f}ms")
```

## Unit Tests

Create unit tests for core functions:

```python
# tests/test_vocabulary.py
import unittest
from src.dataset import Vocabulary

class TestVocabulary(unittest.TestCase):
    def setUp(self):
        self.vocab = Vocabulary()
    
    def test_add_token(self):
        self.vocab.add_token("test")
        self.assertIn("test", self.vocab.token2idx)
    
    def test_encode_decode(self):
        self.vocab.add_sentence("show network")
        encoded = self.vocab.encode("show network")
        decoded = self.vocab.decode(encoded)
        self.assertEqual(decoded, "show network")
    
    def test_special_tokens(self):
        self.assertIn("<PAD>", self.vocab.token2idx)
        self.assertIn("<SOS>", self.vocab.token2idx)
        self.assertIn("<EOS>", self.vocab.token2idx)
        self.assertIn("<UNK>", self.vocab.token2idx)

if __name__ == '__main__':
    unittest.main()
```

## Expected Results

### Training
- Loss should decrease over epochs
- Validation loss < 1.0 after 50 epochs
- Examples should show reasonable translations

### Inference
- Latency: 10-50ms per query on CPU
- Memory: <100MB
- Accuracy: >80% exact match

### Models
- PyTorch model: 2-3 MB
- ONNX models: 1.5-2 MB
- Quantized ONNX: 400-500 KB

## Troubleshooting

### Training Issues

**Issue**: Loss not decreasing
- Solution: Increase learning rate, train longer, or increase model size

**Issue**: Out of memory
- Solution: Reduce batch size, use smaller model dimensions

**Issue**: Poor accuracy on validation
- Solution: Add more diverse training data, increase model capacity

### Export Issues

**Issue**: ONNX export fails
- Solution: Ensure PyTorch model is in eval mode, use CPU for export

**Issue**: ONNX validation fails
- Solution: Update ONNX version, check for unsupported operations

### Inference Issues

**Issue**: Slow inference
- Solution: Use quantized models, reduce max_length, enable optimizations

**Issue**: Poor quality outputs
- Solution: Retrain with more data, use larger model, adjust generation parameters

## Continuous Testing

Set up automated testing:

```bash
# Create test script
cat > test_all.sh << 'EOF'
#!/bin/bash
set -e

echo "Running all tests..."

# Test imports
python -m py_compile src/*.py scripts/*.py
echo "✓ Syntax check passed"

# Test dataset
python -c "import json; assert len(json.load(open('data/commands_dataset.json'))) > 0"
echo "✓ Dataset validation passed"

# Quick training test
python scripts/train.py --epochs 2 --batch_size 8 --output_dir models_test
echo "✓ Training test passed"

# Export test
python scripts/export_onnx.py --model_path models_test/best_model.pt --output_dir models_test
echo "✓ Export test passed"

# Quantization test
python scripts/quantize_onnx.py --encoder_path models_test/encoder.onnx --decoder_path models_test/decoder.onnx --output_dir models_test
echo "✓ Quantization test passed"

echo "All tests passed!"
EOF

chmod +x test_all.sh
```

## Conclusion

Testing should cover:
- ✓ Dataset integrity
- ✓ Model architecture
- ✓ Training pipeline
- ✓ ONNX export
- ✓ Quantization
- ✓ Inference accuracy
- ✓ Performance metrics

Run tests regularly during development to catch issues early.
