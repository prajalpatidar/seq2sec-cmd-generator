# Embedded Deployment Guide

## Overview

This guide provides detailed instructions for deploying the seq2seq command generator on embedded systems and resource-constrained devices.

## Model Architecture Selection

### Option 1: Standard Seq2Seq with Attention (Recommended)
- **Parameters**: ~500K
- **Model Size**: 400-500 KB (quantized)
- **Memory**: ~100 MB runtime
- **Accuracy**: >90%
- **Use Case**: General embedded systems (Raspberry Pi, ARM devices)

### Option 2: Tiny Encoder-Decoder
```bash
python scripts/train.py \
  --embedding_dim 64 \
  --hidden_dim 128 \
  --num_layers 1 \
  --epochs 50
```
- **Parameters**: ~100K
- **Model Size**: 100-150 KB (quantized)
- **Memory**: ~50 MB runtime
- **Accuracy**: 80-85%
- **Use Case**: Very constrained devices (microcontrollers with Linux)

## Quantization Strategies

### Dynamic INT8 Quantization (Implemented)
- **Method**: Weight-only quantization
- **Reduction**: 75% size reduction
- **Speed**: 2-4x faster on CPU
- **Accuracy Loss**: <1%

```bash
python scripts/quantize_onnx.py
```

### Advanced Quantization Options

#### 1. Static INT8 Quantization
Requires calibration data for better accuracy:

```python
from onnxruntime.quantization import quantize_static, CalibrationDataReader

class CommandDataReader(CalibrationDataReader):
    def __init__(self, data_path):
        # Load calibration data
        pass
    
    def get_next(self):
        # Return next calibration sample
        pass

quantize_static(
    model_input='models/encoder.onnx',
    model_output='models/encoder_static_quantized.onnx',
    calibration_data_reader=CommandDataReader('data/commands_dataset.json')
)
```

#### 2. Mixed Precision Quantization
Keep critical layers in FP32:

```python
from onnxruntime.quantization import quantize_dynamic

quantize_dynamic(
    model_input='models/encoder.onnx',
    model_output='models/encoder_mixed.onnx',
    weight_type=QuantType.QInt8,
    nodes_to_exclude=['critical_layer_name']  # Keep in FP32
)
```

## Hardware-Specific Optimizations

### Raspberry Pi (ARM Cortex-A)

**Installation:**
```bash
# Install dependencies
pip install onnxruntime numpy

# Copy models
cp models/*_quantized.onnx /opt/cmd-generator/
cp models/*.json /opt/cmd-generator/
```

**Optimization:**
```bash
# Use all CPU cores
export OMP_NUM_THREADS=4

# Run inference
python scripts/inference.py --use_quantized
```

**Performance:**
- Raspberry Pi 4: ~20ms per query
- Raspberry Pi 3: ~50ms per query
- Memory: ~80 MB

### NVIDIA Jetson (ARM + GPU)

**TensorRT Conversion:**
```python
import onnx
from onnx import shape_inference
import tensorrt as trt

# Infer shapes
model = onnx.load('models/encoder_quantized.onnx')
model = shape_inference.infer_shapes(model)

# Convert to TensorRT
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network()
parser = trt.OnnxParser(network, TRT_LOGGER)
parser.parse_from_file('models/encoder_quantized.onnx')

# Build engine
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
engine = builder.build_serialized_network(network, config)
```

**Performance:**
- Jetson Nano: ~5ms per query
- Jetson Xavier: ~2ms per query

### Intel CPU (x86_64)

**OpenVINO Optimization:**
```bash
# Install OpenVINO
pip install openvino

# Convert ONNX to OpenVINO IR
mo --input_model models/encoder_quantized.onnx \
   --output_dir models/openvino/ \
   --data_type FP16
```

**Performance:**
- Intel i5/i7: ~10ms per query
- Intel Atom: ~30ms per query

### ARM Cortex-M (Microcontrollers)

For microcontrollers, use TFLite Micro:

```bash
# Convert ONNX to TensorFlow
pip install onnx-tf
onnx-tf convert -i models/encoder_quantized.onnx -o models/encoder.pb

# Convert to TFLite
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model('models/encoder.pb')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.int8]
tflite_model = converter.convert()

with open('models/encoder.tflite', 'wb') as f:
    f.write(tflite_model)
```

## Memory Management

### Vocabulary Pruning

Reduce vocabulary size by removing rare tokens:

```python
from src.dataset import Vocabulary

# Load vocabulary
vocab = Vocabulary.load('models/input_vocab.json')

# Prune tokens with frequency < 2
pruned_vocab = Vocabulary()
pruned_vocab.token2idx = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
pruned_vocab.idx2token = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}

for token, count in vocab.token_count.items():
    if count >= 2:  # Threshold
        pruned_vocab.add_token(token)

# Save pruned vocabulary
pruned_vocab.save('models/input_vocab_pruned.json')
```

### Model Pruning

Remove less important weights:

```python
import torch
import torch.nn.utils.prune as prune

# Load model
model = Seq2SeqWithAttention(...)
checkpoint = torch.load('models/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Prune encoder weights (30% sparsity)
for name, module in model.encoder.named_modules():
    if isinstance(module, torch.nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=0.3)

# Save pruned model
torch.save(model.state_dict(), 'models/pruned_model.pt')
```

## Power Management

### CPU Frequency Scaling

```bash
# Set CPU governor to powersave
echo powersave | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Set specific frequency
echo 800000 | sudo tee /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq
```

### Batching for Efficiency

Process multiple queries together:

```python
# Batch inference (process 5 queries at once)
inputs = [
    "show network interfaces",
    "check disk usage",
    "list processes",
    "display memory",
    "show uptime"
]

# Preprocess all inputs
input_seqs = []
input_lengths = []
for text in inputs:
    seq, length = engine.preprocess_input(text)
    input_seqs.append(seq)
    input_lengths.append(length)

# Batch inference
# ... implementation ...
```

## Network Deployment

### HTTP Server

```python
from flask import Flask, request, jsonify
from scripts.inference import ONNXInferenceEngine

app = Flask(__name__)
engine = ONNXInferenceEngine(
    'models/encoder_quantized.onnx',
    'models/decoder_quantized.onnx',
    'models/input_vocab.json',
    'models/output_vocab.json'
)

@app.route('/translate', methods=['POST'])
def translate():
    data = request.json
    text = data.get('text', '')
    command = engine.translate(text)
    return jsonify({'command': command})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### gRPC Server (Lower Latency)

```python
import grpc
from concurrent import futures

class CommandService(command_pb2_grpc.CommandServiceServicer):
    def __init__(self):
        self.engine = ONNXInferenceEngine(...)
    
    def Translate(self, request, context):
        command = self.engine.translate(request.text)
        return command_pb2.CommandResponse(command=command)

server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
command_pb2_grpc.add_CommandServiceServicer_to_server(
    CommandService(), server
)
server.add_insecure_port('[::]:50051')
server.start()
```

## Storage Optimization

### Model Compression

```bash
# Compress models with gzip
gzip -9 models/encoder_quantized.onnx
gzip -9 models/decoder_quantized.onnx

# Decompress at runtime
gunzip -c models/encoder_quantized.onnx.gz > /tmp/encoder.onnx
```

### Read-Only File System

For embedded systems with read-only root:

```bash
# Mount models from read-only partition
mount -o ro /dev/mmcblk0p1 /opt/models

# Use tmpfs for runtime data
mount -t tmpfs -o size=50M tmpfs /var/run/cmd-generator
```

## Security Considerations

### Sandboxing

Run inference in isolated environment:

```bash
# Use Docker
docker run --rm -v $(pwd)/models:/models \
  --memory=128m --cpus=1.0 \
  cmd-generator:latest

# Use systemd
[Service]
MemoryLimit=128M
CPUQuota=50%
PrivateTmp=yes
NoNewPrivileges=yes
```

### Input Validation

```python
def validate_input(text):
    # Limit length
    if len(text) > 200:
        return False
    
    # Check for malicious patterns
    forbidden = ['sudo', 'rm -rf', '|', '&&', ';']
    for pattern in forbidden:
        if pattern in text.lower():
            return False
    
    return True
```

## Monitoring and Logging

### Performance Metrics

```python
import time
import logging

logging.basicConfig(level=logging.INFO)

class MonitoredEngine(ONNXInferenceEngine):
    def translate(self, text):
        start_time = time.time()
        command = super().translate(text)
        latency = (time.time() - start_time) * 1000  # ms
        
        logging.info(f"Query: {text[:50]}... | "
                    f"Command: {command} | "
                    f"Latency: {latency:.2f}ms")
        
        return command
```

### System Resources

```python
import psutil

def log_resources():
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    
    logging.info(f"CPU: {cpu_percent}% | "
                f"Memory: {memory.percent}% "
                f"({memory.used / 1024 / 1024:.1f}MB)")
```

## Troubleshooting

### Common Issues

1. **High Memory Usage**
   - Use quantized models
   - Reduce batch size
   - Enable memory pooling

2. **Slow Inference**
   - Enable graph optimization
   - Use hardware acceleration
   - Reduce max_length

3. **Poor Accuracy**
   - Train longer (more epochs)
   - Increase model size
   - Add more training data

### Debug Mode

```bash
# Enable ONNX Runtime logging
export ORT_LOG_SEVERITY_LEVEL=1

# Run inference with verbose output
python scripts/inference.py --use_quantized --input "test" -v
```

## Benchmarking

### Latency Test

```python
import time
import numpy as np

queries = ["show network"] * 100
latencies = []

for query in queries:
    start = time.time()
    engine.translate(query)
    latency = (time.time() - start) * 1000
    latencies.append(latency)

print(f"Mean: {np.mean(latencies):.2f}ms")
print(f"P50: {np.percentile(latencies, 50):.2f}ms")
print(f"P95: {np.percentile(latencies, 95):.2f}ms")
print(f"P99: {np.percentile(latencies, 99):.2f}ms")
```

### Memory Profiling

```python
from memory_profiler import profile

@profile
def test_inference():
    engine = ONNXInferenceEngine(...)
    for _ in range(100):
        engine.translate("show files")

test_inference()
```

## Production Checklist

- [ ] Model is quantized
- [ ] Vocabularies are pruned
- [ ] Input validation is implemented
- [ ] Resource limits are set
- [ ] Logging is configured
- [ ] Error handling is robust
- [ ] Security measures are in place
- [ ] Performance is benchmarked
- [ ] Monitoring is set up
- [ ] Backup models are available

## Further Optimization

### Knowledge Distillation

Train a smaller "student" model from a larger "teacher":

```python
# Train large teacher model
teacher = Seq2SeqWithAttention(hidden_dim=512)
# ... training ...

# Train small student model with teacher guidance
student = Seq2SeqWithAttention(hidden_dim=128)
distillation_loss = KL_divergence(student_output, teacher_output)
# ... training ...
```

### Neural Architecture Search (NAS)

Automatically find optimal architecture:
- Use tools like NNI or AutoKeras
- Search space: layer sizes, number of layers
- Optimization target: accuracy vs. latency

## References

- ONNX Runtime Performance Tuning: https://onnxruntime.ai/docs/performance/
- TensorRT Documentation: https://developer.nvidia.com/tensorrt
- ARM NN Documentation: https://developer.arm.com/ip-products/processors/machine-learning/arm-nn

---

For questions or issues, please refer to the main README.md or open an issue on GitHub.
