# Embedded Systems Deployment Guide

This guide provides detailed instructions for deploying the seq2sec-cmd-generator on RDK-B embedded systems and other resource-constrained devices.

## Target Environment

- **Platform**: RDK-B Devices / Embedded Linux Systems
- **CPU**: Intel Atom dual-core or ARM Cortex-A series
- **RAM**: 500MB - 1GB available
- **Storage**: 100MB minimum (for models + runtime)
- **OS**: Linux-based (Ubuntu, Debian, Yocto, RDK-B Linux)

## Deployment Options

### Option A: Python-based Deployment (Recommended for Testing)
- Uses Python + ONNX Runtime
- Easy to modify and debug
- Requires Python 3.8+ on device
- Memory: ~50-60 MB
- See sections 1-6 below

### Option B: C++ Deployment (Recommended for Production)
- Pure C++ implementation
- No Python dependency required
- Single binary executable (~500KB stripped)
- Minimal dependencies (only ONNX Runtime library)
- Memory: ~45-55 MB
- **Ideal for embedded devices without Python**
- See [C++ Deployment Guide](../cpp/README.md)

## Deployment Steps (Python-based)

### 1. Prepare Models on Development Machine

First, train and export the model on your development machine:

```bash
# On development machine (high-performance)
cd seq2sec-cmd-generator

# Train the model
python scripts/train.py

# Export to ONNX with quantization
python scripts/export_onnx.py
```

This creates the following files:

**Standard ONNX Models (FP32):**
- `models/onnx/encoder.onnx` (10.85 MB)
- `models/onnx/decoder.onnx` (14.74 MB)
- Total: 25.59 MB

**Quantized ONNX Models (INT8):**
- `models/onnx/encoder_quantized.onnx` (10.61 MB)
- `models/onnx/decoder_quantized.onnx` (13.84 MB)
- Total: 24.45 MB

**Tokenizers:**
- `models/checkpoints/input_tokenizer.pkl` (~150KB, vocab=326)
- `models/checkpoints/output_tokenizer.pkl` (~200KB, vocab=412)

**Total Deployment Package: ~25-26 MB** (quantized recommended)

### 2. Prepare Embedded System

On your embedded device, create a minimal Python environment:

```bash
# Install Python 3.8+ (if not already installed)
sudo apt-get update
sudo apt-get install python3 python3-pip

# Create virtual environment (recommended)
python3 -m venv cmd-gen-env
source cmd-gen-env/bin/activate
```

### 3. Install ONNX Runtime

Install only the runtime dependencies (not training dependencies):

```bash
# Install ONNX Runtime (CPU version, lightweight)
pip install onnxruntime==1.15.0

# Install minimal dependencies
pip install numpy click pyyaml
```

**Note**: Do NOT install PyTorch on the embedded device. It's only needed for training.

### 4. Transfer Files to Embedded System

Copy only the necessary files:

```bash
# On development machine
# Create deployment package
mkdir -p deployment
cp models/onnx/encoder_quantized.onnx deployment/
cp models/onnx/decoder_quantized.onnx deployment/
cp models/checkpoints/input_tokenizer.pkl deployment/
cp models/checkpoints/output_tokenizer.pkl deployment/
cp cli/cmd_generator.py deployment/
cp scripts/data_utils.py deployment/

# Create archive
tar -czf cmd-gen-deployment.tar.gz deployment/

# Transfer to embedded device (choose method based on your setup)
scp cmd-gen-deployment.tar.gz user@embedded-device:/home/user/
# OR
# Use USB drive, SD card, etc.
```

### 5. Install on Embedded Device

```bash
# On embedded device
cd /home/user
tar -xzf cmd-gen-deployment.tar.gz
cd deployment

# Create directory structure
mkdir -p models/onnx
mkdir -p models/checkpoints
mkdir -p scripts

# Move files to proper locations
mv encoder_quantized.onnx models/onnx/
mv decoder_quantized.onnx models/onnx/
mv input_tokenizer.pkl models/checkpoints/
mv output_tokenizer.pkl models/checkpoints/
mv data_utils.py scripts/
```

### 6. Test the Deployment

```bash
# Test Linux command generation
python cmd_generator.py generate "show network interfaces"
# Expected: ifconfig

# Test RDKB dmcli command generation
python cmd_generator.py generate "show wifi ssid"
# Expected: dmcli eRT getv Device.WiFi.SSID.1.SSID

python cmd_generator.py generate "get device model name"
# Expected: dmcli eRT getv Device.DeviceInfo.ModelName

# Test with quantized models (default for production)
python cmd_generator.py generate "list all files" --quantized
# Expected: ls -la
```

### 7. Create System Service (Optional)

For production deployment, create a systemd service:

```bash
# Create service file
sudo nano /etc/systemd/system/cmdgen.service
```

Add the following content:

```ini
[Unit]
Description=Command Generator Service
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/home/user/deployment
ExecStart=/usr/bin/python3 cmd_generator.py interactive
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Enable and start the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable cmdgen.service
sudo systemctl start cmdgen.service
```

## Performance Optimization

### Memory Optimization

1. **Use quantized models**: Always use `--quantized` flag
2. **Limit vocabulary**: Keep vocab size < 1000
3. **Reduce batch size**: Use batch size of 1 for inference

### Speed Optimization

1. **Use ONNX Runtime optimizations**:
   ```python
   import onnxruntime as ort
   
   # Set optimization level
   sess_options = ort.SessionOptions()
   sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
   
   session = ort.InferenceSession(model_path, sess_options)
   ```

2. **Cache tokenizers**: Load tokenizers once at startup, not per request

3. **Pre-allocate buffers**: Reuse numpy arrays for inference

### Storage Optimization

- Use compressed models (`.tar.gz` or `.zip`)
- Remove unused models (keep only quantized versions)
- Use memory-mapped files for large vocabularies

## Benchmarking

Test performance on your embedded device:

```bash
# Create benchmark script
cat > benchmark.py << 'EOF'
import time
import sys
sys.path.insert(0, 'scripts')

from cmd_generator import CommandGenerator

generator = CommandGenerator(
    'models/onnx/encoder_quantized.onnx',
    'models/onnx/decoder_quantized.onnx',
    'models/checkpoints/input_tokenizer.pkl',
    'models/checkpoints/output_tokenizer.pkl'
)

test_inputs = [
    "show network interfaces",
    "list all files",
    "show disk usage",
    "display running processes",
    "show current directory"
]

total_time = 0
for i in range(10):
    for input_text in test_inputs:
        start = time.time()
        result = generator.generate(input_text)
        end = time.time()
        total_time += (end - start)

avg_time = total_time / (10 * len(test_inputs))
print(f"Average inference time: {avg_time*1000:.2f}ms")
EOF

python benchmark.py
```

Expected performance:
- **Intel Atom**: 40-80ms per command (quantized INT8)
- **Raspberry Pi 4**: 50-100ms per command
- **ARM Cortex-A series**: 80-150ms per command
- **RDK-B Devices**: 60-120ms per command

**Model Specifications:**
- Parameters: 6,705,053 (6.7M)
- Architecture: 2-layer GRU encoder-decoder with attention
- Embedding dimension: 256
- Hidden dimension: 512
- Vocabulary: Input=326, Output=412
- Dataset: 611 samples (519 train, 92 validation)

## Troubleshooting

### Issue: Out of Memory

**Solution**: 
- Use INT8 quantized models (saves ~4.5% memory)
- Process one command at a time (no batching)
- Close other applications
- For extreme constraints, consider:
  - Reducing vocabulary size (edit tokenizer max_vocab_size)
  - Using single-layer model (retrain with num_layers: 1)
  - Reducing hidden_dim to 256 (retrain required)

### Issue: Slow Inference

**Solution**:
- Ensure you're using INT8 quantized models (default)
- Check CPU frequency scaling: `cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor`
- Enable performance mode: `sudo cpupower frequency-set -g performance`
- Use ONNX Runtime execution providers:
  ```python
  providers = ['CPUExecutionProvider']
  session = ort.InferenceSession(model, providers=providers)
  ```
- For Intel devices, consider OpenVINO execution provider

### Issue: Import Errors

**Solution**:
```bash
# Verify installation
pip list | grep onnx
pip list | grep numpy

# Reinstall if needed
pip install --force-reinstall onnxruntime numpy
```

### Issue: Model Files Not Found

**Solution**:
- Verify file paths are correct
- Check file permissions: `chmod 644 models/**/*.onnx`
- Ensure directory structure matches expectations

## Integration Examples

### REST API Wrapper

```python
from flask import Flask, request, jsonify
import onnxruntime as ort
import pickle
import numpy as np
from scripts.data_utils import Tokenizer

app = Flask(__name__)

# Load models once at startup
with open('models/checkpoints/input_tokenizer.pkl', 'rb') as f:
    input_tokenizer = pickle.load(f)
with open('models/checkpoints/output_tokenizer.pkl', 'rb') as f:
    output_tokenizer = pickle.load(f)

encoder_session = ort.InferenceSession('models/onnx/encoder_quantized.onnx')
decoder_session = ort.InferenceSession('models/onnx/decoder_quantized.onnx')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    text = data.get('input', '')
    
    # Tokenize and generate
    tokens = input_tokenizer.encode(text)
    input_tensor = np.array([tokens], dtype=np.int64)
    
    # Run encoder
    encoder_outputs, hidden = encoder_session.run(None, {'input': input_tensor})
    
    # Run decoder (simplified)
    output_tokens = []
    decoder_input = np.array([[output_tokenizer.token2id['<START>']]], dtype=np.int64)
    
    for _ in range(50):
        output, hidden, attn = decoder_session.run(
            None,
            {
                'input': decoder_input,
                'hidden': hidden,
                'encoder_outputs': encoder_outputs
            }
        )
        next_token = np.argmax(output[0, -1, :])
        if next_token == output_tokenizer.token2id['<END>']:
            break
        output_tokens.append(next_token)
        decoder_input = np.array([[next_token]], dtype=np.int64)
    
    command = output_tokenizer.decode(output_tokens)
    return jsonify({'command': command, 'input': text})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model': 'seq2seq-2layer-gru'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### Shell Integration

```bash
# Add to .bashrc or .profile
cmdgen() {
    python /opt/cmd-generator/cmd_generator.py generate "$*"
}

# Advanced: Execute generated command directly
nlcmd() {
    local cmd=$(python /opt/cmd-generator/cmd_generator.py generate "$*" 2>/dev/null | grep -E '^(ifconfig|ls|ps|df|dmcli|cat|grep|find|iptables)' | head -1)
    if [ -n "$cmd" ]; then
        echo "Executing: $cmd"
        eval "$cmd"
    else
        echo "Could not generate command for: $*"
    fi
}

# Usage examples
cmdgen show network interfaces
# Output: ifconfig

nlcmd show wifi ssid
# Executes: dmcli eRT getv Device.WiFi.SSID.1.SSID

nlcmd list all files
# Executes: ls -la
```

## Security Considerations

1. **Input Validation**: Validate user input before processing
2. **Command Sanitization**: Sanitize generated commands before execution
3. **Access Control**: Restrict who can generate commands
4. **Audit Logging**: Log all command generation requests

## Monitoring

Monitor system resources:

```bash
# CPU and memory usage
top -p $(pgrep -f cmd_generator.py)

# Detailed memory analysis
cat /proc/$(pgrep -f cmd_generator.py)/status | grep -E "VmRSS|VmSize"
```

## Updates and Maintenance

To update the model on embedded systems:

1. Train new model on development machine
2. Export to ONNX
3. Transfer only the ONNX files (not full codebase)
4. Replace old models
5. Restart service

```bash
# On embedded device
cd deployment
systemctl stop cmdgen
cp new_encoder_quantized.onnx models/onnx/
cp new_decoder_quantized.onnx models/onnx/
systemctl start cmdgen
```

## C++ Deployment (For Devices Without Python)

For embedded devices that don't have Python support, we provide a pure C++ implementation using ONNX Runtime C++ API.

### Advantages of C++ Deployment

✅ **No Python Dependency**: Single binary executable  
✅ **Smaller Footprint**: ~500KB binary (stripped) + ~5MB ONNX Runtime  
✅ **Lower Memory**: 45-55 MB RAM vs 50-60 MB with Python  
✅ **Faster Startup**: No Python interpreter initialization  
✅ **Static Linking**: Can create fully standalone binary  
✅ **Production Ready**: Battle-tested for embedded systems  

### Quick C++ Build & Deploy

```bash
# On development machine
cd cpp
./build.sh

# This will:
# 1. Download ONNX Runtime for your architecture
# 2. Export vocabularies from Python tokenizers  
# 3. Build the C++ application

# Create deployment package
mkdir -p deploy_cpp
cp build/cmd_generator deploy_cpp/
cp ../models/onnx/encoder_quantized.onnx deploy_cpp/
cp ../models/onnx/decoder_quantized.onnx deploy_cpp/
cp ../models/checkpoints/input_vocab.txt deploy_cpp/
cp ../models/checkpoints/output_vocab.txt deploy_cpp/
cp ../onnxruntime/lib/libonnxruntime.so* deploy_cpp/

# Create archive
tar -czf cmd-generator-cpp.tar.gz deploy_cpp/

# Transfer to device
scp cmd-generator-cpp.tar.gz root@192.168.1.1:/tmp/
```

### Install on Embedded Device (C++)

```bash
# SSH to device
ssh root@192.168.1.1

# Extract
cd /opt
tar -xzf /tmp/cmd-generator-cpp.tar.gz
cd deploy_cpp

# Set library path
export LD_LIBRARY_PATH=/opt/deploy_cpp:$LD_LIBRARY_PATH

# Or install library system-wide
sudo cp libonnxruntime.so* /usr/lib/
sudo ldconfig

# Test
./cmd_generator generate "show wifi ssid"
# Expected: dmcli eRT getv Device.WiFi.SSID.1.SSID
```

### C++ Usage Examples

```bash
# Generate single command
./cmd_generator generate "show network interfaces"
# Output: ifconfig

./cmd_generator generate "show wifi ssid"
# Output: dmcli eRT getv Device.WiFi.SSID.1.SSID

# Interactive mode
./cmd_generator interactive

# Batch processing
echo "show wifi ssid
list all files
show memory usage" > commands.txt
./cmd_generator batch commands.txt
```

### C++ Memory & Performance

**Binary Size:**
- Unstripped: ~2 MB
- Stripped: ~500 KB (`strip cmd_generator`)

**Runtime Memory:**
- Executable: 500 KB
- ONNX Runtime: 5-10 MB
- Models: 24.45 MB
- Inference: 15-25 MB
- **Total: 45-55 MB RAM**

**Performance:**
- Intel Atom: 40-60ms per command
- ARM Cortex-A53: 60-100ms per command
- RDK-B devices: 60-120ms per command

### C++ System Service

Create service for C++ deployment:

```bash
# /etc/systemd/system/cmd-generator-cpp.service
cat << 'EOF' | sudo tee /etc/systemd/system/cmd-generator-cpp.service
[Unit]
Description=Command Generator (C++)
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/deploy_cpp
Environment="LD_LIBRARY_PATH=/opt/deploy_cpp"
ExecStart=/opt/deploy_cpp/cmd_generator interactive
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOF

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable cmd-generator-cpp.service
sudo systemctl start cmd-generator-cpp.service
```

### Cross-Compilation for ARM

For cross-compiling from x86_64 to ARM:

```bash
# Install ARM toolchain
sudo apt-get install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu

# Download ARM ONNX Runtime
cd cpp
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-aarch64-1.16.3.tgz
tar -xzf onnxruntime-linux-aarch64-1.16.3.tgz
mv onnxruntime-linux-aarch64-1.16.3 ../onnxruntime

# Build for ARM
mkdir -p build && cd build
cmake .. -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc \
         -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ \
         -DCMAKE_SYSTEM_NAME=Linux \
         -DCMAKE_SYSTEM_PROCESSOR=aarch64
make -j$(nproc)
```

### C++ Deployment Comparison

| Feature | Python | C++ |
|---------|--------|-----|
| Python Required | ✅ Yes (3.8+) | ❌ No |
| Binary Size | ~50 MB | ~500 KB |
| Runtime Memory | 50-60 MB | 45-55 MB |
| Startup Time | ~1-2s | ~0.1-0.2s |
| Inference Time | 60-120ms | 60-120ms |
| Easy to Modify | ✅ Yes | Moderate |
| Production Ready | ✅ Yes | ✅ Yes |
| Best For | Development/Testing | Production/Embedded |

### Detailed C++ Documentation

For complete C++ deployment documentation including:
- Build instructions
- API reference
- Library integration
- Optimization techniques
- Troubleshooting

See **[cpp/README.md](../cpp/README.md)**

## Deployment Decision Guide

**Use Python Deployment if:**
- Device has Python 3.8+ installed
- You need to quickly iterate and test
- You want to easily modify inference logic
- Development/testing environment

**Use C++ Deployment if:**
- Device has no Python support
- You need minimal resource usage
- You want a single standalone binary
- Production embedded environment
- RDK-B or similar constrained devices

Both options provide the same inference quality and support the full 611-command dataset (Linux + RDKB).

## Support

For deployment issues, please open an issue on GitHub with:
- Device specifications
- Error messages
- Performance metrics
- Deployment configuration

---

Successfully deployed? Share your experience and help improve this guide!
