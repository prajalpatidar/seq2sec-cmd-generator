# Embedded Systems Deployment Guide

This guide provides detailed instructions for deploying the seq2sec-cmd-generator on embedded systems with limited resources.

## Target Environment

- **CPU**: Intel Atom dual-core or similar
- **RAM**: 500MB available
- **Storage**: 50MB minimum
- **OS**: Linux-based (Ubuntu, Debian, Yocto, etc.)

## Deployment Steps

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
- `models/onnx/encoder_quantized.onnx` (~250KB)
- `models/onnx/decoder_quantized.onnx` (~250KB)
- `models/checkpoints/input_tokenizer.pkl` (~50KB)
- `models/checkpoints/output_tokenizer.pkl` (~50KB)

**Total size: ~600KB**

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
# Test command generation
python cmd_generator.py generate "show network interfaces" --quantized

# Should output: ifconfig
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
- **Intel Atom**: 20-50ms per command
- **Raspberry Pi 4**: 30-80ms per command
- **ARM Cortex-A9**: 50-150ms per command

## Troubleshooting

### Issue: Out of Memory

**Solution**: 
- Reduce model size by decreasing `hidden_dim` in config
- Use smaller vocabulary size
- Close other applications

### Issue: Slow Inference

**Solution**:
- Ensure you're using quantized models
- Check CPU frequency scaling settings
- Consider using int8 quantization

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
from cmd_generator import CommandGenerator

app = Flask(__name__)
generator = CommandGenerator(...)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    command = generator.generate(data['input'])
    return jsonify({'command': command})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### Shell Integration

```bash
# Add to .bashrc
cmdgen() {
    python /path/to/cmd_generator.py generate "$*" --quantized
}

# Usage
cmdgen show network interfaces
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

## Support

For deployment issues, please open an issue on GitHub with:
- Device specifications
- Error messages
- Performance metrics
- Deployment configuration

---

Successfully deployed? Share your experience and help improve this guide!
