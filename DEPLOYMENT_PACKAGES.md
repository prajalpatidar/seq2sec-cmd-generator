# Deployment Packages Summary

This document provides an overview of the two deployment options for the seq2seq command generator.

## Package Comparison

| Feature | C++ Deployment | Python Deployment |
|---------|----------------|-------------------|
| **Environment** | No Python required | Python 3.8+ required |
| **Package Size** | 44 MB | 76 MB |
| **Binary Size** | 776 KB (exec) + 1.3 MB (lib) | N/A |
| **Dependencies** | ONNX Runtime only | PyTorch + NumPy + ONNX |
| **Memory Usage** | 45-55 MB RAM | 100-150 MB RAM |
| **Inference Speed** | 60-120ms | 80-150ms (PyTorch) |
| **Cold Start** | 300-500ms | 1-2 seconds |
| **Installation** | Copy and run | pip install |
| **Platform** | Linux x86_64/ARM64 | Cross-platform |
| **Best For** | Embedded devices, production | Development, flexibility |

## C++ Deployment (deployment-cpp/)

### Contents
```
deployment-cpp/
├── cmd_generator              # Executable (776 KB)
├── libseq2sec_lib.so          # Shared library (1.3 MB)
├── models/
│   ├── onnx/
│   │   ├── encoder_quantized.onnx  (11 MB)
│   │   └── decoder_quantized.onnx  (14 MB)
│   └── checkpoints/
│       ├── input_vocab.txt         (2.3 KB)
│       └── output_vocab.txt        (2.9 KB)
├── onnxruntime/               # ONNX Runtime (6.3 MB)
│   └── lib/
│       └── libonnxruntime.so.1.16.3
├── run.sh                     # Launcher script
└── README.md                  # Documentation
```

### Quick Start
```bash
cd deployment-cpp
./run.sh generate "show network interfaces"
```

### Use Cases
- ✅ RDK-B embedded devices (no Python)
- ✅ Production deployments (minimal dependencies)
- ✅ Resource-constrained environments
- ✅ Fast inference requirements
- ✅ Standalone executables

### Advantages
- **No Python Runtime**: Perfect for embedded systems
- **Smaller Memory Footprint**: 45-55 MB vs 100-150 MB
- **Faster Inference**: 60-120ms vs 80-150ms
- **Single Binary**: Easy to deploy
- **Production Ready**: Stable, minimal dependencies

### Deployment
```bash
# Option 1: Direct copy
scp -r deployment-cpp/ user@device:/opt/seq2seq/
ssh user@device "cd /opt/seq2seq/deployment-cpp && ./run.sh generate 'test'"

# Option 2: Tarball
tar -czf seq2seq-cpp.tar.gz deployment-cpp/
scp seq2seq-cpp.tar.gz user@device:/tmp/
ssh user@device "cd /tmp && tar -xzf seq2seq-cpp.tar.gz"

# Option 3: System installation
sudo cp deployment-cpp/cmd_generator /usr/local/bin/
sudo cp deployment-cpp/libseq2sec_lib.so /usr/local/lib/
sudo cp -r deployment-cpp/models /usr/local/share/seq2seq/
sudo cp -r deployment-cpp/onnxruntime/lib/* /usr/local/lib/
sudo ldconfig
```

## Python Deployment (deployment-python/)

### Contents
```
deployment-python/
├── cli/
│   ├── __init__.py
│   └── cmd_generator.py       # CLI tool
├── models/
│   ├── __init__.py
│   ├── seq2seq.py
│   ├── checkpoints/           # PyTorch models (50 MB)
│   └── onnx/                  # ONNX models (25 MB)
├── requirements.txt
├── setup.py
└── README.md
```

### Quick Start
```bash
cd deployment-python
pip install -r requirements.txt
pip install -e .
seq2seq-generate "show network interfaces"
```

### Use Cases
- ✅ Development and testing
- ✅ Research and experimentation
- ✅ Environments with Python already
- ✅ Need for model fine-tuning
- ✅ Web service deployments

### Advantages
- **Flexibility**: Easy to modify and extend
- **Python Ecosystem**: Access to full PyTorch tools
- **Model Training**: Can retrain or fine-tune
- **Cross-Platform**: Windows, Linux, macOS
- **Rapid Development**: Python scripting

### Deployment
```bash
# Option 1: Virtual environment
python3 -m venv venv
source venv/bin/activate
cd deployment-python
pip install -r requirements.txt
pip install -e .

# Option 2: Conda environment
conda create -n seq2seq python=3.8
conda activate seq2seq
cd deployment-python
pip install -r requirements.txt
pip install -e .

# Option 3: Docker
docker build -t seq2seq-py -f deployment-python/Dockerfile .
docker run -it seq2seq-py seq2seq-generate "test"
```

## Choosing the Right Deployment

### Choose C++ Deployment if:
- ✅ Target device has no Python (embedded systems)
- ✅ Minimal resource usage required (< 60 MB RAM)
- ✅ Fastest inference speed needed (60-120ms)
- ✅ Production deployment with stability focus
- ✅ Single binary distribution preferred
- ✅ Targeting RDK-B or similar embedded devices

### Choose Python Deployment if:
- ✅ Python environment already available
- ✅ Need to modify or retrain models
- ✅ Development or testing environment
- ✅ Web service or API deployment
- ✅ Cross-platform compatibility needed
- ✅ Rapid prototyping and iteration

## Performance Comparison

### Inference Speed
```
C++ (ONNX INT8):     60-120ms per command
Python (ONNX INT8):  60-120ms per command (with ONNX Runtime)
Python (PyTorch):    80-150ms per command
```

### Memory Usage
```
C++ Peak:            45-55 MB RAM
Python (ONNX):       80-100 MB RAM
Python (PyTorch):    100-150 MB RAM
```

### Package Size
```
C++ (compressed):    ~15 MB (tar.gz)
C++ (uncompressed):  44 MB
Python (compressed): ~30 MB (tar.gz)
Python (full):       76 MB + dependencies
```

### Cold Start Time
```
C++ (model load):    300-500ms
Python (import):     1-2 seconds
```

## Integration Examples

### C++ Integration
```bash
# Shell script
#!/bin/bash
export LD_LIBRARY_PATH=/opt/seq2seq/deployment-cpp/onnxruntime/lib
QUERY="$1"
CMD=$(/opt/seq2seq/deployment-cpp/run.sh generate "$QUERY")
echo "$CMD"
eval "$CMD"
```

### Python Integration
```python
from cli.cmd_generator import CommandGenerator

generator = CommandGenerator()
command = generator.generate("show network interfaces")
print(f"Command: {command}")

# Execute
import subprocess
subprocess.run(command, shell=True)
```

### Python Web API
```python
from flask import Flask, request, jsonify
from cli.cmd_generator import CommandGenerator

app = Flask(__name__)
generator = CommandGenerator()

@app.route('/generate', methods=['POST'])
def generate():
    query = request.json.get('query')
    command = generator.generate(query)
    return jsonify({'query': query, 'command': command})

app.run(host='0.0.0.0', port=5000)
```

## Testing

### Test C++ Deployment
```bash
cd deployment-cpp

# Linux commands
./run.sh generate "show network interfaces"
./run.sh generate "list all files"
./run.sh generate "check disk space"

# RDKB commands
./run.sh generate "show wifi ssid"
./run.sh generate "enable wifi radio"
./run.sh generate "get lan ip address"
```

### Test Python Deployment
```bash
cd deployment-python
pip install -r requirements.txt
pip install -e .

# Linux commands
seq2seq-generate "show network interfaces"
seq2seq-generate "list all files"
seq2seq-generate "check disk space"

# RDKB commands
seq2seq-generate "show wifi ssid"
seq2seq-generate "enable wifi radio"
seq2seq-generate "get lan ip address"
```

## Troubleshooting

### C++ Deployment Issues

**Library not found:**
```bash
export LD_LIBRARY_PATH=./onnxruntime/lib:$LD_LIBRARY_PATH
# Or use run.sh which sets this automatically
```

**Model files not found:**
```bash
# Ensure directory structure is correct:
deployment-cpp/
├── models/
│   ├── onnx/
│   │   ├── encoder_quantized.onnx
│   │   └── decoder_quantized.onnx
│   └── checkpoints/
│       ├── input_vocab.txt
│       └── output_vocab.txt
```

### Python Deployment Issues

**Import errors:**
```bash
pip install -e .
# Or set PYTHONPATH
export PYTHONPATH=/path/to/deployment-python:$PYTHONPATH
```

**Model not found:**
```python
# Use absolute paths
generator = CommandGenerator(
    encoder_path='/absolute/path/to/models/checkpoints/encoder.pth',
    # ...
)
```

## Version Information

- **Model Version**: 2.0.0
- **Release Date**: November 27, 2025
- **ONNX Runtime**: 1.16.3
- **PyTorch**: 2.5.1
- **Architecture**: 2-layer GRU, 6.7M parameters
- **Training Data**: 611 samples (Linux + RDKB)

## Support & Documentation

- **C++ README**: `deployment-cpp/README.md`
- **Python README**: `deployment-python/README.md`
- **Project Docs**: `docs/`
- **Main README**: `../README.md`
