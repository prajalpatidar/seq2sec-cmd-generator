# Quick Deployment Guide

## Two Deployment Options Available

### 🚀 C++ Deployment (Recommended for RDK-B)
- **Size**: 44 MB (30 MB compressed)
- **Requirements**: None (no Python)
- **Memory**: 45-55 MB RAM
- **Speed**: 60-120ms inference

### 🐍 Python Deployment
- **Size**: 76 MB (71 MB compressed)
- **Requirements**: Python 3.8+
- **Memory**: 100-150 MB RAM
- **Speed**: 80-150ms inference

---

## C++ Deployment (deployment-cpp/)

### Quick Start
```bash
cd deployment-cpp
./run.sh generate "show network interfaces"
```

### Transfer to Device
```bash
# Method 1: Direct transfer
scp seq2seq-cpp-deployment.tar.gz user@device:/tmp/
ssh user@device
cd /tmp
tar -xzf seq2seq-cpp-deployment.tar.gz
cd deployment-cpp
./run.sh generate "show network interfaces"

# Method 2: Direct copy
scp -r deployment-cpp/ user@device:/opt/seq2seq/
ssh user@device "cd /opt/seq2seq/deployment-cpp && ./run.sh generate 'test'"
```

### Usage
```bash
# Generate single command
./run.sh generate "show network interfaces"
# Output: ifconfig

# Interactive mode
./run.sh interactive

# Batch processing
./run.sh batch queries.txt commands.txt
```

### Examples
```bash
# Linux commands
./run.sh generate "show network interfaces"  # → ifconfig
./run.sh generate "list all files"           # → ls -l
./run.sh generate "check disk space"         # → df -h

# RDKB commands
./run.sh generate "show wifi ssid"           # → dmcli ert getv Device.Wifi.Ssid.1.Enable bool
./run.sh generate "enable wifi radio"        # → dmcli ert setv Device.Wifi.Radio.1.Enable bool true
./run.sh generate "get lan ip address"       # → dmcli ert getv Device.LAN.IPAddress string
```

---

## Python Deployment (deployment-python/)

### Quick Start
```bash
cd deployment-python
pip install -r requirements.txt
pip install -e .
python -m cli.cmd_generator generate "show network interfaces"
```

### Installation
```bash
# Option 1: Virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Option 2: Direct installation
pip install -r requirements.txt
pip install -e .
```

### Usage
```bash
# Generate single command
python -m cli.cmd_generator generate "show network interfaces"
# Output: Generated Command: ifconfig

# Interactive mode
python -m cli.cmd_generator interactive
```

### Examples
```bash
# Linux commands
python -m cli.cmd_generator generate "show network interfaces"
python -m cli.cmd_generator generate "list all files"
python -m cli.cmd_generator generate "check disk space"

# RDKB commands
python -m cli.cmd_generator generate "show wifi ssid"
python -m cli.cmd_generator generate "enable wifi radio"
python -m cli.cmd_generator generate "get lan ip address"
```

---

## Package Contents

### C++ Package
```
deployment-cpp/
├── cmd_generator              # Executable (776 KB)
├── libseq2sec_lib.so          # Library (1.3 MB)
├── models/
│   ├── onnx/                  # ONNX models (25 MB)
│   └── checkpoints/           # Vocabularies (5 KB)
├── onnxruntime/               # Runtime (6.3 MB)
├── run.sh                     # Launcher
└── README.md
```

### Python Package
```
deployment-python/
├── cli/                       # CLI tool
├── models/                    # Models + checkpoints (50 MB)
├── scripts/                   # Utilities
├── requirements.txt
├── setup.py
└── README.md
```

---

## Performance Comparison

| Metric | C++ | Python (PyTorch) | Python (ONNX) |
|--------|-----|------------------|---------------|
| Inference Time | 60-120ms | 80-150ms | 60-120ms |
| Memory Usage | 45-55 MB | 100-150 MB | 80-100 MB |
| Cold Start | 300-500ms | 1-2 sec | 800ms-1sec |
| Package Size | 44 MB | 76 MB | 76 MB |
| Python Required | ❌ No | ✅ Yes | ✅ Yes |

---

## Choosing the Right Deployment

### Choose C++ if:
- ✅ Target is embedded device (RDK-B)
- ✅ No Python environment available
- ✅ Minimal resource usage critical
- ✅ Production stability required
- ✅ Fastest inference needed

### Choose Python if:
- ✅ Python already installed
- ✅ Development/testing phase
- ✅ Need model fine-tuning
- ✅ Web service deployment
- ✅ Cross-platform required

---

## Testing

### Test C++ Deployment
```bash
cd deployment-cpp

# Test 1: Linux command
./run.sh generate "show network interfaces"
# Expected: ifconfig

# Test 2: RDKB command
./run.sh generate "show wifi ssid"
# Expected: dmcli ert getv Device.Wifi.Ssid.1.Enable bool

# Test 3: Interactive
./run.sh interactive
# Enter queries interactively
```

### Test Python Deployment
```bash
cd deployment-python
source venv/bin/activate  # if using venv

# Test 1: Linux command
python -m cli.cmd_generator generate "show network interfaces"
# Expected: Generated Command: ifconfig

# Test 2: RDKB command
python -m cli.cmd_generator generate "show wifi ssid"
# Expected: Generated Command: dmcli ert getv Device.Wifi.Ssid.1.Ssid

# Test 3: Interactive
python -m cli.cmd_generator interactive
# Enter queries interactively
```

---

## Troubleshooting

### C++ Deployment

**Error: Library not found**
```bash
# Solution: run.sh automatically sets LD_LIBRARY_PATH
# If running cmd_generator directly:
export LD_LIBRARY_PATH=./onnxruntime/lib:$LD_LIBRARY_PATH
./cmd_generator generate "test"
```

**Error: Model files not found**
```bash
# Solution: Ensure you run from deployment-cpp directory
cd deployment-cpp
./run.sh generate "test"
```

### Python Deployment

**Error: Module not found**
```bash
# Solution: Install package
pip install -e .

# Or set PYTHONPATH
export PYTHONPATH=/path/to/deployment-python:$PYTHONPATH
```

**Error: No such command**
```bash
# Correct usage:
python -m cli.cmd_generator generate "query"
#                          ^^^^^^^^ command name required
```

---

## Support

For detailed documentation, see:
- **C++**: `deployment-cpp/README.md`
- **Python**: `deployment-python/README.md`
- **Comparison**: `DEPLOYMENT_PACKAGES.md`
- **Main Project**: `README.md`

---

**Version**: 2.0.0  
**Date**: November 27, 2025  
**Model**: 2-layer GRU, 6.7M parameters, INT8 quantized
