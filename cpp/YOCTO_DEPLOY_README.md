# C++ Deployment Package (Yocto / Target)

This is a standalone C++ deployment package for the seq2seq command generator, cross-compiled for Yocto (core2 x86_64).

## Contents

```
deployment-cpp-yocto/
├── cmd_generator              # Main executable
├── libseq2sec_lib.so          # Shared library
├── models/
│   ├── onnx/                  # ONNX models
│   └── checkpoints/           # Vocabularies
├── onnxruntime/               # ONNX Runtime library
├── run.sh                     # Launcher script
└── README.md                  # This file
```

**Total Package Size**: ~32 MB

## System Requirements

- Yocto-based Linux (core2 x86_64)
- GLIBC compatible with the build toolchain
- ~45-55 MB RAM at runtime
- No Python required

## Quick Start

### 1. Transfer to Target

Copy this entire directory to your target device (e.g., via SCP or USB).

### 2. Run

Use the provided `run.sh` script which sets up the environment variables:

```bash
./run.sh generate "show network interfaces"
```

Or manually:

```bash
export LD_LIBRARY_PATH=./onnxruntime/lib:$LD_LIBRARY_PATH
./cmd_generator generate "show network interfaces"
```

## Usage Examples

### Linux Commands
```bash
./cmd_generator generate "show network interfaces"
# Output: ifconfig

./cmd_generator generate "list all files"
# Output: ls -l

./cmd_generator generate "check disk space"
# Output: df -h
```

### RDKB Commands
```bash
./cmd_generator generate "show wifi ssid"
# Output: dmcli ert getv Device.Wifi.Ssid.1.Enable bool

./cmd_generator generate "enable wifi radio"
# Output: dmcli ert setv Device.Wifi.Radio.1.Enable bool true

./cmd_generator generate "get lan ip address"
# Output: dmcli ert getv Device.LAN.IPAddress string
```

### Interactive Mode
```bash
./cmd_generator interactive

# Then type queries interactively:
> show network interfaces
Command: ifconfig

> show wifi ssid
Command: dmcli ert getv Device.Wifi.Ssid.1.Enable bool

> exit
```

### Batch Mode
```bash
# Create input file
cat > queries.txt << EOF
show network interfaces
list all files
show wifi ssid
check disk space
EOF

# Process batch
./cmd_generator batch queries.txt commands.txt

# View results
cat commands.txt
```

## Deployment to Embedded Device

### Option 1: Direct Copy
```bash
# Copy entire directory to target device
scp -r deployment-cpp/ user@device:/opt/seq2seq/

# On target device
cd /opt/seq2seq/deployment-cpp
export LD_LIBRARY_PATH=./onnxruntime/lib:$LD_LIBRARY_PATH
./cmd_generator generate "show network interfaces"
```

### Option 2: System-wide Installation
```bash
# Copy binary to system path
sudo cp cmd_generator /usr/local/bin/

# Copy library
sudo cp libseq2sec_lib.so /usr/local/lib/

# Copy ONNX Runtime
sudo cp onnxruntime/lib/libonnxruntime.so.1.16.3 /usr/local/lib/
sudo ln -s /usr/local/lib/libonnxruntime.so.1.16.3 /usr/local/lib/libonnxruntime.so
sudo ldconfig

# Copy models and vocabularies
sudo mkdir -p /usr/local/share/seq2seq/
sudo cp encoder_quantized.onnx decoder_quantized.onnx /usr/local/share/seq2seq/
sudo cp input_vocab.txt output_vocab.txt /usr/local/share/seq2seq/

# Now run from anywhere
cmd_generator generate "show network interfaces"
```

### Option 3: Create Tarball for Transfer
```bash
# Create compressed archive
tar -czf seq2seq-cpp-deployment.tar.gz deployment-cpp/

# Transfer to device
scp seq2seq-cpp-deployment.tar.gz user@device:/tmp/

# Extract on device
ssh user@device
cd /tmp
tar -xzf seq2seq-cpp-deployment.tar.gz
cd deployment-cpp
export LD_LIBRARY_PATH=./onnxruntime/lib:$LD_LIBRARY_PATH
./cmd_generator generate "test command"
```

## Performance Characteristics

- **Inference Time**: 60-120ms per command (quantized INT8)
- **Memory Usage**: 45-55 MB RAM
- **Binary Size**: 776 KB (executable) + 1.3 MB (library)
- **Model Size**: 24.5 MB (both ONNX models)
- **Cold Start**: ~300-500ms (model loading)
- **Warm Inference**: 60-120ms per query

## Troubleshooting

### Library Not Found
```bash
# Error: error while loading shared libraries: libonnxruntime.so.1.16.3
export LD_LIBRARY_PATH=./onnxruntime/lib:$LD_LIBRARY_PATH

# Or add to shell profile
echo 'export LD_LIBRARY_PATH=/path/to/deployment-cpp/onnxruntime/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
```

### Model Files Not Found
```bash
# Error: Cannot open ONNX model
# Ensure you run from deployment-cpp directory, or provide absolute paths
cd /path/to/deployment-cpp
./cmd_generator generate "test"
```

### Permission Denied
```bash
chmod +x cmd_generator
```

## Cross-Compilation

If your target device has a different architecture (e.g., ARM64), rebuild on that architecture:

```bash
# On ARM64 device
cd /path/to/project/cpp
bash build.sh

# Copy build artifacts
cp build/cmd_generator ../deployment-cpp/
cp build/libseq2sec_lib.so ../deployment-cpp/
```

## Optimization

### Reduce Binary Size
```bash
# Strip debug symbols
strip cmd_generator
# Size reduces from 776 KB to ~500 KB

strip libseq2sec_lib.so
# Size reduces from 1.3 MB to ~800 KB
```

### Memory Optimization
- Use INT8 quantized models (already included)
- Run single-threaded inference (default)
- Consider model pruning for further reduction

## Integration Examples

### Shell Script Integration
```bash
#!/bin/bash
export LD_LIBRARY_PATH=/opt/seq2seq/deployment-cpp/onnxruntime/lib:$LD_LIBRARY_PATH
QUERY="$1"
CMD=$(/opt/seq2seq/deployment-cpp/cmd_generator generate "$QUERY")
echo "Generated: $CMD"
eval "$CMD"
```

### C/C++ Program Integration
```cpp
#include "seq2seq_inference.h"

int main() {
    seq2seq::Seq2SeqInference engine;
    engine.initialize(
        "encoder_quantized.onnx",
        "decoder_quantized.onnx",
        "input_vocab.txt",
        "output_vocab.txt"
    );
    
    std::string command = engine.generate("show network interfaces");
    std::cout << "Command: " << command << std::endl;
    return 0;
}
```

## Version Information

- **Model Version**: 2.0.0
- **ONNX Runtime**: 1.16.3
- **C++ Standard**: C++17
- **Build Date**: November 27, 2025
- **Architecture**: x86_64 / ARM64

## Support

For issues or questions, refer to the main project documentation or create an issue on the repository.
