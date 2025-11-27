# C++ ONNX Runtime Deployment

This directory contains a C++ implementation for deploying the seq2seq model using ONNX Runtime. This is ideal for embedded devices that don't have Python support.

## Features

- ✅ Pure C++ implementation (no Python dependency)
- ✅ ONNX Runtime C++ API
- ✅ Single binary executable (~500KB after strip)
- ✅ Minimal dependencies
- ✅ Cross-platform (Linux, Windows, macOS)
- ✅ ARM/x86_64 support
- ✅ Fast inference (40-80ms on embedded CPU)

## Requirements

### Build Time
- CMake 3.15+
- C++17 compiler (GCC 7+, Clang 5+)
- ONNX Runtime C++ library (auto-downloaded)

### Runtime
- ONNX Runtime library (~5-10 MB)
- Model files (~25 MB)
- Vocabulary files (~350 KB)
- Total: ~26 MB

## Quick Build

```bash
cd cpp
chmod +x build.sh
./build.sh
```

This will:
1. Download ONNX Runtime for your architecture
2. Export vocabularies from Python tokenizers
3. Build the C++ application

## Manual Build

If you prefer manual steps:

### 1. Export Vocabularies

```bash
cd ..  # Back to project root
python3 scripts/export_vocab_cpp.py
```

### 2. Download ONNX Runtime

**For x86_64:**
```bash
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz
tar -xzf onnxruntime-linux-x64-1.16.3.tgz
mv onnxruntime-linux-x64-1.16.3 ../onnxruntime
```

**For ARM64/AArch64:**
```bash
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-aarch64-1.16.3.tgz
tar -xzf onnxruntime-linux-aarch64-1.16.3.tgz
mv onnxruntime-linux-aarch64-1.16.3 ../onnxruntime
```

### 3. Build with CMake

```bash
cd cpp
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

## Usage

### Generate Single Command

```bash
./build/cmd_generator generate "show network interfaces"
# Output: ifconfig

./build/cmd_generator generate "show wifi ssid"
# Output: dmcli eRT getv Device.WiFi.SSID.1.SSID

./build/cmd_generator generate "list all files"
# Output: ls -la
```

### Interactive Mode

```bash
./build/cmd_generator interactive
```

Example session:
```
=== Interactive Mode ===
Enter natural language instructions (type 'quit' to exit)

> show network interfaces
Command: ifconfig

> show wifi ssid
Command: dmcli eRT getv Device.WiFi.SSID.1.SSID

> list running processes
Command: ps aux

> quit
Goodbye!
```

### Batch Processing

Create a text file with commands (one per line):

```bash
# batch_commands.txt
show network interfaces
show wifi ssid
list all files
show memory usage
get device model name
```

Run batch:
```bash
./build/cmd_generator batch batch_commands.txt
```

### Custom Model Paths

```bash
./build/cmd_generator generate "show wifi ssid" \
    --encoder models/onnx/encoder_quantized.onnx \
    --decoder models/onnx/decoder_quantized.onnx \
    --input-vocab models/checkpoints/input_vocab.txt \
    --output-vocab models/checkpoints/output_vocab.txt \
    --max-length 50
```

## Deployment to Embedded Device

### 1. Cross-Compile (Optional)

For cross-compilation to ARM:

```bash
# Install ARM toolchain
sudo apt-get install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu

# Configure CMake for ARM
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=../arm-toolchain.cmake
make
```

### 2. Prepare Deployment Package

```bash
# Create deployment directory
mkdir -p deploy

# Copy executable
cp build/cmd_generator deploy/

# Copy models
cp -r ../models/onnx/encoder_quantized.onnx deploy/
cp -r ../models/onnx/decoder_quantized.onnx deploy/

# Copy vocabularies
cp ../models/checkpoints/input_vocab.txt deploy/
cp ../models/checkpoints/output_vocab.txt deploy/

# Copy ONNX Runtime library
cp ../onnxruntime/lib/libonnxruntime.so* deploy/

# Create deployment archive
tar -czf cmd-generator-cpp.tar.gz deploy/
```

### 3. Transfer to Target Device

```bash
# SCP to embedded device
scp cmd-generator-cpp.tar.gz root@192.168.1.1:/tmp/

# SSH and extract
ssh root@192.168.1.1
cd /opt
tar -xzf /tmp/cmd-generator-cpp.tar.gz
cd deploy
```

### 4. Set Library Path

```bash
# Add to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/deploy:$LD_LIBRARY_PATH

# Or copy library to system path
sudo cp libonnxruntime.so* /usr/lib/
sudo ldconfig
```

### 5. Test on Device

```bash
./cmd_generator generate "show wifi ssid"
# Expected: dmcli eRT getv Device.WiFi.SSID.1.SSID
```

## Optimization for Embedded

### Strip Binary

```bash
strip build/cmd_generator
# Reduces size from ~2MB to ~500KB
```

### Static Linking

For a fully standalone binary, edit CMakeLists.txt:

```cmake
set(CMAKE_EXE_LINKER_FLAGS "-static")
target_link_libraries(cmd_generator -static ${ONNXRUNTIME_LIB})
```

This creates a single binary with no runtime dependencies (except kernel).

### Use Quantized Models

Always use INT8 quantized models on embedded devices:
- encoder_quantized.onnx (10.61 MB)
- decoder_quantized.onnx (13.84 MB)

These provide 40-80ms inference vs 50-100ms for FP32.

## Memory Usage

### Build Time
- Compilation: ~500 MB RAM
- Linking: ~200 MB RAM

### Runtime
- Executable: ~500 KB (stripped)
- ONNX Runtime: ~5-10 MB
- Models loaded: ~25 MB
- Inference: 15-25 MB RAM
- **Total: ~45-60 MB RAM**

## Performance Benchmarks

### x86_64 (Intel Atom)
- Inference time: 40-60ms (INT8)
- Memory: 50 MB
- CPU usage: 10-15%

### ARM64 (Cortex-A53)
- Inference time: 60-100ms (INT8)
- Memory: 45 MB
- CPU usage: 15-25%

### RDK-B Device
- Inference time: 60-120ms (INT8)
- Memory: 45-55 MB
- CPU usage: 20-30%

## Integration

### As Library

The build also creates `libseq2sec_lib.so` for integration:

```cpp
#include "seq2seq_inference.h"

int main() {
    Seq2SeqInference inference;
    inference.initialize(
        "encoder_quantized.onnx",
        "decoder_quantized.onnx", 
        "input_vocab.txt",
        "output_vocab.txt"
    );
    
    std::string command = inference.generate("show wifi ssid");
    std::cout << "Command: " << command << std::endl;
    
    return 0;
}
```

Compile:
```bash
g++ -o myapp myapp.cpp -lseq2sec_lib -L./build -I./include -lonnxruntime
```

### System Service

Create systemd service on embedded device:

```ini
# /etc/systemd/system/cmd-generator.service
[Unit]
Description=Seq2Seq Command Generator Service
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/deploy
Environment="LD_LIBRARY_PATH=/opt/deploy"
ExecStart=/opt/deploy/cmd_generator interactive
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Enable:
```bash
systemctl enable cmd-generator.service
systemctl start cmd-generator.service
```

## Troubleshooting

### ONNX Runtime Not Found

```bash
error while loading shared libraries: libonnxruntime.so
```

**Solution:**
```bash
export LD_LIBRARY_PATH=/path/to/onnxruntime/lib:$LD_LIBRARY_PATH
# Or copy to system library path
sudo cp libonnxruntime.so* /usr/lib/
sudo ldconfig
```

### Model Files Not Found

```bash
Failed to load encoder model
```

**Solution:** Ensure you're running from the correct directory or use absolute paths:
```bash
cd /opt/deploy
./cmd_generator generate "test"
```

### Vocabulary Not Found

```bash
Failed to load input vocabulary
```

**Solution:** Export vocabularies first:
```bash
python3 scripts/export_vocab_cpp.py
```

### Slow Performance

**Solution:**
1. Use INT8 quantized models
2. Ensure CPU governor is set to performance:
   ```bash
   echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
   ```
3. Compile with optimizations:
   ```bash
   cmake .. -DCMAKE_BUILD_TYPE=Release
   ```

## Architecture

```
cpp/
├── CMakeLists.txt          # Build configuration
├── build.sh                # Automated build script
├── include/
│   ├── tokenizer.h         # Tokenizer interface
│   └── seq2seq_inference.h # Inference engine interface
├── src/
│   ├── tokenizer.cpp       # Tokenizer implementation
│   ├── seq2seq_inference.cpp # ONNX Runtime inference
│   └── main.cpp            # CLI application
└── README.md               # This file
```

## Dependencies

- **ONNX Runtime**: 1.16.3 (auto-downloaded)
- **CMake**: 3.15+ (build only)
- **GCC/Clang**: C++17 support
- **libstdc++**: Standard library

## License

Same as parent project (MIT License)

## Support

For issues specific to C++ deployment:
- Check ONNX Runtime docs: https://onnxruntime.ai/
- Open issue on GitHub with `[C++]` tag
