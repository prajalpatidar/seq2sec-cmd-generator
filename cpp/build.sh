#!/bin/bash
# Build script for C++ ONNX Runtime deployment

set -e

echo "=================================="
echo "Building seq2sec C++ Application"
echo "=================================="

# Check for ONNX Runtime
if [ ! -d "../onnxruntime" ]; then
    echo "ONNX Runtime not found. Downloading..."
    
    # Detect architecture
    ARCH=$(uname -m)
    if [ "$ARCH" == "x86_64" ]; then
        ONNX_URL="https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz"
    elif [ "$ARCH" == "aarch64" ] || [ "$ARCH" == "arm64" ]; then
        ONNX_URL="https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-aarch64-1.16.3.tgz"
    else
        echo "Unsupported architecture: $ARCH"
        exit 1
    fi
    
    wget $ONNX_URL -O onnxruntime.tgz
    tar -xzf onnxruntime.tgz
    mv onnxruntime-* ../onnxruntime
    rm onnxruntime.tgz
    echo "ONNX Runtime downloaded and extracted"
fi

# Export vocabularies from Python tokenizers
echo "Exporting vocabularies..."
cd ..
python3 scripts/export_vocab_cpp.py
cd cpp

# Create build directory
mkdir -p build
cd build

# Configure with CMake
echo "Configuring with CMake..."
cmake ..

# Build
echo "Building..."
make -j$(nproc)

echo ""
echo "=================================="
echo "Build complete!"
echo "=================================="
echo ""
echo "Executable: build/cmd_generator"
echo "Library: build/libseq2sec_lib.so"
echo ""
echo "Usage:"
echo "  ./build/cmd_generator generate \"show network interfaces\""
echo "  ./build/cmd_generator interactive"
echo ""
