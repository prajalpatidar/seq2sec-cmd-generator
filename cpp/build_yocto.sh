#!/bin/bash
# Cross-compilation build script for Yocto environment
# Target: core2 x86_64bit

set -e

echo "=============================================================================="
echo "                 Yocto Cross-Compilation Build Script"
echo "=============================================================================="

# Check if Yocto SDK environment is sourced
if [ -z "$CC" ] || [ -z "$CXX" ]; then
    if [ -f "/opt/poky/4.0.31/environment-setup-core2-64-poky-linux" ]; then
        echo "Sourcing Yocto SDK environment..."
        source /opt/poky/4.0.31/environment-setup-core2-64-poky-linux
    else
        echo "ERROR: Yocto SDK environment not set up!"
        echo ""
        echo "Please source your Yocto SDK environment script first:"
        echo "  source /opt/poky/4.0.31/environment-setup-core2-64-poky-linux"
        echo ""
        echo "Expected environment variables:"
        echo "  - CC (C compiler)"
        echo "  - CXX (C++ compiler)"
        echo "  - SDKTARGETSYSROOT (sysroot path)"
        echo "  - CMAKE_TOOLCHAIN_FILE (optional)"
        echo ""
        exit 1
    fi
fi

echo "Yocto SDK Environment Detected:"
echo "  CC: $CC"
echo "  CXX: $CXX"
echo "  Target Sysroot: ${SDKTARGETSYSROOT:-Not set}"
echo "  CMAKE_TOOLCHAIN_FILE: ${CMAKE_TOOLCHAIN_FILE:-Not set}"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BUILD_DIR="${SCRIPT_DIR}/build_yocto"
ONNX_DIR="${SCRIPT_DIR}/../onnxruntime"

echo "Build Configuration:"
echo "  Source Directory: ${SCRIPT_DIR}"
echo "  Build Directory: ${BUILD_DIR}"
echo "  ONNX Runtime: ${ONNX_DIR}"
echo ""

# Step 1: Download ONNX Runtime if not present
if [ ! -d "$ONNX_DIR" ]; then
    echo "=============================================================================="
    echo "Step 1: Downloading ONNX Runtime for x86_64"
    echo "=============================================================================="
    
    ONNX_VERSION="1.16.3"
    ONNX_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/onnxruntime-linux-x64-${ONNX_VERSION}.tgz"
    
    echo "Downloading: ${ONNX_URL}"
    cd "${SCRIPT_DIR}/.."
    curl -L -o onnxruntime.tgz "$ONNX_URL"
    tar -xzf onnxruntime.tgz
    mv "onnxruntime-linux-x64-${ONNX_VERSION}" onnxruntime
    rm onnxruntime.tgz
    echo "ONNX Runtime downloaded and extracted!"
    echo ""
else
    echo "ONNX Runtime already present: ${ONNX_DIR}"
    echo ""
fi

# Step 2: Export vocabularies from Python tokenizers
echo "=============================================================================="
echo "Step 2: Exporting Vocabularies"
echo "=============================================================================="

MODELS_DIR="${SCRIPT_DIR}/../models"
INPUT_VOCAB="${MODELS_DIR}/checkpoints/input_vocab.txt"
OUTPUT_VOCAB="${MODELS_DIR}/checkpoints/output_vocab.txt"

if [ ! -f "$INPUT_VOCAB" ] || [ ! -f "$OUTPUT_VOCAB" ]; then
    echo "Exporting vocabularies from Python tokenizers..."
    cd "${SCRIPT_DIR}/.."
    python3 scripts/export_vocab_cpp.py
    echo "Vocabularies exported!"
else
    echo "Vocabularies already exist"
fi
echo ""

# Step 3: Clean and create build directory
echo "=============================================================================="
echo "Step 3: Preparing Build Directory"
echo "=============================================================================="

if [ -d "$BUILD_DIR" ]; then
    echo "Cleaning existing build directory..."
    rm -rf "${BUILD_DIR}"/*
else
    mkdir -p "$BUILD_DIR"
fi
echo "Build directory ready: ${BUILD_DIR}"
echo ""

# Step 4: Configure with CMake (Yocto cross-compilation)
echo "=============================================================================="
echo "Step 4: CMake Configuration (Yocto Cross-Compile)"
echo "=============================================================================="

cd "$BUILD_DIR"

# Build CMake arguments
CMAKE_ARGS=(
    -DCMAKE_BUILD_TYPE=Release
    -DONNXRUNTIME_DIR="${ONNX_DIR}"
)

# Add toolchain file if provided by Yocto SDK
if [ -n "$CMAKE_TOOLCHAIN_FILE" ]; then
    echo "Using Yocto CMake toolchain: $CMAKE_TOOLCHAIN_FILE"
    CMAKE_ARGS+=(-DCMAKE_TOOLCHAIN_FILE="$CMAKE_TOOLCHAIN_FILE")
else
    echo "No CMAKE_TOOLCHAIN_FILE set, using environment CC/CXX"
    
    # Extract compiler command and flags from CC/CXX
    CC_CMD=$(echo "$CC" | awk '{print $1}')
    CXX_CMD=$(echo "$CXX" | awk '{print $1}')
    
    # Extract flags (everything after the first word)
    CC_FLAGS=$(echo "$CC" | cut -d' ' -f2-)
    CXX_FLAGS=$(echo "$CXX" | cut -d' ' -f2-)

    CMAKE_ARGS+=(
        -DCMAKE_C_COMPILER="$CC_CMD"
        -DCMAKE_CXX_COMPILER="$CXX_CMD"
        -DCMAKE_C_FLAGS="$CC_FLAGS"
        -DCMAKE_CXX_FLAGS="$CXX_FLAGS"
    )
fi

# Add sysroot if available
if [ -n "$SDKTARGETSYSROOT" ]; then
    echo "Using sysroot: $SDKTARGETSYSROOT"
    CMAKE_ARGS+=(
        -DCMAKE_SYSROOT="$SDKTARGETSYSROOT"
        -DCMAKE_FIND_ROOT_PATH="$SDKTARGETSYSROOT"
    )
fi

# Set target architecture
CMAKE_ARGS+=(
    -DCMAKE_SYSTEM_NAME=Linux
    -DCMAKE_SYSTEM_PROCESSOR=x86_64
)

echo "CMake Arguments:"
for arg in "${CMAKE_ARGS[@]}"; do
    echo "  $arg"
done
echo ""

cmake "${CMAKE_ARGS[@]}" ..

echo "CMake configuration complete!"
echo ""

# Step 5: Build
echo "=============================================================================="
echo "Step 5: Building (Cross-Compile)"
echo "=============================================================================="

make -j$(nproc)

echo ""
echo "=============================================================================="
echo "                          Build Complete!"
echo "=============================================================================="
echo ""
echo "Built for Yocto target: core2 x86_64"
echo ""
echo "Artifacts:"
echo "  Executable: ${BUILD_DIR}/cmd_generator"
echo "  Library:    ${BUILD_DIR}/libseq2sec_lib.so"
echo ""

# Display binary info
if command -v file &> /dev/null; then
    echo "Binary Information:"
    file "${BUILD_DIR}/cmd_generator"
    echo ""
fi

if command -v ${CROSS_COMPILE}readelf &> /dev/null; then
    echo "ELF Header:"
    ${CROSS_COMPILE}readelf -h "${BUILD_DIR}/cmd_generator" | head -15
    echo ""
fi

echo "To create deployment package:"
echo "  1. Copy to target device:"
echo "     scp ${BUILD_DIR}/cmd_generator root@target:/opt/seq2seq/"
echo "     scp ${BUILD_DIR}/libseq2sec_lib.so root@target:/opt/seq2seq/"
echo "     scp -r ${ONNX_DIR} root@target:/opt/seq2seq/"
echo "     scp -r ${MODELS_DIR}/onnx root@target:/opt/seq2seq/models/"
echo "     scp -r ${MODELS_DIR}/checkpoints/*.txt root@target:/opt/seq2seq/models/checkpoints/"
echo ""
echo "  2. On target device, test:"
echo "     cd /opt/seq2seq"
echo "     export LD_LIBRARY_PATH=./onnxruntime/lib:\$LD_LIBRARY_PATH"
echo "     ./cmd_generator generate \"show network interfaces\""
echo ""
echo "=============================================================================="
