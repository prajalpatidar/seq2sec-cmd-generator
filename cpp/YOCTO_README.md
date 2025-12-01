# Yocto Cross-Compilation Build Directory

This directory is for cross-compiling the seq2seq C++ application for Yocto Project targets (core2 x86_64).

## Prerequisites

### 1. Yocto SDK Setup

You need to have a Yocto SDK environment set up for your target platform. Typically this is done by:

```bash
# Download and install Yocto SDK for your target
# Example for core2-64 target:
wget http://downloads.yoctoproject.org/releases/yocto/yocto-x.x/sdk/...
chmod +x poky-glibc-x86_64-core-image-minimal-core2-64-toolchain-x.x.sh
./poky-glibc-x86_64-core-image-minimal-core2-64-toolchain-x.x.sh

# Or use your custom Yocto build SDK:
# bitbake core-image-minimal -c populate_sdk
```

### 2. Source the SDK Environment

Before building, you must source the Yocto SDK environment:

```bash
# Replace with your actual SDK path
source /opt/poky/4.0.31/environment-setup-core2-64-poky-linux
```

This sets up essential environment variables:
- `CC` - C compiler (e.g., x86_64-poky-linux-gcc)
- `CXX` - C++ compiler (e.g., x86_64-poky-linux-g++)
- `SDKTARGETSYSROOT` - Sysroot path
- `CMAKE_TOOLCHAIN_FILE` - CMake toolchain (if provided)
- `CROSS_COMPILE` - Cross-compilation prefix

## Quick Build

### Option 1: Using build_yocto.sh (Recommended)

```bash
# 1. Source Yocto SDK environment
source /opt/poky/4.0.31/environment-setup-core2-64-poky-linux

# 2. Run build script
cd /path/to/seq2sec-cmd-generator/cpp
./build_yocto.sh
```

The script will:
- Verify Yocto SDK environment is set up
- Download ONNX Runtime (if needed)
- Export vocabularies from Python tokenizers
- Configure CMake with Yocto toolchain
- Build the application
- Display build artifacts and deployment instructions

### Option 2: Manual Build

```bash
# 1. Source Yocto SDK
source /opt/poky/4.0.31/environment-setup-core2-64-poky-linux

# 2. Export vocabularies (if not done)
cd /path/to/seq2sec-cmd-generator
python3 scripts/export_vocab_cpp.py

# 3. Download ONNX Runtime (if not done)
cd /path/to/seq2sec-cmd-generator
curl -L -o onnxruntime.tgz https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz
tar -xzf onnxruntime.tgz
mv onnxruntime-linux-x64-1.16.3 onnxruntime
rm onnxruntime.tgz

# 4. Configure CMake
cd cpp/build_yocto
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_TOOLCHAIN_FILE=$CMAKE_TOOLCHAIN_FILE \
      -DONNXRUNTIME_DIR=../../onnxruntime \
      ..

# 5. Build
make -j$(nproc)
```

## Build Output

After successful build, you'll find:

```
build_yocto/
├── cmd_generator              # Main executable
├── libseq2sec_lib.so          # Shared library
├── CMakeFiles/                # CMake build files
└── CMakeCache.txt             # CMake cache
```

## Verify Cross-Compilation

Check the binary is built for the correct target:

```bash
# Check file type
file build_yocto/cmd_generator
# Should show: ELF 64-bit LSB executable, x86-64, ...

# Check ELF header
readelf -h build_yocto/cmd_generator

# Check dynamic dependencies
readelf -d build_yocto/cmd_generator

# Check required libraries
ldd build_yocto/cmd_generator  # Note: This might not work on host
```

## Deployment to Target Device

### Step 1: Create Deployment Package

```bash
# Option A: Manual file copy
mkdir -p deploy_yocto/models/{onnx,checkpoints}

cp build_yocto/cmd_generator deploy_yocto/
cp build_yocto/libseq2sec_lib.so deploy_yocto/
cp -r ../onnxruntime deploy_yocto/
cp ../models/onnx/encoder_quantized.onnx deploy_yocto/models/onnx/
cp ../models/onnx/decoder_quantized.onnx deploy_yocto/models/onnx/
cp ../models/checkpoints/input_vocab.txt deploy_yocto/models/checkpoints/
cp ../models/checkpoints/output_vocab.txt deploy_yocto/models/checkpoints/

# Option B: Use tar archive
tar -czf deploy_yocto.tar.gz deploy_yocto/
```

### Step 2: Transfer to Target

```bash
# Via SCP
scp deploy_yocto.tar.gz root@target-device:/tmp/

# Via USB/SD Card
# Copy deploy_yocto.tar.gz to removable media
```

### Step 3: Extract and Test on Target

```bash
# SSH to target device
ssh root@target-device

# Extract
cd /opt
tar -xzf /tmp/deploy_yocto.tar.gz
cd deploy_yocto

# Set library path
export LD_LIBRARY_PATH=./onnxruntime/lib:$LD_LIBRARY_PATH

# Test
./cmd_generator generate "show network interfaces"
```

## Expected Output on Target

```
Initializing seq2seq command generator...
Loaded encoder model: models/onnx/encoder_quantized.onnx
Loaded decoder model: models/onnx/decoder_quantized.onnx
Loaded vocabulary: 326 tokens
Loaded vocabulary: 412 tokens
Initialization complete!

Input: show network interfaces
Command: ifconfig
```

## Troubleshooting

### Error: "Yocto SDK environment not set up"

**Solution**: Source your Yocto SDK environment script:
```bash
source /opt/poky/4.0.31/environment-setup-core2-64-poky-linux
./build_yocto.sh
```

### Error: "GLIBC version mismatch"

**Solution**: Ensure your Yocto SDK version matches the target device's libc version. Rebuild SDK if needed.

### Error: "libonnxruntime.so.1.16.3: cannot open shared object file"

**Solution**: Set `LD_LIBRARY_PATH` on target device:
```bash
export LD_LIBRARY_PATH=/opt/deploy_yocto/onnxruntime/lib:$LD_LIBRARY_PATH
```
