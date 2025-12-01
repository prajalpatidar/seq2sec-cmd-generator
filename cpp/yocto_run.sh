#!/bin/bash
# Launcher script for seq2seq C++ command generator
# Usage: ./run.sh [mode] [arguments]

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Set library path
export LD_LIBRARY_PATH="${SCRIPT_DIR}/onnxruntime/lib:${LD_LIBRARY_PATH}"

# Change to script directory (models are in current directory)
cd "${SCRIPT_DIR}"

# Run the command generator
./cmd_generator "$@"
