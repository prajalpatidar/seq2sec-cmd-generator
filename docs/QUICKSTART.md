# Quick Start Guide

This guide will help you get started with the seq2sec-cmd-generator in minutes. Learn how to train a seq2seq model that translates natural language to Linux/RDKB commands.

## Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for training

## Step-by-Step Setup

### 1. Clone the Repository

```bash
git clone https://github.com/prajalpatidar/seq2sec-cmd-generator.git
cd seq2sec-cmd-generator
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

For training with GPU support:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Train the Model

```bash
python scripts/train.py
```

Expected output:
```
Using device: cuda
Dataset loaded: 611 samples
Training samples: 519
Validation samples: 92
Input vocabulary size: 326
Output vocabulary size: 412
Model parameters: 6,705,053

Epoch 1/100
Training: 100%|██████████| 33/33 [00:03<00:00,  9.45it/s]
Evaluating: 100%|██████████| 6/6 [00:00<00:00, 15.23it/s]
Train Loss: 4.2134, Val Loss: 4.0521
Saved best model!

...

Epoch 68/100
Train Loss: 0.9823, Val Loss: 1.3453
Training completed! Best validation loss: 1.3453
```

Training takes approximately 20-30 minutes on a modern GPU (RTX 5060) for 611 samples with 2-layer GRU.

### 4. Export to ONNX

```bash
python scripts/export_onnx.py
```

This creates optimized ONNX models for deployment.

### 5. Test the CLI

```bash
# Single command
python cli/cmd_generator.py generate "show network interfaces"

# Interactive mode
python cli/cmd_generator.py interactive
```

## Example Commands

Try these natural language instructions:

**Linux Commands:**
```bash
python cli/cmd_generator.py generate "show network interfaces"
# Output: ifconfig

python cli/cmd_generator.py generate "list all files"
# Output: ls -la

python cli/cmd_generator.py generate "show disk usage"
# Output: df -h

python cli/cmd_generator.py generate "display running processes"
# Output: ps aux

python cli/cmd_generator.py generate "show memory usage"
# Output: free -h
```

**RDKB Commands:**
```bash
python cli/cmd_generator.py generate "show wifi ssid"
# Output: dmcli eRT getv Device.WiFi.SSID.1.SSID

python cli/cmd_generator.py generate "get device model name"
# Output: dmcli eRT getv Device.DeviceInfo.ModelName

python cli/cmd_generator.py generate "show lan ip address"
# Output: dmcli eRT getv Device.LAN.IPAddress

python cli/cmd_generator.py generate "get wifi password"
# Output: dmcli eRT getv Device.WiFi.AccessPoint.1.Security.KeyPassphrase
```

## Interactive Mode

For a more interactive experience:

```bash
python cli/cmd_generator.py interactive
```

Then type your instructions:
```
Enter instruction: show network interfaces
Command: ifconfig

Enter instruction: list all files
Command: ls -la

Enter instruction: quit
Goodbye!
```

## For Embedded Systems (RDK-B Devices)

To use INT8 quantized models optimized for embedded systems:

```bash
python cli/cmd_generator.py generate "your instruction" --quantized
```

Quantized models (INT8) are:
- 4.5% smaller in size (24.45 MB vs 25.59 MB FP32)
- Faster inference (40-80ms vs 50-100ms)
- Minimal accuracy loss (<2% difference)
- Recommended for RDK-B and resource-constrained devices

**Model Comparison:**
| Model Type | Size | Inference Time | Use Case |
|------------|------|----------------|----------|
| FP32 Standard | 25.59 MB | 50-100ms | Development/Testing |
| INT8 Quantized | 24.45 MB | 40-80ms | Production/Embedded |

## Adding Custom Commands

### Method 1: Edit Existing Dataset

Edit `data/commands-dataset.json`:
```json
[
  {
    "input": "show my IP address",
    "output": "hostname -I"
  },
  {
    "input": "check internet connection",
    "output": "ping -c 4 google.com"
  },
  {
    "input": "get wifi channel",
    "output": "dmcli eRT getv Device.WiFi.Radio.1.Channel"
  }
]
```

### Method 2: Use RDKB Command Generator

Add comprehensive RDKB commands automatically:
```bash
python scripts/add_rdkb_commands.py
```

This adds 111 RDKB dmcli commands for WiFi, LAN, WAN, firewall, and more.

### After Adding Commands

1. Retrain: `python scripts/train.py`
2. Re-export: `python scripts/export_onnx.py`
3. Test: `python cli/cmd_generator.py generate "your new command"`

## Troubleshooting

**Error**: "No module named 'torch'"
- **Solution**: Install PyTorch: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

**Error**: "CUDA out of memory"
- **Solution**: Reduce batch_size in `config/train_config.yaml` from 16 to 8

**Error**: "Models not found"
- **Solution**: Train and export models first:
  ```bash
  python scripts/train.py
  python scripts/export_onnx.py
  ```

**Error**: "Empty output or <END> token only"
- **Solution**: Model needs more training data. Current dataset has 611 samples, which should be sufficient. If you modified the dataset, ensure it has at least 200-300 samples.

**Issue**: "dmcli commands have spaces (device . wifi . ssid)"
- **Solution**: This is automatically fixed by post-processing in `scripts/data_utils.py`. Output should be `Device.WiFi.SSID.1.SSID`.

## Next Steps

- Review the full [README.md](../README.md) for detailed documentation
- Deploy to embedded devices: [DEPLOYMENT.md](DEPLOYMENT.md)
- Check the [Configuration Guide](../config/train_config.yaml) for hyperparameter tuning
- Run unit tests: `pytest tests/`
- Experiment with different model architectures (embedding_dim, hidden_dim, num_layers)
- Add more RDKB commands for your specific device
- Test ONNX Runtime: `python scripts/test_onnx_runtime.py`

## Model Specifications

**Current Model:**
- Architecture: 2-layer GRU encoder-decoder with attention
- Parameters: 6,705,053 (6.7M)
- Embedding: 256-dim, Hidden: 512-dim
- Vocabulary: Input=326, Output=412
- Dataset: 611 samples (Linux + RDKB)
- Best Validation Loss: 1.3453
- ONNX FP32: 25.59 MB
- ONNX INT8: 24.45 MB

## Support

For issues and questions, please open an issue on GitHub.

Happy command generation! 🚀
