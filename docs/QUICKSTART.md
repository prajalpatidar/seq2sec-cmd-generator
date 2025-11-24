# Quick Start Guide

This guide will help you get started with the seq2sec-cmd-generator in minutes.

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
Training samples: 96
Validation samples: 24
Input vocabulary size: 215
Output vocabulary size: 156
Model parameters: 56,472

Epoch 1/50
Training: 100%|██████████| 6/6 [00:02<00:00]
Train Loss: 3.8542
Val Loss: 3.6234
Saved best model!
```

Training takes approximately 5-10 minutes on a modern GPU.

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

```bash
python cli/cmd_generator.py generate "show network interfaces"
# Output: ifconfig

python cli/cmd_generator.py generate "list all files"
# Output: ls -la

python cli/cmd_generator.py generate "show disk usage"
# Output: df -h

python cli/cmd_generator.py generate "display running processes"
# Output: ps aux

python cli/cmd_generator.py generate "show current directory"
# Output: pwd
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

## For Embedded Systems

To use quantized models optimized for embedded systems:

```bash
python cli/cmd_generator.py generate "your instruction" --quantized
```

Quantized models are:
- ~50% smaller in size
- ~2-3x faster inference
- Slightly lower accuracy (typically <5% difference)

## Adding Custom Commands

1. Edit `data/commands_dataset.csv`
2. Add new rows with format: `natural language instruction,linux command`
3. Retrain: `python scripts/train.py`
4. Re-export: `python scripts/export_onnx.py`

Example:
```csv
input,output
show my IP address,hostname -I
check internet connection,ping -c 4 google.com
```

## Troubleshooting

**Error**: "No module named 'torch'"
- **Solution**: Install PyTorch: `pip install torch`

**Error**: "CUDA out of memory"
- **Solution**: Reduce batch_size in `config/train_config.yaml`

**Error**: "Models not found"
- **Solution**: Train and export models first

## Next Steps

- Review the full [README.md](README.md) for detailed documentation
- Check out the [Configuration Guide](config/train_config.yaml)
- Run unit tests: `pytest tests/`
- Experiment with different model architectures

## Support

For issues and questions, please open an issue on GitHub.

Happy command generation! 🚀
