# seq2sec-cmd-generator

A lightweight AI model for embedded systems that translates natural language instructions into Linux commands. This project provides a simple CLI tool that maps phrases like "show network interfaces" → "ifconfig".

## Overview

This project implements a sequence-to-sequence (seq2seq) model optimized for embedded systems with limited resources. The model is trained in a high-performance environment and deployed on resource-constrained devices using ONNX Runtime with quantization.

### Training Environment
- **Platform**: WSL-Ubuntu24
- **Framework**: PyTorch 2.9.1 + CUDA 13.0
- **Hardware**: Core Ultra 9 CPU, NVIDIA RTX5060 GPU

### Inference Environment
- **Platform**: Embedded device
- **Hardware**: Intel Atom dual-core CPU, 500MB RAM
- **Runtime**: ONNX Runtime with quantization

## Features

- 🚀 **Lightweight Architecture**: Simple GRU-based seq2seq model with attention
- 📦 **Small Model Size**: Optimized for embedded systems (~1-5MB)
- ⚡ **Fast Inference**: ONNX Runtime with INT8 quantization
- 🎯 **Simple CLI**: Easy-to-use command-line interface
- 📊 **Extensible Dataset**: Easy to add more command mappings

## Installation

### Prerequisites

```bash
# Python 3.8 or higher
python --version

# For training (with GPU support)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Project Structure

```
seq2sec-cmd-generator/
├── data/
│   └── commands_dataset.csv      # Training data
├── models/
│   ├── seq2seq.py                # Model architecture
│   ├── checkpoints/              # Saved models and tokenizers
│   └── onnx/                     # Exported ONNX models
├── scripts/
│   ├── data_utils.py             # Data preprocessing utilities
│   ├── train.py                  # Training script
│   └── export_onnx.py            # ONNX export script
├── cli/
│   └── cmd_generator.py          # CLI tool
├── config/
│   └── train_config.yaml         # Training configuration
└── tests/                        # Unit tests
```

## Usage

### 1. Training the Model

Train the model on your dataset:

```bash
python scripts/train.py
```

The training script will:
- Load data from `data/commands_dataset.csv`
- Split into train/validation sets (80/20)
- Train the seq2seq model with attention
- Save the best model to `models/checkpoints/best_model.pth`
- Save tokenizers to `models/checkpoints/`

Training outputs:
```
Using device: cuda
Training samples: 96
Validation samples: 24
Input vocabulary size: 215
Output vocabulary size: 156
Model parameters: 56,472

Epoch 1/50
Training: 100%|██████████| 6/6 [00:02<00:00,  2.34it/s]
Evaluating: 100%|██████████| 2/2 [00:00<00:00, 12.45it/s]
Train Loss: 3.8542
Val Loss: 3.6234
Saved best model!
...
```

### 2. Export to ONNX

Export the trained model to ONNX format with quantization:

```bash
python scripts/export_onnx.py
```

This will create:
- `models/onnx/encoder.onnx` - Standard encoder
- `models/onnx/decoder.onnx` - Standard decoder
- `models/onnx/encoder_quantized.onnx` - Quantized encoder (for embedded)
- `models/onnx/decoder_quantized.onnx` - Quantized decoder (for embedded)

### 3. Generate Commands

Use the CLI tool to generate Linux commands:

```bash
# Single command generation
python cli/cmd_generator.py generate "show network interfaces"

# Using quantized models (for embedded deployment)
python cli/cmd_generator.py generate "show network interfaces" --quantized

# Interactive mode
python cli/cmd_generator.py interactive
```

Example output:
```
Input: show network interfaces
Generated Command: ifconfig

Input: list all files
Generated Command: ls -la

Input: show disk usage
Generated Command: df -h
```

## Dataset Format

The dataset is stored in CSV format with two columns:

```csv
input,output
show network interfaces,ifconfig
list all files,ls -la
show disk usage,df -h
...
```

To add more commands, simply edit `data/commands_dataset.csv` and retrain the model.

## Model Architecture

The model uses a simple encoder-decoder architecture with attention:

```
Encoder (GRU)
  ↓
Attention Mechanism
  ↓
Decoder (GRU)
  ↓
Linear Layer → Command Tokens
```

**Key Design Choices for Embedded Systems:**
- Single-layer GRU (instead of LSTM) for reduced memory
- Small embedding dimension (64) and hidden dimension (128)
- Word-level tokenization with limited vocabulary (1000 words)
- Unidirectional encoding for faster inference
- Simple dot-product attention

## Configuration

Edit `config/train_config.yaml` to adjust hyperparameters:

```yaml
# Model hyperparameters
embedding_dim: 64        # Embedding size
hidden_dim: 128          # Hidden layer size
num_layers: 1            # Number of RNN layers

# Training hyperparameters
batch_size: 16
num_epochs: 50
learning_rate: 0.001
teacher_forcing_ratio: 0.5

# Data parameters
max_vocab_size: 1000     # Maximum vocabulary size
train_split: 0.8         # Train/validation split
```

## Performance

**Model Size:**
- PyTorch model: ~1-2 MB
- ONNX model: ~1-2 MB
- Quantized ONNX model: ~500KB - 1MB

**Inference Speed (on Intel Atom):**
- Standard ONNX: ~50-100ms per command
- Quantized ONNX: ~20-50ms per command

**Memory Usage:**
- Training: ~2-4GB GPU memory
- Inference: ~50-100MB RAM

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black .
flake8 .
```

## Deployment to Embedded Systems

For deploying to embedded devices:

1. **Train on high-performance machine**:
   ```bash
   python scripts/train.py
   ```

2. **Export to ONNX with quantization**:
   ```bash
   python scripts/export_onnx.py
   ```

3. **Copy to embedded device**:
   ```bash
   # Copy only the necessary files
   models/onnx/encoder_quantized.onnx
   models/onnx/decoder_quantized.onnx
   models/checkpoints/input_tokenizer.pkl
   models/checkpoints/output_tokenizer.pkl
   cli/cmd_generator.py
   scripts/data_utils.py
   ```

4. **Install ONNX Runtime on embedded device**:
   ```bash
   pip install onnxruntime
   ```

5. **Run inference**:
   ```bash
   python cli/cmd_generator.py generate "your instruction" --quantized
   ```

## Extending the Dataset

To add more command mappings:

1. Edit `data/commands_dataset.csv`
2. Add new rows with format: `input,output`
3. Retrain the model: `python scripts/train.py`
4. Re-export to ONNX: `python scripts/export_onnx.py`

## Troubleshooting

**Issue**: CUDA out of memory during training
- **Solution**: Reduce `batch_size` in `config/train_config.yaml`

**Issue**: Model not converging
- **Solution**: Increase `num_epochs` or adjust `learning_rate`

**Issue**: Generated commands are incorrect
- **Solution**: Add more training data or increase model capacity

**Issue**: Models not found when running CLI
- **Solution**: Train the model first with `python scripts/train.py`, then export with `python scripts/export_onnx.py`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Built with PyTorch and ONNX Runtime
- Inspired by sequence-to-sequence models for neural machine translation
