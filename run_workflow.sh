#!/bin/bash

# Complete workflow demonstration script for seq2seq command generator
# This script demonstrates the entire pipeline from training to inference

set -e  # Exit on error

echo "=============================================="
echo "Seq2Seq Command Generator - Complete Workflow"
echo "=============================================="
echo ""

# Step 1: Train the model
echo "Step 1: Training the model..."
echo "This will take 5-10 minutes depending on your hardware"
echo ""
python scripts/train.py \
  --data_path data/commands_dataset.json \
  --embedding_dim 128 \
  --hidden_dim 256 \
  --num_layers 1 \
  --dropout 0.1 \
  --batch_size 32 \
  --epochs 50 \
  --lr 0.001 \
  --output_dir models

echo ""
echo "✓ Training complete!"
echo ""

# Step 2: Export to ONNX
echo "Step 2: Exporting model to ONNX format..."
echo ""
python scripts/export_onnx.py \
  --model_path models/best_model.pt \
  --output_dir models \
  --device cpu

echo ""
echo "✓ ONNX export complete!"
echo ""

# Step 3: Quantize models
echo "Step 3: Quantizing ONNX models (INT8)..."
echo ""
python scripts/quantize_onnx.py \
  --encoder_path models/encoder.onnx \
  --decoder_path models/decoder.onnx \
  --output_dir models

echo ""
echo "✓ Quantization complete!"
echo ""

# Step 4: Test inference
echo "Step 4: Testing inference..."
echo ""
echo "Testing with sample queries:"
echo ""

queries=(
  "show network interfaces"
  "check disk usage"
  "list running processes"
  "display memory usage"
  "show system uptime"
)

for query in "${queries[@]}"; do
  echo "Input: $query"
  python scripts/inference.py \
    --use_quantized \
    --input "$query"
  echo ""
done

echo "=============================================="
echo "Workflow Complete!"
echo "=============================================="
echo ""
echo "All steps completed successfully:"
echo "  ✓ Model trained and saved"
echo "  ✓ Exported to ONNX format"
echo "  ✓ Quantized for deployment"
echo "  ✓ Inference tested"
echo ""
echo "Next steps:"
echo "  - Try interactive mode: python scripts/inference.py --use_quantized"
echo "  - Deploy to embedded device"
echo "  - Add more training data"
echo ""
echo "Models are ready for deployment!"
echo "=============================================="
