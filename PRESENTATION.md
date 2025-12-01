# Seq2Sec Command Generator
## Neural Command Generation for Embedded Systems

**Author:** Prajal Patidar
**Date:** November 30, 2025

---

# 1. Project Overview

### Problem Statement
*   **Goal:** Translate natural language instructions into specific Linux/RDKB terminal commands.
*   **Example:** "show wifi ssid" $\rightarrow$ `dmcli eRT getv Device.WiFi.SSID.1.SSID`
*   **Constraint:** Must run on resource-constrained embedded devices (e.g., RDK-B routers with Atom processors, ~500MB RAM).

### The Solution
*   **Model:** A lightweight Sequence-to-Sequence (Seq2Seq) neural network.
*   **Deployment:** Optimized C++ application using ONNX Runtime.
*   **Key Feature:** Runs entirely offline on the edge (no cloud dependency).

---

# 2. Model Architecture: Seq2Seq with Attention

We utilize an **Encoder-Decoder** architecture, the standard for translation tasks.

### High-Level Data Flow
1.  **Input Sequence:** "show network interfaces"
2.  **Encoder:** Processes input into a fixed-size "Context Vector".
3.  **Decoder:** Generates output tokens one by one based on context.
4.  **Output Sequence:** `ifconfig`

### Why this architecture?
*   **Flexibility:** Handles variable input and output lengths.
*   **Efficiency:** GRU (Gated Recurrent Unit) is computationally cheaper than LSTM or Transformer, ideal for embedded.

---

# 3. Deep Dive: The Encoder

The Encoder understands the input intent.

### Components
*   **Embedding Layer (256-dim):** Converts words (indices) into dense vectors capturing semantic meaning.
*   **GRU Layers (2 layers, 512-dim):**
    *   Processes the sequence step-by-step.
    *   Updates its **Hidden State** at each step.
    *   The final hidden state summarizes the *entire* input sentence.

### Code Insight (`models/seq2seq.py`)
```python
self.embedding = nn.Embedding(vocab_size, embedding_dim)
self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers, batch_first=True)
```

---

# 4. Deep Dive: The Decoder & Attention

The Decoder generates the command, focusing on relevant parts of the input.

### The Attention Mechanism
*   **Problem:** A single context vector can be a bottleneck for long sentences.
*   **Solution:** Attention allows the decoder to "look back" at specific encoder states at every generation step.
*   **Type:** Dot-product attention. It calculates a "relevance score" between the current decoder state and all encoder outputs.

### Generation Loop
1.  Input: Previous token (starts with `<START>`).
2.  Calculate Attention weights.
3.  Combine Input + Context Vector.
4.  GRU Step $\rightarrow$ New Hidden State.
5.  Linear Layer $\rightarrow$ Probability distribution over Output Vocabulary.
6.  Select highest probability token.

---

# 5. Training Methodology

### Dataset
*   **Source:** `data/commands-dataset.json`
*   **Size:** ~600 samples (Linux + RDKB commands).
*   **Format:** Pairs of `{"input": "...", "output": "..."}`.

### Preprocessing
*   **Tokenization:** Word-level tokenization.
*   **Vocabulary:** Built dynamically. Special tokens: `<PAD>`, `<START>`, `<END>`, `<UNK>`.

### Training Loop (`scripts/train.py`)
*   **Loss Function:** `CrossEntropyLoss` (measures difference between predicted and actual tokens).
*   **Optimizer:** `Adam` (Adaptive Moment Estimation) - converges faster than SGD.
*   **Device:** Trained on NVIDIA GPU (CUDA), deployed on CPU.

---

# 6. Key Concept: Teacher Forcing

A technique to stabilize and speed up training.

### How it works
*   **Standard RNN:** Output of step $t$ is input for step $t+1$. If step $t$ is wrong, the whole sequence drifts.
*   **Teacher Forcing:** During training, we sometimes feed the **actual ground truth** token as input to step $t+1$, regardless of what the model predicted.

### Implementation
*   **Ratio:** 0.3 (30% of the time, we "correct" the model during training).
*   **Benefit:** Helps the model learn the correct grammar and structure faster early in training.

---

# 7. Optimization for Embedded (The Pipeline)

Training is only half the battle. We need to run on a dual-core Atom processor.

### 1. Export to ONNX
*   Converts PyTorch dynamic computation graph to a static **Open Neural Network Exchange** graph.
*   Decouples the model from Python/PyTorch dependencies.

### 2. Quantization (INT8)
*   **Float32 (Standard):** 4 bytes per weight. High precision, high memory.
*   **INT8 (Quantized):** 1 byte per weight.
*   **Result:** ~4x reduction in model size (25MB $\rightarrow$ ~6MB effective weight storage) and faster arithmetic on CPU.
*   **Trade-off:** Slight loss in accuracy (<2%), acceptable for command generation.

---

# 8. Deployment Architecture

We moved away from Python for the target device to minimize overhead.

### C++ Inference Engine (`cpp/src/`)
*   **Runtime:** Microsoft ONNX Runtime (C++ API).
*   **Dependencies:** Minimal (Standard C++ libraries + ONNX Runtime shared lib).
*   **Cross-Compilation:** Built using Yocto SDK (`core2-64`) to ensure binary compatibility with the target RDK-B device.

### Runtime Flexibility
*   **Float32 Mode:** `--model-type float32` (Max accuracy).
*   **INT8 Mode:** `--model-type int8` (Max speed/efficiency).

---

# 9. Validation Results

### Target Hardware
*   **Device:** RDK-B Embedded Device.
*   **Processor:** Intel Atom (x86_64 Dual Core).
*   **RAM:** ~500MB available.

### Performance Metrics
*   **Accuracy:** High accuracy on validation set for both FP32 and INT8.
*   **Inference Speed:** ~60-120ms per command (Real-time feel).
*   **Memory Footprint:** ~45-55 MB RAM (Fits comfortably alongside other system services).

---

# 10. Future Roadmap

### 1. Data Expansion
*   Current dataset (600 samples) is small.
*   Goal: 2000+ samples to cover more edge cases and argument variations.

### 2. Advanced Quantization
*   Explore INT4 quantization for even smaller footprint.

### 3. Beam Search
*   Currently using "Greedy Decoding" (picking best token at each step).
*   Beam Search explores multiple future paths to find the globally optimal sequence.

### 4. RDK-C Support
*   Extend dataset to support Camera (RDK-C) specific commands.

---

# Thank You
## Questions?

**Repository:** `github.com/prajalpatidar/seq2sec-cmd-generator`
