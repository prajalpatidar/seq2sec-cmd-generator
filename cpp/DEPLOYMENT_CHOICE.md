# C++ vs Python Deployment - Quick Reference

## When to Use C++

✅ **Use C++ if your device:**
- Has NO Python installed
- Has limited storage (<100MB available)
- Needs fast startup time (<1 second)
- Requires minimal memory footprint
- Is a production RDK-B device
- Needs a single standalone binary

## When to Use Python

✅ **Use Python if:**
- Device has Python 3.8+ already installed
- You need to quickly test and iterate
- You want to easily modify inference logic
- Development or testing environment

## Quick Comparison

| Aspect | Python | C++ |
|--------|--------|-----|
| **Installation** | `pip install onnxruntime` | Build once, copy binary |
| **Size** | ~50 MB | ~26 MB total |
| **Memory** | 50-60 MB | 45-55 MB |
| **Startup** | 1-2 seconds | 0.1-0.2 seconds |
| **Speed** | 60-120ms/cmd | 60-120ms/cmd (same) |
| **Dependencies** | Python 3.8+ required | Only ONNX Runtime lib |

## Getting Started

### C++ Deployment (3 steps)

```bash
# 1. Build
cd cpp && ./build.sh

# 2. Test
./build/cmd_generator generate "show wifi ssid"

# 3. Deploy
tar -czf deploy.tar.gz build/cmd_generator ../models/onnx/*_quantized.onnx ../models/checkpoints/*_vocab.txt
scp deploy.tar.gz root@device:/opt/
```

### Python Deployment (3 steps)

```bash
# 1. Install
pip install onnxruntime

# 2. Test
python cli/cmd_generator.py generate "show wifi ssid"

# 3. Deploy
scp -r models/ cli/ scripts/ root@device:/opt/
```

## Performance

Both implementations use the same ONNX models and provide identical inference quality.

**Inference Time:** 60-120ms per command (on RDK-B)  
**Accuracy:** Same (1.3453 validation loss)  
**Commands:** All 611 (Linux + RDKB)

## Recommendation

**For Production RDK-B Devices:** Use **C++ deployment**
- No Python dependency
- Smaller footprint
- Faster startup
- Production-ready

**For Development/Testing:** Use **Python deployment**
- Easy to modify
- Quick to iterate
- Good for prototyping

## Need Help?

- C++ Guide: `cpp/README.md`
- Python Guide: `docs/DEPLOYMENT.md`
- Main README: `README.md`
