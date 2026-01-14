# Jetson Orin Nano Webcam Demo Deployment Guide

Complete guide to deploy the webcam captioning demo to your Jetson Orin Nano.

## Overview

This demo runs live image captioning on webcam input using:
- **EMEET S600 4K Webcam** (or any USB webcam)
- **Discrete Diffusion Language Model** (56M params, trained on synthetic data)
- **CLIP ViT-B/32** for image encoding
- **ONNX Runtime** for optimized inference

**Expected Performance on Jetson:**
- Total inference: ~2-5 seconds per caption (25 diffusion steps)
- CLIP features: ~0.3-0.5s
- Caption generation: ~1.5-4s

## Prerequisites on Jetson

### 1. JetPack Installation
Ensure you have JetPack 6.0+ installed:
```bash
cat /etc/nv_tegra_release  # Should show R36.x or higher
```

### 2. Python Environment
```bash
# Check Python version (should be 3.8+)
python3 --version

# Install pip if needed
sudo apt-get update
sudo apt-get install python3-pip
```

### 3. PyTorch Installation
Install PyTorch for Jetson from NVIDIA:
```bash
# Download PyTorch wheel from:
# https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048

# Example for JetPack 6.0 (check forum for latest version)
wget https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.1.0a0+41361538.nv23.06-cp310-cp310-linux_aarch64.whl

# Install
pip3 install torch-*.whl
```

### 4. Required Dependencies
```bash
# Install OpenCV with CUDA support (should be pre-installed with JetPack)
sudo apt-get install python3-opencv

# Install Python packages
pip3 install transformers tokenizers onnxruntime
```

## File Transfer from Desktop to Jetson

You need to transfer these files from your desktop to Jetson:

### Required Files (Total: ~1.5MB)
```
models/caption_poc.onnx              # 798KB - ONNX model
models/caption_poc_config.json       # 141B  - Model config
data_full/tokenizer.json             # ~700KB - Tokenizer
jetson_webcam_demo.py                # 16KB  - Demo script
model.py                             # ~20KB - Model definitions (for PyTorch mode)
diffusion.py                         # ~15KB - Diffusion logic (for PyTorch mode)
```

### Transfer Methods

**Option 1: USB Drive**
```bash
# On desktop: copy files to USB drive
mkdir -p /media/usb/diffusion-demo
cp models/caption_poc.onnx /media/usb/diffusion-demo/
cp models/caption_poc_config.json /media/usb/diffusion-demo/
cp data_full/tokenizer.json /media/usb/diffusion-demo/
cp jetson_webcam_demo.py /media/usb/diffusion-demo/

# On Jetson: copy from USB
mkdir -p ~/diffusion-demo
cp /media/<usb-mount-point>/diffusion-demo/* ~/diffusion-demo/
```

**Option 2: SCP (if on same network)**
```bash
# On desktop: transfer via SCP
scp models/caption_poc.onnx jetson@<jetson-ip>:~/diffusion-demo/
scp models/caption_poc_config.json jetson@<jetson-ip>:~/diffusion-demo/
scp data_full/tokenizer.json jetson@<jetson-ip>:~/diffusion-demo/
scp jetson_webcam_demo.py jetson@<jetson-ip>:~/diffusion-demo/
```

**Option 3: Git Clone (if repository is public)**
```bash
# On Jetson: clone the repo
git clone <your-repo-url>
cd diffusion-llm

# Download models separately if not in git
# (ONNX files are typically not committed due to size)
```

## Setup on Jetson

### 1. Organize Files
```bash
cd ~/diffusion-demo
ls -lh

# Expected structure:
# caption_poc.onnx
# caption_poc_config.json
# tokenizer.json
# jetson_webcam_demo.py
```

### 2. Connect Webcam
```bash
# Plug in EMEET S600 webcam via USB

# Verify camera is detected
ls /dev/video*

# Should show /dev/video0 (or video1, video2, etc.)

# Test camera with v4l2
v4l2-ctl --list-devices
```

### 3. Verify CLIP Model Download
```bash
# On first run, CLIP model will download (~350MB)
# Ensure Jetson has internet access or pre-download on desktop:

# To pre-download CLIP on desktop:
python3 -c "from transformers import CLIPVisionModel, CLIPImageProcessor; \
            CLIPVisionModel.from_pretrained('openai/clip-vit-base-patch32'); \
            CLIPImageProcessor.from_pretrained('openai/clip-vit-base-patch32')"

# Transfer cache to Jetson:
# Desktop: ~/.cache/huggingface/hub/
# Jetson:  ~/.cache/huggingface/hub/
```

## Running the Demo

### Basic Usage (ONNX Backend - Recommended)
```bash
cd ~/diffusion-demo

python3 jetson_webcam_demo.py \
    --onnx caption_poc.onnx \
    --config caption_poc_config.json \
    --tokenizer tokenizer.json \
    --max_len 12 \
    --steps 25 \
    --temperature 0.8
```

### PyTorch Backend (Alternative)
If you transferred the full checkpoint instead:
```bash
python3 jetson_webcam_demo.py \
    --checkpoint checkpoints_caption_poc/final.pt \
    --tokenizer tokenizer.json \
    --max_len 12 \
    --steps 25
```

### Command-Line Options
```bash
--onnx PATH              # ONNX model path (for ONNX backend)
--config PATH            # Config JSON path (for ONNX backend)
--checkpoint PATH        # PyTorch checkpoint (for PyTorch backend)
--tokenizer PATH         # Tokenizer path (default: data_full/tokenizer.json)
--camera_id N            # Camera device ID (default: 0)
--max_len N              # Max caption length (default: 12)
--steps N                # Diffusion steps (default: 25, lower=faster but worse quality)
--temperature FLOAT      # Sampling temperature (default: 0.8, lower=more conservative)
--width N                # Camera width (default: 640)
--height N               # Camera height (default: 480)
```

## Usage Instructions

Once the demo is running:

1. **Wait for initialization** (~5-10 seconds for models to load)
2. **Camera preview window** will appear
3. **Press SPACEBAR** to capture current frame and generate caption
4. **Caption appears** overlaid at bottom of screen
5. **Inference time** shown in bottom-right corner
6. **Press 'q'** to quit

## Performance Tuning

### Faster Inference (Trade Quality)
```bash
# Reduce diffusion steps (25 → 10)
python3 jetson_webcam_demo.py --onnx caption_poc.onnx --config caption_poc_config.json \
    --tokenizer tokenizer.json --steps 10 --max_len 8

# Expected speedup: 2-3x faster (~1-2 seconds total)
```

### Better Quality (Slower)
```bash
# Increase steps (25 → 50)
python3 jetson_webcam_demo.py --onnx caption_poc.onnx --config caption_poc_config.json \
    --tokenizer tokenizer.json --steps 50 --max_len 12

# Expected slowdown: 2x slower (~4-8 seconds total)
```

### Memory Management
```bash
# If you get CUDA OOM errors, reduce batch operations
# The demo already uses batch_size=1, so try reducing --max_len:

python3 jetson_webcam_demo.py --onnx caption_poc.onnx --config caption_poc_config.json \
    --tokenizer tokenizer.json --max_len 8
```

## Troubleshooting

### Camera Not Found
```bash
# Check camera device
ls /dev/video*

# Try different camera ID
python3 jetson_webcam_demo.py --camera_id 1 ...
python3 jetson_webcam_demo.py --camera_id 2 ...

# Check camera permissions
sudo chmod 666 /dev/video0
```

### CUDA Errors
```bash
# Ensure CUDA is available
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'CUDA device: {torch.cuda.get_device_name(0)}')"

# If CUDA not available, demo will fall back to CPU (slower)
```

### Import Errors
```bash
# Missing transformers
pip3 install transformers

# Missing tokenizers
pip3 install tokenizers

# Missing cv2
sudo apt-get install python3-opencv

# Missing onnxruntime
pip3 install onnxruntime
```

### Performance Issues
```bash
# Check GPU utilization
sudo tegrastats

# Monitor while running demo - should see GPU usage increase during inference

# Ensure Jetson is in max performance mode
sudo nvpmodel -m 0  # Max performance mode
sudo jetson_clocks  # Lock clocks to max
```

### Model Quality Issues
**Note:** The POC model is trained on synthetic colored shapes, so:
- ✅ **Will work well**: Colored objects, simple scenes, size/color descriptions
- ❌ **Won't work well**: Complex real-world scenes, people, animals, etc.

To get better real-world captions, you need to train on COCO (happens tomorrow when GPU arrives!).

## Expected Behavior (Synthetic POC Model)

**Good Examples:**
- Point camera at **red mug** → "a red circle"
- Point camera at **blue notebook** → "a large blue square"
- Point camera at **yellow ball** → "a small yellow circle"

**Poor Examples:**
- Point camera at person → Random color/shape description
- Point camera at landscape → Random output

This is expected! Once the COCO model is trained, it will understand real scenes.

## Next Steps After Demo

1. **Benchmark Performance**: Note inference times for different `--steps` values
2. **Test Edge Cases**: Try different lighting, distances, object sizes
3. **Profile Bottlenecks**: Identify if CLIP or diffusion decoder is slower
4. **Plan Optimizations**: TensorRT compilation, FP16 precision, custom kernels

## Advanced: TensorRT Optimization (Optional)

For maximum performance, compile ONNX model to TensorRT:

```bash
# Requires NVIDIA TensorRT (included in JetPack)
pip3 install tensorrt

# Convert ONNX to TensorRT engine
trtexec --onnx=caption_poc.onnx \
        --saveEngine=caption_poc.trt \
        --fp16 \
        --workspace=4096

# Expected speedup: 1.5-2x over ONNX
```

Then modify `jetson_webcam_demo.py` to load `.trt` engine instead of `.onnx`.

## File Sizes Reference

| File | Size | Purpose |
|------|------|---------|
| caption_poc.onnx | 798KB | Decoder model weights |
| caption_poc_config.json | 141B | Model configuration |
| tokenizer.json | ~700KB | BPE tokenizer |
| CLIP (auto-downloaded) | ~350MB | Vision encoder |
| **Total** | **~1.5MB + 350MB** | (CLIP cached after first run) |

## Demonstration Tips

For best demo impact:
1. **Use solid colored objects** (POC model specialty)
2. **Good lighting** (helps CLIP features)
3. **Center objects** in frame
4. **Press spacebar** deliberately for dramatic effect
5. **Show inference time** to emphasize edge computing

## Questions or Issues?

If you encounter problems, check:
1. All files transferred correctly: `ls -lh`
2. Dependencies installed: `pip3 list | grep -E "torch|transformers|onnx"`
3. Camera working: `v4l2-ctl --list-devices`
4. CUDA available: `python3 -c "import torch; print(torch.cuda.is_available())"`

---

**Ready to run?** Execute:
```bash
python3 jetson_webcam_demo.py --onnx caption_poc.onnx --config caption_poc_config.json --tokenizer tokenizer.json
```

Point the camera at colored objects and press SPACEBAR! 🎥✨
