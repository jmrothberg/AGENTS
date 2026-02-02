# LFM-2.5 Inference Tools

**Written by Jonathan M Rothberg**

Local inference scripts for LiquidAI's LFM-2.5 models. Supports both:
- **macOS** (Apple Silicon) using MLX for optimized inference
- **Ubuntu/Linux** (NVIDIA GPUs) using transformers/PyTorch

## Models

| Model | Size | Description |
|-------|------|-------------|
| LFM2.5-1.2B-Thinking | 1.2B params | Text-only reasoning model |
| LFM2.5-VL-1.6B | 1.6B params | Vision-Language model (images + video) |

## Scripts

### `lfm_thinking.py`
Interactive chat interface supporting text, images, and video analysis.

### `glm_flash.py`
GLM-4.7-Flash-FP8 script (note: FP8 not yet supported on Blackwell GPUs).

## Usage

```bash
python lfm_thinking.py
```

Select model at startup:
- **1** = Text-only reasoning
- **2** = Vision-Language (images + video)

### Media Options (VL Model)
- `i` = Image (opens file dialog)
- `v` = Video analysis
- `n` = Text only

## Video Analysis

### How It Works

1. Select a video file via file dialog
2. Choose sampling interval (default: 2 seconds)
3. Script extracts frames at the specified interval
4. Each frame is sent to the VL model with your prompt
5. Descriptions are printed with timestamps
6. Option to save results to a timestamped text file

### Supported Video Formats

OpenCV (cv2) handles video decoding. Supported formats depend on your system's codecs:

| Format | Extension | Notes |
|--------|-----------|-------|
| MP4 | `.mp4` | H.264/H.265 codec, most common |
| AVI | `.avi` | Legacy format, widely supported |
| MOV | `.mov` | QuickTime format |
| MKV | `.mkv` | Matroska container |
| WebM | `.webm` | VP8/VP9 codec |

### Frame Sampling

The script does **not** process every frame. Instead:

- Calculates `frame_interval = FPS × interval_seconds`
- Reads frames sequentially but only analyzes every Nth frame
- Example: 30 FPS video with 2s interval = analyze every 60th frame

This allows fast processing of long videos while capturing scene changes.

### Save Output

After video analysis, type `y` when prompted to save results:

```
Save results? (y/n): y
Saved to: drone_test_analysis_20260201_143052.txt
```

Output filename format: `{video_name}_analysis_{YYYYMMDD_HHMMSS}.txt`

## Requirements

```bash
pip install -r requirements.txt
```

### Core Dependencies (All Platforms)
- `pillow` - Image processing
- `opencv-python` - Video processing
- `pyttsx3` - Text-to-speech (optional)

### macOS (Apple Silicon)
- `mlx-vlm` - MLX Vision Language Models
- `torchvision` - Required by mlx_vlm processor

### Ubuntu/Linux
- `transformers` (5.0+)
- `torch` (with CUDA support)
- `accelerate` (model loading)

## Hardware

### macOS (Apple Silicon)
Tested on Mac Studio with M-series chips. Uses MLX framework for optimized inference on Apple Neural Engine.

### Ubuntu/Linux (NVIDIA GPUs)
Tested on NVIDIA Blackwell GPUs (DGX Spark). The scripts include Blackwell-specific optimizations:
- `CUDA_DEVICE_MAX_CONNECTIONS=1`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

## Local Model Paths

### macOS
```
/Users/jonathanrothberg/MLX_Models/LFM2.5-VL-1.6B-MLX-8bit
```

### Ubuntu/Linux
```
/home/jonathan/Models_Transformer/LFM2.5-1.2B-Thinking
/home/jonathan/Models_Transformer/LFM2.5-VL-1.6B
```

## License

MIT
