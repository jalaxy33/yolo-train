# yolo-train

A generic YOLO model training and export framework built on [ultralytics](https://github.com/ultralytics/ultralytics).

## Quick Start

```bash
# Install dependencies
uv sync

# Run training
uv run src/train.py
```

## Dependencies

- **uv**: An extremely fast Python package manager. [Installation guide](https://docs.astral.sh/uv/).

## Configuration

Edit `assets/configs/<config>.toml` to customize:
- `datasets_dir` - Training data location
- `weights_dir` - Pre-trained weights location
- `runs_dir` - Output directory for results

## Requirements

- Python 3.12+
- CUDA-capable GPU (recommended)

## Project Structure

```
yolo-train/
├── src/
│   ├── train.py         # Training & export entry point
│   └── utils/
│       ├── yolo_settings.py  # TOML config loader
│       └── path_utils.py     # Path helpers
├── assets/
│   └── configs/         # TOML configuration files
└── Taskfile.yml         # Automation tasks
```

## Tech Stack

- **ultralytics**: YOLO model training/export
- **torch/torchvision**: PyTorch backend
- **onnx/onnxruntime**: ONNX export and inference
- **uv**: Python package management
- **task**: Task automation runner
