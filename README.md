# ImgSeg

NN-based segmentation for $2\mu m$ Visium HD data. Use Residual Attention U-Net for sub-cellular accuracy.

## Features
- **Residual Blocks**: Deep gradient flow. No vanishing.
- **Attention Gates**: Focus on cell boundaries. Ignore background noise.
- **GroupNorm**: Stable training on small batches (high-res tiles).
- **Hybrid Loss**: Dice + BCE. Handle class imbalance (tiny cells vs large background).
- **Inference Optimization**: FP16 mixed precision + overlapping patch prediction.

## Structure
```text
.
├── models/         # U-Net, Attention, Residual modules
├── utils/          # Tiling, normalization, patching
├── train.py        # Training script (Mixed Precision)
├── predict.py      # Inference on .btf or .tif images
└── requirements.txt
```

## Installation
```bash
pip install torch torchvision torchaudio
pip install opencv-python tifffile segmentation-models-pytorch
```

## Usage

### 1. Data Prep
Visium HD provide $2\mu m$ `.btf` images. Slice into $256 \times 256$ or $512 \times 512$ patches.
```python
# Tile high-res image
from utils import tile_image
tile_image("visium_hd_image.btf", patch_size=256, overlap=32)
```

### 2. Training
```bash
python train.py --data_dir ./patches --epochs 50 --lr 1e-4 --batch_size 16
```

### 3. Inference
Predict masks on full slide using sliding window.
```bash
python predict.py --input tissue_image.btf --model_path best_model.pth --output mask.tif
```

## Performance Tuning
- **Overlapping Patches**: Use 32px-64px overlap to fix edge artifacts.
- **TTA**: Flip/rotate input at test time. Average results.
- **Mixed Precision**:
```python
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    output = model(images)
    loss = criterion(output, masks)
```

## Integration
Masks compatible with `SpaceRanger` custom segmentation or `bin2cell` for UMI-to-cell assignment.

## License
MIT
