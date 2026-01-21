# Pneumonia Detection from Chest X-rays

Deep learning model for detecting pneumonia in chest X-ray images. Uses ConvNeXt V2 with various training tricks to achieve ~95% accuracy.

## Results

**Test Set Performance (624 images):**

| Metric          | Score  |
| --------------- | ------ |
| Accuracy        | 94.87% |
| ROC AUC         | 0.9787 |
| Sensitivity     | 96.41% |
| Specificity     | 92.31% |
| False Positives | 18     |
| False Negatives | 14     |

```
              precision    recall  f1-score   support

      Normal       0.94      0.92      0.93       234
   Pneumonia       0.95      0.96      0.96       390

    accuracy                           0.95       624
```

## Requirements

- Python 3.9+
- CUDA GPU (training takes ~30 min on RTX 3080)
- ~4GB VRAM minimum

## Quick Start

```bash
# Install deps
pip install -r requirements.txt

# Train
python train.py --data_dir ./dataset --epochs 40

# Predict
python predict.py --model_path output/pneumonia_model_TIMESTAMP.pth --data_dir dataset/test --use_tta
```

## Dataset Structure

```
dataset/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

Get it from [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia).

## Training

```bash
python train.py --data_dir /path/to/data --batch_size 16 --epochs 40
```

**Key arguments:**

- `--loss_type`: `combined` (focal + label smoothing), `focal`, `smooth_bce`, `bce`
- `--use_mixup` / `--no_mixup`: Mixup augmentation (on by default)
- `--use_swa` / `--no_swa`: Stochastic Weight Averaging (on by default)
- `--resume_from`: Resume from checkpoint

**Resume training:**

```bash
python train.py --data_dir ./dataset --resume_from checkpoints/checkpoint_epoch_10.pth
```

## Prediction

```bash
python predict.py --model_path output/model.pth --data_dir dataset/test
```

**Key arguments:**

- `--use_tta` / `--no_tta`: Test-time augmentation (averages 5 augmented predictions)
- `--optimize_threshold`: Finds best classification threshold (not just 0.5)
- `--threshold 0.7`: Use specific threshold

## What's Under the Hood

**Model:** ConvNeXt V2 Base pretrained on ImageNet-22k.

**Loss:** Combined focal loss + label smoothing BCE. Focal loss helps with the class imbalance, label smoothing prevents overconfident predictions.

**Augmentation:** Heavy augmentation via Albumentations - elastic transforms, grid distortion, coarse dropout, brightness/contrast jitter, etc. Also uses mixup during training.

**Training tricks:**

- Mixed precision (AMP) for faster training
- OneCycleLR scheduler
- Stochastic Weight Averaging in final epochs
- Gradient clipping
- AdamW with weight decay

**Inference tricks:**

- Test-time augmentation (5 augmented versions averaged)
- Threshold optimization (finds optimal cutoff, not just 0.5)

## Checkpoints

Training saves:

- `checkpoints/best_model.pth` - Best validation accuracy
- `checkpoints/swa_model.pth` - SWA averaged weights
- `checkpoints/checkpoint_epoch_N.pth` - Periodic saves

Output saves:

- `output/pneumonia_model_TIMESTAMP.pth` - Full checkpoint
- `output/pneumonia_model_TIMESTAMP_weights_only.pth` - Just weights (smaller)
- `output/pneumonia_model_TIMESTAMP_swa.pth` - SWA model

## Prediction Output

Running predict.py generates in `predictions/`:

- `predictions.csv` - All predictions with probabilities
- `summary.txt` - Accuracy metrics
- `classification_report.txt` - Precision/recall/F1
- `confusion_matrix.png`
- `roc_curve.png`
- `precision_recall_curve.png`
- `threshold_analysis.png` - How metrics change with threshold
