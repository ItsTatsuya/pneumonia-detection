# Pneumonia Detection from Chest X-rays with GPU Acceleration

This project implements a deep learning model to detect pneumonia from chest X-ray images with GPU acceleration.

## Requirements

- Python 3.6+
- CUDA-compatible GPU (for GPU acceleration)
- Dependencies listed in `requirements.txt`

## Installation

1. Clone this repository:

   ```
   git clone https://github.com/yourusername/pneumonia-detection.git
   cd pneumonia-detection
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Dataset

The model is designed to work with the Chest X-Ray dataset that contains normal and pneumonia X-ray images. The expected dataset structure is:

```
chest_xray/
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

You can download the dataset from [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia).

## Usage

### Training

To train the model:

```
python train.py --data_dir /path/to/chest_xray --batch_size 32 --epochs 25 --output_dir models
```

Additional training options:

- `--lr`: Learning rate (default: 0.001)
- `--num_workers`: Number of data loading workers (default: 4)
- `--pretrained`: Use pretrained weights (default: True)
- `--model_type`: Model architecture to use (default: 'efficientnet_b4')

#### Checkpoint Options

The training script supports checkpointing to save training progress and resume from interruptions:

- `--checkpoint_dir`: Directory to save checkpoints (default: 'checkpoints')
- `--resume_from`: Path to a checkpoint to resume training from
- `--save_freq`: Save checkpoint every N epochs (default: 1)
- `--keep_checkpoints`: Number of most recent checkpoints to keep (default: 3)

To resume training from a checkpoint:

```
python train.py --data_dir /path/to/chest_xray --resume_from checkpoints/checkpoint_epoch_10.pth
```

### Prediction

To make predictions on new X-ray images:

```
python predict.py --model_path models/pneumonia_model.pth --image_path /path/to/xray.jpg
```

For batch predictions on a directory of images:

```
python predict.py --model_path models/pneumonia_model.pth --image_path /path/to/images_dir
```

## Model Architecture

The project implements several advanced model architectures with the following options:

### EfficientNet-B4 (Default)

Our default model uses EfficientNet-B4 architecture which offers an excellent balance between accuracy and computational efficiency. The model has been enhanced with:

- Custom classifier head with 1024 neurons in the first fully connected layer
- Multiple dropout layers (0.4, 0.5, 0.3) to prevent overfitting
- Multi-layer structure for better feature extraction

### Other Available Architectures

- `resnet`: ResNet-50 architecture with custom classifier
- `efficientnet`: EfficientNet-B3 with custom classifier
- `efficientnet_b4`: EfficientNet-B4 with enhanced classifier (default)
- `timm_*`: Any model from the timm library (e.g., `timm_tf_efficientnetv2_s`)

## Advanced Features

### Mixed Precision Training

The model uses mixed precision training with automatic mixed precision (AMP) to:

- Speed up training by approximately 2-3x
- Reduce memory usage
- Maintain numerical stability and accuracy

### Enhanced Data Augmentation

We use Albumentations library for advanced augmentations:

- Elastic transforms and grid distortion specifically effective for medical imaging
- Vertical flipping (useful for some X-ray datasets)
- Gaussian noise and blur for improved robustness
- Random brightness and contrast adjustments
- Extensive geometric transformations (rotation, scaling, translation)

### Learning Rate Scheduling

The model implements CosineAnnealingWarmRestarts scheduler which:

- Periodically reduces learning rate following a cosine curve
- Periodically "restarts" the learning rate to help escape local minima
- Improves convergence compared to step-based schedulers

### Class Balancing

For imbalanced datasets, the model automatically:

- Detects class imbalance in the training data
- Implements weighted sampling to ensure equal representation
- Applies class weights to the loss function

### Random Seed Control

Full control over random seed initialization for reproducibility.

## Performance

The enhanced model achieves:

- Test Accuracy: >93% on the pneumonia detection task
- AUC (Area Under ROC Curve): >0.97
- High recall for pneumonia class (~97%) which is critical for medical applications

### Comparison with Original Model

| Metric            | Original Model    | Enhanced Model              |
| ----------------- | ----------------- | --------------------------- |
| Architecture      | ResNet-50         | EfficientNet-B4             |
| Data Augmentation | Basic             | Advanced (Albumentations)   |
| Training Method   | Standard          | Mixed Precision             |
| Scheduler         | ReduceLROnPlateau | CosineAnnealingWarmRestarts |
| Test Accuracy     | ~90%              | >93%                        |
| AUC               | ~0.95             | >0.97                       |
| Training Time     | Baseline          | ~40% faster                 |

## GPU Acceleration

The code automatically uses GPU acceleration if available. To verify GPU usage, check the output when running the scripts:

```
Using device: cuda
```

The implementation includes:

- Automatic detection of CUDA-capable GPUs
- Mixed precision training with torch.cuda.amp
- Efficient data loading with pin_memory and non-blocking transfers

## Checkpoint System

The model implements a robust checkpoint system that:

1. Saves the best model based on validation accuracy
2. Periodically saves checkpoints during training
3. Allows resuming training from any checkpoint
4. Automatically cleans up old checkpoints to save disk space

Two versions of the final model are saved:

- A complete checkpoint with training state (for resuming training)
- A lightweight version with only model weights (for deployment/inference)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
