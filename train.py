import argparse
import os
import torch
import json
from model import (PneumoniaModel, ChestXRayDataset, get_transforms, train_model,
                   evaluate_model, CombinedLoss, FocalLoss, LabelSmoothingBCEWithLogitsLoss,
                   find_optimal_threshold, set_seed)
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description='Train a pneumonia detection model')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=40, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--pretrained', action='store_true', default=True, help='Use pretrained model (default: True)')
    parser.add_argument('--no_pretrained', dest='pretrained', action='store_false', help='Disable pretrained weights')
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory for saving models')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory for saving checkpoints')
    parser.add_argument('--resume_from', type=str, default=None, help='Path to checkpoint to resume training from')
    parser.add_argument('--save_freq', type=int, default=1, help='Save checkpoint every N epochs')
    parser.add_argument('--keep_checkpoints', type=int, default=3, help='Number of most recent checkpoints to keep')
    parser.add_argument('--early_stop_patience', type=int, default=15, help='Early stopping patience in epochs')
    parser.add_argument('--early_stop_min_delta', type=float, default=0.001, help='Minimum improvement to reset patience')
    parser.add_argument('--loss_type', type=str, default='combined',
                        help='Loss function: combined, focal, smooth_bce, bce')
    parser.add_argument('--use_mixup', action='store_true', default=True, help='Use mixup augmentation')
    parser.add_argument('--no_mixup', dest='use_mixup', action='store_false', help='Disable mixup augmentation')
    parser.add_argument('--use_swa', action='store_true', default=True, help='Use Stochastic Weight Averaging')
    parser.add_argument('--no_swa', dest='use_swa', action='store_false', help='Disable SWA')
    parser.add_argument('--swa_start', type=int, default=25, help='Epoch to start SWA')
    parser.add_argument('--image_size', type=int, default=256, help='Input image size before crop')
    parser.add_argument('--crop_size', type=int, default=224, help='Crop size for training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--threshold_metric', type=str, default='f1', choices=['f1', 'balanced'],
                        help='Metric used to select validation threshold')
    return parser.parse_args()

def cleanup_old_checkpoints(checkpoint_dir, keep_n, exclude=None):
    """Delete all but the keep_n most recent checkpoints, excluding any specific files."""
    if not os.path.exists(checkpoint_dir):
        return

    # Get all checkpoint files
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_') and f.endswith('.pth')]
    # Sort by epoch number (descending)
    checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]), reverse=True)

    # Keep the first keep_n checkpoints and delete the rest
    for checkpoint in checkpoints[keep_n:]:
        if exclude and checkpoint == os.path.basename(exclude):
            continue
        try:
            os.remove(os.path.join(checkpoint_dir, checkpoint))
            print(f"Removed old checkpoint: {checkpoint}")
        except Exception as e:
            print(f"Failed to remove {checkpoint}: {e}")

def main():
    args = parse_args()

    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'val')
    test_dir = os.path.join(args.data_dir, 'test')

    train_transform, val_transform = get_transforms(args.image_size, args.crop_size)

    print("Creating datasets...")
    train_dataset = ChestXRayDataset(train_dir, transform=train_transform)
    val_dataset = ChestXRayDataset(val_dir, transform=val_transform)
    test_dataset = ChestXRayDataset(test_dir, transform=val_transform)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    class_counts = train_dataset.class_counts
    imbalance = abs(class_counts['normal'] - class_counts['pneumonia'])

    sampler = None
    if imbalance > 50:
        class_weights = {
            0: 1.0 / max(1, class_counts['normal']),
            1: 1.0 / max(1, class_counts['pneumonia'])
        }
        sample_weights = [class_weights[label] for label in train_dataset.labels]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print("Initializing model...")
    model = PneumoniaModel(use_pretrained=args.pretrained).to(device)

    # Select loss function based on argument
    if args.loss_type == 'combined':
        criterion = CombinedLoss(focal_weight=0.5, smoothing=0.05, alpha=0.25, gamma=2.0)
        print("Using Combined Loss (Focal + Label Smoothing BCE)")
    elif args.loss_type == 'focal':
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        print("Using Focal Loss")
    elif args.loss_type == 'smooth_bce':
        criterion = LabelSmoothingBCEWithLogitsLoss(smoothing=0.1)
        print("Using Label Smoothing BCE Loss")
    else:
        if sampler is not None:
            pos_weight = torch.tensor([class_counts['normal'] / max(1, class_counts['pneumonia'])], device=device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            criterion = nn.BCEWithLogitsLoss()
        print("Using BCE with Logits Loss")

    # Use AdamW with weight decay for better generalization
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("Starting training...")
    model, history, swa_model = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        num_epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume_from,
        save_freq=args.save_freq,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        use_mixup=args.use_mixup,
        use_swa=args.use_swa,
        swa_start=args.swa_start
    )

    cleanup_old_checkpoints(args.checkpoint_dir, args.keep_checkpoints)

    best_model_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
    final_model_path = os.path.join(args.output_dir, f"pneumonia_model_{timestamp}.pth")

    checkpoint = None
    if os.path.exists(best_model_path):
        # Copy the full checkpoint for resuming training later
        shutil.copy2(best_model_path, final_model_path)

        # Save a lightweight version with only the model weights for inference
        checkpoint = torch.load(best_model_path, map_location=device)
        torch.save(
            checkpoint['model_state_dict'],
            os.path.join(args.output_dir, f"pneumonia_model_{timestamp}_weights_only.pth")
        )

        print(f"Best model saved to {final_model_path}")
        print(
            "Lightweight model for inference saved to "
            f"{os.path.join(args.output_dir, f'pneumonia_model_{timestamp}_weights_only.pth')}"
        )

    # Also save SWA model if available
    swa_model_path = os.path.join(args.checkpoint_dir, 'swa_model.pth')
    if os.path.exists(swa_model_path):
        shutil.copy2(swa_model_path, os.path.join(args.output_dir, f"pneumonia_model_{timestamp}_swa.pth"))
        print(f"SWA model saved to {os.path.join(args.output_dir, f'pneumonia_model_{timestamp}_swa.pth')}")

    print("Evaluating model on test set...")

    print("Selecting classification threshold on validation set (strict holdout protocol)...")
    _, val_probs, val_labels, val_metrics = evaluate_model(
        model,
        val_loader,
        threshold=0.5,
        save_roc_path=os.path.join(args.output_dir, 'roc_curve_val.png'),
        verbose=False
    )

    optimal_threshold, threshold_score = find_optimal_threshold(
        val_labels,
        val_probs,
        metric=args.threshold_metric
    )
    print(
        f"Validation-selected threshold ({args.threshold_metric}): "
        f"{optimal_threshold:.3f} (score={threshold_score:.4f})"
    )

    threshold_artifact = {
        'threshold': optimal_threshold,
        'metric': args.threshold_metric,
        'val_metrics_at_0_5': val_metrics
    }
    threshold_path = os.path.join(args.output_dir, f"threshold_{timestamp}.json")
    with open(threshold_path, 'w') as f:
        json.dump(threshold_artifact, f, indent=2)

    test_acc, all_preds, all_labels, test_metrics = evaluate_model(
        model,
        test_loader,
        threshold=optimal_threshold,
        save_roc_path=os.path.join(args.output_dir, 'roc_curve_test.png'),
        verbose=True
    )

    print(
        "Training complete! "
        f"Final test metrics -> Accuracy: {test_metrics['accuracy']:.4f}, "
        f"F1: {test_metrics['f1']:.4f}, AUC: {test_metrics['auc']:.4f}"
    )

    with open(os.path.join(args.output_dir, f"training_summary_{timestamp}.txt"), "w") as f:
        f.write(f"Model: ConvNeXt V2 Base\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Learning rate: {args.lr}\n")
        f.write(f"Pretrained: {args.pretrained}\n")
        f.write(f"Loss type: {args.loss_type}\n")
        f.write(f"Use mixup: {args.use_mixup}\n")
        f.write(f"Use SWA: {args.use_swa}\n")
        f.write(f"SWA start epoch: {args.swa_start}\n")
        f.write(f"Image size: {args.image_size}\n")
        f.write(f"Crop size: {args.crop_size}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Threshold metric: {args.threshold_metric}\n")
        f.write(f"Selected threshold (from val only): {optimal_threshold:.3f}\n")
        f.write(f"Final test accuracy: {test_metrics['accuracy']:.4f}\n")
        f.write(f"Final test F1: {test_metrics['f1']:.4f}\n")
        f.write(f"Final test AUC: {test_metrics['auc']:.4f}\n")
        f.write(f"Threshold artifact: {threshold_path}\n")
        if checkpoint:
            f.write(f"Best validation accuracy: {checkpoint.get('best_val_acc', 0.0):.4f}\n")
            f.write(f"Best validation F1: {checkpoint.get('best_val_f1', 0.0):.4f}\n")
            f.write(f"Best validation AUC: {checkpoint.get('best_val_auc', 0.0):.4f}\n")
            f.write(f"Best model epoch: {checkpoint['epoch']}\n")

if __name__ == "__main__":
    main()
