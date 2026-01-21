import argparse
import os
import torch
from model import PneumoniaModel, ChestXRayDataset, get_transforms, train_model, evaluate_model
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description='Train a pneumonia detection model')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=60, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--pretrained', action='store_true', default=True, help='Use pretrained model (default: True)')
    parser.add_argument('--no_pretrained', dest='pretrained', action='store_false', help='Disable pretrained weights')
    parser.add_argument('--model_type', type=str, default='efficientnet_b4', help='Model architecture to use')
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory for saving models')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory for saving checkpoints')
    parser.add_argument('--resume_from', type=str, default=None, help='Path to checkpoint to resume training from')
    parser.add_argument('--save_freq', type=int, default=1, help='Save checkpoint every N epochs')
    parser.add_argument('--keep_checkpoints', type=int, default=3, help='Number of most recent checkpoints to keep')
    parser.add_argument('--early_stop_patience', type=int, default=10, help='Early stopping patience in epochs')
    parser.add_argument('--early_stop_min_delta', type=float, default=0.001, help='Minimum improvement to reset patience')
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

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'val')
    test_dir = os.path.join(args.data_dir, 'test')

    train_transform, val_transform = get_transforms()

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
    model = PneumoniaModel(use_pretrained=args.pretrained, model_type=args.model_type).to(device)

    if sampler is not None:
        pos_weight = torch.tensor([class_counts['normal'] / max(1, class_counts['pneumonia'])], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("Starting training...")
    model, history = train_model(
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
        early_stop_min_delta=args.early_stop_min_delta
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

    print("Evaluating model on test set...")
    test_acc, all_preds, all_labels = evaluate_model(model, test_loader)

    print(f"Training complete! Final test accuracy: {test_acc:.4f}")

    with open(os.path.join(args.output_dir, f"training_summary_{timestamp}.txt"), "w") as f:
        f.write(f"Model: {args.model_type}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Learning rate: {args.lr}\n")
        f.write(f"Pretrained: {args.pretrained}\n")
        f.write(f"Final test accuracy: {test_acc:.4f}\n")
        if checkpoint:
            f.write(f"Best validation accuracy: {checkpoint['best_val_acc']:.4f}\n")
            f.write(f"Best model epoch: {checkpoint['epoch']}\n")

if __name__ == "__main__":
    main()
