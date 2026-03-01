import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, WeightedRandomSampler
from torch.amp import autocast, GradScaler
from torch.optim.swa_utils import AveragedModel, SWALR
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_curve, auc, f1_score, roc_auc_score
from tqdm import tqdm
import random
from torch.optim.lr_scheduler import OneCycleLR
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance - focuses on hard examples"""
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        F_loss = alpha_t * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        return F_loss


class LabelSmoothingBCEWithLogitsLoss(nn.Module):
    """BCE with label smoothing to prevent overconfident predictions"""
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingBCEWithLogitsLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, inputs, targets):
        targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return F.binary_cross_entropy_with_logits(inputs, targets)


class CombinedLoss(nn.Module):
    """Combines Focal Loss and Label Smoothing BCE for robust training"""
    def __init__(self, focal_weight=0.5, smoothing=0.05, alpha=0.25, gamma=2.0):
        super(CombinedLoss, self).__init__()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.smooth_bce = LabelSmoothingBCEWithLogitsLoss(smoothing=smoothing)
        self.focal_weight = focal_weight

    def forward(self, inputs, targets):
        return self.focal_weight * self.focal_loss(inputs, targets) + \
               (1 - self.focal_weight) * self.smooth_bce(inputs, targets)

class PneumoniaModel(nn.Module):
    def __init__(self, use_pretrained=True):
        super(PneumoniaModel, self).__init__()

        # ConvNeXt V2 Base pretrained on ImageNet-22k
        self.model = timm.create_model('convnextv2_base.fcmae_ft_in22k_in1k',
                                       pretrained=use_pretrained, num_classes=0)
        in_features = self.model.num_features
        self.classifier = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(0.4),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        features = self.model(x)
        return self.classifier(features)

class ChestXRayDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        self.normal_images = [os.path.join(image_dir, 'NORMAL', img)
                             for img in os.listdir(os.path.join(image_dir, 'NORMAL'))]
        self.pneumonia_images = [os.path.join(image_dir, 'PNEUMONIA', img)
                                for img in os.listdir(os.path.join(image_dir, 'PNEUMONIA'))]

        self.all_images = self.normal_images + self.pneumonia_images
        self.labels = [0] * len(self.normal_images) + [1] * len(self.pneumonia_images)

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        img_path = self.all_images[idx]
        image = np.array(Image.open(img_path).convert('RGB'))
        label = self.labels[idx]

        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']

        return image, torch.tensor(label, dtype=torch.float32)

    @property
    def class_counts(self):
        return {
            'normal': len(self.normal_images),
            'pneumonia': len(self.pneumonia_images)
        }

def get_transforms(image_size=256, crop_size=224):
    """Enhanced transforms with stronger augmentation for better generalization"""
    train_transform = A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.RandomCrop(height=crop_size, width=crop_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Affine(
            translate_percent=(-0.1, 0.1),
            scale=(0.85, 1.15),
            rotate=(-15, 15),
            p=0.5
        ),
        A.OneOf([
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
            A.ElasticTransform(alpha=1, sigma=50, p=1.0),
            A.OpticalDistortion(distort_limit=0.3, p=1.0),
        ], p=0.3),
        A.OneOf([
            A.GaussNoise(std_range=(0.12, 0.28), p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=7, p=1.0),
        ], p=0.2),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
            A.RandomGamma(gamma_limit=(80, 120), p=1.0),
        ], p=0.5),
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(crop_size // 32, crop_size // 16),
            hole_width_range=(crop_size // 32, crop_size // 16),
            p=0.3
        ),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Resize(height=crop_size, width=crop_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    return train_transform, val_transform


def get_tta_transforms(crop_size=224):
    """Test-time augmentation transforms for more robust predictions"""
    return [
        # Original
        A.Compose([
            A.Resize(height=crop_size, width=crop_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        # Horizontal flip
        A.Compose([
            A.Resize(height=crop_size, width=crop_size),
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        # Slight rotation
        A.Compose([
            A.Resize(height=crop_size, width=crop_size),
            A.Rotate(limit=5, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        # Scale up slightly
        A.Compose([
            A.Resize(height=int(crop_size*1.1), width=int(crop_size*1.1)),
            A.CenterCrop(height=crop_size, width=crop_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        # Slight brightness adjustment
        A.Compose([
            A.Resize(height=crop_size, width=crop_size),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
    ]

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mixup_data(x, y, alpha=0.2):
    """Performs mixup augmentation on input data"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Loss function for mixup training"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def find_optimal_threshold(y_true, y_prob, metric='f1'):
    """Find an operating threshold using validation probabilities."""
    thresholds = np.arange(0.1, 0.901, 0.01)
    y_true = np.array(y_true).astype(int)
    y_prob = np.array(y_prob)

    best_threshold = 0.5
    best_score = -1.0

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        if metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'balanced':
            tp = np.sum((y_pred == 1) & (y_true == 1))
            tn = np.sum((y_pred == 0) & (y_true == 0))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            fn = np.sum((y_pred == 0) & (y_true == 1))
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            score = (recall + specificity) / 2
        else:
            score = f1_score(y_true, y_pred, zero_division=0)

        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    return best_threshold, best_score


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=15,
                checkpoint_dir='checkpoints', resume_from=None, save_freq=1,
                early_stop_patience=10, early_stop_min_delta=0.001, use_mixup=True,
                use_swa=True, swa_start=15, mixup_alpha=0.2):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    best_val_acc = 0.0
    best_val_f1 = 0.0
    best_val_auc = 0.0
    epochs_without_improve = 0
    start_epoch = 0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [], 'val_auc': []
    }

    if resume_from and os.path.isfile(resume_from):
        print(f"Loading checkpoint '{resume_from}'")
        checkpoint = torch.load(resume_from)
        start_epoch = checkpoint['epoch']
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        best_val_f1 = checkpoint.get('best_val_f1', best_val_acc)
        best_val_auc = checkpoint.get('best_val_auc', 0.0)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        history = checkpoint['history']
        print(f"Loaded checkpoint '{resume_from}' (epoch {checkpoint['epoch']})")
    else:
        print("No checkpoint found, starting from scratch")

    scaler = GradScaler()

    # Use OneCycleLR for better convergence
    scheduler = OneCycleLR(
        optimizer,
        max_lr=optimizer.param_groups[0]['lr'],
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy='cos',
        div_factor=25,
        final_div_factor=1000
    )

    # Setup SWA for better generalization
    swa_model = None
    swa_scheduler = None
    if use_swa:
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=1e-5)

    for epoch in range(start_epoch, num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}:')

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        train_pbar = tqdm(train_loader, desc="Training")

        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                # Apply mixup augmentation
                if use_mixup and epoch < num_epochs - 5:  # Disable mixup for last 5 epochs
                    inputs_mixed, targets_a, targets_b, lam = mixup_data(inputs, labels, mixup_alpha)
                    outputs = model(inputs_mixed).squeeze()
                    loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                    # For accuracy calculation, use original predictions
                    with torch.no_grad():
                        orig_outputs = model(inputs).squeeze()
                        predicted = (torch.sigmoid(orig_outputs) > 0.5).float()
                else:
                    outputs = model(inputs).squeeze()
                    loss = criterion(outputs, labels)
                    predicted = (torch.sigmoid(outputs) > 0.5).float()

            scaler.scale(loss).backward()

            # Gradient clipping for stability
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            # Step the scheduler
            if epoch < swa_start or not use_swa:
                scheduler.step()

            running_loss += loss.item() * inputs.size(0)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            train_pbar.set_postfix(loss=loss.item(), acc=(predicted == labels).float().mean().item())

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_probs = []
        val_labels = []

        val_pbar = tqdm(val_loader, desc="Validation", leave=False)

        with torch.no_grad():
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                probs = torch.sigmoid(outputs)
                predicted = (probs > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                val_probs.extend(probs.detach().cpu().numpy())
                val_labels.extend(labels.detach().cpu().numpy())

                val_pbar.set_postfix(loss=loss.item(), acc=(predicted == labels).float().mean().item())

        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_acc = val_correct / val_total
        val_epoch_f1 = f1_score(np.array(val_labels).astype(int), (np.array(val_probs) >= 0.5).astype(int), zero_division=0)
        try:
            val_epoch_auc = roc_auc_score(np.array(val_labels).astype(int), np.array(val_probs))
        except ValueError:
            val_epoch_auc = 0.0

        history['val_loss'].append(val_epoch_loss)
        history['val_acc'].append(val_epoch_acc)
        history['val_f1'].append(val_epoch_f1)
        history['val_auc'].append(val_epoch_auc)

        print(f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}')
        print(
            f'Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}, '
            f'Val F1: {val_epoch_f1:.4f}, Val AUC: {val_epoch_auc:.4f}'
        )

        # Update SWA model after swa_start epochs
        if use_swa and epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        print(f'Current Learning Rate: {current_lr:.6f}')

        if val_epoch_f1 > best_val_f1 + early_stop_min_delta:
            best_val_acc = val_epoch_acc
            best_val_f1 = val_epoch_f1
            best_val_auc = val_epoch_auc
            epochs_without_improve = 0
            best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'best_val_f1': best_val_f1,
                'best_val_auc': best_val_auc,
                'history': history
            }, best_model_path)
            print(
                f"Saved best model to {best_model_path} with validation F1: {best_val_f1:.4f} "
                f"(Acc: {best_val_acc:.4f}, AUC: {best_val_auc:.4f})"
            )
        else:
            epochs_without_improve += 1

        if (epoch + 1) % save_freq == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'best_val_f1': best_val_f1,
                'best_val_auc': best_val_auc,
                'history': history
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

        if early_stop_patience > 0 and epochs_without_improve >= early_stop_patience:
            print(f"Early stopping triggered after {epoch + 1} epochs with no improvement.")
            break

    best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
    if os.path.isfile(best_model_path):
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(
            "Loaded best model with validation metrics: "
            f"Acc={checkpoint.get('best_val_acc', 0.0):.4f}, "
            f"F1={checkpoint.get('best_val_f1', 0.0):.4f}, "
            f"AUC={checkpoint.get('best_val_auc', 0.0):.4f}"
        )

    # Update batch normalization statistics for SWA model
    if use_swa and swa_model is not None:
        print("Updating SWA batch normalization statistics...")
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)

        # Save SWA model
        swa_model_path = os.path.join(checkpoint_dir, 'swa_model.pth')
        torch.save({
            'model_state_dict': swa_model.module.state_dict(),
            'epoch': num_epochs,
            'best_val_acc': best_val_acc,
            'best_val_f1': best_val_f1,
            'best_val_auc': best_val_auc,
            'history': history
        }, swa_model_path)
        print(f"Saved SWA model to {swa_model_path}")

    return model, history, swa_model

def evaluate_model(model, test_loader, threshold=0.5, save_roc_path='roc_curve.png', verbose=True):
    model.eval()
    all_preds = []
    all_labels = []
    test_correct = 0
    test_total = 0

    test_pbar = tqdm(test_loader, desc="Testing")

    with torch.no_grad():
        for inputs, labels in test_pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs).squeeze()
            probs = torch.sigmoid(outputs)
            predicted = (probs > threshold).float()

            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

            current_acc = (predicted == labels).float().mean().item()
            test_pbar.set_postfix(acc=current_acc)

            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_acc = test_correct / test_total
    if verbose:
        print(f'Test Accuracy: {test_acc:.4f}')

    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    roc_auc = auc(fpr, tpr)

    binary_preds = [1 if p > threshold else 0 for p in all_preds]
    test_f1 = f1_score(np.array(all_labels).astype(int), np.array(binary_preds).astype(int), zero_division=0)
    report = classification_report(all_labels, binary_preds, target_names=['Normal', 'Pneumonia'])

    if verbose:
        print(report)
        print(f'AUC: {roc_auc:.4f}')
        print(f'F1 (@{threshold:.2f}): {test_f1:.4f}')

    if save_roc_path:
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(save_roc_path)

    metrics = {
        'accuracy': test_acc,
        'auc': roc_auc,
        'f1': test_f1,
        'threshold': threshold
    }

    return test_acc, all_preds, all_labels, metrics

def predict_image(model, image_path, transform):
    model.eval()
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image).squeeze().item()
        probability = torch.sigmoid(torch.tensor(output)).item()

    prediction = "Pneumonia" if probability > 0.5 else "Normal"

    return prediction, probability
