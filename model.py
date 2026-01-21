import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import models
from torchvision.models import ResNet50_Weights, EfficientNet_B3_Weights, EfficientNet_B4_Weights
from torch.amp import autocast, GradScaler
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_curve, auc
from tqdm import tqdm
import random
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PneumoniaModel(nn.Module):
    def __init__(self, use_pretrained=True, model_type='efficientnet_b4'):
        super(PneumoniaModel, self).__init__()

        if model_type == 'resnet':
            weights = ResNet50_Weights.IMAGENET1K_V1 if use_pretrained else None
            self.model = models.resnet50(weights=weights)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 1)
            )
        elif model_type == 'efficientnet':
            weights = EfficientNet_B3_Weights.IMAGENET1K_V1 if use_pretrained else None
            self.model = models.efficientnet_b3(weights=weights)
            in_features = self.model.classifier[1].in_features
            self.model.classifier = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 1)
            )
        elif model_type == 'efficientnet_b4':
            weights = EfficientNet_B4_Weights.IMAGENET1K_V1 if use_pretrained else None
            self.model = models.efficientnet_b4(weights=weights)
            in_features = self.model.classifier[1].in_features
            self.model.classifier = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(in_features, 1024),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 1)
            )
        elif model_type == 'ensemble':
            raise NotImplementedError("Ensemble model not implemented yet")

        elif 'timm_' in model_type:
            model_name = model_type.replace('timm_', '')
            self.model = timm.create_model(model_name, pretrained=use_pretrained, num_classes=0)
            in_features = self.model.num_features
            self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(in_features, 1024),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(1024, 1)
            )

    def forward(self, x):
        if hasattr(self, 'classifier'):
            features = self.model(x)
            return self.classifier(features)
        return self.model(x)

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

def get_transforms():
    train_transform = A.Compose([
        A.Resize(height=256, width=256),
        A.RandomCrop(height=224, width=224),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.Affine(translate_percent=0.1, scale=(0.8, 1.2), rotate=(-15, 15), p=0.5),
        A.OneOf([
            A.GridDistortion(p=1.0),
            A.ElasticTransform(p=1.0)
        ], p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(p=0.2),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Resize(height=224, width=224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    return train_transform, val_transform

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=15,
                checkpoint_dir='checkpoints', resume_from=None, save_freq=1,
                early_stop_patience=10, early_stop_min_delta=0.001):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    best_val_acc = 0.0
    epochs_without_improve = 0
    start_epoch = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    if resume_from and os.path.isfile(resume_from):
        print(f"Loading checkpoint '{resume_from}'")
        checkpoint = torch.load(resume_from)
        start_epoch = checkpoint['epoch']
        best_val_acc = checkpoint['best_val_acc']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        history = checkpoint['history']
        print(f"Loaded checkpoint '{resume_from}' (epoch {checkpoint['epoch']})")
    else:
        print("No checkpoint found, starting from scratch")

    scaler = GradScaler()

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1, eta_min=1e-6)

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
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
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

        val_pbar = tqdm(val_loader, desc="Validation", leave=False)

        with torch.no_grad():
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                val_pbar.set_postfix(loss=loss.item(), acc=(predicted == labels).float().mean().item())

        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_acc = val_correct / val_total
        history['val_loss'].append(val_epoch_loss)
        history['val_acc'].append(val_epoch_acc)

        print(f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}')
        print(f'Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}')

        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        print(f'Current Learning Rate: {current_lr:.6f}')

        if val_epoch_acc > best_val_acc + early_stop_min_delta:
            best_val_acc = val_epoch_acc
            epochs_without_improve = 0
            best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'history': history
            }, best_model_path)
            print(f"Saved best model to {best_model_path} with validation accuracy: {best_val_acc:.4f}")
        else:
            epochs_without_improve += 1

        if (epoch + 1) % save_freq == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
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
        print(f"Loaded best model with validation accuracy: {checkpoint['best_val_acc']:.4f}")

    return model, history

def evaluate_model(model, test_loader):
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
            predicted = (probs > 0.5).float()

            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

            current_acc = (predicted == labels).float().mean().item()
            test_pbar.set_postfix(acc=current_acc)

            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_acc = test_correct / test_total
    print(f'Test Accuracy: {test_acc:.4f}')

    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    roc_auc = auc(fpr, tpr)

    binary_preds = [1 if p > 0.5 else 0 for p in all_preds]
    report = classification_report(all_labels, binary_preds, target_names=['Normal', 'Pneumonia'])

    print(report)
    print(f'AUC: {roc_auc:.4f}')

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')

    return test_acc, all_preds, all_labels

def predict_image(model, image_path, transform):
    model.eval()
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image).squeeze().item()
        probability = torch.sigmoid(torch.tensor(output)).item()

    prediction = "Pneumonia" if probability > 0.5 else "Normal"

    return prediction, probability

def main():
    set_seed(42)

    data_dir = './chest_xray'  # Update this to your dataset path
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')

    # Get transforms
    train_transform, val_transform = get_transforms()

    # Create datasets
    train_dataset = ChestXRayDataset(train_dir, transform=train_transform)
    val_dataset = ChestXRayDataset(val_dir, transform=val_transform)
    test_dataset = ChestXRayDataset(test_dir, transform=val_transform)

    # Get class counts and create weighted sampler to handle imbalance
    class_counts = train_dataset.class_counts
    print(f"Class distribution in training: {class_counts}")

    # Calculate class weights if classes are imbalanced
    weights = None
    if abs(class_counts['normal'] - class_counts['pneumonia']) > 50:  # Check for imbalance
        class_weights = {
            0: 1.0 / class_counts['normal'],
            1: 1.0 / class_counts['pneumonia']
        }
        sample_weights = [class_weights[label] for label in train_dataset.labels]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        train_shuffle = False  # Don't shuffle when using sampler
    else:
        sampler = None
        train_shuffle = True

    # Create data loaders with potential sampler
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=train_shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=4
    )
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Use a more powerful model
    model = PneumoniaModel(use_pretrained=True, model_type='efficientnet_b4').to(device)

    # Define loss function and optimizer
    # Use BCEWithLogitsLoss instead of BCELoss
    if weights is not None:
        class_weights_tensor = torch.tensor([class_weights[0], class_weights[1]], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights_tensor[1])
    else:
        criterion = nn.BCEWithLogitsLoss()

    # Use AdamW instead of Adam for better weight decay handling
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Define checkpoint directory
    checkpoint_dir = 'checkpoints'

    # Increase training epochs and enable mixed precision
    model, history = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        num_epochs=25,  # Increase epochs
        checkpoint_dir=checkpoint_dir,
        resume_from=None,  # Change to a checkpoint path to resume training
        save_freq=1
    )

    # Load best model
    best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from {best_model_path} with validation accuracy: {checkpoint['best_val_acc']:.4f}")
    else:
        print(f"Warning: Best model file {best_model_path} not found. Using the last trained model.")

    # Evaluate model
    test_acc, all_preds, all_labels = evaluate_model(model, test_loader)

    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Accuracy')
    plt.legend()

    plt.savefig('training_history.png')

    print('Training and evaluation complete!')

if __name__ == '__main__':
    main()
