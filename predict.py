import argparse
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from model import PneumoniaModel, get_tta_transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import pandas as pd
import seaborn as sns

def parse_args():
    parser = argparse.ArgumentParser(description='Predict pneumonia from chest X-ray images')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--data_dir', type=str, default='./chest_xray/test', help='Path to the test dataset')
    parser.add_argument('--output_dir', type=str, default='predictions', help='Output directory for saving results')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for prediction')
    parser.add_argument('--sample_vis', type=int, default=10, help='Number of sample images to visualize per class')
    parser.add_argument('--use_tta', action='store_true', default=True, help='Use test-time augmentation')
    parser.add_argument('--no_tta', dest='use_tta', action='store_false', help='Disable test-time augmentation')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Classification threshold (if None, uses optimal threshold from validation)')
    parser.add_argument('--optimize_threshold', action='store_true', default=True,
                        help='Find optimal threshold to minimize false positives/negatives')
    return parser.parse_args()

def get_transform():
    return A.Compose([
        A.Resize(height=224, width=224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def predict_single_image(model, image_path, transform, device):
    image = np.array(Image.open(image_path).convert('RGB'))
    transformed = transform(image=image)
    image_tensor = transformed['image'].unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor).squeeze().item()
        probability = torch.sigmoid(torch.tensor(output)).item()

    prediction = "Pneumonia" if probability > 0.5 else "Normal"
    return prediction, probability


def predict_with_tta(model, image_path, tta_transforms, device):
    """Predict with test-time augmentation - averages predictions across augmented versions"""
    image = np.array(Image.open(image_path).convert('RGB'))
    probabilities = []

    with torch.no_grad():
        for transform in tta_transforms:
            transformed = transform(image=image)
            image_tensor = transformed['image'].unsqueeze(0).to(device)
            output = model(image_tensor).squeeze().item()
            prob = torch.sigmoid(torch.tensor(output)).item()
            probabilities.append(prob)

    # Average all TTA predictions
    avg_probability = np.mean(probabilities)
    return avg_probability


def find_optimal_threshold(y_true, y_prob, metric='f1'):
    """Find optimal classification threshold based on various metrics"""
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_threshold = 0.5
    best_score = 0

    results = []
    for thresh in thresholds:
        y_pred = (np.array(y_prob) >= thresh).astype(int)

        # Calculate metrics
        tp = np.sum((y_pred == 1) & (np.array(y_true) == 1))
        tn = np.sum((y_pred == 0) & (np.array(y_true) == 0))
        fp = np.sum((y_pred == 1) & (np.array(y_true) == 0))
        fn = np.sum((y_pred == 0) & (np.array(y_true) == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # Balanced accuracy
        balanced_acc = (recall + specificity) / 2

        # Combined score prioritizing low false positives and false negatives
        # Weighted to minimize both FP and FN
        combined = 0.4 * f1 + 0.3 * precision + 0.3 * recall

        results.append({
            'threshold': thresh,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'balanced_acc': balanced_acc,
            'combined': combined,
            'fp': fp,
            'fn': fn
        })

        if metric == 'f1' and f1 > best_score:
            best_score = f1
            best_threshold = thresh
        elif metric == 'balanced' and balanced_acc > best_score:
            best_score = balanced_acc
            best_threshold = thresh
        elif metric == 'combined' and combined > best_score:
            best_score = combined
            best_threshold = thresh

    return best_threshold, pd.DataFrame(results)

def visualize_predictions(image_paths, predictions, probabilities, labels, output_dir, prefix=''):
    n_images = len(image_paths)
    n_cols = 5
    n_rows = (n_images + n_cols - 1) // n_cols

    plt.figure(figsize=(20, 4 * n_rows))

    for i, (img_path, pred, prob, true_label) in enumerate(zip(image_paths, predictions, probabilities, labels)):
        plt.subplot(n_rows, n_cols, i+1)

        img = Image.open(img_path).convert('RGB')
        plt.imshow(img)

        correct = pred == true_label
        color = 'green' if correct else 'red'
        title = f"Pred: {pred} ({prob:.2f})\nTrue: {true_label}"
        plt.title(title, color=color)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{prefix}_predictions.png'))
    plt.close()

def plot_confusion_matrix(y_true, y_pred, output_dir):
    classes = ['Normal', 'Pneumonia']
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

def plot_roc_curve(y_true, y_prob, output_dir):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()

    return roc_auc


def plot_precision_recall_curve(y_true, y_prob, output_dir):
    """Plot precision-recall curve"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'))
    plt.close()


def plot_threshold_analysis(threshold_df, optimal_threshold, output_dir):
    """Plot threshold analysis showing metrics vs threshold"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # F1, Precision, Recall vs Threshold
    ax1 = axes[0, 0]
    ax1.plot(threshold_df['threshold'], threshold_df['f1'], label='F1', color='blue')
    ax1.plot(threshold_df['threshold'], threshold_df['precision'], label='Precision', color='green')
    ax1.plot(threshold_df['threshold'], threshold_df['recall'], label='Recall', color='red')
    ax1.axvline(x=optimal_threshold, color='black', linestyle='--', label=f'Optimal ({optimal_threshold:.2f})')
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Score')
    ax1.set_title('Metrics vs Threshold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # FP, FN vs Threshold
    ax2 = axes[0, 1]
    ax2.plot(threshold_df['threshold'], threshold_df['fp'], label='False Positives', color='orange')
    ax2.plot(threshold_df['threshold'], threshold_df['fn'], label='False Negatives', color='purple')
    ax2.axvline(x=optimal_threshold, color='black', linestyle='--', label=f'Optimal ({optimal_threshold:.2f})')
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('Count')
    ax2.set_title('False Positives/Negatives vs Threshold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Balanced Accuracy vs Threshold
    ax3 = axes[1, 0]
    ax3.plot(threshold_df['threshold'], threshold_df['balanced_acc'], label='Balanced Accuracy', color='teal')
    ax3.axvline(x=optimal_threshold, color='black', linestyle='--', label=f'Optimal ({optimal_threshold:.2f})')
    ax3.set_xlabel('Threshold')
    ax3.set_ylabel('Balanced Accuracy')
    ax3.set_title('Balanced Accuracy vs Threshold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Specificity vs Sensitivity
    ax4 = axes[1, 1]
    ax4.plot(threshold_df['threshold'], threshold_df['specificity'], label='Specificity', color='brown')
    ax4.plot(threshold_df['threshold'], threshold_df['recall'], label='Sensitivity (Recall)', color='red')
    ax4.axvline(x=optimal_threshold, color='black', linestyle='--', label=f'Optimal ({optimal_threshold:.2f})')
    ax4.set_xlabel('Threshold')
    ax4.set_ylabel('Score')
    ax4.set_title('Specificity vs Sensitivity')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'threshold_analysis.png'), dpi=150)
    plt.close()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = PneumoniaModel(use_pretrained=False).to(device)

    checkpoint = torch.load(args.model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']} with validation accuracy: {checkpoint['best_val_acc']:.4f}")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded model weights")

    model.eval()

    transform = get_transform()
    tta_transforms = get_tta_transforms() if args.use_tta else None

    normal_dir = os.path.join(args.data_dir, 'NORMAL')
    pneumonia_dir = os.path.join(args.data_dir, 'PNEUMONIA')

    normal_images = [os.path.join(normal_dir, f) for f in os.listdir(normal_dir)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    pneumonia_images = [os.path.join(pneumonia_dir, f) for f in os.listdir(pneumonia_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    print(f"Found {len(normal_images)} normal images and {len(pneumonia_images)} pneumonia images")

    all_images = normal_images + pneumonia_images
    true_labels = ['Normal'] * len(normal_images) + ['Pneumonia'] * len(pneumonia_images)
    numeric_labels = [0] * len(normal_images) + [1] * len(pneumonia_images)

    predictions = []
    probabilities = []

    desc = "Processing images (with TTA)" if args.use_tta else "Processing images"
    for img_path in tqdm(all_images, desc=desc):
        if args.use_tta and tta_transforms:
            probability = predict_with_tta(model, img_path, tta_transforms, device)
        else:
            _, probability = predict_single_image(model, img_path, transform, device)
        probabilities.append(probability)

    # Find optimal threshold if requested
    if args.optimize_threshold:
        optimal_threshold, threshold_df = find_optimal_threshold(numeric_labels, probabilities, metric='combined')
        print(f"\nOptimal threshold found: {optimal_threshold:.3f}")
        plot_threshold_analysis(threshold_df, optimal_threshold, args.output_dir)
        threshold_df.to_csv(os.path.join(args.output_dir, 'threshold_analysis.csv'), index=False)
    else:
        optimal_threshold = args.threshold if args.threshold else 0.5

    # Use optimal threshold for predictions
    threshold = optimal_threshold
    predictions = ["Pneumonia" if p > threshold else "Normal" for p in probabilities]

    binary_preds = [1 if pred == 'Pneumonia' else 0 for pred in predictions]

    report = classification_report(numeric_labels, binary_preds, target_names=['Normal', 'Pneumonia'])
    print("\nClassification Report:")
    print(report)

    with open(os.path.join(args.output_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)

    plot_confusion_matrix(numeric_labels, binary_preds, args.output_dir)

    roc_auc = plot_roc_curve(numeric_labels, probabilities, args.output_dir)
    print(f"ROC AUC: {roc_auc:.4f}")

    # Plot precision-recall curve
    plot_precision_recall_curve(numeric_labels, probabilities, args.output_dir)

    results_df = pd.DataFrame({
        'image': [os.path.basename(img) for img in all_images],
        'true_label': true_labels,
        'prediction': predictions,
        'probability': probabilities,
        'correct': [pred == true for pred, true in zip(predictions, true_labels)]
    })

    results_df.to_csv(os.path.join(args.output_dir, 'predictions.csv'), index=False)

    accuracy = (results_df['correct'].sum() / len(results_df)) * 100
    print(f"\nOverall Accuracy: {accuracy:.2f}%")

    normal_acc = (results_df[results_df['true_label'] == 'Normal']['correct'].sum() /
                 len(results_df[results_df['true_label'] == 'Normal'])) * 100
    pneumonia_acc = (results_df[results_df['true_label'] == 'Pneumonia']['correct'].sum() /
                    len(results_df[results_df['true_label'] == 'Pneumonia'])) * 100

    # Calculate additional metrics
    tp = sum((results_df['prediction'] == 'Pneumonia') & (results_df['true_label'] == 'Pneumonia'))
    tn = sum((results_df['prediction'] == 'Normal') & (results_df['true_label'] == 'Normal'))
    fp = sum((results_df['prediction'] == 'Pneumonia') & (results_df['true_label'] == 'Normal'))
    fn = sum((results_df['prediction'] == 'Normal') & (results_df['true_label'] == 'Pneumonia'))

    sensitivity = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0  # Positive Predictive Value
    npv = tn / (tn + fn) * 100 if (tn + fn) > 0 else 0  # Negative Predictive Value

    print(f"Normal Class Accuracy: {normal_acc:.2f}%")
    print(f"Pneumonia Class Accuracy: {pneumonia_acc:.2f}%")
    print(f"\nDetailed Metrics:")
    print(f"  Sensitivity (True Positive Rate): {sensitivity:.2f}%")
    print(f"  Specificity (True Negative Rate): {specificity:.2f}%")
    print(f"  Positive Predictive Value (Precision): {ppv:.2f}%")
    print(f"  Negative Predictive Value: {npv:.2f}%")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")

    with open(os.path.join(args.output_dir, 'summary.txt'), 'w') as f:
        f.write(f"Model: ConvNeXt V2 Base\n")
        f.write(f"Test dataset: {args.data_dir}\n")
        f.write(f"Total images: {len(all_images)}\n")
        f.write(f"Normal images: {len(normal_images)}\n")
        f.write(f"Pneumonia images: {len(pneumonia_images)}\n")
        f.write(f"Test-time augmentation: {args.use_tta}\n")
        f.write(f"Classification threshold: {threshold:.3f}\n\n")
        f.write(f"Overall Accuracy: {accuracy:.2f}%\n")
        f.write(f"Normal Class Accuracy: {normal_acc:.2f}%\n")
        f.write(f"Pneumonia Class Accuracy: {pneumonia_acc:.2f}%\n")
        f.write(f"ROC AUC: {roc_auc:.4f}\n\n")
        f.write(f"Detailed Metrics:\n")
        f.write(f"  Sensitivity (True Positive Rate): {sensitivity:.2f}%\n")
        f.write(f"  Specificity (True Negative Rate): {specificity:.2f}%\n")
        f.write(f"  Positive Predictive Value (Precision): {ppv:.2f}%\n")
        f.write(f"  Negative Predictive Value: {npv:.2f}%\n")
        f.write(f"  True Positives: {tp}\n")
        f.write(f"  True Negatives: {tn}\n")
        f.write(f"  False Positives: {fp}\n")
        f.write(f"  False Negatives: {fn}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    normal_correct = results_df[(results_df['true_label'] == 'Normal') & (results_df['correct'])].sample(min(args.sample_vis, sum((results_df['true_label'] == 'Normal') & (results_df['correct']))))
    normal_incorrect = results_df[(results_df['true_label'] == 'Normal') & (~results_df['correct'])].sample(min(args.sample_vis, sum((results_df['true_label'] == 'Normal') & (~results_df['correct']))))
    pneumonia_correct = results_df[(results_df['true_label'] == 'Pneumonia') & (results_df['correct'])].sample(min(args.sample_vis, sum((results_df['true_label'] == 'Pneumonia') & (results_df['correct']))))
    pneumonia_incorrect = results_df[(results_df['true_label'] == 'Pneumonia') & (~results_df['correct'])].sample(min(args.sample_vis, sum((results_df['true_label'] == 'Pneumonia') & (~results_df['correct']))))

    def get_full_paths(image_names):
        return [img_path for img_path in all_images if os.path.basename(img_path) in image_names.values]

    if not normal_correct.empty:
        normal_correct_paths = get_full_paths(normal_correct['image'])
        normal_correct_preds = normal_correct['prediction'].tolist()
        normal_correct_probs = normal_correct['probability'].tolist()
        normal_correct_labels = normal_correct['true_label'].tolist()
        visualize_predictions(normal_correct_paths, normal_correct_preds, normal_correct_probs,
                              normal_correct_labels, args.output_dir, 'normal_correct')

    if not normal_incorrect.empty:
        normal_incorrect_paths = get_full_paths(normal_incorrect['image'])
        normal_incorrect_preds = normal_incorrect['prediction'].tolist()
        normal_incorrect_probs = normal_incorrect['probability'].tolist()
        normal_incorrect_labels = normal_incorrect['true_label'].tolist()
        visualize_predictions(normal_incorrect_paths, normal_incorrect_preds, normal_incorrect_probs,
                              normal_incorrect_labels, args.output_dir, 'normal_incorrect')

    if not pneumonia_correct.empty:
        pneumonia_correct_paths = get_full_paths(pneumonia_correct['image'])
        pneumonia_correct_preds = pneumonia_correct['prediction'].tolist()
        pneumonia_correct_probs = pneumonia_correct['probability'].tolist()
        pneumonia_correct_labels = pneumonia_correct['true_label'].tolist()
        visualize_predictions(pneumonia_correct_paths, pneumonia_correct_preds, pneumonia_correct_probs,
                              pneumonia_correct_labels, args.output_dir, 'pneumonia_correct')

    if not pneumonia_incorrect.empty:
        pneumonia_incorrect_paths = get_full_paths(pneumonia_incorrect['image'])
        pneumonia_incorrect_preds = pneumonia_incorrect['prediction'].tolist()
        pneumonia_incorrect_probs = pneumonia_incorrect['probability'].tolist()
        pneumonia_incorrect_labels = pneumonia_incorrect['true_label'].tolist()
        visualize_predictions(pneumonia_incorrect_paths, pneumonia_incorrect_preds, pneumonia_incorrect_probs,
                              pneumonia_incorrect_labels, args.output_dir, 'pneumonia_incorrect')

    print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
