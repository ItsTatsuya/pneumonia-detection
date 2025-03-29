import argparse
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from model import PneumoniaModel
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import pandas as pd
import seaborn as sns

def parse_args():
    parser = argparse.ArgumentParser(description='Predict pneumonia from chest X-ray images')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--data_dir', type=str, default='./chest_xray/test', help='Path to the test dataset')
    parser.add_argument('--output_dir', type=str, default='predictions', help='Output directory for saving results')
    parser.add_argument('--model_type', type=str, default='efficientnet_b4', help='Model type (resnet, efficientnet, efficientnet_b4)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for prediction')
    parser.add_argument('--sample_vis', type=int, default=10, help='Number of sample images to visualize per class')
    return parser.parse_args()

def get_transform():
    # Updated to use Albumentations for consistency with training
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

def visualize_predictions(image_paths, predictions, probabilities, labels, output_dir, prefix=''):
    # Create a figure with a grid of images
    n_images = len(image_paths)
    n_cols = 5
    n_rows = (n_images + n_cols - 1) // n_cols

    plt.figure(figsize=(20, 4 * n_rows))

    for i, (img_path, pred, prob, true_label) in enumerate(zip(image_paths, predictions, probabilities, labels)):
        plt.subplot(n_rows, n_cols, i+1)

        # Load image
        img = Image.open(img_path).convert('RGB')
        plt.imshow(img)

        # Create title with prediction info
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

def main():
    args = parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model = PneumoniaModel(use_pretrained=False, model_type=args.model_type).to(device)

    # Check if the model is a checkpoint or just weights
    checkpoint = torch.load(args.model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # This is a full checkpoint
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']} with validation accuracy: {checkpoint['best_val_acc']:.4f}")
    else:
        # This is just the model weights
        model.load_state_dict(checkpoint)
        print("Loaded model weights")

    model.eval()

    # Set up transform
    transform = get_transform()

    # Define test data directories
    normal_dir = os.path.join(args.data_dir, 'NORMAL')
    pneumonia_dir = os.path.join(args.data_dir, 'PNEUMONIA')

    # Get all image paths
    normal_images = [os.path.join(normal_dir, f) for f in os.listdir(normal_dir)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    pneumonia_images = [os.path.join(pneumonia_dir, f) for f in os.listdir(pneumonia_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    print(f"Found {len(normal_images)} normal images and {len(pneumonia_images)} pneumonia images")

    # Process all images
    all_images = normal_images + pneumonia_images
    true_labels = ['Normal'] * len(normal_images) + ['Pneumonia'] * len(pneumonia_images)
    numeric_labels = [0] * len(normal_images) + [1] * len(pneumonia_images)

    # Prepare results containers
    predictions = []
    probabilities = []

    # Process all images with a progress bar
    for img_path in tqdm(all_images, desc="Processing images"):
        prediction, probability = predict_single_image(model, img_path, transform, device)
        predictions.append(prediction)
        probabilities.append(probability)

    # Calculate metrics
    binary_preds = [1 if pred == 'Pneumonia' else 0 for pred in predictions]

    # Generate classification report
    report = classification_report(numeric_labels, binary_preds, target_names=['Normal', 'Pneumonia'])
    print("\nClassification Report:")
    print(report)

    # Save report to file
    with open(os.path.join(args.output_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)

    # Plot confusion matrix
    plot_confusion_matrix(numeric_labels, binary_preds, args.output_dir)

    # Plot ROC curve
    roc_auc = plot_roc_curve(numeric_labels, probabilities, args.output_dir)
    print(f"ROC AUC: {roc_auc:.4f}")

    # Save all predictions to CSV
    results_df = pd.DataFrame({
        'image': [os.path.basename(img) for img in all_images],
        'true_label': true_labels,
        'prediction': predictions,
        'probability': probabilities,
        'correct': [pred == true for pred, true in zip(predictions, true_labels)]
    })

    results_df.to_csv(os.path.join(args.output_dir, 'predictions.csv'), index=False)

    # Calculate accuracy
    accuracy = (results_df['correct'].sum() / len(results_df)) * 100
    print(f"\nOverall Accuracy: {accuracy:.2f}%")

    # Separate accuracies for each class
    normal_acc = (results_df[results_df['true_label'] == 'Normal']['correct'].sum() /
                 len(results_df[results_df['true_label'] == 'Normal'])) * 100
    pneumonia_acc = (results_df[results_df['true_label'] == 'Pneumonia']['correct'].sum() /
                    len(results_df[results_df['true_label'] == 'Pneumonia'])) * 100

    print(f"Normal Class Accuracy: {normal_acc:.2f}%")
    print(f"Pneumonia Class Accuracy: {pneumonia_acc:.2f}%")

    # Create summary file
    with open(os.path.join(args.output_dir, 'summary.txt'), 'w') as f:
        f.write(f"Model: {args.model_type}\n")
        f.write(f"Test dataset: {args.data_dir}\n")
        f.write(f"Total images: {len(all_images)}\n")
        f.write(f"Normal images: {len(normal_images)}\n")
        f.write(f"Pneumonia images: {len(pneumonia_images)}\n\n")
        f.write(f"Overall Accuracy: {accuracy:.2f}%\n")
        f.write(f"Normal Class Accuracy: {normal_acc:.2f}%\n")
        f.write(f"Pneumonia Class Accuracy: {pneumonia_acc:.2f}%\n")
        f.write(f"ROC AUC: {roc_auc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    # Visualize some sample predictions from each class
    # Sample correct and incorrect predictions for each class
    normal_correct = results_df[(results_df['true_label'] == 'Normal') & (results_df['correct'])].sample(min(args.sample_vis, sum((results_df['true_label'] == 'Normal') & (results_df['correct']))))
    normal_incorrect = results_df[(results_df['true_label'] == 'Normal') & (~results_df['correct'])].sample(min(args.sample_vis, sum((results_df['true_label'] == 'Normal') & (~results_df['correct']))))
    pneumonia_correct = results_df[(results_df['true_label'] == 'Pneumonia') & (results_df['correct'])].sample(min(args.sample_vis, sum((results_df['true_label'] == 'Pneumonia') & (results_df['correct']))))
    pneumonia_incorrect = results_df[(results_df['true_label'] == 'Pneumonia') & (~results_df['correct'])].sample(min(args.sample_vis, sum((results_df['true_label'] == 'Pneumonia') & (~results_df['correct']))))

    # Find the full paths for these samples
    def get_full_paths(image_names):
        return [img_path for img_path in all_images if os.path.basename(img_path) in image_names.values]

    # Visualize correct predictions
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
