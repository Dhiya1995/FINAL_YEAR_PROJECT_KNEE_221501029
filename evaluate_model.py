
import os
import sys
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from models.classification_model import (
    KneeXRayDataset, get_transforms, KLGradeClassifier, load_trained_model
)
from torch.utils.data import DataLoader
from config import (
    DATASET_DIR, MODEL_NAME, NUM_CLASSES, BATCH_SIZE, 
    DEVICE, MODEL_FOLDER
)

def evaluate_model():
    """Evaluate the trained model on test set"""
    print("=" * 60)
    print("Model Evaluation on Test Set")
    print("=" * 60)
    
    # Load test dataset
    print("\nLoading test dataset...")
    test_dataset = KneeXRayDataset(DATASET_DIR, split='test', transform=get_transforms('val'))
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=2 if torch.cuda.is_available() else 0
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Load trained model
    print("\nLoading trained model...")
    model_path = os.path.join(MODEL_FOLDER, 'best_kl_classifier.pth')
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first using: python train_model.py")
        sys.exit(1)
    
    model = load_trained_model(model_path, model_name=MODEL_NAME, num_classes=NUM_CLASSES)
    model.to(DEVICE)
    model.eval()
    
    # Evaluate
    print("\nEvaluating on test set...")
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='weighted', zero_division=0
    )
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        all_labels, all_predictions, average=None, zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Print results
    print("\n" + "=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"  F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    
    print(f"\nPer-Class Metrics:")
    print(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 60)
    for i in range(NUM_CLASSES):
        print(f"Grade {i:<6} {precision_per_class[i]:<12.4f} {recall_per_class[i]:<12.4f} "
              f"{f1_per_class[i]:<12.4f} {support[i]:<10}")
    
    # Plot confusion matrix
    print("\nGenerating confusion matrix...")
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[f'Grade {i}' for i in range(NUM_CLASSES)],
                yticklabels=[f'Grade {i}' for i in range(NUM_CLASSES)])
    plt.title('Confusion Matrix - Test Set', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    cm_path = os.path.join(MODEL_FOLDER, 'test_confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to: {cm_path}")
    plt.close()
    
    print("\n" + "=" * 60)
    print("Evaluation completed!")
    print("=" * 60)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }

if __name__ == '__main__':
    evaluate_model()

