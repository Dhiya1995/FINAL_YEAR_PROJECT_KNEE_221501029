import os
import sys
from models.classification_model import (
    KneeXRayDataset, get_transforms, KLGradeClassifier, train_model
)
from torch.utils.data import DataLoader
from config import (
    DATASET_DIR, MODEL_NAME, NUM_CLASSES, BATCH_SIZE, 
    NUM_EPOCHS, DEVICE, MODEL_FOLDER
)
import torch

if __name__ == '__main__':
    print("=" * 60)
    print("Knee Osteoarthritis Classification Model Training")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {DEVICE}")
    print(f"Dataset: {DATASET_DIR}")
    print("=" * 60)
    
    if not os.path.exists(DATASET_DIR):
        print(f"Error: Dataset directory not found: {DATASET_DIR}")
        sys.exit(1)
    
    print("\nLoading datasets...")
    train_dataset = KneeXRayDataset(DATASET_DIR, split='train', transform=get_transforms('train'))
    val_dataset = KneeXRayDataset(DATASET_DIR, split='val', transform=get_transforms('val'))
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    if len(train_dataset) == 0:
        print("Error: No training samples found!")
        sys.exit(1)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=2 if torch.cuda.is_available() else 0
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=2 if torch.cuda.is_available() else 0
    )
    
    print(f"\nInitializing {MODEL_NAME} model...")
    print("Freezing first 75-80% of backbone layers...")
    model = KLGradeClassifier(model_name=MODEL_NAME, num_classes=NUM_CLASSES, pretrained=True, freeze_backbone=True)
    model.to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print(f"\nStarting training for {NUM_EPOCHS} epochs...")
    print("=" * 60)
    
    use_ordinal = input("\nUse Ordinal Loss? (better for KL grades) [y/N]: ").strip().lower() == 'y'
    
    trained_model, train_losses, val_losses, val_accuracies = train_model(
        model, train_loader, val_loader, 
        num_epochs=NUM_EPOCHS, 
        device=DEVICE,
        use_ordinal_loss=use_ordinal
    )
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print(f"Best model saved to: {os.path.join(MODEL_FOLDER, 'best_kl_classifier.pth')}")
    print("=" * 60)
