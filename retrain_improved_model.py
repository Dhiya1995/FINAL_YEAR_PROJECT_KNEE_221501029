
import os
import sys

def main():
    print("=" * 70)
    print("🔄 IMPROVED MODEL RETRAINING WORKFLOW")
    print("=" * 70)
    print("\nThis will:")
    print("1. ✅ Check dataset balance")
    print("2. ✅ Generate synthetic images for minority classes (optional)")
    print("3. ✅ Retrain classifier with ordinal loss")
    print("4. ✅ Evaluate new model performance")
    print("=" * 70)
    
    # Step 1: Check current dataset balance
    print("\n📊 STEP 1: Checking Dataset Balance...")
    print("-" * 70)
    os.system("python check_dataset_balance.py")
    
    # Step 2: Ask about GAN augmentation
    print("\n" + "=" * 70)
    print("💡 OPTIONAL: GAN Augmentation")
    print("=" * 70)
    print("GAN can generate synthetic images for minority classes.")
    print("This improves model performance but takes time (~1-2 hours).")
    print()
    
    response = input("Generate synthetic images with GAN? (y/n): ").strip().lower()
    
    if response == 'y':
        print("\n🎨 Generating synthetic images...")
        print("This will take 1-2 hours. Progress will be shown.")
        print("-" * 70)
        
        # Train GAN for each minority class
        for grade in [0, 1, 4]:  # Minority classes
            print(f"\n📦 Training GAN for Grade {grade}...")
            os.system(f"python training/train_gan.py --class_idx {grade} --num_epochs 50")
        
        # Augment dataset
        print("\n📂 Augmenting dataset with synthetic images...")
        os.system("python augment_dataset.py")
        
        # Check balance again
        print("\n📊 New Dataset Balance:")
        os.system("python check_dataset_balance.py")
    else:
        print("\n⏭️  Skipping GAN augmentation. Using current dataset.")
    
    # Step 3: Retrain classifier
    print("\n" + "=" * 70)
    print("🧠 STEP 2: Training Improved Classifier")
    print("=" * 70)
    print("Training with:")
    print("  - Ordinal Loss (better for KL grades)")
    print("  - Class weights (handles imbalance)")
    print("  - Early stopping (prevents overfitting)")
    print("  - 15 epochs (or until early stopping)")
    print()
    
    input("Press Enter to start training...")
    
    print("\n🚀 Starting training...")
    print("-" * 70)
    
    # Create a temporary training script that auto-answers 'y' for ordinal loss
    train_script = """
import sys
import os

# Simulate 'y' input for ordinal loss
class MockInput:
    def __init__(self):
        self.called = False
    
    def __call__(self, prompt):
        if not self.called and 'Ordinal Loss' in prompt:
            self.called = True
            print(prompt + "y")
            return 'y'
        return input(prompt)

# Replace input temporarily
original_input = __builtins__.input
__builtins__.input = MockInput()

# Import and run training
from train_model import *

# Restore original input
__builtins__.input = original_input
"""
    
    with open('_temp_train.py', 'w') as f:
        f.write(train_script)
    
    os.system("python _temp_train.py")
    
    # Clean up
    if os.path.exists('_temp_train.py'):
        os.remove('_temp_train.py')
    
    # Step 4: Evaluate new model
    print("\n" + "=" * 70)
    print("📈 STEP 3: Evaluating New Model")
    print("=" * 70)
    print("Testing on held-out test set...")
    print("-" * 70)
    
    os.system("python evaluate_model.py")
    
    # Final summary
    print("\n" + "=" * 70)
    print("✅ RETRAINING COMPLETE!")
    print("=" * 70)
    print("\nNew model saved to: models/best_kl_classifier.pth")
    print("\nNext steps:")
    print("1. Check test_confusion_matrix.png to see improvements")
    print("2. Run the web application: python app.py")
    print("3. Test with real X-ray images")
    print("=" * 70)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        sys.exit(1)

