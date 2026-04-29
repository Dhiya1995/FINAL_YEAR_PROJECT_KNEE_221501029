"""
Script to augment training dataset with GAN-generated synthetic images
Copies synthetic images from synthetic_images/ to new-dataset/train/ folders
"""
import os
import shutil
from config import DATASET_DIR, SYNTHETIC_IMAGES_DIR

def augment_dataset_with_synthetic(minority_classes=[0, 1, 4], copy_all=False):
    """
    Copy synthetic images to training dataset
    
    Args:
        minority_classes: List of classes to augment
        copy_all: If True, copy all synthetic images. If False, only copy to reach target count
    """
    print("=" * 60)
    print("Augmenting Training Dataset with Synthetic Images")
    print("=" * 60)
    
    target_samples = 2000  # Target samples per class
    
    for class_idx in minority_classes:
        print(f"\nProcessing Grade {class_idx}...")
        
        # Source directory (synthetic images)
        synthetic_dir = os.path.join(SYNTHETIC_IMAGES_DIR, str(class_idx))
        if not os.path.exists(synthetic_dir):
            print(f"  ⚠️  No synthetic images found for Grade {class_idx}")
            continue
        
        # Destination directory (training dataset)
        train_dir = os.path.join(DATASET_DIR, 'train', str(class_idx))
        if not os.path.exists(train_dir):
            os.makedirs(train_dir, exist_ok=True)
        
        # Count existing training images
        existing_images = [f for f in os.listdir(train_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        current_count = len(existing_images)
        
        # Count synthetic images
        synthetic_images = [f for f in os.listdir(synthetic_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        synthetic_count = len(synthetic_images)
        
        print(f"  Current training images: {current_count}")
        print(f"  Available synthetic images: {synthetic_count}")
        
        if copy_all:
            # Copy all synthetic images
            to_copy = synthetic_images
            print(f"  Copying all {synthetic_count} synthetic images...")
        else:
            # Copy only what's needed to reach target
            needed = max(0, target_samples - current_count)
            to_copy = synthetic_images[:needed]
            print(f"  Copying {len(to_copy)} synthetic images to reach target...")
        
        # Copy images
        copied = 0
        skipped = 0
        errors = 0
        
        for img_file in to_copy:
            src = os.path.join(synthetic_dir, img_file)
            dst = os.path.join(train_dir, f'synthetic_{img_file}')
            
            if os.path.exists(dst):
                skipped += 1
                continue
            
            try:
                shutil.copy2(src, dst)
                copied += 1
            except Exception as e:
                errors += 1
                print(f"  ⚠️  Error copying {img_file}: {e}")
        
        print(f"  ✅ Copied {copied} images to training dataset")
        if skipped > 0:
            print(f"  ⏭️  Skipped {skipped} images (already exist)")
        if errors > 0:
            print(f"  ❌ Errors: {errors} images")
        print(f"  New total: {current_count + copied} images")
    
    print("\n" + "=" * 60)
    print("Dataset Augmentation Completed!")
    print("=" * 60)
    print("\nNext step: Retrain the classifier with augmented dataset:")
    print("  python train_model.py")

if __name__ == '__main__':
    # Augment dataset with synthetic images for classes 0, 1, and 4
    augment_dataset_with_synthetic(minority_classes=[0, 1, 4], copy_all=False)

