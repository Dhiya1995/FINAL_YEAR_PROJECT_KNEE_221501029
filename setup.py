
import os
from database import init_db
from config import (
    UPLOAD_FOLDER, OUTPUT_FOLDER, MODEL_FOLDER, STATIC_FOLDER,
    DATASET_DIR
)

def setup_project():
    """Initialize project directories and database"""
    print("Setting up Knee OA Classification System...")
    
    # Create directories
    directories = [
        UPLOAD_FOLDER,
        OUTPUT_FOLDER,
        MODEL_FOLDER,
        STATIC_FOLDER,
        os.path.join(OUTPUT_FOLDER, 'gradcam'),
        os.path.join(STATIC_FOLDER, 'gradcam')
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    
    # Initialize database
    print("\nInitializing database...")
    init_db()
    
    # Check dataset
    if os.path.exists(DATASET_DIR):
        print(f"\nDataset found at: {DATASET_DIR}")
        # Count images
        total_images = 0
        for split in ['train', 'val', 'test']:
            split_dir = os.path.join(DATASET_DIR, split)
            if os.path.exists(split_dir):
                for class_idx in range(5):
                    class_dir = os.path.join(split_dir, str(class_idx))
                    if os.path.exists(class_dir):
                        count = len([f for f in os.listdir(class_dir) 
                                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                        total_images += count
                        if count > 0:
                            print(f"  {split}/class_{class_idx}: {count} images")
        print(f"Total images: {total_images}")
    else:
        print(f"\nWarning: Dataset directory not found: {DATASET_DIR}")
        print("Please ensure your dataset is in the 'new-dataset' folder")
    
    print("\n" + "=" * 60)
    print("Setup completed successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Set environment variables (SECRET_KEY, GEMINI_API_KEY, etc.)")
    print("2. Train the model: python train_model.py")
    print("3. Run the web app: python app.py")
    print("=" * 60)

if __name__ == '__main__':
    setup_project()

