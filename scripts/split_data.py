import os
import shutil
import random
import argparse
from pathlib import Path
from tqdm import tqdm

# Default paths
DEFAULT_DATASET_PATH = "./dataset/Monkeypox Skin Image Dataset/"
OUTPUT_PATH = "./data"

def split_dataset(dataset_path, output_path, train_split=0.7, val_split=0.15, test_split=0.15):
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    
    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        return

    # Classes are subfolders in the dataset directory
    classes = [d.name for d in dataset_path.iterdir() if d.is_dir()]
    print(f"Found classes: {classes}")

    # Create output directories
    for split in ["train", "val", "test"]:
        for cls in classes:
            (output_path / split / cls).mkdir(parents=True, exist_ok=True)

    total_images = 0
    
    for cls in classes:
        cls_path = dataset_path / cls
        images = [f for f in cls_path.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
        random.shuffle(images)
        
        n_total = len(images)
        total_images += n_total
        
        n_train = int(n_total * train_split)
        n_val = int(n_total * val_split)
        # n_test = rest
        
        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train+n_val]
        test_imgs = images[n_train+n_val:]
        
        splits = {
            "train": train_imgs,
            "val": val_imgs,
            "test": test_imgs
        }
        
        print(f"Class {cls}: {n_total} images -> Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test: {len(test_imgs)}")
        
        for split, imgs in splits.items():
            for img in imgs:
                dest = output_path / split / cls / img.name
                shutil.copy2(img, dest) # Using copy2 to be safe
                
    print(f"Finished. Processed {total_images} images.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into train/val/test")
    parser.add_argument("--dataset_path", type=str, default=DEFAULT_DATASET_PATH, help="Path to original dataset")
    parser.add_argument("--output_path", type=str, default=OUTPUT_PATH, help="Path to output data directory")
    args = parser.parse_args()
    
    split_dataset(args.dataset_path, args.output_path)
