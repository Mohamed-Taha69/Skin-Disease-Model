import os
from torch.utils.data import DataLoader
from torchvision import datasets
from src.utils.transforms import get_transforms

def build_dataloaders(config):
    """
    Builds DataLoaders for train, val, and test sets.
    """
    data_dir = "data" # Assumes data is in 'data' folder relative to execution
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")
    
    train_transform, test_transform = get_transforms(config['img_size'])
    
    # Check if data exists
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training data not found at {train_dir}. Please run scripts/split_data.py first.")

    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=test_transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
