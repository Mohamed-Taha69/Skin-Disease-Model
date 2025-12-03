import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from src.utils.config import load_config
from src.utils.seed import set_global_seed
from src.data.dataset_builder import build_dataloaders
from src.models.efficientnet import EfficientNetB3
from src.training.trainer import Trainer

def main():
    parser = argparse.ArgumentParser(description="Train Monkeypox Classifier")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    args = parser.parse_args()
    
    # 1. Load Config & Set Seed
    config = load_config(args.config)
    set_global_seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. Build DataLoaders
    print("Building DataLoaders...")
    train_loader, val_loader, test_loader = build_dataloaders(config)
    
    # 3. Build Model
    print("Building Model...")
    model = EfficientNetB3(num_classes=config['num_classes'])
    model = model.to(device)
    
    # 4. Define Optimizer & Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=float(config['lr']))
    
    # 5. Initialize Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device
    )
    
    # 6. Start Training
    print("Starting Training...")
    trainer.train(epochs=config['epochs'])

if __name__ == "__main__":
    main()
