import torch
import argparse
from src.utils.config import load_config
from src.data.dataset_builder import build_dataloaders
from src.models.efficientnet import EfficientNetB3
from src.evaluation.evaluate import evaluate_model

def main():
    parser = argparse.ArgumentParser(description="Evaluate Monkeypox Classifier")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--model_path", type=str, default="checkpoints/best_model.pth", help="Path to trained model")
    args = parser.parse_args()
    
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Data
    _, _, test_loader = build_dataloaders(config)
    
    # Load Model
    model = EfficientNetB3(num_classes=config['num_classes'])
    
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Loaded model from {args.model_path}")
    except FileNotFoundError:
        print(f"Error: Model checkpoint not found at {args.model_path}. Please train the model first.")
        return

    model = model.to(device)
    
    # Evaluate
    evaluate_model(model, test_loader, device, config['classes'])

if __name__ == "__main__":
    main()
