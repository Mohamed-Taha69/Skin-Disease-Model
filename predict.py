import torch
import argparse
from src.utils.config import load_config
from src.inference.predict import Predictor

def main():
    parser = argparse.ArgumentParser(description="Predict Single Image")
    parser.add_argument("image_path", type=str, help="Path to image file")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--model_path", type=str, default="checkpoints/best_model.pth", help="Path to trained model")
    args = parser.parse_args()
    
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        predictor = Predictor(args.model_path, config, device)
        class_name, confidence = predictor.predict(args.image_path)
        print(f"Prediction: {class_name}")
        print(f"Confidence: {confidence:.4f}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
