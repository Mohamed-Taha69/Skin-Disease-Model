import torch
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from tqdm import tqdm

def evaluate_model(model, test_loader, device, classes):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Calculate metrics
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    
    accuracy = np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    print(f"\nTest Accuracy: {accuracy*100:.2f}%")
    
    return accuracy, cm
