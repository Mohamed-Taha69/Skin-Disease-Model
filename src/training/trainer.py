import torch
import os
from tqdm import tqdm
import numpy as np

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device, save_dir="checkpoints"):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.save_dir = save_dir
        
        os.makedirs(self.save_dir, exist_ok=True)
        
    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': running_loss/total, 'acc': 100 * correct / total})
            
        return running_loss / len(self.train_loader), 100 * correct / total
        
    def validate(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                pbar.set_postfix({'loss': running_loss/total, 'acc': 100 * correct / total})
                
        return running_loss / len(self.val_loader), 100 * correct / total
        
    def save_checkpoint(self, filename="best_model.pth"):
        path = os.path.join(self.save_dir, filename)
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
        
    def train(self, epochs):
        best_acc = 0.0
        
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            if val_acc > best_acc:
                best_acc = val_acc
                self.save_checkpoint()
                
        print(f"Training complete. Best Validation Accuracy: {best_acc:.2f}%")
