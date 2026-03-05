import torch
import os
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        scheduler=None,
        save_dir="checkpoints",
        early_stopping_patience=None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = save_dir
        self.early_stopping_patience = early_stopping_patience

        os.makedirs(self.save_dir, exist_ok=True)

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc="Training")
        for i, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            pbar.set_postfix({
                'loss': running_loss / (i + 1),
                'acc': 100 * correct / total
            })

        return running_loss / len(self.train_loader), 100 * correct / total

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            for i, (images, labels) in enumerate(pbar):
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                pbar.set_postfix({
                    'loss': running_loss / (i + 1),
                    'acc': 100 * correct / total
                })

        return running_loss / len(self.val_loader), 100 * correct / total

    def save_checkpoint(self, filename="best_model.pth"):
        path = os.path.join(self.save_dir, filename)
        torch.save({
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict()
        }, path)
        print(f"Model saved to {path}")

    def train(self, epochs):
        best_acc = 0.0
        epochs_without_improve = 0

        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()

            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            if self.scheduler:
                self.scheduler.step()

            if val_acc > best_acc:
                best_acc = val_acc
                epochs_without_improve = 0
                self.save_checkpoint("best_model.pth")
            else:
                epochs_without_improve += 1

            if (
                self.early_stopping_patience is not None
                and epochs_without_improve >= self.early_stopping_patience
            ):
                print(
                    f"\nEarly stopping triggered after {epoch+1} epochs. "
                    f"Best Validation Accuracy: {best_acc:.2f}%"
                )
                break

        print(f"\nTraining complete. Best Validation Accuracy: {best_acc:.2f}%")