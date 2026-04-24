import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os

from src.utils.config import load_config
from src.utils.seed import set_global_seed
from src.data.dataset_builder import build_dataloaders, build_kfold_dataloaders
from src.models.efficientnet import EfficientNetB3
from src.training.trainer import Trainer


def build_optimizer(model, train_cfg):
    lr = float(train_cfg["lr"])
    wd = float(train_cfg["weight_decay"])
    opt_name = train_cfg["optimizer"].lower()

    if opt_name == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "sgd":
        return optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")


def build_scheduler(optimizer, train_cfg):
    sched_name = train_cfg["scheduler"].lower()

    if sched_name == "none" or sched_name is None:
        return None
    elif sched_name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg["epochs"])
    elif sched_name == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    else:
        raise ValueError(f"Unknown scheduler: {sched_name}")


def main():
    parser = argparse.ArgumentParser(description="Monkeypox Classification Training Script")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config["data"]
    train_cfg = config["train"]
    project_cfg = config["project"]

    set_global_seed(config.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Loss (with optional class weights & label smoothing)
    class_weights = train_cfg.get("class_weights")
    label_smoothing = float(train_cfg.get("label_smoothing", 0.0))

    if class_weights is not None:
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    else:
        weight_tensor = None

    use_kfold = bool(train_cfg.get("kfold", {}).get("enabled", False))

    if use_kfold:
        print("Building k-fold dataloaders...")
        fold_loaders = build_kfold_dataloaders(config)
        n_splits = len(fold_loaders)
        fold_best_accs = []

        for fold_idx, (train_loader, val_loader) in enumerate(fold_loaders, start=1):
            print(f"\n========== Fold {fold_idx}/{n_splits} ==========")
            print("Building model...")
            model = EfficientNetB3(num_classes=data_cfg["num_classes"]).to(device)
            criterion = nn.CrossEntropyLoss(
                weight=weight_tensor,
                label_smoothing=label_smoothing if label_smoothing > 0.0 else 0.0,
            )
            optimizer = build_optimizer(model, train_cfg)
            scheduler = build_scheduler(optimizer, train_cfg)

            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                save_dir=os.path.join(project_cfg["checkpoints_dir"], f"fold_{fold_idx}"),
                early_stopping_patience=train_cfg.get("early_stopping_patience"),
            )

            print("Starting training...\n")
            best_acc = trainer.train(epochs=train_cfg["epochs"])
            fold_best_accs.append(best_acc)

        mean_acc = sum(fold_best_accs) / len(fold_best_accs)
        print("\n========== K-Fold Summary ==========")
        for i, acc in enumerate(fold_best_accs, start=1):
            print(f"Fold {i} best val acc: {acc:.2f}%")
        print(f"Mean best val acc: {mean_acc:.2f}%")
    else:
        print("Building dataloaders...")
        train_loader, val_loader = build_dataloaders(config)

        print("Building model...")
        model = EfficientNetB3(num_classes=data_cfg["num_classes"]).to(device)
        criterion = nn.CrossEntropyLoss(
            weight=weight_tensor,
            label_smoothing=label_smoothing if label_smoothing > 0.0 else 0.0,
        )
        optimizer = build_optimizer(model, train_cfg)
        scheduler = build_scheduler(optimizer, train_cfg)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            save_dir=project_cfg["checkpoints_dir"],
            early_stopping_patience=train_cfg.get("early_stopping_patience"),
        )

        print("Starting training...\n")
        trainer.train(epochs=train_cfg["epochs"])


if __name__ == "__main__":
    main()
