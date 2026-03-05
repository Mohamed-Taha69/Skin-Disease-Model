import os
from torch.utils.data import DataLoader
from torchvision import datasets
from src.utils.transforms import get_transforms


def build_dataloaders(config):

    # قراءة إعدادات الداتا من ملف config
    data_cfg = config["data"]

    train_dir = data_cfg["train_dir"]
    val_dir = data_cfg["val_dir"]
    test_dir = data_cfg["test_dir"]

    img_size = data_cfg["img_size"]           # ← هنا مكان img_size الصحيح
    batch_size = config["train"]["batch_size"]
    num_workers = data_cfg.get("num_workers", 2)

    # التأكد من وجود المسارات
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training data not found at {train_dir}")
    if not os.path.exists(val_dir):
        raise FileNotFoundError(f"Validation data not found at {val_dir}")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test data not found at {test_dir}")

    # التحويلات (باستخدام إعدادات الـ augmentation من الكونفج لو موجودة)
    aug_cfg = config.get("aug", {})
    train_tf, test_tf = get_transforms(img_size, aug_cfg=aug_cfg)

    # تحميل الداتا
    train_dataset = datasets.ImageFolder(train_dir, transform=train_tf)
    val_dataset = datasets.ImageFolder(val_dir, transform=test_tf)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_tf)

    # الداتا لودر
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader
