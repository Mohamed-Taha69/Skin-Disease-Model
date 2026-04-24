import os
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.datasets.folder import default_loader
from sklearn.model_selection import StratifiedKFold
from src.utils.transforms import get_transforms


class _PathDataset:
    def __init__(self, samples, targets, transform=None):
        self.samples = samples
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = default_loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def build_dataloaders(config):

    # قراءة إعدادات الداتا من ملف config
    data_cfg = config["data"]

    train_dir = data_cfg["train_dir"]
    val_dir = data_cfg["val_dir"]

    img_size = data_cfg["img_size"]           # ← هنا مكان img_size الصحيح
    batch_size = config["train"]["batch_size"]
    num_workers = data_cfg.get("num_workers", 2)

    # التأكد من وجود المسارات
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training data not found at {train_dir}")
    if not os.path.exists(val_dir):
        raise FileNotFoundError(f"Validation data not found at {val_dir}")

    # التحويلات (باستخدام إعدادات الـ augmentation من الكونفج لو موجودة)
    aug_cfg = config.get("aug", {})
    train_tf, test_tf = get_transforms(img_size, aug_cfg=aug_cfg)

    # تحميل الداتا
    train_dataset = datasets.ImageFolder(train_dir, transform=train_tf)
    val_dataset = datasets.ImageFolder(val_dir, transform=test_tf)

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
    return train_loader, val_loader


def build_kfold_dataloaders(config):
    """Build stratified k-fold train/val dataloaders."""
    data_cfg = config["data"]
    train_cfg = config["train"]
    kfold_cfg = train_cfg.get("kfold", {})

    train_dir = data_cfg["train_dir"]
    val_dir = data_cfg["val_dir"]

    img_size = data_cfg["img_size"]
    batch_size = train_cfg["batch_size"]
    num_workers = data_cfg.get("num_workers", 2)
    n_splits = int(kfold_cfg.get("n_splits", 5))
    shuffle = bool(kfold_cfg.get("shuffle", True))
    random_state = int(kfold_cfg.get("random_state", config.get("seed", 42)))

    if n_splits < 2:
        raise ValueError("k-fold requires n_splits >= 2")

    for path_value, split_name in (
        (train_dir, "Training"),
        (val_dir, "Validation"),
    ):
        if not os.path.exists(path_value):
            raise FileNotFoundError(f"{split_name} data not found at {path_value}")

    aug_cfg = config.get("aug", {})
    train_tf, test_tf = get_transforms(img_size, aug_cfg=aug_cfg)

    train_meta = datasets.ImageFolder(train_dir)
    val_meta = datasets.ImageFolder(val_dir)
    samples = list(train_meta.samples) + list(val_meta.samples)
    targets = list(train_meta.targets) + list(val_meta.targets)
    indices = list(range(len(targets)))

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state if shuffle else None,
    )

    fold_loaders = []
    for train_idx, val_idx in splitter.split(indices, targets):
        train_samples = [samples[i] for i in train_idx.tolist()]
        train_targets = [targets[i] for i in train_idx.tolist()]
        val_samples = [samples[i] for i in val_idx.tolist()]
        val_targets = [targets[i] for i in val_idx.tolist()]

        train_subset = _PathDataset(train_samples, train_targets, transform=train_tf)
        val_subset = _PathDataset(val_samples, val_targets, transform=test_tf)

        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        fold_loaders.append((train_loader, val_loader))

    return fold_loaders
