from torchvision import transforms


def get_transforms(img_size=224, aug_cfg=None):
    """
    Get training and testing transformations.

    If aug_cfg is provided (from the YAML config under 'aug'),
    it will be used to configure the training augmentations.
    """
    aug_cfg = aug_cfg or {}

    hflip_p = float(aug_cfg.get("hflip", 0.5))
    vflip_p = float(aug_cfg.get("vflip", 0.0))
    rotate_deg = float(aug_cfg.get("rotate", 15))
    cj = aug_cfg.get("color_jitter", [0.2, 0.2, 0.1, 0.1])
    random_crop = bool(aug_cfg.get("random_crop", False))
    random_erasing_p = float(aug_cfg.get("random_erasing", 0.0))

    color_jitter = transforms.ColorJitter(
        brightness=cj[0],
        contrast=cj[1],
        saturation=cj[2],
        hue=cj[3],
    )

    train_tf_list = [
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=hflip_p),
    ]

    if vflip_p > 0:
        train_tf_list.append(transforms.RandomVerticalFlip(p=vflip_p))

    if random_crop:
        train_tf_list.append(
            transforms.RandomResizedCrop(
                img_size,
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1),
            )
        )
    else:
        train_tf_list.append(transforms.RandomRotation(rotate_deg))

    train_tf_list.extend(
        [
            color_jitter,
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    if random_erasing_p > 0:
        train_tf_list.append(
            transforms.RandomErasing(
                p=random_erasing_p,
                scale=(0.02, 0.2),
                ratio=(0.3, 3.3),
                value="random",
            )
        )

    train_transforms = transforms.Compose(train_tf_list)

    test_transforms = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    return train_transforms, test_transforms
