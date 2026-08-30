from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


IMAGE_SIZE = 224
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


def _image_transform(training: bool) -> transforms.Compose:
    """Return the preprocessing pipeline used for train or validation images."""
    operations = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
    if training:
        operations.append(transforms.RandomHorizontalFlip())
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
        ]
    )
    return transforms.Compose(operations)


def get_dataloaders(
    data_dir: str,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation loaders from an ImageFolder-style dataset.

    ``data_dir`` must contain ``train`` and ``val`` directories. Each split must
    contain one directory per class, as required by ``torchvision.ImageFolder``.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    root = Path(data_dir).expanduser()
    train_dir = root / "train"
    val_dir = root / "val"
    missing_splits = [str(path) for path in (train_dir, val_dir) if not path.is_dir()]
    if missing_splits:
        raise FileNotFoundError(
            "Expected ImageFolder splits at " + ", ".join(missing_splits) + "."
        )

    train_dataset = datasets.ImageFolder(train_dir, transform=_image_transform(True))
    val_dataset = datasets.ImageFolder(val_dir, transform=_image_transform(False))
    if train_dataset.classes != val_dataset.classes:
        raise ValueError("Training and validation splits must contain the same classes.")

    loader_options = {
        "batch_size": batch_size,
        "num_workers": 0,
        "pin_memory": torch.cuda.is_available(),
    }
    return (
        DataLoader(train_dataset, shuffle=True, **loader_options),
        DataLoader(val_dataset, shuffle=False, **loader_options),
    )
