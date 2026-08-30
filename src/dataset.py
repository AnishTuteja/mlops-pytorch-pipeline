from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


_DATASET_SPECS = {
    "cifar10": {
        "loader": datasets.CIFAR10,
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
        "grayscale": False,
        "crop_size": 32,
    },
    "fashion_mnist": {
        "loader": datasets.FashionMNIST,
        "mean": (0.2860, 0.2860, 0.2860),
        "std": (0.3530, 0.3530, 0.3530),
        "grayscale": True,
        "crop_size": 28,
    },
}


def _dataset_spec(dataset: str) -> dict:
    try:
        return _DATASET_SPECS[dataset]
    except KeyError:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Supported: {sorted(_DATASET_SPECS)}."
        ) from None


def get_transform(dataset: str, training: bool) -> transforms.Compose:
    """Return the preprocessing pipeline used for train or eval images.

    Shared by dataset loading and serving so both stay in sync with whatever
    dataset a given checkpoint was actually trained on.
    """
    spec = _dataset_spec(dataset)
    operations = []
    if spec["grayscale"]:
        operations.append(transforms.Grayscale(num_output_channels=3))
    if training:
        operations.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(spec["crop_size"], padding=4),
            ]
        )
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=spec["mean"], std=spec["std"]),
        ]
    )
    return transforms.Compose(operations)


def get_dataloaders(
    data_dir: str,
    batch_size: int,
    dataset: str = "cifar10",
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation loaders for a supported torchvision dataset.

    Downloads the dataset into ``data_dir`` on first use.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    spec = _dataset_spec(dataset)
    loader_cls = spec["loader"]
    root = Path(data_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)

    train_dataset = loader_cls(
        root=root, train=True, download=True, transform=get_transform(dataset, training=True)
    )
    val_dataset = loader_cls(
        root=root, train=False, download=True, transform=get_transform(dataset, training=False)
    )

    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    return (
        DataLoader(train_dataset, shuffle=True, **loader_options),
        DataLoader(val_dataset, shuffle=False, **loader_options),
    )