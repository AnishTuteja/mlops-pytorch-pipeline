"""Train an image-classification model and emit JSON-lines metrics."""

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml

try:
    from .dataset import get_dataloaders
    from .model import get_model
except ImportError:  # Supports `python src/train.py` as well as package imports.
    from dataset import get_dataloaders
    from model import get_model


DEFAULT_CONFIG_PATH = Path(
    os.getenv(
        "TRAINING_CONFIG",
        Path(__file__).resolve().parents[1] / "configs" / "training_config.yaml",
    )
)


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a non-empty YAML training configuration."""
    path = Path(config_path)
    with path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError(f"Training configuration must be a YAML mapping: {path}")
    return config


def _require_mapping(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration requires a '{name}' mapping.")
    return value


def _metrics(total_loss: float, correct: int, total: int, split: str) -> tuple[float, float]:
    if total == 0:
        raise ValueError(f"The {split} data loader produced no samples.")
    return total_loss / total, correct / total


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Run one optimization epoch and return mean loss and accuracy."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        correct += outputs.argmax(dim=1).eq(targets).sum().item()
        total += targets.size(0)

    return _metrics(total_loss, correct, total, "training")


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate a model without recording gradients."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * inputs.size(0)
        correct += outputs.argmax(dim=1).eq(targets).sum().item()
        total += targets.size(0)

    return _metrics(total_loss, correct, total, "validation")


def _checkpoint_path(output_config: dict[str, Any]) -> Path:
    """Return the configured checkpoint destination.

    Prefer ``output.checkpoint_path``. ``checkpoint_dir`` plus ``model_name`` is
    also accepted for compatibility with container-oriented configurations.
    """
    if "checkpoint_path" in output_config:
        return Path(output_config["checkpoint_path"])
    if "checkpoint_dir" in output_config:
        return Path(output_config["checkpoint_dir"]) / output_config.get(
            "model_name", "model.pt"
        )
    raise ValueError(
        "Configuration requires output.checkpoint_path or output.checkpoint_dir."
    )


def _log(**event: Any) -> None:
    print(json.dumps(event, sort_keys=True), flush=True)


def run_training(config: dict[str, Any]) -> Path:
    """Train from a parsed configuration and return the best checkpoint path."""
    data_config = _require_mapping(config, "data")
    model_config = _require_mapping(config, "model")
    training_config = _require_mapping(config, "training")
    output_config = _require_mapping(config, "output")

    epochs = int(training_config["epochs"])
    patience = int(training_config["early_stopping_patience"])
    if epochs <= 0:
        raise ValueError("training.epochs must be greater than zero.")
    if patience < 0:
        raise ValueError("training.early_stopping_patience cannot be negative.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(
        architecture=model_config["architecture"],
        num_classes=model_config["num_classes"],
    ).to(device)
    train_loader, val_loader = get_dataloaders(
        data_dir=data_config["data_dir"],
        batch_size=int(training_config["batch_size"]),
        dataset=data_config["dataset"],
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
    )
    criterion = nn.CrossEntropyLoss()
    checkpoint_path = _checkpoint_path(output_config)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    patience_counter = 0
    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
        _log(
            event="epoch_complete",
            epoch=epoch,
            train_loss=round(train_loss, 6),
            train_accuracy=round(train_accuracy, 6),
            val_loss=round(val_loss, 6),
            val_accuracy=round(val_accuracy, 6),
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    "config": config,
                },
                checkpoint_path,
            )
            _log(event="checkpoint_saved", epoch=epoch, path=str(checkpoint_path))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                _log(event="early_stopping", epoch=epoch, patience=patience)
                break

    _log(
        event="training_complete",
        best_val_loss=round(best_val_loss, 6),
        checkpoint_path=str(checkpoint_path),
    )
    return checkpoint_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the YAML training configuration.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_training(load_config(args.config))


if __name__ == "__main__":
    main()