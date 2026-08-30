"""Flask inference service for image-classification checkpoints."""

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from flask import Flask, jsonify, request
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

try:
    from .model import get_model
except ImportError:  # Supports `python src/serve.py` as well as package imports.
    from model import get_model


IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ]
)
DEFAULT_CHECKPOINT_PATH = Path("/app/models/model.pt")


def _model_settings(checkpoint: Mapping[str, Any]) -> tuple[str, int]:
    """Read model settings embedded by training or supplied through the environment."""
    config = checkpoint.get("config", {})
    model_config = config.get("model", {}) if isinstance(config, Mapping) else {}
    architecture = model_config.get("architecture") or os.getenv("MODEL_ARCHITECTURE")
    num_classes = model_config.get("num_classes") or os.getenv("MODEL_NUM_CLASSES")
    if not architecture or num_classes is None:
        raise ValueError(
            "Checkpoint must include config.model architecture and num_classes, or set "
            "MODEL_ARCHITECTURE and MODEL_NUM_CLASSES."
        )
    return str(architecture), int(num_classes)


def load_model(checkpoint_path: str | Path) -> tuple[torch.nn.Module, torch.device]:
    """Load a trained model in evaluation mode from a checkpoint file."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model checkpoint does not exist: {path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Checkpoint must be a mapping containing model_state_dict.")

    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("Checkpoint does not contain a model_state_dict.")

    architecture, num_classes = _model_settings(checkpoint)
    model = get_model(architecture=architecture, num_classes=num_classes).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, device


def create_app(checkpoint_path: str | Path | None = None) -> Flask:
    """Create the service app and attempt to load its model once at startup."""
    app = Flask(__name__)
    path = Path(checkpoint_path or os.getenv("MODEL_PATH", DEFAULT_CHECKPOINT_PATH))
    app.config["MODEL_PATH"] = str(path)
    app.extensions["model"] = None
    app.extensions["model_device"] = None
    app.extensions["model_error"] = None

    try:
        model, device = load_model(path)
        app.extensions["model"] = model
        app.extensions["model_device"] = device
    except Exception as error:  # Keep health available when startup loading fails.
        app.extensions["model_error"] = str(error)

    @app.get("/health")
    def health() -> tuple[Any, int] | Any:
        if app.extensions["model"] is None:
            return jsonify(status="unhealthy", detail=app.extensions["model_error"]), 503
        return jsonify(status="healthy"), 200

    @app.post("/predict")
    def predict() -> tuple[Any, int] | Any:
        model = app.extensions["model"]
        device = app.extensions["model_device"]
        if model is None or device is None:
            return jsonify(error="Model is not loaded."), 503

        image_file = request.files.get("image") or request.files.get("file")
        if image_file is None or not image_file.filename:
            return jsonify(error="Submit an image as multipart field 'image'."), 400

        try:
            image = Image.open(image_file.stream).convert("RGB")
        except (UnidentifiedImageError, OSError):
            return jsonify(error="The uploaded file is not a valid image."), 400

        image_tensor = IMAGE_TRANSFORM(image).unsqueeze(0).to(device)
        with torch.no_grad():
            probabilities = torch.softmax(model(image_tensor), dim=1)[0].cpu().tolist()

        predicted_class = max(range(len(probabilities)), key=probabilities.__getitem__)
        return jsonify(
            probabilities=probabilities,
            predicted_class=predicted_class,
            predicted_probability=probabilities[predicted_class],
        )

    return app


app = create_app()


if __name__ == "__main__":
    app.run(
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8080")),
        debug=False,
    )
