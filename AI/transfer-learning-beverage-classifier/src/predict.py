from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.config import Paths


def _load_metadata(metadata_path: Path) -> dict:
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _load_image(image_path: Path, image_size: tuple[int, int]) -> tf.Tensor:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = tf.keras.utils.load_img(image_path, target_size=image_size)
    arr = tf.keras.utils.img_to_array(img)
    return tf.expand_dims(arr, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict beverage class from an image.")
    parser.add_argument("--image", required=True, help="Path to an input image.")
    args = parser.parse_args()

    paths = Paths.from_project_root()
    model_path = paths.models_dir / "beverage_classifier.keras"
    metadata_path = paths.models_dir / "metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. Train first: python -m src.train"
        )

    metadata = _load_metadata(metadata_path)
    image_size = tuple(metadata["image_size"])
    class_names = list(metadata["class_names"])

    model = tf.keras.models.load_model(model_path)

    x = _load_image(Path(args.image), image_size=image_size)
    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))

    print("Prediction:")
    print(f"  class: {class_names[idx]}")
    print(f"  confidence: {float(probs[idx]):.4f}")
    print("  probabilities:")
    for name, p in zip(class_names, probs):
        print(f"    {name}: {float(p):.4f}")


if __name__ == "__main__":
    main()
