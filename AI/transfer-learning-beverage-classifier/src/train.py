from __future__ import annotations

import json

import matplotlib.pyplot as plt
import tensorflow as tf

from src.config import Paths, TrainConfig
from src.data import load_datasets
from src.model import build_model


def _ensure_dirs(paths: Paths) -> None:
    paths.models_dir.mkdir(parents=True, exist_ok=True)
    paths.reports_dir.mkdir(parents=True, exist_ok=True)


def _plot_history(history: tf.keras.callbacks.History, output_path) -> None:
    metrics = history.history
    epochs = range(1, len(metrics.get("loss", [])) + 1)

    plt.figure()
    plt.plot(list(epochs), metrics.get("loss", []), label="train_loss")
    plt.plot(list(epochs), metrics.get("val_loss", []), label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    paths = Paths.from_project_root()
    cfg = TrainConfig()
    _ensure_dirs(paths)

    train_ds, val_ds, info = load_datasets(
        train_dir=paths.train_dir,
        val_dir=paths.val_dir,
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
    )

    model = build_model(num_classes=info.num_classes, image_size=cfg.image_size)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        )
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate
    results = model.evaluate(val_ds, verbose=0)
    metrics = dict(zip(model.metrics_names, [float(x) for x in results]))

    # Save model and metadata
    model_path = paths.models_dir / "beverage_classifier.keras"
    model.save(model_path)

    metadata = {
        "image_size": list(cfg.image_size),
        "class_names": info.class_names,
        "metrics": metrics,
    }
    (paths.models_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Reports
    _plot_history(history, paths.reports_dir / "training_loss.png")
    (paths.reports_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Training completed.")
    print(f"Saved model: {model_path}")
    print(f"Saved metadata: {paths.models_dir / 'metadata.json'}")
    print(f"Saved reports: {paths.reports_dir}")


if __name__ == "__main__":
    main()
