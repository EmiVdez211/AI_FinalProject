from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    root: Path
    data_dir: Path
    train_dir: Path
    val_dir: Path
    models_dir: Path
    reports_dir: Path

    @staticmethod
    def from_project_root() -> "Paths":
        root = Path(__file__).parent.parent
        data_dir = root / "data"
        return Paths(
            root=root,
            data_dir=data_dir,
            train_dir=data_dir / "train",
            val_dir=data_dir / "val",
            models_dir=root / "models",
            reports_dir=root / "reports",
        )


@dataclass(frozen=True)
class TrainConfig:
    image_size: tuple[int, int] = (224, 224)
    batch_size: int = 16
    epochs: int = 8
    learning_rate: float = 1e-4
    seed: int = 42
