"""Registry of the consolidated Fashion-MNIST experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Experiment:
    name: str
    objective: str
    output: str
    primary_metrics: tuple[str, ...]

    def to_dict(self) -> dict[str, str | list[str]]:
        result = asdict(self)
        result["primary_metrics"] = list(self.primary_metrics)
        return result


EXPERIMENTS = {
    "classification": Experiment(
        name="classification",
        objective="Predict one of ten garment classes",
        output="class probabilities",
        primary_metrics=("accuracy", "macro_f1", "per_class_recall"),
    ),
    "autoencoder": Experiment(
        name="autoencoder",
        objective="Compress and reconstruct garment images",
        output="reconstructed images",
        primary_metrics=("mse", "mae", "psnr"),
    ),
    "gan": Experiment(
        name="gan",
        objective="Generate garment-like images from random latent vectors",
        output="generated images",
        primary_metrics=("sample_diversity", "pixel_mean", "pixel_std"),
    ),
    "dcgan": Experiment(
        name="dcgan",
        objective="Preserve spatial structure during garment generation",
        output="generated images",
        primary_metrics=("sample_diversity", "pixel_mean", "pixel_std"),
    ),
}


def get_experiment(name: str) -> Experiment:
    try:
        return EXPERIMENTS[name]
    except KeyError as exc:
        choices = ", ".join(sorted(EXPERIMENTS))
        raise ValueError(f"Unknown experiment '{name}'. Choose from: {choices}") from exc
