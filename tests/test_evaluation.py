from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fashion_vision.evaluation import (
    classification_summary,
    generation_summary,
    reconstruction_summary,
)
from fashion_vision.experiments import EXPERIMENTS, get_experiment
from fashion_vision.models import build_autoencoder


class EvaluationTests(unittest.TestCase):
    def test_classification_summary(self) -> None:
        labels = np.array([0, 1, 1, 2])
        probabilities = np.array([
            [0.9, 0.1, 0.0],
            [0.1, 0.8, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
        ])
        report = classification_summary(labels, probabilities, ["top", "shoe", "bag"])
        self.assertEqual(report["accuracy"], 1.0)
        self.assertEqual(report["macro_f1"], 1.0)

    def test_reconstruction_summary(self) -> None:
        original = np.zeros((2, 4, 4, 1))
        reconstructed = np.full_like(original, 0.5)
        report = reconstruction_summary(original, reconstructed)
        self.assertAlmostEqual(report["mse"], 0.25)
        self.assertAlmostEqual(report["psnr"], 10 * np.log10(4))

    def test_generation_diversity_distinguishes_duplicates(self) -> None:
        collapsed = np.zeros((4, 4, 4, 1))
        varied = collapsed.copy()
        varied[1:] = np.arange(1, 4)[:, None, None, None]
        self.assertEqual(generation_summary(collapsed)["sample_diversity"], 0.0)
        self.assertGreater(generation_summary(varied)["sample_diversity"], 0.0)

    def test_experiment_registry(self) -> None:
        self.assertEqual(set(EXPERIMENTS), {"classification", "autoencoder", "gan", "dcgan"})
        self.assertEqual(get_experiment("dcgan").output, "generated images")
        with self.assertRaisesRegex(ValueError, "Unknown experiment"):
            get_experiment("unknown")

    def test_tensorflow_error_is_actionable_when_missing(self) -> None:
        try:
            import tensorflow  # noqa: F401
        except ImportError:
            with self.assertRaisesRegex(RuntimeError, "requirements-train"):
                build_autoencoder()


if __name__ == "__main__":
    unittest.main()
