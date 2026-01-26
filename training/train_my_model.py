"""Train the from-scratch NeuralNetwork and save a portable weights-only artifact."""

from __future__ import annotations

import json
from pathlib import Path

import idx2numpy
import numpy as np

from neural_network import NeuralNetwork

# ---------------------------
# Adjustable training settings
# ---------------------------
EPOCHS: int = 3
LAYOUT: list[int] = [25, 10]
LEARNING_RATE: float = 0.01

# ---------------------------
# Paths
# ---------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
MNIST_DIR = REPO_ROOT / "data" / "mnist"

TRAIN_IMAGE_FILENAME = MNIST_DIR / "train-images.idx3-ubyte"
TRAIN_LABEL_FILENAME = MNIST_DIR / "train-labels.idx1-ubyte"
TEST_IMAGE_FILENAME = MNIST_DIR / "t10k-images.idx3-ubyte"
TEST_LABEL_FILENAME = MNIST_DIR / "t10k-labels.idx1-ubyte"


def read_mnist_data():
    """
    Reads and formats MNIST data set for our use
    Returns training and testing input and labels
    """
    train_images = idx2numpy.convert_from_file(str(TRAIN_IMAGE_FILENAME))
    train_labels = idx2numpy.convert_from_file(str(TRAIN_LABEL_FILENAME))
    test_images = idx2numpy.convert_from_file(str(TEST_IMAGE_FILENAME))
    test_labels = idx2numpy.convert_from_file(str(TEST_LABEL_FILENAME))

    # Reformat and standardize
    x_train = train_images.reshape(60000, 784).astype(np.float32)
    mean = float(np.mean(x_train))
    stddev = float(np.std(x_train))
    if stddev == 0.0:
        stddev = 1.0
    x_train = (x_train - mean) / stddev

    x_test = test_images.reshape(10000, 784).astype(np.float32)
    x_test = (x_test - mean) / stddev

    # One-hot encoded output
    y_train = np.zeros((60000, 10), dtype=np.float32)
    y_test = np.zeros((10000, 10), dtype=np.float32)

    for i, y in enumerate(train_labels):
        y_train[i][int(y)] = 1.0
    for i, y in enumerate(test_labels):
        y_test[i][int(y)] = 1.0

    return x_train, y_train, x_test, y_test, mean, stddev



def main() -> None:
    print("Loading MNIST...")
    x_train, y_train, x_test, y_test, mean, stddev = read_mnist_data()
    print(f"Loaded MNIST: x_train={x_train.shape}, x_test={x_test.shape}")
    print(f"Preprocess stats: mean={mean:.6f}, stddev={stddev:.6f}")

    print(f"Training scratch model: layout={LAYOUT}, epochs={EPOCHS}, lr={LEARNING_RATE}")
    model = NeuralNetwork(LAYOUT, x_train, y_train, x_test, y_test, learning_rate=LEARNING_RATE)

    print("Starting training...")
    for epoch in range(EPOCHS):
        print(f"--- Epoch {epoch + 1}/{EPOCHS} ---")
        model.training_loop(1)
        print(f"Finished epoch {epoch + 1}/{EPOCHS}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Serializing trained model state...")
    state_path = MODELS_DIR / "scratch_model_state.json"
    state = model.state_dict()
    with state_path.open("w", encoding="utf-8") as f:
        json.dump(state, f)

    print("Saved:")
    print(f"  - {state_path}")


if __name__ == "__main__":
    main()
