"""Train a simple PyTorch MNIST model and export a portable artifact."""

from __future__ import annotations

import json
import time
from pathlib import Path

import idx2numpy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

EPOCHS: int = 3
BATCH_SIZE: int = 256
LEARNING_RATE: float = 1e-3
HIDDEN: int = 128
USE_COMPILE: bool = False  # torch.compile can be faster, but keep False for max compatibility

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
MNIST_DIR = REPO_ROOT / "data" / "mnist"

TRAIN_IMAGE_FILENAME = MNIST_DIR / "train-images.idx3-ubyte"
TRAIN_LABEL_FILENAME = MNIST_DIR / "train-labels.idx1-ubyte"
TEST_IMAGE_FILENAME = MNIST_DIR / "t10k-images.idx3-ubyte"
TEST_LABEL_FILENAME = MNIST_DIR / "t10k-labels.idx1-ubyte"


def read_mnist_data():
    """Read MNIST IDX files, flatten to 784, standardize (train mean/std), return arrays + stats."""
    train_images = idx2numpy.convert_from_file(str(TRAIN_IMAGE_FILENAME))
    train_labels = idx2numpy.convert_from_file(str(TRAIN_LABEL_FILENAME))
    test_images = idx2numpy.convert_from_file(str(TEST_IMAGE_FILENAME))
    test_labels = idx2numpy.convert_from_file(str(TEST_LABEL_FILENAME))

    # Flatten
    x_train = train_images.reshape(60000, 784).astype(np.float32)
    x_test = test_images.reshape(10000, 784).astype(np.float32)

    # Standardize using train stats
    mean = float(np.mean(x_train))
    stddev = float(np.std(x_train))
    if stddev == 0.0:
        stddev = 1.0

    x_train = (x_train - mean) / stddev
    x_test = (x_test - mean) / stddev

    y_train = train_labels.astype(np.int64)
    y_test = test_labels.astype(np.int64)

    return x_train, y_train, x_test, y_test, mean, stddev


class MLP(nn.Module):
    """Tiny MLP for MNIST: 784 -> HIDDEN -> 10."""

    def __init__(self, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += yb.numel()
    return correct / max(total, 1)


def main() -> None:
    # Device: keep CPU by default for Cloud Run parity; allow CUDA locally if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading MNIST...")
    x_train, y_train, x_test, y_test, mean, stddev = read_mnist_data()
    print(f"Loaded MNIST: x_train={x_train.shape}, x_test={x_test.shape}")
    print(f"Preprocess stats: mean={mean:.6f}, stddev={stddev:.6f}")
    print(f"Device: {device}")

    # Torch tensors
    x_train_t = torch.from_numpy(x_train)
    y_train_t = torch.from_numpy(y_train)
    x_test_t = torch.from_numpy(x_test)
    y_test_t = torch.from_numpy(y_test)

    train_ds = TensorDataset(x_train_t, y_train_t)
    test_ds = TensorDataset(x_test_t, y_test_t)

    # DataLoaders: num_workers=0 for max portability (works on macOS + Cloud Run)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = MLP(hidden=HIDDEN).to(device)
    if USE_COMPILE and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore[attr-defined]

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Warm-up (helps stabilize first-iteration overhead)
    model.train()
    xb0, _ = next(iter(train_loader))
    xb0 = xb0.to(device)
    _ = model(xb0)

    print(f"Training: epochs={EPOCHS}, batch_size={BATCH_SIZE}, lr={LEARNING_RATE}, hidden={HIDDEN}")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.perf_counter()
        model.train()

        running_loss = 0.0
        n_seen = 0

        for step, (xb, yb) in enumerate(train_loader, start=1):
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            bs = yb.numel()
            running_loss += loss.item() * bs
            n_seen += bs

            # Minimal progress output
            if step % 100 == 0:
                avg_loss = running_loss / max(n_seen, 1)
                print(f"Epoch {epoch}/{EPOCHS} step {step}/{len(train_loader)} loss={avg_loss:.4f}")

        train_loss = running_loss / max(n_seen, 1)
        test_acc = evaluate(model, test_loader, device)
        dt = time.perf_counter() - t0
        print(f"Epoch {epoch}/{EPOCHS} done: train_loss={train_loss:.4f} test_acc={test_acc*100:.2f}% time={dt:.2f}s")

    # Save artifacts
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    state_path = MODELS_DIR / "torch_mlp_state.pt"
    torch.save(model.state_dict(), state_path)

    meta_path = MODELS_DIR / "torch_mlp_meta.json"
    meta = {
        "schema_version": 1,
        "model": "mlp",
        "input": 784,
        "hidden": HIDDEN,
        "output": 10,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "device_trained_on": str(device),
        "preprocess": {"mean": mean, "std": stddev, "expected_input": "flattened_784"},
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f)

    print("Saved:")
    print(f"  - {state_path}")
    print(f"  - {meta_path}")


if __name__ == "__main__":
    main()
