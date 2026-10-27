"""Train and evaluate the sequential CNN on MNIST."""

import argparse
import time
from pathlib import Path

import numpy as np

from cnn import SequentialCNN
from mnist import load_mnist


THRESHOLD = 1e-2


def train(model: SequentialCNN, images: np.ndarray, labels: np.ndarray, epochs: int) -> None:
    batch_size = 1
    batches = len(images)
    progress_step = max(1, batches // 20)
    print(f"Training images: {len(images)} | batch size: {batch_size} | batches per epoch: {batches}", flush=True)
    training_start = time.perf_counter()
    for epoch in range(epochs):
        error = 0.0
        start = time.perf_counter()
        for i, (image, label) in enumerate(zip(images, labels)):
            output, cache = model.forward(image)
            residual = -output.copy()
            residual[int(label)] += 1.0
            error += float(np.linalg.norm(residual))
            model.backprop(int(label), cache)
            done = i + 1
            if done % progress_step == 0 or done == batches:
                print(f"Epoch {epoch + 1}/{epochs}: {done}/{batches} batches ({done / batches:.0%})", flush=True)

        mean_error = error / len(images)
        elapsed = time.perf_counter() - start
        print(f"epoch: {epoch + 1}, error: {mean_error:e}, time_on_cpu: {elapsed:.6f}", flush=True)
        if mean_error < THRESHOLD:
            print("Training complete, error less than threshold\n", flush=True)
            break
    print(f"\nTime - {time.perf_counter() - training_start:.6f}", flush=True)


def evaluate(model: SequentialCNN, images: np.ndarray, labels: np.ndarray) -> None:
    errors = sum(model.classify(image) != int(label) for image, label in zip(images, labels))
    rate = errors / len(labels) * 100.0
    print(f"Error Rate: {rate:.2f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the sequential MNIST CNN.")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")

    train_images, train_labels = load_mnist(args.data_dir / "train-images.idx3-ubyte", args.data_dir / "train-labels.idx1-ubyte")
    test_images, test_labels = load_mnist(args.data_dir / "t10k-images.idx3-ubyte", args.data_dir / "t10k-labels.idx1-ubyte")
    print(f"Dataset: {len(train_images)} training images, {len(test_images)} test images")
    model = SequentialCNN(seed=args.seed)
    train(model, train_images, train_labels, args.epochs)
    evaluate(model, test_images, test_labels)
    for name, elapsed_ms in model.times.items():
        print(f"Total {name.title()} Time: {elapsed_ms:.3f} ms")


if __name__ == "__main__":
    main()
