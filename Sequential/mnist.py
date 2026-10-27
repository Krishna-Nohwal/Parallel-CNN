"""Reader for MNIST IDX image and label files."""

import struct
from pathlib import Path

import numpy as np


def load_mnist(images_path: Path, labels_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load and validate an IDX image/label pair, scaling pixels to [0, 1]."""
    with images_path.open("rb") as images_file, labels_path.open("rb") as labels_file:
        image_magic, image_count, rows, columns = struct.unpack(">IIII", images_file.read(16))
        label_magic, label_count = struct.unpack(">II", labels_file.read(8))
        if image_magic != 2051 or label_magic != 2049:
            raise ValueError(f"Invalid MNIST IDX files: {images_path}, {labels_path}")
        if image_count != label_count or (rows, columns) != (28, 28):
            raise ValueError("MNIST image and label dimensions/counts do not match")

        images = np.frombuffer(images_file.read(), dtype=np.uint8)
        labels = np.frombuffer(labels_file.read(), dtype=np.uint8)
    expected_pixels = image_count * rows * columns
    if images.size != expected_pixels or labels.size != label_count:
        raise ValueError("MNIST IDX files are truncated")
    return images.reshape(image_count, rows, columns).astype(np.float32) / 255.0, labels
