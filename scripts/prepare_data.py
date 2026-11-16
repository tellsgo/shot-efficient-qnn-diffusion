#!/usr/bin/env python3
"""Download and prepare the datasets used by the image experiments."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Mapping

import numpy as np
from sklearn.datasets import fetch_openml, load_digits


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "data"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save_npz(
    path: Path, arrays: Mapping[str, np.ndarray], *, force: bool
) -> None:
    if path.exists() and not force:
        raise FileExistsError(
            f"{path} already exists; pass --force to replace generated data"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        with temporary_path.open("wb") as handle:
            np.savez(handle, **arrays)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)

    summary = ", ".join(
        f"{name}={array.shape}/{array.dtype}" for name, array in arrays.items()
    )
    print(f"Wrote {path}")
    print(f"  {summary}")
    print(f"  sha256={_sha256(path)}")


def prepare_digits(output_dir: Path, *, force: bool) -> None:
    """Prepare scikit-learn's built-in 8x8 handwritten-digits data."""
    dataset = load_digits()
    arrays = {
        "images": np.asarray(dataset.images, dtype=np.float32),
        "data": np.asarray(dataset.data, dtype=np.float32),
        "target": np.asarray(dataset.target, dtype=np.int64),
    }
    if arrays["images"].shape != (1797, 8, 8):
        raise ValueError(f"Unexpected digits image shape: {arrays['images'].shape}")
    if arrays["data"].shape != (1797, 64):
        raise ValueError(f"Unexpected digits data shape: {arrays['data'].shape}")
    if arrays["target"].shape != (1797,):
        raise ValueError(f"Unexpected digits target shape: {arrays['target'].shape}")
    _save_npz(output_dir / "digits8x8.npz", arrays, force=force)


def prepare_mnist(output_dir: Path, *, force: bool) -> None:
    """Download OpenML MNIST version 1 and prepare the 28x28 arrays."""
    dataset = fetch_openml(
        "mnist_784",
        version=1,
        as_frame=False,
        parser="liac-arff",
        data_home=str(output_dir / ".cache"),
    )
    arrays = {
        "images": np.asarray(dataset.data, dtype=np.float32).reshape(-1, 28, 28),
        "target": np.asarray(dataset.target, dtype=np.int16),
    }
    if arrays["images"].shape != (70000, 28, 28):
        raise ValueError(f"Unexpected MNIST image shape: {arrays['images'].shape}")
    if arrays["target"].shape != (70000,):
        raise ValueError(f"Unexpected MNIST target shape: {arrays['target'].shape}")
    _save_npz(output_dir / "mnist28x28.npz", arrays, force=force)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the generated datasets used by QUDDPM experiments."
    )
    parser.add_argument(
        "--dataset",
        choices=("all", "mnist", "digits"),
        default="all",
        help="dataset to prepare (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing generated dataset",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if args.dataset in ("all", "digits"):
        prepare_digits(output_dir, force=args.force)
    if args.dataset in ("all", "mnist"):
        prepare_mnist(output_dir, force=args.force)


if __name__ == "__main__":
    main()
