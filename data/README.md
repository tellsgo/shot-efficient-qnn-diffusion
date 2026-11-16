# Data

The datasets in this directory are generated locally and are not tracked in
version control. From the repository root, run:

```bash
python scripts/prepare_data.py
```

This creates:

- `mnist28x28.npz`, downloaded as OpenML dataset `mnist_784`, version 1. It
  contains `images` with shape `(70000, 28, 28)` and dtype `float32`, and
  `target` with shape `(70000,)` and dtype `int16`.
- `digits8x8.npz`, generated from `sklearn.datasets.load_digits`. It contains
  `images` with shape `(1797, 8, 8)` and dtype `float32`, `data` with shape
  `(1797, 64)` and dtype `float32`, and `target` with shape `(1797,)` and dtype
  `int64`.

Use `--dataset mnist` or `--dataset digits` to generate only one dataset. The
script does not overwrite existing files unless `--force` is supplied.
