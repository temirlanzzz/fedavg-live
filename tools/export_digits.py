"""Export scikit-learn's load_digits (UCI optical digits, 8x8, values 0-16) to data/digits.json.

Split: 500 held-out test images (stratified, 50 per class), the remaining 1,297 for training.
Each image is encoded as a 64-char string, pixel value v -> chr(ord('a') + v), to keep the file small.

    pip install scikit-learn
    python tools/export_digits.py
"""
import json
import os

import numpy as np
from sklearn.datasets import load_digits

rng = np.random.default_rng(7)
d = load_digits()
X, y = d.data.astype(int), d.target.astype(int)

test_idx = []
for c in range(10):
    idx = np.flatnonzero(y == c)
    test_idx.extend(rng.choice(idx, 50, replace=False).tolist())
test_mask = np.zeros(len(y), bool)
test_mask[test_idx] = True
train_idx = rng.permutation(np.flatnonzero(~test_mask))
test_idx = rng.permutation(np.flatnonzero(test_mask))


def enc(rows):
    return ["".join(chr(97 + int(v)) for v in r) for r in rows]


out = {
    "source": "sklearn.datasets.load_digits (UCI ML optical recognition of handwritten digits)",
    "shape": [8, 8],
    "maxValue": 16,
    "train": {"x": enc(X[train_idx]), "y": y[train_idx].tolist()},
    "test": {"x": enc(X[test_idx]), "y": y[test_idx].tolist()},
}
path = os.path.join(os.path.dirname(__file__), "..", "data", "digits.json")
os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, "w") as f:
    json.dump(out, f, separators=(",", ":"))
print(f"train {len(train_idx)}  test {len(test_idx)}  -> {os.path.normpath(path)}")
