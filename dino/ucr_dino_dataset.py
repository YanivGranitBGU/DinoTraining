import os
from typing import List, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from aeon.datasets import load_classification


class MultiUCRDinoDataset(Dataset):
    """Combine multiple UCR multivariate datasets into a single DINO-ready Dataset.

    Each *channel* of each time-series sample is treated as its own 1D
    time series. Concretely, for an array X with shape
    (n_samples, n_channels, n_timepoints), this dataset yields
    n_samples * n_channels items. Each item is a 3-channel "image" of
    shape (1, n_timepoints, 3), so it can be processed by the existing
    DINO image augmentations.
    """

    def __init__(self, root_path: str, transform=None, split: str = "train") -> None:
        """Args:
            root_path: Directory containing one subfolder per UCR dataset
                (e.g. "./data/multivariate").
            transform: Transform to apply (should be DataAugmentationDINO).
            split: Which split to load from each dataset ("train" or "test").
        """
        self.root_path = root_path
        self.transform = transform
        self.split = split
        # Fixed deterministic base geometry before DINO augmentations.
        self._base_height = 256
        self._base_width = 256
        # Adaptive delay-embedding window heuristic parameters.
        self._embed_ratio = 0.6
        self._embed_lmin = 48
        self._embed_lmax = 192

        self._datasets: List[str] = []
        self._data: List[np.ndarray] = []
        # Per-dataset, per-channel normalization stats (min/max over all samples & time)
        self._channel_mins: List[np.ndarray] = []  # shape (n_channels,)
        self._channel_maxs: List[np.ndarray] = []  # shape (n_channels,)
        # (dataset_idx, sample_idx, channel_idx)
        self._index_map: List[Tuple[int, int, int]] = []

        self._load_all_datasets()

    def _load_all_datasets(self) -> None:
        if not os.path.isdir(self.root_path):
            raise ValueError(f"root_path does not exist or is not a directory: {self.root_path}")

        dataset_folders = [
            f
            for f in os.listdir(self.root_path)
            if os.path.isdir(os.path.join(self.root_path, f))
        ]
        dataset_folders.sort()

        for name in dataset_folders:
            try:
                X, y = load_classification(name, extract_path=self.root_path, split=self.split)
            except Exception:
                # If a dataset cannot be loaded for some reason, just skip it.
                continue

            # Expect X to be array-like with shape (n_samples, n_channels, n_timepoints)
            X = np.asarray(X)
            if X.ndim < 2:
                # Not a valid multivariate time-series array, skip.
                continue

            dataset_idx = len(self._data)
            self._datasets.append(name)
            self._data.append(X)

            # Compute per-channel min/max across all samples & time for this dataset
            # X shape: (n_samples, n_channels, n_timepoints)
            mins = X.min(axis=(0, 2)).astype(np.float32)  # (n_channels,)
            maxs = X.max(axis=(0, 2)).astype(np.float32)  # (n_channels,)
            self._channel_mins.append(mins)
            self._channel_maxs.append(maxs)

            n_samples, n_channels = X.shape[0], X.shape[1]
            # Create one index entry per (sample, channel) pair so that each
            # channel becomes its own time series for DINO.
            for i in range(n_samples):
                for c in range(n_channels):
                    self._index_map.append((dataset_idx, i, c))

        if not self._index_map:
            raise RuntimeError(f"No usable datasets found under {self.root_path}")

    def __len__(self) -> int:
        return len(self._index_map)

    def _delay_embed_2d(self, x: np.ndarray, height: int, width: int) -> np.ndarray:
        """Convert a 1D time-series to a 2D delay-embedding matrix.

        Each output column is a sliding window over the 1D signal.
        The output shape is always (height, width).
        """
        if x.ndim != 1:
            raise ValueError(f"Expected 1D time-series for delay embedding, got shape {x.shape}")

        if height <= 0 or width <= 0:
            raise ValueError(f"height and width must be positive, got {(height, width)}")

        if x.size == 0:
            x = np.zeros(height, dtype=np.float32)

        length = int(x.size)
        raw_l = int(math.floor(self._embed_ratio * length))
        l = min(self._embed_lmax, max(self._embed_lmin, raw_l))
        # Never exceed the available signal length.
        l = max(1, min(l, length))

        max_start = max(0, length - l)
        delay = (max_start / float(width - 1)) if width > 1 else 0.0

        out = np.empty((height, width), dtype=np.float32)
        for col in range(width):
            start = int(round(col * delay))
            if start > max_start:
                start = max_start
            window = x[start:start + l]
            if l == height:
                out[:, col] = window
            elif l == 1:
                out[:, col] = window[0]
            else:
                src = np.linspace(0.0, 1.0, num=l, dtype=np.float32)
                dst = np.linspace(0.0, 1.0, num=height, dtype=np.float32)
                out[:, col] = np.interp(dst, src, window).astype(np.float32)
        return out

    def _to_pil_image(self, x: np.ndarray, vmin: float, vmax: float) -> Image.Image:
        """Convert a 1D array into a delay-embedded 3-channel PIL image.

        Normalization is done per-dataset & per-channel using precomputed
        vmin/vmax, so samples from the same dataset/channel share the same scale.
        """
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 1:
            x = np.squeeze(x)
            if x.ndim != 1:
                raise ValueError(f"Expected 1D time-series, got shape {x.shape}")

        # Normalize using dataset-level channel stats to [0, 1]
        if vmax > vmin:
            x = (x - vmin) / (vmax - vmin)
        else:
            x = np.zeros_like(x)

        x = np.clip(x, 0.0, 1.0)

        # Build a deterministic 2D delay-embedding matrix.
        x2d = self._delay_embed_2d(x, self._base_height, self._base_width)

        # Map to [0, 255] uint8 and create a 3-channel image
        img = (x2d * 255.0).astype(np.uint8)   # (H, W)
        img = np.stack([img] * 3, axis=-1)     # (H, W, 3)
        return Image.fromarray(img)

    def __getitem__(self, index: int):
        dataset_idx, sample_idx, channel_idx = self._index_map[index]
        X_ds = self._data[dataset_idx]
        # X_ds has shape (n_samples, n_channels, n_timepoints)
        # Select a single channel: shape (n_timepoints,)
        x = X_ds[sample_idx, channel_idx]

        # Use per-dataset, per-channel min/max for normalization
        mins = self._channel_mins[dataset_idx]
        maxs = self._channel_maxs[dataset_idx]
        vmin = float(mins[channel_idx])
        vmax = float(maxs[channel_idx])

        img = self._to_pil_image(x, vmin, vmax)
        if self.transform is not None:
            crops = self.transform(img)
        else:
            crops = img

        # DINO ignores labels, so we can safely return a dummy 0 label.
        return crops, 0
