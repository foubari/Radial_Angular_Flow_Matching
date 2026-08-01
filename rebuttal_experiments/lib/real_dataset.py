"""Real tabular dataset with CHRONOLOGICAL splits, compatible with rafm.Trainer/Sampler.

Loads a saved (N, d) float tensor and exposes the BaseDataset interface used by the
pipeline: get_train_data / get_val_data / get_test_data / sample_train / dim / name.
Splits are chronological (train = earliest, test = latest) to avoid look-ahead leakage.
No analytic oracle source (real data) — use gaussian / radial_empirical only.
"""
from pathlib import Path
import torch


class RealTabularDataset:
    def __init__(self, pt_path, name, train_frac=0.6, val_frac=0.2, split="chrono", split_seed=0):
        """split: 'chrono' (train=earliest, realistic, no leakage) or 'random' (iid shuffle,
        approximately stationary — use ONLY as a diagnostic ablation)."""
        data = torch.load(pt_path, map_location="cpu")
        if not torch.is_tensor(data):
            data = torch.tensor(data, dtype=torch.float32)
        self._data = data.float()
        self.dim = data.shape[1]
        self.name = name
        self.split = split
        n = data.shape[0]
        n_tr = int(n * train_frac); n_va = int(n * val_frac)
        if split == "random":
            g = torch.Generator().manual_seed(split_seed)
            perm = torch.randperm(n, generator=g)
        else:
            perm = torch.arange(n)
        self._train_idx = perm[:n_tr]
        self._val_idx = perm[n_tr:n_tr + n_va]
        self._test_idx = perm[n_tr + n_va:]
        self.A = None  # no oracle

    def get_train_data(self): return self._data[self._train_idx]
    def get_val_data(self): return self._data[self._val_idx]
    def get_test_data(self): return self._data[self._test_idx]

    def sample_train(self, k):
        idx = self._train_idx[torch.randint(len(self._train_idx), (k,))]
        return self._data[idx]

    @property
    def n_train(self): return len(self._train_idx)
    @property
    def n_test(self): return len(self._test_idx)
