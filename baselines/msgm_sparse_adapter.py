"""MSGM-SPARSE adapter — same MSGM sliced-score-matching baseline as baselines/msgm_adapter.py, but the
generator is the authors' SPARSE multiplicative-noise SDE (`MSGMsde(denseTensor=False)`: COO tensor, 2n
non-zeros, L_G=1/2 I) from sdeflow-light, instead of the DENSE (d,d,d) `multiplicativeNoise` from
18727_Multiplicative_Diffusion_code. EVERYTHING ELSE is identical to the dense adapter: same drift MLP
(vendored verbatim from 18727 NN.py), same lr/steps/batch/nfe, same eval. This is the sparse variant used
for the DC-AE / AudioMNIST runs (where the dense generator is memory-infeasible), applied here to the
tabular datasets so there is a consistent MSGM-sparse row across every dataset.

Sparse generator provenance: github.com/vressegu/sdeflow-light @ 590ec4b417a3cb4a136d4d80d37129fb52f6a241
(vendored UNMODIFIED under baselines/_sdeflow/, MIT). The dense MSGM baseline uses a different codebase
(18727) — same model/authors, noted as a caveat; do not treat dense and sparse as bit-identical stacks.
"""
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

_SDEFLOW = Path(__file__).parent / "_sdeflow"
if str(_SDEFLOW) not in sys.path:
    sys.path.insert(0, str(_SDEFLOW))   # sdeflow SDEs.py / sde_scheme.py + vendored 18727 MLP (msgm_nn.py)


class _MLPDrift(nn.Module):
    """Wrap the (unchanged) MSGM MLP so sdeflow's PluginReverseSDE can call it as a(x, t): coerce whatever
    time shape the SSM passes into the (batch, 1) the MLP expects. No change to the network itself."""
    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp

    def forward(self, x, t):
        n = x.shape[0]
        if torch.is_tensor(t):
            tt = t.reshape(-1).float()
            tt = (tt[:n] if tt.numel() >= n else tt.expand(n)).reshape(n, 1)
        else:
            tt = torch.full((n, 1), float(t), device=x.device)
        return self.mlp(x, tt)


class MSGMSparseAdapter:
    """Adapter for the SPARSE Multiplicative Diffusion baseline. Same interface (build/train/sample) and the
    same training data, architecture, and seeds as the FM methods and the dense MSGM adapter."""

    def __init__(self, train_data: torch.Tensor, cfg: dict, device: str = "cpu"):
        self.train_data = train_data
        self.cfg = cfg
        self.device = device
        self.dim = train_data.shape[1]
        self._sde = None
        self._model = None
        self._gen_sde = None

    def build(self) -> None:
        from SDEs import MSGMsde           # sdeflow (vendored); sparse via denseTensor=False
        from msgm_nn import MLP            # verbatim 18727 MLP — identical drift to the dense adapter

        self._sde = MSGMsde(
            self.train_data.to(self.device),
            beta_min=0.1, beta_max=20.0, T=1.0,
            denseTensor=False,             # <-- the only model difference vs dense MSGM
            norm_sampler="ecdf",           # matches the dense adapter's default
            device=self.device,
            estim_cst_norm_dens_r_T=False,
        )
        # sde_scheme.py expects sde.T to be a tensor (uses sde.T.device); MSGMsde stores a float.
        if not torch.is_tensor(self._sde.T):
            self._sde.T = torch.tensor(self._sde.T, device=self.device)

        mlp = MLP(input_dim=self.dim, hidden_dim=self.cfg.get("hidden_dim", 128)).to(self.device)
        self._model = _MLPDrift(mlp).to(self.device)

    def train(self, seed: int = 0, ckpt_dir: Path | None = None) -> dict:
        if self._sde is None:
            self.build()
        from SDEs import PluginReverseSDE
        from torch.optim import Adam
        import random, numpy as np, csv

        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

        self._gen_sde = PluginReverseSDE(
            self._sde, self._model, self._sde.T,
            vtype="rademacher", debias=False, ssm_intT=False, deviceReverseSDE=self.device,
        ).to(self.device)

        optimizer = Adam(self._gen_sde.parameters(), lr=self.cfg.get("lr", 1e-3))
        n_steps = self.cfg.get("n_train_steps", 10_000)
        batch_size = self.cfg.get("batch_size", 256)
        ckpt_every = self.cfg.get("ckpt_every", 5000)

        start_step, elapsed_before = 0, 0.0
        ckpt_path = Path(ckpt_dir) / "msgm_sparse_ckpt.pt" if ckpt_dir else None
        if ckpt_path and ckpt_path.exists():
            ck = torch.load(ckpt_path, weights_only=False)
            self._model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"])
            start_step = ck["step"]; elapsed_before = ck.get("elapsed_s", 0.0)
            print(f"  Resuming msgm_sparse from checkpoint at step {start_step}")

        train_gpu = self.train_data.to(self.device); n_train = train_gpu.shape[0]
        log_path = (ckpt_dir / "train_log.csv") if ckpt_dir else None
        log_rows = []
        if log_path and log_path.exists():
            with open(log_path) as f:
                log_rows = list(csv.DictReader(f))

        from tqdm import tqdm
        t0 = time.time()
        pbar = tqdm(range(start_step + 1, n_steps + 1), desc="msgm_sparse",
                    dynamic_ncols=True, initial=start_step, total=n_steps)
        for step in pbar:
            idx = torch.randint(n_train, (batch_size,), device=self.device)
            x = train_gpu[idx]
            optimizer.zero_grad(set_to_none=True)
            loss = self._gen_sde.ssm(x).mean()
            loss.backward()
            optimizer.step()
            if step % self.cfg.get("log_every", 200) == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}")
                if log_path:
                    log_rows.append({"step": step, "loss": loss.item(),
                                     "elapsed_s": elapsed_before + (time.time() - t0)})
                    with open(log_path, "w", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=["step", "loss", "elapsed_s"])
                        w.writeheader(); w.writerows(log_rows)
            if ckpt_path and step % ckpt_every == 0:
                torch.save({"step": step, "model": self._model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "elapsed_s": elapsed_before + (time.time() - t0)}, ckpt_path)

        total_time = elapsed_before + (time.time() - t0)
        if ckpt_path and ckpt_path.exists():
            ckpt_path.unlink()
        return {"total_train_time_s": total_time}

    @torch.no_grad()
    def sample(self, n: int) -> dict:
        from sde_scheme import rk4_stratonovich_sampler
        if self._gen_sde is None:
            raise RuntimeError("call train() before sample() (needs the trained reverse SDE)")
        x0 = self._sde.latent_sample(n, self.dim)
        t0 = time.time()
        xs = rk4_stratonovich_sampler(
            self._gen_sde, x0, num_steps=self.cfg.get("nfe", 128), keep_all_samples=False)
        return {"samples": xs.cpu(), "sample_time_s": time.time() - t0}
