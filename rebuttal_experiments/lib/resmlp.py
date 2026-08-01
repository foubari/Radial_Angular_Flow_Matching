"""Residual MLP for the higher-capacity ablation (drop-in for rafm.models.mlp.MLP).

Same input/output contract as MLP: forward(x, t) -> velocity of shape like x.
Time is concatenated at the input (matching the base MLP). Residual blocks:
  h = h + Swish(Linear(Swish(Linear(h)))), width `hidden_dim`, `n_blocks` blocks.
"""
import torch
import torch.nn as nn


class Swish(nn.Module):
    def forward(self, x): return torch.sigmoid(x) * x


class ResidualBlock(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(w, w), Swish(), nn.Linear(w, w))
        self.act = Swish()

    def forward(self, h):
        return self.act(h + self.net(h))


class ResidualMLP(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=256, n_blocks=4, premodule=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = input_dim
        self.inp = nn.Sequential(nn.Linear(input_dim + 1, hidden_dim), Swish())
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(n_blocks)])
        self.out = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, t):
        sz = x.shape
        x = x.view(-1, self.input_dim)
        t = t.view(-1, 1).float().expand(x.shape[0], 1)
        h = self.inp(torch.cat([x, t], dim=1))
        for b in self.blocks:
            h = b(h)
        return self.out(h).view(*sz)
