#!/usr/bin/env python3
"""Efficiency / FLOPs analysis for the Angular RAFM paper. No full training reruns.

- Parameter counts: built models (tabular MLP, ResMLP big, DiT-SiT, audio UNet, MSGM MLP).
- FLOPs/forward: analytic MAC counter via forward hooks on Linear/Conv2d (MACs*2 = FLOPs).
- Training compute: FLOPs/iter (fwd + bwd ~= 3x fwd) * batch * steps (steps/batch from configs/logs).
- Sampling cost: exact RK4 NFE accounting (n_steps * 4 model calls) * FLOPs/sample.
- MSGM: measured runtime + slowdown + code-supported reason (see ANGULAR_AUDIT / cost note).

Writes efficiency.json + table_efficiency.tex + prints a summary.
"""
import json, sys, time
from pathlib import Path
import torch, torch.nn as nn

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "rebuttal_experiments"))
sys.path.insert(0, str(REPO / "third_party/SiT"))


def count_macs(model, inputs):
    """Return (params, MACs) for one forward. MACs from Linear/Conv2d hooks."""
    macs = [0]
    hooks = []

    def lin_hook(m, i, o):
        macs[0] += m.in_features * m.out_features * (o.numel() // o.shape[-1] if o.dim() > 1 else 1)

    def conv_hook(m, i, o):
        out_elems = o.numel() // o.shape[0]  # per-sample output elements
        macs[0] += out_elems * m.in_channels // m.groups * m.kernel_size[0] * m.kernel_size[1] * o.shape[0]

    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            hooks.append(mod.register_forward_hook(lin_hook))
        elif isinstance(mod, nn.Conv2d):
            hooks.append(mod.register_forward_hook(conv_hook))
    model.eval()
    with torch.no_grad():
        model(*inputs)
    for h in hooks:
        h.remove()
    params = sum(p.numel() for p in model.parameters())
    return params, macs[0]


def human(n):
    for u in ["", "K", "M", "G", "T"]:
        if abs(n) < 1000:
            return f"{n:.2f}{u}"
        n /= 1000
    return f"{n:.2f}P"


rows = {}


def add(name, params, macs_per_sample, batch, steps, ms_per_iter, sampler, note=""):
    fwd_flops = 2 * macs_per_sample                       # 1 MAC = 2 FLOP
    train_iter_flops = 3 * fwd_flops * batch              # fwd + bwd ~= 3x fwd
    total_train_flops = train_iter_flops * steps
    rows[name] = {
        "params": params, "flops_per_forward": fwd_flops,
        "batch": batch, "steps": steps, "ms_per_iter": ms_per_iter,
        "train_iter_flops": train_iter_flops, "total_train_flops": total_train_flops,
        "wallclock_train_s": (ms_per_iter * steps / 1000) if ms_per_iter else None,
        "sampler": sampler, "note": note,
    }
    print(f"  {name:22s} params={human(params):>8s} fwd={human(fwd_flops):>8s}FLOP "
          f"train={human(total_train_flops):>8s}FLOP  {sampler}")


def main():
    dev = "cpu"
    from rafm.models.mlp import MLP
    from lib.resmlp import ResidualMLP

    # tabular MLP (finance d49) — represents FM/RAFM/Angular (identical net & cost)
    for tag, dim, batch, steps in [("MLP d49 (Finance)", 49, 4096, 10000),
                                   ("MLP d96 (Weather)", 96, 2048, 10000),
                                   ("MLP d64 (PIV)", 64, 4096, 10000)]:
        m = MLP(input_dim=dim, hidden_dim=128, n_layers=3)
        p, mac = count_macs(m, (torch.randn(8, dim), torch.rand(8)))
        add(f"FM/RAFM/Angular {tag}", p, mac / 8, batch, steps, None,
            "RK4 128 steps=512 evals (baselines) / 512 steps=2048 evals (Angular runs)")

    # ResMLP big (weather 256x4)
    m = ResidualMLP(input_dim=96, hidden_dim=256, n_blocks=4)
    p, mac = count_macs(m, (torch.randn(8, 96), torch.rand(8)))
    add("ResMLP 256x4 (Weather big)", p, mac / 8, 2048, 10000, None, "RK4 512 steps=2048 evals")

    # MSGM MLP (reference NN.MLP hidden 128) — cost dominated by SSM, not the net (see note)
    try:
        from baselines.msgm_adapter import MSGM_DIR  # noqa
        sys.path.insert(0, str(MSGM_DIR))
        from NN import MLP as MSGM_MLP
        mm = MSGM_MLP(input_dim=49, hidden_dim=128)
        p, mac = count_macs(mm, (torch.randn(8, 49), torch.ones(8, 1) * 0.5))
        rows["MSGM (Finance d49)"] = {
            "params": p, "flops_per_forward": 2 * mac / 8, "batch": 4096, "steps": 10000,
            "ms_per_iter": 1800.0, "measured_train_s_per_seed": "18000-20000",
            "slowdown_vs_rafm": "~525x", "sampler": "RK4 Stratonovich",
            "note": "Per-step cost dominated NOT by the MLP but by the multiplicative-noise SDE: dense "
                    "(dim,dim,dim) generator G -> O(batch*dim^3) drift/diffusion, wrapped in sliced-score-"
                    "matching (autograd.grad create_graph=True -> double backward). Cost scales with dim^3.",
        }
        print(f"  {'MSGM (Finance d49)':22s} params={human(p):>8s} net-fwd small; cost = SSM+dim^3 generator; "
              f"measured 18-20k s/seed (~525x RAFM)")
    except Exception as e:
        print(f"  MSGM MLP profile skipped: {e}")

    # DiT SiT (DC-AE) — profile one forward
    try:
        from models import SiT
        sit = SiT(input_size=8, patch_size=1, in_channels=32, hidden_size=384, depth=12,
                  num_heads=6, num_classes=10, class_dropout_prob=0.1, learn_sigma=False)
        x = torch.randn(4, 32, 8, 8); t = torch.rand(4); y = torch.randint(0, 10, (4,))
        p, mac = count_macs(sit, (x, t, y))
        add("DiT-SiT (DC-AE image)", p, mac / 4, 64, 40000, 83.0,
            "RK4 25 steps=100 evals (all methods, matched)", "linear-attn MACs approximate")
    except Exception as e:
        print(f"  DiT profile skipped ({e}); using known params 32.5M")
        rows["DiT-SiT (DC-AE image)"] = {"params": 32_500_000, "batch": 64, "steps": 40000,
                                         "ms_per_iter": 83.0, "sampler": "RK4 25 steps=100 evals",
                                         "note": "FLOPs not profiled (SiT import unavailable)"}

    # Audio UNet — profile one forward
    try:
        sys.path.insert(0, str(REPO / "experiments/poc_audio"))
        from audio_flow import make_model
        net = make_model("unet", ch=96, depth=None, ncls=10) if "ncls" in make_model.__code__.co_varnames \
            else make_model("unet", 96)
        x = torch.randn(2, 2, 129, 63); t = torch.rand(2); y = torch.randint(0, 10, (2,))
        try:
            p, mac = count_macs(net, (x, t, y))
        except Exception:
            p, mac = count_macs(net, (x, t))
        add("UNet ch96 (AudioMNIST)", p, mac / 2, 32, 24000, None,
            "RK4 (see audio_eval nfe)", "conv MACs")
    except Exception as e:
        print(f"  Audio UNet profile skipped ({e}); using known params ~27.9M")
        rows["UNet ch96 (AudioMNIST)"] = {"params": 27_900_000, "batch": 32, "steps": 24000,
                                          "sampler": "RK4", "note": "FLOPs not profiled"}

    (HERE / "efficiency.json").write_text(json.dumps(rows, indent=2))

    # LaTeX table
    tex = [r"\begin{table}[t]\centering",
           r"\caption{Model complexity and compute. FLOPs from analytic MAC counting (Linear/Conv2d). "
           r"Training compute $=3\times$ forward FLOPs (fwd+bwd) $\times$ batch $\times$ steps. "
           r"RK4 sampling: $n$ nominal steps $=4n$ model evaluations. FM, standard RAFM and Angular RAFM "
           r"share the \emph{same} network and per-step cost (Angular adds only a scalar norm+multiply). "
           r"MSGM is $\sim$525$\times$ slower for the reason in the note.}",
           r"\label{tab:efficiency}", r"\small", r"\begin{tabular}{lrrrl}", r"\toprule",
           r"Model & Params & FLOPs/fwd & Train FLOPs & Sampling \\", r"\midrule"]
    for name, r in rows.items():
        pf = human(r["params"]); ff = (human(r["flops_per_forward"]) if r.get("flops_per_forward") else "--")
        tf = (human(r["total_train_flops"]) if r.get("total_train_flops") else "--")
        tex.append(f"{name} & {pf} & {ff} & {tf} & {r['sampler']} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (HERE / "tables/table_efficiency.tex").write_text("\n".join(tex))
    print("\nwrote efficiency.json + tables/table_efficiency.tex")


if __name__ == "__main__":
    main()
