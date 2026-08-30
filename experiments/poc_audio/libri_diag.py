"""LibriSpeech PoC companion diagnostics (not in audio_eval): radial drift, target-norm-vs-radius
correlation (std full-velocity vs angular scale-free), and real-reference Whisper word-presence.

Reuses audio_flow / audio_eval building blocks + saved 5k checkpoints. No retraining.
Usage: python experiments/poc_audio/libri_diag.py --step 5000 --n 300 --nfe 6
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(HERE))
from audio_flow import make_model, build, FREQ, FRAMES, D
from audio_data import istft, SR
from rafm.flow_matching.sampler import _project_tangent
from rafm.paths.spherical_geodesic import SphericalGeodesicPath
from rafm.utils.sphere import uniform_on_sphere

METHODS = [("matched_euclidean", "runs_libri5", "Matched-Eucl."),
           ("rafm", "runs_libri5", "RAFM (std)"),
           ("rafm", "runs_libri5_ang", "Angular RAFM")]


@torch.no_grad()
def sample_with_drift(model, x0, y, spherical, nfe, angular):
    dev = x0.device; x = x0; n = x0.shape[0]; dt = 1.0 / nfe
    r0 = x0.norm(dim=1)
    def v(xx, tt):
        out = model(xx.reshape(n, 2, FREQ, FRAMES), torch.full((n,), tt, device=dev), y).reshape(n, -1)
        if angular:
            out = xx.norm(dim=1, keepdim=True) * out
        return _project_tangent(out, xx) if spherical else out
    for i in range(nfe):
        t0 = i * dt; k1 = v(x, t0); k2 = v(x + dt / 2 * k1, t0 + dt / 2)
        k3 = v(x + dt / 2 * k2, t0 + dt / 2); k4 = v(x + dt * k3, t0 + dt)
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    r1 = x.norm(dim=1)
    drift = ((r1 - r0).abs() / r0.clamp(min=1e-8)).mean().item()   # relative radial drift
    return x, drift


def target_norm_corr(xtr, device, n=20000, seed=0):
    """Geometry: corr of full-velocity ‖u_t‖ and angular ‖u_t‖/‖x_t‖ with radius ‖x_t‖ on the spherical path."""
    torch.manual_seed(seed)
    path = SphericalGeodesicPath()
    r_a, un_a, ang_a = [], [], []
    for s in range(0, n, 2000):
        b = min(2000, n - s)
        idx = torch.randint(len(xtr), (b,))
        xb = xtr[idx].to(device); R = xb.norm(dim=1, keepdim=True)
        u0 = uniform_on_sphere(b, xb.shape[1], device=device); x0 = R * u0
        t = torch.rand(b, device=device)
        xt = path.sample_path(x0, xb, t); ut = path.conditional_vector_field(x0, xb, t)
        rt = xt.norm(dim=1).clamp(min=1e-8); un = ut.norm(dim=1)
        r_a.append(rt.cpu().numpy()); un_a.append(un.cpu().numpy()); ang_a.append((un / rt).cpu().numpy())
    r = np.concatenate(r_a); un = np.concatenate(un_a); ang = np.concatenate(ang_a)
    return {"corr_fullvel_radius": round(float(np.corrcoef(un, r)[0, 1]), 3),
            "corr_angular_radius": round(float(np.corrcoef(ang, r)[0, 1]), 3),
            "fullvel_CoV": round(float(un.std() / un.mean()), 3),
            "angular_CoV": round(float(ang.std() / ang.mean()), 3),
            "angular_median": round(float(np.median(ang)), 3)}


def whisper_wordpresence(X, ywords, words, device, tag, nmax=300):
    import torchaudio
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    proc = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
    wm = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en").to(device).eval()
    up = torchaudio.transforms.Resample(SR, 16000); nw = min(nmax, X.shape[0]); hit = 0
    with torch.no_grad():
        for i in range(0, nw, 16):
            wav16 = up(istft(X[i:i + 16].reshape(-1, 2, FREQ, FRAMES).cpu()))
            feat = proc(wav16.numpy(), sampling_rate=16000, return_tensors="pt").input_features.to(device)
            ids = wm.generate(feat, max_new_tokens=32); txt = proc.batch_decode(ids, skip_special_tokens=True)
            for j, t in enumerate(txt):
                if words[int(ywords[i + j])] in t.lower().split():
                    hit += 1
    del wm; torch.cuda.empty_cache()
    return round(hit / nw, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=5000); ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--nfe", type=int, default=6); ap.add_argument("--out", default="libri_diag_5k.json")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr = torch.load(HERE / "data/libri_word_train.pt", map_location="cpu")
    te = torch.load(HERE / "data/libri_word_test.pt", map_location="cpu")
    xtr = tr["x"].reshape(len(tr["x"]), -1).float(); ytr = tr["word"].long()
    words = json.loads((HERE / "data/libri_word_list.json").read_text())["words"]

    res = {"step": a.step, "n": a.n, "nfe_steps": a.nfe, "nfe_model_calls": a.nfe * 4}

    # geometry: target-norm vs radius (dataset property, method-independent)
    res["target_norm_corr"] = target_norm_corr(xtr, dev)
    print("target-norm corr:", res["target_norm_corr"], flush=True)

    # radial drift per method
    res["radial_drift"] = {}
    for method, rd, lbl in METHODS:
        run = HERE / rd / method
        meta = json.loads((run / "meta.json").read_text())["args"]
        st = build(method, xtr, ytr, 0)
        model = make_model(meta.get("arch", "unet"), meta.get("ch", 96), meta.get("depth", 5), meta.get("ncls", 20)).to(dev).eval()
        model.load_state_dict(torch.load(run / f"ema_{a.step}.pt", map_location="cpu")["ema"])
        torch.manual_seed(0)
        y = torch.arange(20).repeat_interleave(int(np.ceil(a.n / 20)))[:a.n].to(dev)
        r = st["src"].sample(a.n, D).norm(dim=1, keepdim=True).to(dev)
        u0 = torch.randn(a.n, D, device=dev); x0 = r * u0 / u0.norm(dim=1, keepdim=True)
        _, drift = sample_with_drift(model, x0, y, st["spherical"], a.nfe, bool(meta.get("angular", False)))
        res["radial_drift"][lbl] = round(drift, 5)
        print(f"drift {lbl}: {drift:.5f}", flush=True)
        del model; torch.cuda.empty_cache()

    # real-reference Whisper word-presence (upper bound / sanity of the evaluator)
    res["whisper_real_reference"] = whisper_wordpresence(
        te["x"].float(), te["word"].long(), words, dev, "real", nmax=a.n)
    print("real-ref whisper word-presence:", res["whisper_real_reference"], flush=True)

    (HERE / a.out).write_text(json.dumps(res, indent=2))
    print("wrote", HERE / a.out)


if __name__ == "__main__":
    main()
