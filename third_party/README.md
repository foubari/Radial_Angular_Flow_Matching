# third_party — pinned external implementations (not committed; reproduce by cloning)

External code is **not** vendored into git history. Reproduce exactly by cloning at the
pinned commit SHA below. Licenses remain with the upstream repositories.

| repo | URL | pinned SHA | used for |
|---|---|---|---|
| RAE | https://github.com/bytetriper/RAE | a4d18c4db766419cbe7cb8c02cd9f7ceb0ec9041 | DINOv2-B encoder + ViTXL decoder + imagenet1k norm stats (Stage-2 latent) |
| RJF | https://github.com/amandpkr/RJF | (to pin when used) | spherical/geodesic flow reference (Phase 3+) |
| flow_matching | https://github.com/facebookresearch/flow_matching | (to pin when used) | FM reference |

## Reproduce RAE setup
```bash
git clone https://github.com/bytetriper/RAE third_party/RAE && (cd third_party/RAE && git checkout a4d18c4db766419cbe7cb8c02cd9f7ceb0ec9041)
# weights (DINOv2-B decoder + stats only) from HuggingFace nyu-visionx/RAE-collections:
#   decoders/dinov2/wReg_base/ViTXL_n08/model.pt  -> third_party/RAE/models/decoders/dinov2/wReg_base/ViTXL_n08/model.pt
#   stats/dinov2/wReg_base/imagenet1k/stat.pt     -> third_party/RAE/models/stats/dinov2/wReg_base/imagenet1k/stat.pt
# isolated venv (reuses system torch): python -m venv --system-site-packages experiments/image_latents/.venv_img
#   then: pip install transformers==4.56.2 timm==0.9.16 omegaconf==2.3.0
```
Encoder auto-downloaded by transformers: facebook/dinov2-with-registers-base.
