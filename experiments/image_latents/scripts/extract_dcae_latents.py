"""Extract DC-AE latents (flattened 2048-d) for Imagenette train+val -> dcae_latents.pt."""
import torch, glob, sys
from diffusers import AutoencoderDC
from torchvision import transforms
from PIL import Image
dev = "cuda" if torch.cuda.is_available() else "cpu"
ae = AutoencoderDC.from_pretrained("mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers", torch_dtype=torch.float32).to(dev).eval()
paths = sorted(glob.glob("experiments/image_latents/data/imagenette2-320/**/*.JPEG", recursive=True))
print("total images:", len(paths), flush=True)
tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(256), transforms.ToTensor()])
lat, buf = [], []
with torch.no_grad():
    for k, p in enumerate(paths):
        try: buf.append(tf(Image.open(p).convert("RGB")))
        except Exception: continue
        if len(buf) == 32 or k == len(paths) - 1:
            x = (torch.stack(buf).to(dev) * 2 - 1)
            z = ae.encode(x).latent
            lat.append(z.reshape(z.shape[0], -1).cpu()); buf = []
            if (k // 128) % 10 == 0: print(f"  {k+1}/{len(paths)}", flush=True)
L = torch.cat(lat).float()
torch.save(L, "experiments/image_latents/data/dcae_latents.pt")
r = L.norm(dim=1)
print("SAVED dcae_latents.pt shape %s mean-norm %.1f CoV %.4f" % (tuple(L.shape), r.mean(), r.std()/r.mean()), flush=True)
