"""Measure UNet velocity net: param count, ms/it (fwd+bwd), peak VRAM, at batch 32 and 64."""
import sys, time
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE))
from audio_flow import make_model, FREQ, FRAMES
dev="cuda"
for ch in [96,128]:
    m=make_model("unet",ch,0).to(dev); npar=sum(p.numel() for p in m.parameters())/1e6
    opt=torch.optim.AdamW(m.parameters(),lr=1e-4)
    print(f"\nUNet ch={ch}: {npar:.1f}M params")
    for B in [32,64]:
        try:
            torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
            x=torch.randn(B,2,FREQ,FRAMES,device=dev); t=torch.rand(B,device=dev); y=torch.randint(0,10,(B,),device=dev)
            for i in range(8):
                with torch.autocast("cuda",dtype=torch.bfloat16): v=m(x,t,y); loss=(v.float()**2).mean()
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            torch.cuda.synchronize(); t0=time.time()
            for i in range(15):
                with torch.autocast("cuda",dtype=torch.bfloat16): v=m(x,t,y); loss=(v.float()**2).mean()
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            torch.cuda.synchronize(); ms=(time.time()-t0)/15*1000; peak=torch.cuda.max_memory_allocated()/1e9
            print(f"  batch {B}: {ms:.0f} ms/it | peak VRAM {peak:.1f} GB")
        except torch.cuda.OutOfMemoryError:
            print(f"  batch {B}: OOM"); torch.cuda.empty_cache()
    del m,opt; torch.cuda.empty_cache()
