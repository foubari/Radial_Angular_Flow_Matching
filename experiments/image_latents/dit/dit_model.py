"""Compact class-conditional DiT (adaLN-zero) for DC-AE latents [B, C, H, W].

Closest official architecture to LightningDiT/DiT, self-contained (no VAE coupling).
Predicts a velocity of the same shape (used with flow matching). Supports a null class
index (= num_classes) for classifier-free guidance.
Reference: Peebles & Xie, "Scalable Diffusion Models with Transformers" (DiT), adaLN-zero.
"""
import math, torch, torch.nn as nn

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden, freq=256):
        super().__init__(); self.freq=freq
        self.mlp=nn.Sequential(nn.Linear(freq,hidden),nn.SiLU(),nn.Linear(hidden,hidden))
    def forward(self,t):
        half=self.freq//2
        f=torch.exp(-math.log(10000)*torch.arange(half,device=t.device)/half)
        a=t[:,None].float()*f[None]*1000.0
        emb=torch.cat([torch.cos(a),torch.sin(a)],dim=-1)
        return self.mlp(emb)

class Attention(nn.Module):
    def __init__(self,dim,heads):
        super().__init__(); self.h=heads; self.qkv=nn.Linear(dim,dim*3); self.proj=nn.Linear(dim,dim)
    def forward(self,x):
        B,N,D=x.shape; qkv=self.qkv(x).reshape(B,N,3,self.h,D//self.h).permute(2,0,3,1,4)
        q,k,v=qkv[0],qkv[1],qkv[2]
        x=nn.functional.scaled_dot_product_attention(q,k,v)
        return self.proj(x.transpose(1,2).reshape(B,N,D))

class DiTBlock(nn.Module):
    def __init__(self,hidden,heads,mlp_ratio=4.0):
        super().__init__()
        self.n1=nn.LayerNorm(hidden,elementwise_affine=False,eps=1e-6)
        self.attn=Attention(hidden,heads)
        self.n2=nn.LayerNorm(hidden,elementwise_affine=False,eps=1e-6)
        m=int(hidden*mlp_ratio)
        self.mlp=nn.Sequential(nn.Linear(hidden,m),nn.GELU(approximate="tanh"),nn.Linear(m,hidden))
        self.ada=nn.Sequential(nn.SiLU(),nn.Linear(hidden,6*hidden))
    def forward(self,x,c):
        sa,ba,ga,sm,bm,gm=self.ada(c).chunk(6,dim=1)
        x=x+ga.unsqueeze(1)*self.attn(modulate(self.n1(x),sa,ba))
        x=x+gm.unsqueeze(1)*self.mlp(modulate(self.n2(x),sm,bm))
        return x

class DiT(nn.Module):
    def __init__(self,in_ch=32,size=8,patch=1,hidden=384,depth=12,heads=6,num_classes=10,class_dropout=0.1):
        super().__init__()
        self.in_ch=in_ch; self.size=size; self.patch=patch; self.num_classes=num_classes; self.class_dropout=class_dropout
        self.nt=(size//patch)**2
        self.x_embed=nn.Conv2d(in_ch,hidden,kernel_size=patch,stride=patch)
        self.pos=nn.Parameter(torch.zeros(1,self.nt,hidden))
        self.t_embed=TimestepEmbedder(hidden)
        self.y_embed=nn.Embedding(num_classes+1,hidden)   # +1 = null for CFG
        self.blocks=nn.ModuleList([DiTBlock(hidden,heads) for _ in range(depth)])
        self.nf=nn.LayerNorm(hidden,elementwise_affine=False,eps=1e-6)
        self.adaf=nn.Sequential(nn.SiLU(),nn.Linear(hidden,2*hidden))
        self.head=nn.Linear(hidden,patch*patch*in_ch)
        self._init()
    def _init(self):
        nn.init.trunc_normal_(self.pos,std=0.02)
        for b in self.blocks: nn.init.zeros_(b.ada[-1].weight); nn.init.zeros_(b.ada[-1].bias)
        nn.init.zeros_(self.adaf[-1].weight); nn.init.zeros_(self.adaf[-1].bias)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
    def unpatchify(self,x):
        B=x.shape[0]; p=self.patch; hp=self.size//p
        x=x.reshape(B,hp,hp,p,p,self.in_ch).permute(0,5,1,3,2,4).reshape(B,self.in_ch,self.size,self.size)
        return x
    def forward(self,x,t,y=None,drop=False):
        # x: (B,C,H,W); t:(B,); y:(B,) class or None
        B=x.shape[0]
        h=self.x_embed(x).flatten(2).transpose(1,2)+self.pos
        if y is None:
            y=torch.full((B,),self.num_classes,device=x.device,dtype=torch.long)
        elif drop and self.training and self.class_dropout>0:
            m=torch.rand(B,device=x.device)<self.class_dropout
            y=torch.where(m,torch.full_like(y,self.num_classes),y)
        c=self.t_embed(t)+self.y_embed(y)
        for blk in self.blocks: h=blk(h,c)
        sf,bf=self.adaf(c).chunk(2,dim=1)
        h=modulate(self.nf(h),sf,bf)
        return self.unpatchify(self.head(h))
    def num_params(self): return sum(p.numel() for p in self.parameters())
