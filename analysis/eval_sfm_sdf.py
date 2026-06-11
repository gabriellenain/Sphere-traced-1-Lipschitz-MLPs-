"""Print f(x) statistics at COLMAP sparse points."""
import torch
import numpy as np
from pathlib import Path
from lip_tracer.model import FTheta
from lip_tracer.data import load_colmap_points

CKPT  = Path("outputs/run_20260507_162756_scan122/checkpoint_best_photo.pt")
SCENE = Path("/scratch/_projets_/willow/1-lip-tracer-new/data/dtu_idr/scan122")

ckpt = torch.load(CKPT, map_location="cpu")
f = FTheta(depth=ckpt["depth"], hidden=256, group_size=ckpt["group_size"],
           activation=ckpt["activation"], input_encoding=ckpt["input_encoding"],
           multires=ckpt["multires"])
f.load_state_dict(ckpt["f"])
f.eval()

pts = load_colmap_points(SCENE)
print(f"COLMAP pts: {len(pts)}")

with torch.no_grad():
    vals = f(pts).abs()

print(f"|f(x)| mean   {vals.mean():.4f}")
print(f"|f(x)| median {vals.median():.4f}")
print(f"|f(x)| p90    {vals.quantile(0.9):.4f}")
print(f"% < 0.01      {(vals < 0.01).float().mean()*100:.1f}%")

# gradient magnitude at surface points
pts.requires_grad_(True)
v = f(pts)
grad = torch.autograd.grad(v.sum(), pts)[0]
gnorm = grad.norm(dim=-1)
print(f"\n|∇f| mean   {gnorm.mean():.4f}  (ideal=1.0)")
print(f"|∇f| median {gnorm.median():.4f}")
print(f"|∇f| p10    {gnorm.quantile(0.1):.4f}")
print(f"% > 0.9     {(gnorm > 0.9).float().mean()*100:.1f}%")
print(f"% < 0.1     {(gnorm < 0.1).float().mean()*100:.1f}%")
print(f"\nstep: {ckpt['step']}  photo: {ckpt['photo']:.4f}")
