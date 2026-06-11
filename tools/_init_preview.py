"""Init-only preview: hull warm-start WITH vs WITHOUT the SFM ray free-space term.

Fits f_theta to the scan24 visual hull twice (w_sfm_free = 0 and 1.0), using the
launched run's exact model + trace config, sphere-traces 3 real cameras, and
overlays the shaded surface on each photo. Nothing here touches the training loop.
Output columns: [photo | w_sfm_free=0 | w_sfm_free=1.0].
"""
import json, numpy as np, torch
from pathlib import Path
from PIL import Image
from scipy.ndimage import binary_erosion

from lip_tracer.config import ModelConfig, InitConfig, TraceConfig
from lip_tracer.train import fit_hull_init
from lip_tracer.data import load_views
from lip_tracer.sphere_tracing import trace_nograd

RUN = Path("/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/run_20260601_143409_scan24_4933894")
OUT = Path("/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/outputs/hull_fix/init_preview_sfmfree.png")
PREVIEW_STEPS = 10000
DOWN = 4
WEIGHTS = [0.0, 1.0]

cfg = json.loads((RUN / "config.json").read_text())
scene = Path(cfg["scene"]); bound = float(cfg["eval"]["bound_dtu"])
mc, tc, ic = cfg["model"], cfg["trace"], cfg["init"]
model_cfg = ModelConfig(hidden=mc["hidden"], depth=mc["depth"], group_size=mc["group_size"],
                        activation=mc["activation"], input_encoding=mc["input_encoding"],
                        multires=mc["multires"], architecture=mc["architecture"],
                        lipschitz_mode=mc["lipschitz_mode"])
trace_cfg = TraceConfig(iters=tc["iters"], eps=tc["eps"], t_far=tc["t_far"],
                        eik_stride=tc["eik_stride"], newton_steps=tc["newton_steps"],
                        grad_mode=tc["grad_mode"], bsphere_radius=tc["bsphere_radius"],
                        sdf_min_beta=tc["sdf_min_beta"])
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={DEV}  scene={scene}")

views = load_views(scene)
V = views["c2w"].shape[0]
sel = np.linspace(0, V - 1, 3, dtype=int)
H, W = views["H"], views["W"]
Hd, Wd = H // DOWN, W // DOWN
light = torch.tensor([0.4, 0.5, 0.8], device=DEV); light = light / light.norm()


def render(f, vi):
    c2w = views["c2w"][vi].numpy(); K = views["K"][vi].numpy()
    ys, xs = np.meshgrid(np.arange(Hd), np.arange(Wd), indexing="ij")
    xf = (xs + 0.5) * DOWN - 0.5; yf = (ys + 0.5) * DOWN - 0.5
    dc = np.stack([(xf - K[0, 2]) / K[0, 0], (yf - K[1, 2]) / K[1, 1], np.ones_like(xf)], -1)
    d = dc @ c2w[:3, :3].T; d = d / np.linalg.norm(d, axis=-1, keepdims=True)
    o = np.broadcast_to(c2w[:3, 3], d.shape).reshape(-1, 3)
    ot = torch.from_numpy(o.copy()).float().to(DEV); dt = torch.from_numpy(d.reshape(-1, 3)).float().to(DEV)
    xh, _, hit = trace_nograd(f, ot, dt, cfg=trace_cfg)
    with torch.enable_grad():
        xr = xh.detach().clone().requires_grad_(True)
        n = torch.autograd.grad(f(xr).sum(), xr)[0]
    n = n / n.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    sh = (0.3 + 0.7 * (n @ light).abs()).clamp(0, 1)
    return hit.view(Hd, Wd).cpu().numpy(), (sh * hit.float()).view(Hd, Wd).cpu().numpy()


def overlay(vi, hitm, sh):
    photo = views["images"][vi].numpy()
    photo = np.array(Image.fromarray((photo * 255).astype(np.uint8)).resize((Wd, Hd), Image.BILINEAR)) / 255.0
    out = photo.copy(); a = 0.55
    hull_rgb = np.stack([sh, sh * 0.55, sh * 0.1], -1)
    out[hitm] = (1 - a) * photo[hitm] + a * hull_rgb[hitm]
    edge = hitm & ~binary_erosion(hitm, iterations=2)
    out[edge] = [0.1, 1.0, 0.2]
    return photo, out


cols_per_w = {}
for w in WEIGHTS:
    print(f"\n===== fitting init  w_sfm_free={w}  steps={PREVIEW_STEPS} =====")
    init_cfg = InitConfig(init="hull", steps=PREVIEW_STEPS, batch=ic["batch"], lr=ic["lr"],
                          hull_res=ic["hull_res"], hull_sfm_roi=True, hull_border_aware=True,
                          w_sfm_free=w, sfm_free_eps=0.02)
    f = fit_hull_init(model_cfg, init_cfg, scene, bound)
    f.eval()
    cols_per_w[w] = [render(f, vi) for vi in sel]

rows = []
for r, vi in enumerate(sel):
    photo, ov0 = overlay(vi, *cols_per_w[WEIGHTS[0]][r])
    _, ov1 = overlay(vi, *cols_per_w[WEIGHTS[1]][r])
    rows.append(np.concatenate([photo, ov0, ov1], axis=1))
grid = np.clip(np.concatenate(rows, axis=0), 0, 1)
OUT.parent.mkdir(parents=True, exist_ok=True)
Image.fromarray((grid * 255).astype(np.uint8)).save(OUT)
print(f"\nviews={sel.tolist()}  cols=[photo | w=0 | w=1.0]  saved → {OUT}")
