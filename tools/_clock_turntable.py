"""Sphere-trace a clay turntable of a checkpoint around world-Y to find the
frontal (ornate-face) viewpoint. Uses the run's EXACT trace params.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).parent))
from lip_tracer.config import TraceConfig
from lip_tracer.model import make_model
from lip_tracer.sphere_tracing import trace_nograd

RUN = Path("outputs/run_20260604_094256_bmvs_clock_mvsformer_carved_init_4972680")
CKPT = RUN / "ckpt" / "checkpoint_step_070000.pt"
OUT = RUN / "turntable_step70000"
CENTER = np.array([-0.098, 0.025, 0.051], dtype=np.float32)  # object centroid (norm frame)
RADIUS = 2.6
RES = 1000
SS = 2            # supersample: trace at RES*SS, downscale for clean edges
FOV_DEG = 22.0
BASE = np.array([0.55, 0.80, 0.78], np.float32)   # teal clay like the reference
# Final front view: face points down (-Y), az 0 = roof up. -72 = slight tilt, -85 = head-on.
ELEVS = [-72, -85]
AZIMS = [0]
HEADLIGHT = True   # light along view dir so the camera-facing surface is always lit


def look_at(cam, target):
    f = target - cam; f /= np.linalg.norm(f)
    up = np.array([0, 1, 0], np.float32)
    if abs(np.dot(f, up)) > 0.95:      # near-vertical view: pick a horizontal up
        up = np.array([0, 0, 1], np.float32)
    r = np.cross(f, up); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = r; c2w[:3, 1] = -u; c2w[:3, 2] = f; c2w[:3, 3] = cam
    return c2w  # camera looks +z, y down (image convention)


def main():
    OUT.mkdir(exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = json.loads((RUN / "config.json").read_text())
    tc = TraceConfig(**cfg["trace"]); print("[trace]", tc)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    m = cfg["model"]
    f = make_model(hidden=m["hidden"], depth=m["depth"], group_size=m["group_size"],
                   activation=m["activation"], input_encoding=m["input_encoding"],
                   multires=m["multires"], architecture=m["architecture"]).to(dev)
    f.load_state_dict(ck["f"], strict=False); f.eval()

    H = W = RES * SS
    fpix = 0.5 * W / np.tan(np.deg2rad(FOV_DEG) / 2)
    K = np.array([[fpix, 0, W / 2], [0, fpix, H / 2], [0, 0, 1]], np.float32)
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    dcam = np.stack([(xs + .5 - K[0, 2]) / K[0, 0],
                     (ys + .5 - K[1, 2]) / K[1, 1], np.ones_like(xs, np.float32)], -1)
    light = np.array([0.3, 0.7, 0.55], np.float32); light /= np.linalg.norm(light)

    import imageio.v2 as imageio
    poses = [(ed, ad) for ed in ELEVS for ad in AZIMS]
    for ed, ad in poses:
        e = np.deg2rad(ed); az = np.deg2rad(ad)
        cam = CENTER + RADIUS * np.array([np.cos(az) * np.cos(e), np.sin(e),
                                          np.sin(az) * np.cos(e)], np.float32)
        c2w = look_at(cam, CENTER)
        dw = (dcam @ c2w[:3, :3].T).astype(np.float32)
        dw /= np.linalg.norm(dw, axis=-1, keepdims=True)
        o = torch.from_numpy(np.broadcast_to(c2w[:3, 3], dw.shape).copy().reshape(-1, 3)).float().to(dev)
        d = torch.from_numpy(dw.reshape(-1, 3)).float().to(dev)
        xh, hh = [], []
        for i in range(0, o.shape[0], 65536):
            x_, _, h_ = trace_nograd(f, o[i:i+65536], d[i:i+65536], tc)
            xh.append(x_); hh.append(h_)
        xh = torch.cat(xh); hit = torch.cat(hh).cpu().numpy().reshape(H, W, 1)
        g = []
        for i in range(0, xh.shape[0], 8192):
            xr = xh[i:i+8192].detach().requires_grad_(True)
            with torch.enable_grad():
                g.append(torch.autograd.grad(f(xr).sum(), xr)[0].detach())
        n = torch.cat(g).cpu().numpy()
        n /= np.linalg.norm(n, axis=-1, keepdims=True).clip(1e-6)
        if HEADLIGHT:
            ldir = -dw.reshape(-1, 3)                      # toward camera, per-pixel
            diff = np.clip((n * ldir).sum(-1, keepdims=True), 0, 1)
        else:
            diff = np.clip((n * light).sum(-1, keepdims=True), 0, 1)
        shaded = (0.35 + 0.65 * diff) * BASE
        img = np.where(hit, shaded.reshape(H, W, 3), 0.30)
        u8 = np.clip(img*255+.5, 0, 255).astype(np.uint8)
        if SS > 1:
            from PIL import Image as _PIL
            u8 = np.array(_PIL.fromarray(u8).resize((RES, RES), _PIL.LANCZOS))
        name = f"front_e{ed:+03d}_a{ad:03d}.png"
        imageio.imwrite(OUT / name, u8)
        print(f"  elev={ed:+3d} az={ad:3d}  hit={hit.mean():.3f}  -> {name}", flush=True)
    print("[done]", OUT)


if __name__ == "__main__":
    main()
