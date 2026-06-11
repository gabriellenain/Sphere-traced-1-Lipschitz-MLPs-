"""Sphere-trace the clock's ornate FRONT face (a synthesized below-looking-up
camera, not a dataset view) into a screen-space mesh + a camera npz, so
tools/render_blender.py can AO-render it exactly like the other views.

Everything stays in the model's normalized frame: the PLY is normalized and the
camera npz holds (K, R world->cam, center) in that same frame, so render_blender
is run with --normalized-mesh --cam-npz (no S_inv round-trip).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from dataclasses import replace
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).parent))
from lip_tracer.config import TraceConfig
from lip_tracer.model import make_model
from sphere_traced_screen_mesh import trace, grad_at, build_screen_mesh, write_ply

RUN = Path("outputs/run_20260604_094256_bmvs_clock_mvsformer_carved_init_4972680")
CKPT = RUN / "ckpt" / "checkpoint_step_070000.pt"
OUT = RUN / "blender_st_front_step70000"
CENTER = np.array([-0.098, 0.025, 0.051], dtype=np.float32)  # object centroid
RADIUS = 2.9          # slightly looser than the 2.6 turntable for top/bottom margin
ELEV_DEG = -85.0      # head-on to the front face (points -Y); matches the reference
AZ_DEG = 0.0          # roof apex up
RES = 1100
FOV_DEG = 22.0
RELGAP = 0.02
GRAZING_COS = 0.0     # keep every hit the tracer produced (faithful)


def look_at(cam, target):
    f = target - cam; f /= np.linalg.norm(f)
    up = np.array([0, 1, 0], np.float32)
    if abs(np.dot(f, up)) > 0.95:
        up = np.array([0, 0, 1], np.float32)
    r = np.cross(f, up); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = r; c2w[:3, 1] = -u; c2w[:3, 2] = f; c2w[:3, 3] = cam
    return c2w


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

    H = W = RES
    fpix = 0.5 * W / np.tan(np.deg2rad(FOV_DEG) / 2)
    K = np.array([[fpix, 0, W / 2], [0, fpix, H / 2], [0, 0, 1]], np.float64)
    e = np.deg2rad(ELEV_DEG); az = np.deg2rad(AZ_DEG)
    cam = CENTER + RADIUS * np.array([np.cos(az) * np.cos(e), np.sin(e),
                                      np.sin(az) * np.cos(e)], np.float32)
    c2w = look_at(cam, CENTER)

    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    dcam = np.stack([(xs + .5 - K[0, 2]) / K[0, 0],
                     (ys + .5 - K[1, 2]) / K[1, 1], np.ones_like(xs, np.float64)], -1)
    dw = (dcam @ c2w[:3, :3].T.astype(np.float64))
    dw /= np.linalg.norm(dw, axis=-1, keepdims=True)
    o = torch.from_numpy(np.broadcast_to(c2w[:3, 3], dw.shape).copy()
                         .reshape(-1, 3)).float().to(dev)
    d = torch.from_numpy(dw.reshape(-1, 3)).float().to(dev)

    iters = cfg["trace"]["iters"]
    t, hit = trace(f, o, d, replace(tc, iters=iters), 32768)
    print(f"hits: {int(hit.sum()):,} / {len(hit):,}", flush=True)
    pts = (o + torch.from_numpy(t).to(dev).unsqueeze(-1) * d)
    nraw = grad_at(f, pts, 32768)
    n = (nraw / np.linalg.norm(nraw, axis=1, keepdims=True).clip(1e-9)).astype(np.float32)

    verts, vnorm, faces = build_screen_mesh(
        hit, pts.detach().cpu().numpy().astype(np.float32), t.astype(np.float32),
        n, H, W, RELGAP, GRAZING_COS, dw.reshape(-1, 3).astype(np.float32))
    ply = OUT / "st_front_world.ply"          # actually normalized frame
    write_ply(ply, verts.astype(np.float32),
              (vnorm / np.linalg.norm(vnorm, axis=1, keepdims=True).clip(1e-9)).astype(np.float32),
              faces)

    # camera npz in normalized frame for render_blender --cam-npz
    R_opencv = c2w[:3, :3].T.astype(np.float64)   # world->cam
    np.savez(OUT / "front_cam.npz", K=K, R=R_opencv, center=cam.astype(np.float64),
             H=H, W=W)
    print("[cam] saved", OUT / "front_cam.npz", "  cam=", cam.round(3))
    print("[done]", OUT)


if __name__ == "__main__":
    main()
