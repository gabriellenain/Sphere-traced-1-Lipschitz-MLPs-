"""Phong-render the current SDF + sanity-check scene/camera normalization for the MVMannequin run."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from lip_tracer.data import load_views
from lip_tracer.model import make_model
from lip_tracer.train import _render_poses


def main() -> None:
    run_dir = Path("outputs/run_20260528_172115_mvm_kinette-cos-hx_4921642")
    scene = Path("data/mvmannequin_neus/kinette-cos-hx")
    ckpt_path = run_dir / "ckpt" / "checkpoint_latest.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    architecture = ckpt.get("architecture", "cpl")
    hidden = ckpt["f"]["head_weight"].shape[0] if "head_weight" in ckpt["f"] else 256
    group_size = ckpt.get("group_size", 2)
    activation = ckpt.get("activation", "groupsort")
    input_encoding = ckpt.get("input_encoding", "pe")
    multires = ckpt.get("multires", 6)
    depth = ckpt.get("depth",
                     sum(1 for k in ckpt["f"]
                         if k.startswith("net.") and k.endswith(".weight") and "_u" not in k))
    print(f"  arch={architecture}  hidden={hidden}  depth={depth}  group={group_size}  "
          f"act={activation}  enc={input_encoding}  L={multires}")

    f = make_model(hidden=hidden, depth=depth, group_size=group_size, activation=activation,
                   input_encoding=input_encoding, multires=multires,
                   architecture=architecture).to(device)
    f.load_state_dict(ckpt["f"], strict=False)
    with torch.enable_grad():
        f(torch.zeros(1, 3, device=device))  # warm caches
    f.eval()

    views = load_views(scene)
    print(f"\nviews: V={views['c2w'].shape[0]} H,W=({views['H']},{views['W']})")

    cam_pos = views["c2w"][:, :3, 3].numpy()
    cam_dist = np.linalg.norm(cam_pos, axis=-1)
    print(f"\n=== Camera positions (normalized frame, should be outside unit sphere) ===")
    print(f"  N cameras: {len(cam_pos)}")
    print(f"  ||c|| min={cam_dist.min():.3f}  mean={cam_dist.mean():.3f}  max={cam_dist.max():.3f}")
    print(f"  cam[0]:    pos={cam_pos[0]}  dist={cam_dist[0]:.3f}")
    # angle of each camera ray to scene origin
    look_dirs = -cam_pos / np.linalg.norm(cam_pos, axis=-1, keepdims=True)
    z_cam = views["c2w"][:, :3, 2].numpy()
    cos_to_origin = (z_cam * look_dirs).sum(-1)
    print(f"  cos(camera_z, -cam_position): min={cos_to_origin.min():.3f}  mean={cos_to_origin.mean():.3f}  max={cos_to_origin.max():.3f}")
    print(f"  (close to 1.0 means cameras point at origin)")

    print(f"\n=== Scene normalization check (GT mesh in normalized frame) ===")
    try:
        import trimesh
        m = trimesh.load(scene / "gt_mesh.ply", process=False)
        v = np.asarray(m.vertices, dtype=np.float64)
        r = np.linalg.norm(v, axis=-1)
        print(f"  N verts: {len(v)}")
        print(f"  GT mesh bbox: min={v.min(0)} max={v.max(0)}")
        print(f"  GT vertex ||x||: min={r.min():.3f}  mean={r.mean():.3f}  max={r.max():.3f}  (<1 means inside unit sphere)")
    except Exception as e:
        print(f"  (skipped, {e})")

    # Probe model at the origin and at unit-sphere samples
    print(f"\n=== Current SDF probes ===")
    with torch.no_grad():
        o = f(torch.zeros(1, 3, device=device)).item()
        print(f"  f(0,0,0) = {o:.4f}")
        rng = np.random.default_rng(0)
        x = rng.normal(size=(1024, 3)).astype(np.float32)
        x /= np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9
        vals = []
        for r_try in [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
            v_r = f(torch.from_numpy(x * r_try).to(device)).cpu().numpy()
            vals.append((r_try, v_r.mean(), v_r.std()))
        print("  Sphere shells (r → f mean ± std):")
        for r_try, mean, std in vals:
            print(f"    r={r_try:.2f}  f={mean:+.4f} ± {std:.4f}")

    print(f"\n=== Rendering 4 views ===")
    _render_poses(f, views, step=ckpt.get("step", 7600), run_dir=run_dir,
                  device=device, res=400)
    print("\nDone. See:")
    print(f"  {run_dir}/render/render_*.png")


if __name__ == "__main__":
    main()
