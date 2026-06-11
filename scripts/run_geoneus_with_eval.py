"""Run Geo-Neus unmodified, mirror its TB writer to wandb, and after every
mesh validation call DTUeval-python directly — the same code path our model
uses in `_run_dtu_official_eval` — and log {chamfer, accuracy, completeness}
to wandb at the same step. IDR cleaning (largest connected component) is
applied before each chamfer.

Geo-Neus code is imported as-is from baselines/Geo-Neus — no edits to that
repo's algorithm. Only minimal device hints on tensor-creation sites are
applied via monkey-patching from this wrapper.
"""
from __future__ import annotations
import argparse, json, os, re, subprocess, sys
from pathlib import Path

import numpy as np
import wandb

REPO = Path(__file__).resolve().parents[1]
GEONEUS_DIR = REPO / "baselines" / "Geo-Neus"
DTU_EVAL_SCRIPT = REPO / "DTUeval-python" / "eval.py"
PIXI_PYTHON = Path("/home/glenain/nerfstudio/.pixi/envs/default/bin/python3")


def parse_scan_id(case: str) -> int:
    m = re.search(r"\d+", case)
    if not m:
        raise ValueError(f"Could not parse scan id from case={case!r}")
    return int(m.group(0))


def _eval_python() -> str:
    try:
        import open3d  # noqa: F401
        return sys.executable
    except ImportError:
        return str(PIXI_PYTHON) if PIXI_PYTHON.exists() else sys.executable


def idr_clean_largest_component(mesh_ply: Path) -> Path:
    """IDR cleaning: largest connected component by area."""
    import trimesh
    m = trimesh.load(str(mesh_ply), force="mesh", process=False)
    comps = m.split(only_watertight=False)
    if len(comps) <= 1:
        return mesh_ply
    areas = np.array([c.area for c in comps], dtype=np.float64)
    cleaned = comps[int(areas.argmax())]
    out = mesh_ply.with_name(mesh_ply.stem + "_clean.ply")
    cleaned.export(str(out))
    print(f"[clean] {mesh_ply.name}: {len(comps)} components → kept largest "
          f"(frac={areas.max()/areas.sum():.3f}) → {out.name}", flush=True)
    return out


def run_dtu_eval(mesh_ply: Path, scan_id: int, dtu_eval_dir: Path,
                 out_dir: Path) -> dict | None:
    if not DTU_EVAL_SCRIPT.exists():
        print(f"[dtu_eval] missing {DTU_EVAL_SCRIPT}", flush=True)
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        _eval_python(), str(DTU_EVAL_SCRIPT),
        "--data", str(mesh_ply),
        "--scan", str(scan_id),
        "--mode", "mesh",
        "--dataset_dir", str(dtu_eval_dir),
        "--vis_out_dir", str(out_dir),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    (out_dir / "official_stdout.txt").write_text(res.stdout)
    (out_dir / "official_stderr.txt").write_text(res.stderr)
    if res.returncode != 0:
        print(f"[dtu_eval] failed rc={res.returncode}; see {out_dir}", flush=True)
        return None
    lines = [ln.strip() for ln in res.stdout.splitlines() if ln.strip()]
    try:
        acc, comp, chamfer = [float(v) for v in lines[-1].split()]
    except (IndexError, ValueError):
        print(f"[dtu_eval] could not parse metrics; see {out_dir}", flush=True)
        return None
    payload = {"accuracy": acc, "completeness": comp, "chamfer": chamfer}
    (out_dir / "dtu_official.json").write_text(json.dumps(payload, indent=2))
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", required=True)
    ap.add_argument("--case", required=True)
    ap.add_argument("--mode", default="train")
    ap.add_argument("--is_continue", action="store_true")
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--dtu_eval_dir", required=True)
    ap.add_argument("--scan_id", type=int, default=None)
    ap.add_argument("--eval_every", type=int, default=10000)
    ap.add_argument("--mesh_res_periodic", type=int, default=384)
    ap.add_argument("--mesh_res_final", type=int, default=512)
    ap.add_argument("--wandb_project", default="geoneus-dtu")
    ap.add_argument("--wandb_name", default=None)
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    dtu_eval_dir = Path(args.dtu_eval_dir).resolve()
    scan_id = args.scan_id if args.scan_id is not None else parse_scan_id(args.case)

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name or run_dir.name,
        dir=str(run_dir),
        config={**vars(args), "scan_id": scan_id},
        sync_tensorboard=True,
    )

    sys.path.insert(0, str(GEONEUS_DIR))
    os.chdir(GEONEUS_DIR)

    # Stub out Geo-Neus's `models.mesh_filtering` (requires pytorch3d, which
    # doesn't build against torch 2.8/cu128). The wrapper applies IDR cleaning
    # (largest connected component) to every mesh before chamfer eval — same
    # cleaning we use for NeuS, so the NeuS↔Geo-Neus comparison stays fair.
    import types as _types
    _stub = _types.ModuleType("models.mesh_filtering")
    _stub.mesh_filter = lambda *a, **kw: print("[mesh_filter] stubbed (pytorch3d unavailable); IDR cleaning runs in the wrapper instead.", flush=True)
    sys.modules["models.mesh_filtering"] = _stub

    from exp_runner import Runner  # noqa: E402

    runner = Runner(args.conf, args.mode, args.case, args.is_continue)

    # ---- Geo-Neus-on-modern-PyTorch device fix --------------------------------
    # Geo-Neus dataset builds pixel coords on CPU then matmuls with CUDA
    # intrinsics. Patch the two ray-gen methods at the instance level so we
    # don't touch the Geo-Neus source.
    import types, torch
    def _gen_random_rays_at(self, img_idx, batch_size):
        src_idx = self.src_idx[img_idx][:9]
        idx_list = torch.cat([img_idx.clone().detach().unsqueeze(0), src_idx], dim=0).cuda()
        poses_pair = self.pose_all[idx_list]
        intrinsics_pair = self.intrinsics_all[idx_list]
        intrinsics_inv_pair = self.intrinsics_all_inv[idx_list]
        images_gray_pair = self.images_gray[idx_list]
        dev = self.intrinsics_all_inv.device
        pixels_x = torch.randint(low=0, high=self.W, size=[batch_size])
        pixels_y = torch.randint(low=0, high=self.H, size=[batch_size])
        color = self.images[img_idx][(pixels_y, pixels_x)]
        mask  = self.masks[img_idx][(pixels_y, pixels_x)]
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float().to(dev)
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze()
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)
        rays_v = torch.matmul(self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()
        rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape)
        return (torch.cat([rays_o.cpu(), rays_v.cpu(), color, mask[:, :1]], dim=-1).cuda(),
                intrinsics_pair, intrinsics_inv_pair, poses_pair, images_gray_pair)

    def _gen_rays_at(self, img_idx, resolution_level=1):
        src_idx = self.src_idx[img_idx][:9]
        idx_list = torch.cat([torch.tensor(img_idx).unsqueeze(0), src_idx], dim=0)
        poses_pair = self.pose_all[idx_list]
        intrinsics_pair = self.intrinsics_all[idx_list]
        intrinsics_inv_pair = self.intrinsics_all_inv[idx_list]
        images_gray_pair = self.images_gray[idx_list]
        dev = self.intrinsics_all_inv.device
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty, indexing="ij")
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).to(dev)
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, None, :3, :3], p[:, :, :, None]).squeeze()
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)
        rays_v = torch.matmul(self.pose_all[img_idx, None, None, :3, :3], rays_v[:, :, :, None]).squeeze()
        rays_o = self.pose_all[img_idx, None, None, :3, 3].expand(rays_v.shape)
        return (rays_o.transpose(0, 1), rays_v.transpose(0, 1),
                intrinsics_pair, intrinsics_inv_pair, poses_pair, images_gray_pair)

    runner.dataset.gen_random_rays_at = types.MethodType(_gen_random_rays_at, runner.dataset)
    runner.dataset.gen_rays_at        = types.MethodType(_gen_rays_at,        runner.dataset)
    # ---------------------------------------------------------------------------

    orig_validate_mesh = runner.validate_mesh

    def validate_and_eval(world_space=False, resolution=64, threshold=0.0):
        step_now = runner.iter_step
        is_periodic_eval = (args.eval_every > 0 and step_now > 0
                            and step_now % args.eval_every == 0)
        if is_periodic_eval:
            world_space = True
            resolution = args.mesh_res_periodic
            # Geo-Neus uses `self.suffix` in the mesh filename
            # (`output_mesh{suffix}.ply`). Stamping the step here makes each
            # periodic save unique instead of overwriting the previous one.
            runner.suffix = f"_{step_now:08d}"
        orig_validate_mesh(world_space=world_space,
                           resolution=resolution, threshold=threshold)
        if not is_periodic_eval:
            return
        step = step_now
        mesh_ply = Path(runner.base_exp_dir) / "meshes" / f"output_mesh_{step:08d}.ply"
        if not mesh_ply.exists():
            print(f"[dtu_eval] mesh not found: {mesh_ply}", flush=True)
            return
        mesh_ply = idr_clean_largest_component(mesh_ply)
        out_dir = run_dir / "dtu_eval" / f"iter{step:08d}"
        payload = run_dtu_eval(mesh_ply, scan_id, dtu_eval_dir, out_dir)
        if payload is None:
            return
        wandb.log({"dtu/chamfer":      payload["chamfer"],
                   "dtu/accuracy":     payload["accuracy"],
                   "dtu/completeness": payload["completeness"]}, step=step)
        print(f"[dtu_eval@{step:5d}] chamfer={payload['chamfer']:.4f}mm  "
              f"acc={payload['accuracy']:.4f}mm  "
              f"comp={payload['completeness']:.4f}mm", flush=True)

    runner.validate_mesh = validate_and_eval

    if args.mode == "train":
        runner.train()
        runner.validate_mesh(world_space=True, resolution=args.mesh_res_final, threshold=0.0)
    elif args.mode == "validate_mesh":
        runner.validate_mesh(world_space=True, resolution=args.mesh_res_final, threshold=0.0)

    wandb.finish()


if __name__ == "__main__":
    main()
