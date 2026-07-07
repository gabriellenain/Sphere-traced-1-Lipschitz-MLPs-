"""Run NeuS unmodified, mirror its TB writer to wandb, and after every
mesh validation call DTUeval-python directly — the same code path our model
uses in `_run_dtu_official_eval` — and log {chamfer, accuracy, completeness}
to wandb at the same step.

NeuS code is imported as-is from baselines/NeuS — no edits to that repo.
The eval hook is installed by monkey-patching Runner.validate_mesh on the
instance.
"""
from __future__ import annotations
import argparse, json, os, re, subprocess, sys
from pathlib import Path

import numpy as np
import wandb

REPO = Path(__file__).resolve().parents[1]
NEUS_DIR = REPO / "baselines" / "NeuS"
DTU_EVAL_SCRIPT = REPO / "DTUeval-python" / "eval.py"
TNT_EVAL_SCRIPT = REPO / "analysis" / "eval_tnt_official.py"
BMVS_EVAL_SCRIPT = REPO / "analysis" / "eval_bmvs_chamfer.py"
PIXI_PYTHON = Path("/home/glenain/nerfstudio/.pixi/envs/default/bin/python3")


def parse_scan_id(case: str) -> int:
    m = re.search(r"\d+", case)
    if not m:
        raise ValueError(f"Could not parse scan id from case={case!r}")
    return int(m.group(0))


def _eval_python() -> str:
    """Same logic as lip_tracer.train._run_dtu_official_eval: prefer the current
    env if it has open3d, else fall back to the pixi env that has it."""
    try:
        import open3d  # noqa: F401
        return sys.executable
    except ImportError:
        return str(PIXI_PYTHON) if PIXI_PYTHON.exists() else sys.executable


def idr_clean_largest_component(mesh_ply: Path) -> Path:
    """IDR's cleaning (eval.py): take the largest connected component by area.
    This is the cleaning step used to produce published NeuS/VolSDF/Geo-Neus
    DTU numbers. Writes <name>_clean.ply alongside the input and returns it.
    """
    import trimesh
    mesh = trimesh.load(str(mesh_ply), force="mesh", process=False)
    components = mesh.split(only_watertight=False)
    if len(components) == 0:
        return mesh_ply
    areas = np.array([c.area for c in components], dtype=np.float64)
    cleaned = components[int(areas.argmax())]
    out = mesh_ply.with_name(mesh_ply.stem + "_clean.ply")
    cleaned.export(str(out))
    print(f"[clean] {mesh_ply.name}: {len(components)} components "
          f"→ kept largest (area={areas.max():.2f}, frac={areas.max()/areas.sum():.3f}) "
          f"→ {out.name}", flush=True)
    return out


def run_dtu_eval(mesh_ply: Path, scan_id: int, dtu_eval_dir: Path,
                 out_dir: Path) -> dict | None:
    """Direct DTUeval-python call. Parses last stdout line as `acc comp chamfer`,
    matching lip_tracer.train._run_dtu_official_eval."""
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
        print(f"[dtu_eval] could not parse metrics from {out_dir}/official_stdout.txt", flush=True)
        return None
    payload = {"accuracy": acc, "completeness": comp, "chamfer": chamfer}
    (out_dir / "dtu_official.json").write_text(json.dumps(payload, indent=2))
    return payload


def run_tnt_eval(mesh_ply: Path, scene: str, scene_dir: Path, gt_dir: Path,
                 out_dir: Path) -> dict | None:
    """Run the official Tanks&Temples F-score on a world-frame mesh via
    analysis/eval_tnt_official.py (--mesh, --frame colmap-pose). The NeuS mesh
    is extracted with world_space=True, i.e. already in the raw pose/*.txt
    (own-COLMAP) frame the eval aligns against. Parses out/fscore.json."""
    if not TNT_EVAL_SCRIPT.exists():
        print(f"[tnt_eval] missing {TNT_EVAL_SCRIPT}", flush=True)
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(TNT_EVAL_SCRIPT),
        "--mesh",      str(mesh_ply),
        "--scene",     scene,
        "--scene-dir", str(scene_dir),
        "--gt-dir",    str(gt_dir),
        "--frame",     "colmap-pose",
        "--out",       str(out_dir),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    (out_dir / "official_stdout.txt").write_text(res.stdout)
    (out_dir / "official_stderr.txt").write_text(res.stderr)
    fscore_json = out_dir / "fscore.json"
    if res.returncode != 0 or not fscore_json.exists():
        print(f"[tnt_eval] failed rc={res.returncode}; see {out_dir}", flush=True)
        return None
    return json.loads(fscore_json.read_text())


def run_bmvs_eval(mesh_ply: Path, case: str, scene_dir: Path, gt_mesh: Path,
                  out_dir: Path) -> dict | None:
    """Official BMVS Chamfer on a world-frame NeuS mesh via
    analysis/eval_bmvs_chamfer.py. NeuS meshes are exported world_space=True, so
    --mesh-space world brings them into the normalized frame; the rest mirrors
    the b15 in-training recipe (probesdf protocol, point-to-mesh, bound 1.5,
    n=300k, mask-crop dilate12/all/0.95, normalized GroundTruth). Parses
    out/bmvs_chamfer.json."""
    if not BMVS_EVAL_SCRIPT.exists():
        print(f"[bmvs_eval] missing {BMVS_EVAL_SCRIPT}", flush=True)
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        _eval_python(), str(BMVS_EVAL_SCRIPT),
        "--mesh", str(mesh_ply), "--mesh-space", "world",
        "--scene", str(scene_dir),
        "--gt-mesh", str(gt_mesh), "--gt-space", "normalized",
        "--protocol", "probesdf", "--metric", "point-to-mesh",
        "--bound", "1.5", "--n-points", "300000",
        "--mask-crop", "--mask-dilate", "12", "--mask-mode", "all",
        "--mask-min-ratio", "0.95", "--mask-min-views", "1",
        "--device", "cpu", "--out", str(out_dir),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    (out_dir / "official_stdout.txt").write_text(res.stdout)
    (out_dir / "official_stderr.txt").write_text(res.stderr)
    cham_json = out_dir / "bmvs_chamfer.json"
    if res.returncode != 0 or not cham_json.exists():
        print(f"[bmvs_eval] failed rc={res.returncode}; see {out_dir}", flush=True)
        return None
    return json.loads(cham_json.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", required=True, help="path to NeuS .conf (already case-resolved)")
    ap.add_argument("--case", required=True, help="e.g. scan65")
    ap.add_argument("--mode", default="train", help="train | validate_mesh | interpolate_<i>_<j>")
    ap.add_argument("--is_continue", action="store_true")
    ap.add_argument("--run_dir", required=True, help="single output dir (base_exp_dir)")
    ap.add_argument("--dataset", choices=["dtu", "tnt", "bmvs"], default="dtu",
                    help="dtu = DTU chamfer eval; tnt = official Tanks&Temples F-score; "
                         "bmvs = train + mesh only (no in-loop GT eval)")
    ap.add_argument("--dtu_eval_dir", default=None, help="DTU SampleSet/MVS Data path (dataset=dtu)")
    ap.add_argument("--tnt_scene", default=None, help="TnT scene name, e.g. Barn (dataset=tnt)")
    ap.add_argument("--tnt_scene_dir", default=None, help="TnT scene dir, e.g. data/tnt/Barn (dataset=tnt)")
    ap.add_argument("--tnt_gt_dir", default=None, help="TnT GT dir with <scene>.ply/_trans.txt/.json (dataset=tnt)")
    ap.add_argument("--scan_id", type=int, default=None, help="override; else parsed from --case")
    ap.add_argument("--eval_every", type=int, default=10000)
    ap.add_argument("--mesh_res_periodic", type=int, default=384,
                    help="MC resolution for periodic (every eval_every) chamfer — matches dtu_official_res")
    ap.add_argument("--mesh_res_final", type=int, default=512,
                    help="MC resolution for the final post-train mesh — matches analysis/eval_dtu_official.py --res")
    ap.add_argument("--wandb_project", default="neus-dtu")
    ap.add_argument("--wandb_name", default=None)
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if args.dataset == "dtu":
        if not args.dtu_eval_dir:
            raise SystemExit("--dtu_eval_dir is required for dataset=dtu")
        dtu_eval_dir = Path(args.dtu_eval_dir).resolve()
        scan_id = args.scan_id if args.scan_id is not None else parse_scan_id(args.case)
    elif args.dataset == "bmvs":
        scan_id = None
        bmvs_scene_dir = REPO / "data" / "bmvs" / args.case
        bmvs_gt_mesh = (REPO / "data" / "bmvs_gt" / "meshes" / "BMVS"
                        / args.case.replace("bmvs_", "") / "GroundTruth.ply")
        if not bmvs_scene_dir.exists():
            raise SystemExit(f"missing BMVS scene dir: {bmvs_scene_dir}")
        if not bmvs_gt_mesh.exists():
            raise SystemExit(f"missing BMVS GT mesh: {bmvs_gt_mesh}")
    else:
        for name in ("tnt_scene", "tnt_scene_dir", "tnt_gt_dir"):
            if not getattr(args, name):
                raise SystemExit(f"--{name} is required for dataset=tnt")
        tnt_scene_dir = Path(args.tnt_scene_dir).resolve()
        tnt_gt_dir = Path(args.tnt_gt_dir).resolve()
        scan_id = None

    # wandb FIRST so sync_tensorboard patches the SummaryWriter NeuS creates.
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name or run_dir.name,
        dir=str(run_dir),
        config={**vars(args), "scan_id": scan_id},
        sync_tensorboard=True,
    )

    sys.path.insert(0, str(NEUS_DIR))
    os.chdir(NEUS_DIR)  # NeuS resolves some paths relative to cwd
    from exp_runner import Runner  # noqa: E402

    runner = Runner(args.conf, args.mode, args.case, args.is_continue)

    # ---- NeuS-on-modern-PyTorch device fix --------------------------------
    # NeuS's dataset builds pixel coords on CPU, then matmuls with CUDA
    # intrinsics. Newer torch refuses implicit cross-device matmul. Patch
    # the dataset methods at the instance level (no edits to NeuS source).
    import types, torch
    def _gen_random_rays_at(self, img_idx, batch_size):
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
        return torch.cat([rays_o.cpu(), rays_v.cpu(), color, mask[:, :1]], dim=-1).cuda()

    def _gen_rays_at(self, img_idx, resolution_level=1):
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
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    runner.dataset.gen_random_rays_at = types.MethodType(_gen_random_rays_at, runner.dataset)
    runner.dataset.gen_rays_at        = types.MethodType(_gen_rays_at,        runner.dataset)
    # -----------------------------------------------------------------------

    orig_validate_mesh = runner.validate_mesh

    def validate_and_eval(world_space=False, resolution=64, threshold=0.0):
        step_now = runner.iter_step
        is_periodic_eval = (args.eval_every > 0 and step_now > 0
                            and step_now % args.eval_every == 0)
        if is_periodic_eval:
            # Override NeuS's res=64 default + ensure world frame for fair eval.
            world_space = True
            resolution = args.mesh_res_periodic
        orig_validate_mesh(world_space=world_space,
                           resolution=resolution, threshold=threshold)
        if not is_periodic_eval:
            return
        step = step_now
        mesh_ply = Path(runner.base_exp_dir) / "meshes" / f"{step:08d}.ply"
        if args.dataset == "bmvs":
            if not mesh_ply.exists():
                print(f"[eval] mesh not found: {mesh_ply}", flush=True)
                return
            out_dir = run_dir / "bmvs_eval" / f"iter{step:08d}"
            payload = run_bmvs_eval(mesh_ply, args.case, bmvs_scene_dir,
                                    bmvs_gt_mesh, out_dir)
            if payload is None:
                return
            wandb.log({"bmvs/chamfer":      payload["chamfer"],
                       "bmvs/accuracy":     payload["accuracy"],
                       "bmvs/completeness": payload["completeness"]}, step=step)
            print(f"[bmvs_eval@{step:5d}] chamfer={payload['chamfer']:.6f}  "
                  f"acc={payload['accuracy']:.6f}  "
                  f"comp={payload['completeness']:.6f}  (normalized units)", flush=True)
            return
        if not mesh_ply.exists():
            print(f"[eval] mesh not found: {mesh_ply}", flush=True)
            return
        if args.dataset == "tnt":
            # Mesh is already in the raw pose (own-COLMAP) frame (world_space=True);
            # the official eval crops to the GT region, so keep the full mesh.
            out_dir = run_dir / "tnt_eval" / f"iter{step:08d}"
            payload = run_tnt_eval(mesh_ply, args.tnt_scene, tnt_scene_dir,
                                   tnt_gt_dir, out_dir)
            if payload is None:
                return
            wandb.log({"tnt/fscore":    payload["fscore"],
                       "tnt/precision": payload["precision"],
                       "tnt/recall":    payload["recall"]}, step=step)
            print(f"[tnt_eval@{step:5d}] F={payload['fscore']:.4f}  "
                  f"P={payload['precision']:.4f}  "
                  f"R={payload['recall']:.4f}  (tau={payload['tau']})", flush=True)
            return
        # IDR cleaning (largest connected component) — same step used to produce
        # published NeuS/VolSDF/Geo-Neus DTU numbers.
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
        # Final hi-res mesh + eval at end_iter (in case it's not a multiple of val_mesh_freq).
        runner.validate_mesh(world_space=True, resolution=args.mesh_res_final, threshold=0.0)
    elif args.mode == "validate_mesh":
        runner.validate_mesh(world_space=True, resolution=args.mesh_res_final, threshold=0.0)
    elif args.mode.startswith("interpolate"):
        _, i, j = args.mode.split("_")
        runner.interpolate_view(int(i), int(j))

    wandb.finish()


if __name__ == "__main__":
    main()
