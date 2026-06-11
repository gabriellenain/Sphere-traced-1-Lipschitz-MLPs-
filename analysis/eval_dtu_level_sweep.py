#!/usr/bin/env python3
"""Evaluate DTU official metrics across marching-cubes iso-levels."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import trimesh

from eval_dtu_official import DTUEVAL_DIR, PIXI_PYTHON, _ensure_dtueval
from lip_tracer.data import load_views
from render_paper_marching import (
    _camera_rays,
    _checkpoint_model_kwargs,
    _level_tag,
    _mesh_from_volume,
    _mesh_hits,
    _resolve_trained_scene,
)
from lip_tracer.model import make_model


def _parse_levels(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def _infer_scan_id(scene: Path) -> int:
    import re

    m = re.search(r"scan(\d+)", str(scene))
    if not m:
        raise ValueError(f"Could not infer scan id from scene path: {scene}")
    return int(m.group(1))


def _eval_python() -> str:
    try:
        import open3d  # noqa: F401

        return sys.executable
    except ImportError:
        return str(PIXI_PYTHON) if PIXI_PYTHON.exists() else sys.executable


def _extract_sdf_volume(ckpt_path: Path, bound: float, res: int,
                        device: str, chunk: int | None = None) -> np.ndarray:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    f = make_model(**_checkpoint_model_kwargs(ckpt)).to(device).eval()
    f.load_state_dict(ckpt["f"], strict=False)
    with torch.enable_grad():
        f(torch.zeros(1, 3, device=device))

    if chunk is None:
        chunk = 4096 if device == "cuda" else 65536

    vox = torch.linspace(-bound, bound, res)
    grid = torch.stack(torch.meshgrid(vox, vox, vox, indexing="ij"), dim=-1).reshape(-1, 3)
    n_total = len(grid)
    n_chunks = (n_total + chunk - 1) // chunk
    log_every = max(1, n_chunks // 20)
    out = []
    t0 = time.time()
    print(
        f"[sdf] evaluating {n_total:,} grid points "
        f"({n_chunks} chunks of {chunk}, device={device})",
        flush=True,
    )
    with torch.no_grad():
        for ci, i in enumerate(range(0, n_total, chunk)):
            out.append(f(grid[i:i + chunk].to(device)).detach().cpu())
            if (ci + 1) % log_every == 0 or ci + 1 == n_chunks:
                done = min(i + chunk, n_total)
                elapsed = time.time() - t0
                eta = elapsed * (n_total / done - 1) if done else 0.0
                print(
                    f"      [{done:>10,} / {n_total:,}] "
                    f"{100.0 * done / n_total:5.1f}%  "
                    f"elapsed {elapsed:6.1f}s  eta {eta:6.1f}s",
                    flush=True,
                )
    vol = torch.cat(out).reshape(res, res, res).numpy()
    del f, grid, out
    if device != "cpu":
        torch.cuda.empty_cache()
    print(f"[sdf] range=[{vol.min():.4f}, {vol.max():.4f}]", flush=True)
    return vol


def _to_world_mesh(mesh: trimesh.Trimesh, scale_mat: np.ndarray) -> trimesh.Trimesh:
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    pts_h = np.concatenate([verts, np.ones((len(verts), 1), dtype=np.float32)], axis=1)
    verts_world = (scale_mat @ pts_h.T).T[:, :3].astype(np.float32)
    return trimesh.Trimesh(vertices=verts_world, faces=mesh.faces, process=False)


def _boundary_edges(mesh: trimesh.Trimesh) -> int:
    faces = np.asarray(mesh.faces, dtype=np.int64)
    edges = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]],
        axis=0,
    )
    edges.sort(axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return int((counts == 1).sum())


def _missing_fg(mesh: trimesh.Trimesh, scene: Path, view_ids: list[int],
                down: int, ss: int) -> tuple[int, int, float]:
    intersector = mesh.ray
    views = load_views(scene=scene, down=down)
    masks = views.get("masks")
    if masks is None:
        imgs = views["images"]
        masks = imgs.mean(dim=-1) > 1e-3

    missing_total = 0
    fg_total = 0
    for vi in view_ids:
        origins, dirs, Hs, Ws = _camera_rays(
            views["c2w"][vi].numpy(),
            views["K"][vi].numpy(),
            views["H"],
            views["W"],
            ss,
        )
        hit, _, _, _ = _mesh_hits(mesh, intersector, origins, dirs)
        hit_alpha = hit.astype(np.float64).reshape(Hs, Ws, 1)
        hit_alpha = hit_alpha.reshape(views["H"], ss, views["W"], ss, 1).mean(axis=(1, 3))[..., 0]
        gt_fg = masks[vi].cpu().numpy().astype(bool)
        missing = gt_fg & (hit_alpha == 0.0)
        missing_total += int(missing.sum())
        fg_total += int(gt_fg.sum())

    pct = 100.0 * missing_total / max(fg_total, 1)
    return missing_total, fg_total, pct


def _run_official(eval_script: Path, eval_python: str, mesh_path: Path,
                  scan_id: int, dtu_eval_dir: Path, out_dir: Path) -> tuple[float, float, float]:
    cmd = [
        eval_python,
        str(eval_script),
        "--data",
        str(mesh_path),
        "--scan",
        str(scan_id),
        "--mode",
        "mesh",
        "--dataset_dir",
        str(dtu_eval_dir),
        "--vis_out_dir",
        str(out_dir),
    ]
    print(f"[official] {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    (out_dir / "official_stdout.txt").write_text(result.stdout)
    (out_dir / "official_stderr.txt").write_text(result.stderr)
    if result.stdout:
        print(result.stdout, end="", flush=True)
    if result.stderr:
        print("[stderr]\n" + result.stderr, end="", flush=True)
    if result.returncode != 0:
        raise RuntimeError(f"DTUeval-python failed with code {result.returncode}")
    acc, comp, chamfer = [float(v) for v in result.stdout.strip().splitlines()[-1].split()]
    return chamfer, acc, comp


def _write_tables(rows: list[dict], out_dir: Path) -> None:
    fields = ["level", "chamfer", "acc", "comp", "missing_fg", "boundary_edges"]
    with (out_dir / "level_sweep.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fields})
    (out_dir / "level_sweep.json").write_text(json.dumps(rows, indent=2))

    lines = [
        "| level | chamfer | acc | comp | missing_fg | boundary_edges |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['level']:+.4f} | {row['chamfer']:.4f} | "
            f"{row['acc']:.4f} | {row['comp']:.4f} | "
            f"{row['missing_fg']:.2f}% | {row['boundary_edges']} |"
        )
    (out_dir / "level_sweep.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--scene", type=Path, default=None)
    ap.add_argument("--scan-id", type=int, default=None)
    ap.add_argument("--dtu-eval-dir", type=Path, required=True)
    ap.add_argument("--levels", type=str, default="-0.005,0.000,0.0025,0.005,0.0075,0.010")
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--bound", type=float, default=1.0)
    ap.add_argument("--chunk", type=int, default=None)
    ap.add_argument("--out", type=Path, default=Path("outputs/dtu_level_sweep"))
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--views", type=str, default=None,
                    help="comma-separated view ids for missing_fg; default: evenly spaced train views")
    ap.add_argument("--n-views", type=int, default=5)
    ap.add_argument("--down", type=int, default=1)
    ap.add_argument("--ss", type=int, default=1)
    args = ap.parse_args()

    trained_scene = _resolve_trained_scene(args.ckpt)
    if trained_scene is not None:
        args.scene = trained_scene if args.scene is None else args.scene
    if args.scene is None:
        ap.error("--scene required when checkpoint config does not contain a scene")

    scan_id = args.scan_id if args.scan_id is not None else _infer_scan_id(args.scene)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    levels = _parse_levels(args.levels)
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    views = load_views(scene=args.scene, down=args.down)
    n_total = views["c2w"].shape[0]
    if args.views:
        view_ids = [int(v) for v in args.views.split(",")]
    else:
        k = min(args.n_views, n_total)
        view_ids = np.linspace(0, n_total - 1, k).round().astype(int).tolist()

    print(f"[sweep] levels={levels}", flush=True)
    print(f"[sweep] scan={scan_id} scene={args.scene}", flush=True)
    print(f"[sweep] missing_fg views={view_ids} down={args.down} ss={args.ss}", flush=True)

    eval_script = _ensure_dtueval()
    eval_python = _eval_python()
    scale_mat = np.load(args.scene / "cameras.npz")["scale_mat_0"].astype(np.float64)
    vol = _extract_sdf_volume(args.ckpt, args.bound, args.res, device, chunk=args.chunk)

    rows = []
    for level in levels:
        tag = _level_tag(level)
        level_dir = out_dir / f"level_{tag}"
        level_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== level {level:+.4f} ===", flush=True)
        mesh = _mesh_from_volume(vol, args.bound, args.res, level)
        mesh_path = level_dir / "pred_world_mesh.ply"
        _to_world_mesh(mesh, scale_mat).export(mesh_path)
        missing_px, fg_px, missing_pct = _missing_fg(mesh, args.scene, view_ids, args.down, args.ss)
        boundary = _boundary_edges(mesh)
        chamfer, acc, comp = _run_official(
            eval_script, eval_python, mesh_path, scan_id, args.dtu_eval_dir, level_dir
        )
        rows.append({
            "level": float(level),
            "chamfer": float(chamfer),
            "acc": float(acc),
            "comp": float(comp),
            "missing_fg": float(missing_pct),
            "missing_pixels": int(missing_px),
            "foreground_pixels": int(fg_px),
            "boundary_edges": int(boundary),
            "mesh": str(mesh_path),
        })
        _write_tables(rows, out_dir)

    print(f"[done] {out_dir}", flush=True)


if __name__ == "__main__":
    main()
