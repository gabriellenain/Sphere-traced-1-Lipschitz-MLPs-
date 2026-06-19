#!/usr/bin/env python3
"""Average sorted source-view residual curves for sphere-traced points.

For sampled reference-view hits x, evaluate the training ZNCC residual
    r_j(x) = 1 - ZNCC(ref patch, source_j patch)
for every source view whose projected patch is usable.  Per point, sort the
valid residuals and average each rank k across points that have rank k.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lip_tracer import data as data_mod
from lip_tracer.data import load_views
from lip_tracer.loss import pmvs_ncc_loss
from lip_tracer.model import make_model
from lip_tracer.sphere_tracing import trace_nograd, TraceConfig


def _load_config(run_dir: Path) -> dict:
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        cands = sorted(run_dir.glob("source_config*.json"))
        if not cands:
            raise FileNotFoundError(f"no config.json/source_config*.json under {run_dir}")
        cfg_path = cands[0]
    return json.loads(cfg_path.read_text())


def _load_checkpoint(run_dir: Path, ckpt: Path | None) -> tuple[Path, dict]:
    ckpt_path = ckpt
    if ckpt_path is None:
        ckpt_path = run_dir / "ckpt" / "checkpoint_latest.pt"
        if not ckpt_path.exists():
            steps = sorted((run_dir / "ckpt").glob("checkpoint_step_*.pt"))
            if not steps:
                raise FileNotFoundError(f"no checkpoint under {run_dir / 'ckpt'}")
            ckpt_path = steps[-1]
    sd = torch.load(ckpt_path, map_location="cpu")
    return ckpt_path, sd


def _grad_normals(f, x: torch.Tensor, chunk: int = 8192) -> torch.Tensor:
    outs = []
    for i in range(0, x.shape[0], chunk):
        xc = x[i:i + chunk].detach().requires_grad_(True)
        with torch.enable_grad():
            g = torch.autograd.grad(f(xc).sum(), xc)[0].detach()
        outs.append(g)
    g = torch.cat(outs, dim=0)
    return g / g.norm(dim=-1, keepdim=True).clamp(min=1e-6)


def _make_random_rays(
    views: dict,
    ref_views: list[int],
    n_candidates_per_ref: int,
    down: int,
    rng: np.random.Generator,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return origins, directions, ref_ids, linear pixel ids."""
    H, W = int(views["H"]), int(views["W"])
    c2w = views["c2w"].numpy()
    K = views["K"].numpy()
    origins, dirs, refs, pix = [], [], [], []
    for v in ref_views:
        ids = rng.choice(H * W, size=n_candidates_per_ref, replace=False)
        y = ids // W
        x = ids % W
        xf = (x.astype(np.float32) + 0.5) * down - 0.5
        yf = (y.astype(np.float32) + 0.5) * down - 0.5
        d_cam = np.stack([
            (xf - K[v, 0, 2]) / K[v, 0, 0],
            (yf - K[v, 1, 2]) / K[v, 1, 1],
            np.ones_like(xf),
        ], axis=-1)
        d_w = d_cam @ c2w[v, :3, :3].T
        d_w = d_w / np.linalg.norm(d_w, axis=-1, keepdims=True)
        origins.append(np.broadcast_to(c2w[v, :3, 3], d_w.shape).copy())
        dirs.append(d_w.astype(np.float32))
        refs.append(np.full(ids.shape, v, np.int64))
        pix.append(ids.astype(np.int64))
    return (
        torch.from_numpy(np.concatenate(origins)).float().to(device),
        torch.from_numpy(np.concatenate(dirs)).float().to(device),
        torch.from_numpy(np.concatenate(refs)).long().to(device),
        torch.from_numpy(np.concatenate(pix)).long().to(device),
    )


def _trace_hits(f, origins, dirs, trace_cfg, chunk: int) -> tuple[torch.Tensor, torch.Tensor]:
    xs, hs = [], []
    for i in range(0, origins.shape[0], chunk):
        with torch.no_grad():
            x, _, h = trace_nograd(f, origins[i:i + chunk], dirs[i:i + chunk], trace_cfg)
        xs.append(x)
        hs.append(h)
    return torch.cat(xs, dim=0), torch.cat(hs, dim=0)


def _project_centers(x3d, K, w2c):
    xc = x3d @ w2c[:3, :3].T + w2c[:3, 3]
    uvh = xc @ K.T
    uv = uvh[:, :2] / uvh[:, 2:3].clamp(min=1e-6)
    return uv, xc[:, 2]


def compute_scene(
    label: str,
    run_dir: Path,
    ckpt: Path | None,
    args,
) -> dict:
    device = args.device
    rng = np.random.default_rng(args.seed)
    cfg = _load_config(run_dir)
    ckpt_path, sd = _load_checkpoint(run_dir, ckpt)
    step = int(sd.get("step", 0)) if isinstance(sd, dict) else -1

    if "bake_background" in cfg.get("train", {}):
        data_mod.BAKE_BACKGROUND = bool(cfg["train"]["bake_background"])

    model = make_model(**cfg["model"]).to(device).eval()
    state = sd["f"] if isinstance(sd, dict) and "f" in sd else sd
    model.load_state_dict(state, strict=False)

    tcfg = cfg["train"]
    down = int(args.down if args.down is not None else tcfg.get("down", 1))
    views = load_views(Path(cfg["scene"]), down=down)
    H, W = int(views["H"]), int(views["W"])
    V = int(views["c2w"].shape[0])

    K_all = views["K"].float().to(device)
    c2w_all = views["c2w"].float().to(device)
    w2c_all = torch.linalg.inv(c2w_all)
    images = views["images"].float().to(device)

    if args.views:
        ref_views = [int(v) for v in args.views.split(",") if v.strip()]
    else:
        ref_views = list(np.linspace(0, V - 1, args.n_ref_views, dtype=int))

    trace_cfg = TraceConfig(**cfg["trace"])
    all_rows: list[np.ndarray] = []
    point_counts = []
    valid_counts = []
    sampled = 0
    attempts = 0

    source_all = torch.arange(V, device=device)
    print(f"[{label}] run={run_dir} ckpt={ckpt_path.name} step={step} V={V} HxW={H}x{W}")
    print(f"[{label}] ref_views={ref_views} target_points={args.n_points}")

    while sampled < args.n_points and attempts < args.max_rounds:
        attempts += 1
        o, d, ref, _ = _make_random_rays(
            views, ref_views, args.candidates_per_ref, down, rng, device)
        x, hit = _trace_hits(model, o, d, trace_cfg, args.trace_chunk)
        if not hit.any():
            continue
        x = x[hit]
        ref = ref[hit]
        if x.shape[0] > args.points_per_round:
            sel = torch.from_numpy(
                rng.choice(x.shape[0], args.points_per_round, replace=False)
            ).long().to(device)
            x = x[sel]
            ref = ref[sel]
        n = _grad_normals(model, x, chunk=args.normal_chunk)

        for start in range(0, x.shape[0], args.point_chunk):
            xb = x[start:start + args.point_chunk]
            nb = n[start:start + args.point_chunk]
            rb = ref[start:start + args.point_chunk]
            B = xb.shape[0]
            residual_cols = []
            for s in range(V):
                src = torch.full((B,), s, device=device, dtype=torch.long)
                same = src == rb
                if same.all():
                    continue
                # Cheap center gate before patch construction. pmvs_ncc_loss
                # still enforces full-patch in-bounds and texture validity.
                uv, z = _project_centers(xb, K_all[s], w2c_all[s])
                cam = c2w_all[s, :3, 3]
                dp = cam[None] - xb
                dp = dp / dp.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                cos_ok = (nb * dp).sum(dim=-1).abs() > float(tcfg.get("cos_thresh", 0.1))
                center_ok = (
                    (~same) & cos_ok & (z > 0)
                    & (uv[:, 0] >= 0) & (uv[:, 0] < W)
                    & (uv[:, 1] >= 0) & (uv[:, 1] < H)
                )
                if not center_ok.any():
                    residual_cols.append(torch.full((B,), float("nan"), device=device))
                    continue
                zncc = torch.full((B,), float("nan"), device=device)
                zvals = pmvs_ncc_loss(
                    images, xb[center_ok], nb[center_ok],
                    rb[center_ok], src[center_ok],
                    K_all, w2c_all, H, W,
                    int(tcfg.get("ncc_patch", 5)),
                    float(tcfg.get("ncc_half_pix", 2.0)),
                    tcfg.get("sample_mode", "bilinear"),
                    float(tcfg.get("gaussian_sigma", 2.0)),
                    int(tcfg.get("gaussian_radius", 1)),
                    float(tcfg.get("ncc_min", 0.0)),
                    return_full=True,
                    ncc_color=tcfg.get("ncc_color", "gray"),
                    ncc_grad_alpha=float(tcfg.get("ncc_grad_alpha", 0.0)),
                    patch_wsigma=float(tcfg.get("ncc_patch_wsigma", 0.0)),
                    patch_bilateral_gamma=float(tcfg.get("ncc_bilateral_gamma", 0.0)),
                )[-1]
                zncc[center_ok] = zvals
                residual_cols.append(1.0 - zncc)
            R = torch.stack(residual_cols, dim=1).detach().cpu().numpy()
            R.sort(axis=1)
            counts = np.isfinite(R).sum(axis=1)
            keep = counts >= args.min_valid_sources
            if keep.any():
                all_rows.append(R[keep])
                valid_counts.extend(counts[keep].tolist())
                sampled += int(keep.sum())
            if sampled >= args.n_points:
                break
        print(f"[{label}] round={attempts} sampled={sampled} latest_hits={int(hit.sum())}")

    if not all_rows:
        raise RuntimeError(f"{label}: no points with >= {args.min_valid_sources} valid sources")

    R = np.concatenate(all_rows, axis=0)[:args.n_points]
    counts = np.asarray(valid_counts[:R.shape[0]], dtype=np.int32)
    max_rank = int(np.nanmax(counts))
    mean = np.full(max_rank, np.nan, dtype=np.float64)
    median = np.full(max_rank, np.nan, dtype=np.float64)
    n_rank = np.zeros(max_rank, dtype=np.int32)
    for k in range(max_rank):
        vals = R[:, k]
        vals = vals[np.isfinite(vals)]
        if vals.size:
            mean[k] = vals.mean()
            median[k] = np.median(vals)
            n_rank[k] = vals.size

    return {
        "label": label,
        "run_dir": str(run_dir),
        "ckpt": str(ckpt_path),
        "step": step,
        "n_points": int(R.shape[0]),
        "valid_counts": counts,
        "mean": mean,
        "median": median,
        "n_rank": n_rank,
        "residual_rows": R,
    }


def _write_scene_outputs(result: dict, out_dir: Path) -> None:
    label = result["label"]
    np.savez_compressed(
        out_dir / f"{label}_sorted_residuals.npz",
        residual_rows=result["residual_rows"],
        valid_counts=result["valid_counts"],
        mean=result["mean"],
        median=result["median"],
        n_rank=result["n_rank"],
        step=result["step"],
        run_dir=result["run_dir"],
        ckpt=result["ckpt"],
    )
    csv = out_dir / f"{label}_sorted_residual_curve.csv"
    with csv.open("w") as f:
        f.write("rank,mean_residual,median_residual,n_points_with_rank\n")
        for i, (m, md, n) in enumerate(zip(result["mean"], result["median"], result["n_rank"]), start=1):
            f.write(f"{i},{m:.8g},{md:.8g},{int(n)}\n")


def _plot(results: list[dict], out_dir: Path) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    for i, res in enumerate(results):
        k = np.arange(1, len(res["mean"]) + 1)
        ax.plot(k, res["mean"], lw=1.7, marker="o", ms=2.4,
                color=colors[i % len(colors)],
                label=f"{res['label']} (n={res['n_points']})")
    ax.set_xlabel("sorted source-view rank k")
    ax.set_ylabel("mean residual E[1 - ZNCC_(k)]")
    ax.set_title("Average sorted valid source-view residuals")
    ax.grid(True, color="#dddddd", lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "sorted_source_residual_curves.png", dpi=300, facecolor="white")
    fig.savefig(out_dir / "sorted_source_residual_curves.pdf", facecolor="white")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", action="append", nargs=4, metavar=("LABEL", "RUN_DIR", "CKPT", "VIEWS"),
                    required=True,
                    help="Scene spec. Use CKPT='auto' and VIEWS='' for defaults.")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/sorted_source_residual_curves"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--down", type=int, default=None)
    ap.add_argument("--n-ref-views", type=int, default=8)
    ap.add_argument("--views", default="", help="global comma-list override for every scene")
    ap.add_argument("--n-points", type=int, default=800)
    ap.add_argument("--min-valid-sources", type=int, default=2)
    ap.add_argument("--candidates-per-ref", type=int, default=1200)
    ap.add_argument("--points-per-round", type=int, default=512)
    ap.add_argument("--point-chunk", type=int, default=128)
    ap.add_argument("--trace-chunk", type=int, default=8192)
    ap.add_argument("--normal-chunk", type=int, default=4096)
    ap.add_argument("--max-rounds", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for label, run_s, ckpt_s, views_s in args.scene:
        local_args = argparse.Namespace(**vars(args))
        local_args.views = args.views if args.views else views_s
        ckpt = None if ckpt_s == "auto" else Path(ckpt_s)
        res = compute_scene(label, Path(run_s), ckpt, local_args)
        _write_scene_outputs(res, args.out_dir)
        results.append(res)
        vc = res["valid_counts"]
        print(f"[{label}] valid sources: mean={vc.mean():.2f} median={np.median(vc):.1f} "
              f"p90={np.percentile(vc, 90):.1f} max={vc.max()}")
    _plot(results, args.out_dir)
    print(f"[done] wrote {args.out_dir / 'sorted_source_residual_curves.png'}")
    print(f"[done] wrote {args.out_dir / 'sorted_source_residual_curves.pdf'}")


if __name__ == "__main__":
    main()
