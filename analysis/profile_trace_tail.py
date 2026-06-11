"""Profile active-ray count per sphere-tracing iteration for one checkpoint.

Loads the model from --pt, builds deterministic rays for --scene at --down,
runs an instrumented nograd trace at high iters, and prints the per-iter
active-count curve (also dumps a PNG).

Usage:
    python -u profile_trace_tail.py --pt PATH.pt --scene PATH --views 16,32 \
        --iters 256 --down 2 --out outputs/trace_tail_<jobid>.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from lip_tracer.config import TraceConfig
from lip_tracer.data import load_views, make_deterministic_rays
from lip_tracer.model import make_model
from lip_tracer.sphere_tracing import ray_sphere_exit


def build_model_from_ckpt(ckpt_path: Path, device: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    architecture = ckpt.get("architecture", "cpl")
    if architecture == "neus":
        hidden = ckpt["f"]["layers.0.weight"].shape[0]
    elif "head_weight" in ckpt["f"]:
        hidden = ckpt["f"]["head_weight"].shape[0]
    else:
        hidden = next(v.shape[1] for k, v in ckpt["f"].items()
                      if k.endswith(".weight") and v.ndim == 2 and v.shape[0] != 1
                      and not k.startswith("encoder"))
    group_size = ckpt.get("group_size", 2)
    activation = ckpt.get("activation", "groupsort")
    input_encoding = ckpt.get("input_encoding", "pe")
    multires = ckpt.get("multires", 6)
    if architecture == "neus":
        depth = ckpt.get("depth", sum(1 for k in ckpt["f"]
                                      if k.startswith("layers.") and k.endswith(".weight")))
    else:
        depth = ckpt.get("depth", sum(1 for k in ckpt["f"]
                                      if k.startswith("net.") and k.endswith(".weight")
                                      and "_u" not in k))
    print(f"  arch={architecture} hidden={hidden} depth={depth} "
          f"group_size={group_size} act={activation} enc={input_encoding} mr={multires}",
          flush=True)
    f = make_model(hidden=hidden, depth=depth, group_size=group_size,
                   activation=activation, input_encoding=input_encoding,
                   multires=multires, architecture=architecture).to(device)
    f.load_state_dict(ckpt["f"], strict=False)
    with torch.enable_grad():
        f(torch.zeros(1, 3, device=device))  # populate caches
    f.eval()
    return f


@torch.no_grad()
def trace_with_tail_stats(f, o, d, iters, eps, bsphere_radius, chunk=65536):
    """Run nograd sphere trace, return (active_per_iter, sdf_min, hit, t)."""
    B = o.shape[0]
    active_per_iter_total = torch.zeros(iters, dtype=torch.long)
    sdf_min_all = torch.full((B,), float("inf"))
    hit_all = torch.zeros(B, dtype=torch.bool)
    t_all = torch.zeros(B)

    for s in range(0, B, chunk):
        oc = o[s:s+chunk]
        dc = d[s:s+chunk]
        bs = oc.shape[0]
        if bsphere_radius > 0:
            t_far = ray_sphere_exit(oc, dc, bsphere_radius)
        else:
            t_far = torch.full((bs,), 5.0, device=oc.device)
        t = torch.zeros(bs, device=oc.device)
        converged = torch.zeros(bs, dtype=torch.bool, device=oc.device)
        sdf_min = torch.full((bs,), float("inf"), device=oc.device)
        for i in range(iters):
            escaped = t >= t_far
            active = ~(converged | escaped)
            active_per_iter_total[i] += int(active.sum().item())
            if not active.any():
                break
            x = oc + t.unsqueeze(-1) * dc
            sdf = f(x)
            sdf_min = torch.minimum(sdf_min, sdf)
            converged = converged | (sdf.abs() < eps)
            step = torch.where(converged | escaped, torch.zeros_like(sdf), sdf)
            t = t + step
        hit = converged & (t < t_far) & (t >= 0)
        sdf_min_all[s:s+chunk] = sdf_min.cpu()
        hit_all[s:s+chunk] = hit.cpu()
        t_all[s:s+chunk] = t.cpu()
    return active_per_iter_total, sdf_min_all, hit_all, t_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--views", default="16,32",
                    help="comma-separated view indices to profile (whole image each)")
    ap.add_argument("--iters", type=int, default=256)
    ap.add_argument("--eps", type=float, default=1e-4)
    ap.add_argument("--down", type=int, default=2)
    ap.add_argument("--bsphere", type=float, default=1.5,
                    help="bounding-sphere radius for t_far (matches typical TraceConfig)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)
    print(f"loading {args.pt}", flush=True)
    f = build_model_from_ckpt(Path(args.pt), device)

    print(f"loading scene {args.scene}", flush=True)
    views = load_views(Path(args.scene), down=1)
    rays = make_deterministic_rays(views, down=args.down, device=device)
    o_all, d_all, vi_all = rays["o"], rays["d"], rays["vi"]
    rays_per_view = rays["rays_per_view"]
    print(f"  views={int(vi_all.max())+1} rays_per_view={rays_per_view} "
          f"total={o_all.shape[0]}", flush=True)

    view_ids = [int(v) for v in args.views.split(",")]
    masks = torch.zeros_like(vi_all, dtype=torch.bool)
    for v in view_ids:
        masks |= (vi_all == v)
    o = o_all[masks].to(device)
    d = d_all[masks].to(device)
    print(f"profiling views {view_ids}: {o.shape[0]} rays, iters={args.iters}",
          flush=True)

    active_per_iter, sdf_min, hit, t = trace_with_tail_stats(
        f, o, d, iters=args.iters, eps=args.eps, bsphere_radius=args.bsphere
    )

    B = o.shape[0]
    apc = active_per_iter.numpy()
    print("\n--- per-iter active counts ---", flush=True)
    print(f"  iter   active     %B    cum_done%", flush=True)
    cum_done = 0
    for i in range(args.iters):
        if apc[i] == 0 and i > 0 and apc[i-1] == 0:
            continue
        done = B - apc[i]
        cum_done = done
        pct = 100.0 * apc[i] / B
        if i < 30 or i % 10 == 0 or apc[i] < B * 0.01 or apc[i] == 0:
            print(f"  {i:4d}   {apc[i]:7d}   {pct:6.2f}   {100.0*cum_done/B:6.2f}",
                  flush=True)
        if apc[i] == 0:
            break

    print(f"\nfinal hit:    {int(hit.sum())} / {B} ({100.0*hit.sum()/B:.2f}%)", flush=True)
    print(f"final escape: {int(((~hit) & (t.cpu() >= 0)).sum())} / {B}", flush=True)

    # iterations needed to reach % thresholds
    for thr in (0.5, 0.9, 0.99, 0.999):
        target = B * (1 - thr)
        idxs = np.where(apc <= target)[0]
        if len(idxs) > 0:
            print(f"  {thr*100:5.1f}% done at iter {idxs[0]:4d} "
                  f"(active={apc[idxs[0]]})", flush=True)
        else:
            print(f"  {thr*100:5.1f}% never reached (min active={apc[apc>0].min() if (apc>0).any() else 0})",
                  flush=True)

    # cost ratio: full-batch vs compacted
    full_cost = args.iters * B
    compact_cost = int(apc.sum())
    print(f"\ncost full-batch:  {full_cost:>12d}", flush=True)
    print(f"cost compacted:   {compact_cost:>12d}", flush=True)
    print(f"speedup ratio:    {full_cost / max(compact_cost, 1):>12.2f}x", flush=True)

    # plot
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    iters_axis = np.arange(args.iters)
    axes[0].plot(iters_axis, apc, lw=1.5)
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("active rays")
    axes[0].set_title(f"active rays per iter (B={B}, views={view_ids})")
    axes[0].grid(alpha=0.3)
    axes[1].semilogy(iters_axis, np.maximum(apc, 1), lw=1.5)
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("active rays (log)")
    axes[1].set_title(f"log scale — cost saving = {full_cost/max(compact_cost,1):.1f}×")
    axes[1].grid(alpha=0.3, which="both")
    fig.suptitle(f"{Path(args.pt).name}  scene={Path(args.scene).name}")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"\nsaved plot: {out}", flush=True)

    # also dump raw counts as npz for later replotting
    npz = out.with_suffix(".npz")
    np.savez(npz, active_per_iter=apc, B=B, iters=args.iters,
             views=np.array(view_ids), pt=str(args.pt), scene=str(args.scene))
    print(f"saved data: {npz}", flush=True)


if __name__ == "__main__":
    main()
