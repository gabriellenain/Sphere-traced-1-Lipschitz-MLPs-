"""One-shot compute / memory profile of the SDF network.

Static facts (params, analytic FLOPs per SDF eval) plus a single measured
forward and forward+backward pass with peak CUDA memory. Called once at
startup behind TrainConfig.profile — never on the training hot path.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from .model import ConvexPotentialLayer


def _fmt(n: float) -> str:
    for unit in ("", "K", "M", "G", "T"):
        if abs(n) < 1000.0:
            return f"{n:7.2f}{unit}"
        n /= 1000.0
    return f"{n:7.2f}P"


def _eval_flops(m: nn.Module, batch: int) -> int:
    """Analytic FLOPs (mul+add = 2) for one SDF eval over `batch` points."""
    if isinstance(m, ConvexPotentialLayer):
        d = m.weight.shape[0]
        # two d×d matvecs per point (Wx and Wᵀ·relu)
        return 2 * (2 * batch * d * d)
    if isinstance(m, nn.Linear):
        return 2 * batch * m.in_features * m.out_features
    return 0


def profile_model(
    f: nn.Module,
    device: torch.device | str,
    batch: int = 4096,
    iters: int = 1,
) -> dict:
    """Print a per-component param/FLOP table + measured fwd / fwd+bwd / peak mem.

    `iters` mirrors trace iters so the FLOP total reflects the real per-step
    cost (the SDF is evaluated `iters` times per ray inside sphere tracing).
    Returns a flat dict of scalar metrics (suitable for wandb.summary).
    """
    device = torch.device(device)
    f = f.to(device).eval()

    # ---- static: params + analytic FLOPs, grouped by named child ----
    rows: list[tuple[str, int, int]] = []
    counted: set[int] = set()
    for name, child in f.named_children():
        p = sum(t.numel() for t in child.parameters())
        fl = 0
        for sub in child.modules():
            if id(sub) in counted:
                continue
            counted.add(id(sub))
            fl += _eval_flops(sub, batch)
        rows.append((name or "<root>", p, fl))
    head_p = sum(
        t.numel() for n, t in f.named_parameters() if not any(n.startswith(c + ".") for c, _ in f.named_children())
    )
    if head_p:
        rows.append(("head/params", head_p, 2 * batch * getattr(f, "hidden", 0)))

    total_p = sum(t.numel() for t in f.parameters())
    total_fl = sum(r[2] for r in rows)
    param_mb = sum(t.numel() * t.element_size() for t in f.parameters()) / 1e6

    print("  ── model profile " + "─" * 52)
    print(f"  {'component':<22}{'params':>12}{'FLOPs/eval':>14}")
    for name, p, fl in rows:
        print(f"  {name:<22}{p:>12,}{_fmt(fl):>14}")
    print(f"  {'TOTAL':<22}{total_p:>12,}{_fmt(total_fl):>14}")
    print(f"  param memory: {param_mb:.2f} MB   "
          f"batch={batch}  trace_iters={iters}  "
          f"FLOPs/step ≈ {_fmt(total_fl * iters)}")

    # ---- measured: latency + peak memory ----
    x = torch.randn(batch, 3, device=device)
    cuda = device.type == "cuda"

    def _sync():
        if cuda:
            torch.cuda.synchronize()

    metrics = {
        "profile/params": total_p,
        "profile/flops_per_eval": total_fl,
        "profile/flops_per_step": total_fl * iters,
        "profile/param_mb": param_mb,
    }

    for _ in range(3):  # warm up
        f(x).sum()
    _sync()

    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    t0 = torch.cuda.Event(enable_timing=True) if cuda else None
    t1 = torch.cuda.Event(enable_timing=True) if cuda else None
    import time
    with torch.no_grad():
        if cuda:
            t0.record()
        else:
            _w = time.perf_counter()
        for _ in range(10):
            f(x)
        if cuda:
            t1.record()
            _sync()
            fwd_ms = t0.elapsed_time(t1) / 10
        else:
            fwd_ms = (time.perf_counter() - _w) * 1e3 / 10
    fwd_mem = torch.cuda.max_memory_allocated(device) / 1e6 if cuda else 0.0

    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    xg = x.clone().requires_grad_(True)
    _wb = time.perf_counter()
    if cuda:
        t0.record()
    for _ in range(10):
        f.zero_grad(set_to_none=True)
        out = f(xg)
        g = torch.autograd.grad(out.sum(), xg, create_graph=False)[0]
        g.sum()
    if cuda:
        t1.record()
        _sync()
        bwd_ms = t0.elapsed_time(t1) / 10
    else:
        bwd_ms = (time.perf_counter() - _wb) * 1e3 / 10
    bwd_mem = torch.cuda.max_memory_allocated(device) / 1e6 if cuda else 0.0

    metrics.update({
        "profile/fwd_ms": fwd_ms,
        "profile/fwd_bwd_ms": bwd_ms,
        "profile/peak_mem_fwd_mb": fwd_mem,
        "profile/peak_mem_fwd_bwd_mb": bwd_mem,
    })
    print(f"  fwd: {fwd_ms:.3f} ms   fwd+grad: {bwd_ms:.3f} ms   "
          f"(batch={batch}, single SDF eval)")
    if cuda:
        print(f"  peak CUDA mem  fwd: {fwd_mem:.1f} MB   "
              f"fwd+grad: {bwd_mem:.1f} MB")
    print("  " + "─" * 69)

    f.train()
    return metrics
