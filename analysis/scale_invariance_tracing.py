"""Scale-(non)invariance of finite-budget sphere tracing.

A 1-Lipschitz SDF f_theta and its rescaled twin f_c(x) = f_theta(x) / c share the
*exact same zero set* {f_c = 0} = {f_theta = 0}: the converged surface is invariant
to c. But sphere tracing is the recurrence

    t_{k+1} = t_k + f_c(x_k) = t_k + f_theta(x_k) / c ,

so every step is rescaled by 1/c. With a *finite* budget of K steps the rendered
depth t_K is NOT scale invariant: c > 1 shrinks the steps (under-stepping, the ray
stalls short of the surface), c < 1 inflates them (over-stepping / oscillation).
Sphere tracing relies on |grad f| <= 1 to take maximal safe steps; rescaling the
field rescales that step budget. This script makes the effect explicit on a trained
checkpoint, holding the zero set fixed.

Oracle root t_star is computed per ray by a high-precision trace (large budget,
c = 1) followed by bisection of the first downward sign-change bracket to ~1e-7 --
t_star is scale independent by construction (it is a property of the sign of f).

Centrepiece: the number of iterations a ray needs to converge, N_conv(c), grows
linearly in c. For a unit-Lipschitz field a step shrinks residual by ~(1 - cos/c)
per iteration, so N_conv(c) ~= c * N_conv(1): rescaling by c demands c x more steps,
and once c * N_conv(1) exceeds the budget K the ray simply misses.

Note on training's bracketing: the production trace_nograd brackets a sign flip
f(x_i).f(x_{i+1}) < 0 and Newton-refines it, but that only fires on an *overshoot*.
Under-stepping (c > 1, the regime here) approaches the surface monotonically with
no sign flip, so bracketing never triggers -- the N_conv law below is exactly what
the training tracer does. (Over-stepping, c < 1, does flip sign and is recovered;
the +bracketing console column shows it.)

Three panels:
  A. Convergence curves -- residual mean|f(x_k)| vs iteration, ladder of scales c.
  B. Iterations-to-converge N_conv(c) vs c, with the c*N_conv(1) unit-Lip law and
     the budget K_train (HERO).
  C. Budget sweep -- median depth error |t_K - t*| vs K for {1, sqrt(gamma), gamma};
     extended K so the c=gamma curve converges.

Usage:
    python analysis/scale_invariance_tracing.py \
        --run outputs/run_20260613_134200_scan69_sphere_nomask_init_5014496 \
        --view 16 --gamma 4.0
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))                   # analysis/ siblings
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root for `lip_tracer`

from lip_tracer.config import TraceConfig
from lip_tracer.model import make_model
from lip_tracer.sphere_tracing import trace_nograd
import lip_tracer.data as data_mod
from plot_sphere_trace_steps import rays_for_view


class ScaledField(torch.nn.Module):
    """Wrap f as f_c(x) = f(x) / c, forwarding the attributes the production
    tracer dispatches on (`input_encoding`, `architecture`) so that running the
    *real* trace_nograd on this wrapper reproduces training EXACTLY -- same PE
    bracketing on sign flips, same Newton refinement -- just on the rescaled
    field. Used to show what training's bracketing does and does not rescue."""

    def __init__(self, f: torch.nn.Module, c: float) -> None:
        super().__init__()
        self.f = f
        self.c = float(c)
        self.input_encoding = getattr(f, "input_encoding", "identity")
        self.architecture = getattr(f, "architecture", "cpl")

    def forward(self, x):
        return self.f(x) / self.c


def load_run(run_dir: Path, ckpt_name: str, device: str):
    cfg = json.loads((run_dir / "config.json").read_text())
    ckpt_path = run_dir / "ckpt" / ckpt_name
    if not ckpt_path.exists():
        ckpt_path = run_dir / ckpt_name
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m = cfg["model"]
    f = make_model(
        hidden=m["hidden"], depth=m["depth"],
        group_size=m.get("group_size", 2),
        activation=m.get("activation", "groupsort"),
        input_encoding=m.get("input_encoding", "pe"),
        multires=m.get("multires", 6),
        architecture=m.get("architecture", "cpl"),
    ).to(device)
    f.load_state_dict(ckpt.get("f", ckpt), strict=False)
    with torch.enable_grad():
        f(torch.zeros(1, 3, device=device))
    f.eval()
    return f, TraceConfig(**cfg["trace"]), Path(cfg["scene"])


@torch.no_grad()
def trace_scaled(f, o, d, c: float, K: int, t_far: float, chunk: int = 200_000):
    """Pure sphere trace on the rescaled field f/c, recording the full history.

    Returns:
        t_hist : (K+1, B)  ray depth after each of the K steps (t_hist[0] = 0).
        f_hist : (K,  B)   true (unscaled) f(x_k) at each step -- the geometric
                           residual, comparable across scales.
        lo, hi : (B,)      first downward (+ -> -) sign-change bracket [lo, hi].
        brk    : (B,) bool whether such a bracket was seen (ray crosses surface).
    """
    B = o.shape[0]
    t = torch.zeros(B, device=o.device)
    prev_t = torch.zeros(B, device=o.device)
    prev_f = torch.zeros(B, device=o.device)
    have_prev = torch.zeros(B, dtype=torch.bool, device=o.device)
    lo = torch.zeros(B, device=o.device)
    hi = torch.zeros(B, device=o.device)
    brk = torch.zeros(B, dtype=torch.bool, device=o.device)
    t_hist = [t.clone()]
    f_hist = []
    for _ in range(K):
        x = o + t.unsqueeze(-1) * d
        fv = torch.empty(B, device=o.device)
        for s in range(0, B, chunk):
            fv[s:s + chunk] = f(x[s:s + chunk])
        f_hist.append(fv.clone())
        cross = have_prev & (prev_f > 0) & (fv <= 0) & (~brk)
        lo = torch.where(cross, prev_t, lo)
        hi = torch.where(cross, t, hi)
        brk = brk | cross
        prev_t = t.clone()
        prev_f = fv.clone()
        have_prev = torch.ones_like(have_prev)
        t = (t + fv / c).clamp(0.0, t_far)
        t_hist.append(t.clone())
    return torch.stack(t_hist), torch.stack(f_hist), lo, hi, brk


@torch.no_grad()
def trace_nconv(f, o, d, c: float, K: int, t_far: float, eps_geo: float,
                k_probe: int, chunk: int = 200_000):
    """Sphere trace on f/c; return per-ray iterations-to-converge (low memory).

    Active-ray compaction: each step evaluates f only on rays that have neither
    converged (|f| < eps_geo, a *fixed geometric* tolerance, comparable across
    scales) nor escaped (t >= t_far). Converged rays freeze.

    Returns:
        nconv   : (B,) long  first step index k with |f(x_k)| < eps_geo, else K+1.
        t_probe : (B,)       ray depth after exactly k_probe steps (the finite-
                             budget render at the training budget).
        t_final : (B,)       ray depth after all K steps.
    """
    B = o.shape[0]
    t = torch.zeros(B, device=o.device)
    nconv = torch.full((B,), K + 1, dtype=torch.long, device=o.device)
    done = torch.zeros(B, dtype=torch.bool, device=o.device)
    t_probe = None
    for k in range(K):
        if k == k_probe:
            t_probe = t.clone()
        active = ~(done | (t >= t_far))
        idx = active.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            break
        ta = t[idx]
        xa = o[idx] + ta.unsqueeze(-1) * d[idx]
        fa = torch.empty(idx.numel(), device=o.device)
        for s in range(0, idx.numel(), chunk):
            fa[s:s + chunk] = f(xa[s:s + chunk])
        newly = fa.abs() < eps_geo
        ii = idx[newly]
        nconv[ii] = k
        done[ii] = True
        t[idx] = (ta + fa / c).clamp(0.0, t_far)
    if t_probe is None:            # k_probe never reached (K <= k_probe): use final t
        t_probe = t.clone()
    return nconv, t_probe, t


@torch.no_grad()
def oracle_root(f, o, d, t_far: float, K_oracle: int = 400, bisect: int = 60):
    """High-precision per-ray root depth t_star (scale invariant).

    A large-budget c=1 trace finds the first + -> - crossing bracket, then plain
    bisection on sign(f) refines it to ~1e-7. Rays that never cross are invalid.
    """
    _, _, lo, hi, brk = trace_scaled(f, o, d, c=1.0, K=K_oracle, t_far=t_far)
    lo = lo.clone()
    hi = hi.clone()
    for _ in range(bisect):
        mid = 0.5 * (lo + hi)
        fm = f(o + mid.unsqueeze(-1) * d)
        pos = fm > 0
        lo = torch.where(pos, mid, lo)
        hi = torch.where(pos, hi, mid)
    return 0.5 * (lo + hi), brk


def directional_lipschitz(f, o, d, t_star, valid):
    """Mean |grad f . d| at the oracle hit -- where c=1 sits vs a unit-step SDF."""
    idx = valid.nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return float("nan")
    x = (o[idx] + t_star[idx].unsqueeze(-1) * d[idx]).detach().requires_grad_(True)
    with torch.enable_grad():
        g = torch.autograd.grad(f(x).sum(), x)[0]
    return (g * d[idx]).sum(-1).abs().mean().item()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--ckpt", default="checkpoint_final.pt")
    ap.add_argument("--view", type=int, default=16)
    ap.add_argument("--gamma", type=float, default=None,
                    help="max scale; highlighted scales are {1, sqrt(gamma), gamma}. "
                         "Default = the raw-PE world-space Lipschitz gain "
                         "K = sqrt((4^L + 2)/3) with L = multires (the factor by which "
                         "positional encoding can inflate |grad f| above 1, i.e. the "
                         "scale that restores unit sphere-tracing steps).")
    ap.add_argument("--n-rays", type=int, default=4000,
                    help="foreground rays subsampled for the per-ray panels (A/C/D)")
    ap.add_argument("--k-max", type=int, default=384,
                    help="min display horizon for the convergence-curve panel")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    f, cfg, scene = load_run(args.run, args.ckpt, device)
    K_train = cfg.iters
    t_far = cfg.t_far
    eps = cfg.eps
    # gamma = the raw-PE Lipschitz gain (repo's `uniform` normaliser, see
    # PositionalEncoding): K = sqrt((4^L + 2)/3), L = multires. This is the worst-
    # case factor by which PE inflates |grad f| in world space, i.e. the scale that
    # would divide the field back down to unit-step. L=6 -> 36.96.
    if args.gamma is not None:
        gamma = args.gamma
    elif getattr(f, "encoder", None) is not None and f.multires > 0:
        L = f.multires
        gamma = float(np.sqrt((4.0 ** L + 2.0) / 3.0))
        print(f"gamma = PE Lipschitz gain sqrt((4^{L}+2)/3) = {gamma:.3f}")
    else:
        gamma = 16.0
        print(f"no PE encoder; falling back to gamma = {gamma}")
    scales3 = {"c=1": 1.0, "c=sqrt(gamma)": float(np.sqrt(gamma)), "c=gamma": gamma}
    colors3 = {"c=1": "tab:green", "c=sqrt(gamma)": "tab:orange", "c=gamma": "tab:red"}
    print(f"scene={scene.name} view={args.view} K_train={K_train} eps={eps} "
          f"t_far={t_far} gamma={gamma}")

    views = data_mod.load_views(scene, down=1)
    o_full, d_full, img, mask, H, W = rays_for_view(views, args.view, device)

    # ---- per-ray subset (foreground) for panels A / C / D ----
    fg_idx = np.flatnonzero(mask.reshape(-1))
    rng = np.random.default_rng(args.seed)
    sub = fg_idx[rng.permutation(len(fg_idx))[:args.n_rays]]
    sub = torch.from_numpy(sub).to(device)
    o, d = o_full[sub], d_full[sub]
    print(f"foreground rays: {len(fg_idx)}  subset: {len(sub)}")

    # oracle root (scale invariant) on the subset
    t_star, valid = oracle_root(f, o, d, t_far)
    n_valid = int(valid.sum())
    lip = directional_lipschitz(f, o, d, t_star, valid)
    print(f"valid (surface-crossing) rays: {n_valid}/{len(sub)}  "
          f"mean |grad f . d| at hit = {lip:.3f}  (1.0 == ideal unit-step SDF)")
    vt_star = t_star[valid]

    # eps_geo: a *fixed geometric* convergence tolerance (|f(x_k)| < eps_geo),
    # identical across scales, so iteration counts are directly comparable.
    eps_geo = eps

    # ---- iteration budget for the N_conv analysis ----
    # A unit-Lipschitz field needs ~c x more steps when rescaled by c, so cap the
    # analysis budget a few x above gamma * N_conv(c=1) so even c=gamma converges.
    nconv1, _, _ = trace_nconv(f, o, d, 1.0, 512, t_far, eps_geo, k_probe=K_train)
    N1 = float(nconv1[valid].clamp(max=512).median().item())
    K_ana = int(min(6000, max(512, 4.0 * gamma * N1)))
    print(f"median iters-to-converge at c=1: N1={N1:.0f}  ->  analysis budget K_ana={K_ana}")

    # ---- continuous scale sweep: iters-to-converge + finite-budget consequence ----
    c_sweep = np.geomspace(1.0 / gamma, gamma, 41)
    nconv_med, nconv_q25, nconv_q75 = [], [], []
    err_pure, hit_pure, hit_brkt, err_brkt = [], [], [], []
    for c in c_sweep:
        c = float(c)
        nconv, t_probe, _ = trace_nconv(f, o, d, c, K_ana, t_far, eps_geo, k_probe=K_train)
        nc = nconv[valid].float()
        nconv_med.append(nc.median().item())
        nconv_q25.append(nc.quantile(0.25).item())
        nconv_q75.append(nc.quantile(0.75).item())
        # pure stepping at the training budget: converged within K_train?
        hit_pure.append((nconv[valid] <= K_train).float().mean().item())
        err_pure.append((t_probe[valid] - vt_star).abs().median().item())
        # faithful: the *real* training tracer (PE bracketing + Newton) on f/c
        _, t_b, hit_b = trace_nograd(ScaledField(f, c), o, d,
                                     replace(cfg, iters=K_train))
        hit_brkt.append(hit_b[valid].float().mean().item())
        err_brkt.append((t_b[valid] - vt_star).abs().median().item())

    # ---- single trace per highlighted scale out to the full analysis budget ----
    # Feeds BOTH the convergence-curve panel (residual f_hist) and the budget-sweep
    # panel (depth error from t_hist), so A and C share the same scales and K range.
    budget_hist = {}
    for name, c in scales3.items():
        t_hist, f_hist, *_ = trace_scaled(f, o, d, c, K_ana, t_far)
        budget_hist[name] = (t_hist, f_hist)

    # ===================== figure (ICLR style) =====================
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 11, "axes.titlesize": 11.5, "axes.labelsize": 10.5,
        "axes.titleweight": "bold", "legend.fontsize": 8.5,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "figure.dpi": 200,
        "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    })
    nconv_med = np.array(nconv_med); nconv_q25 = np.array(nconv_q25)
    nconv_q75 = np.array(nconv_q75)
    floor = max(1e-5, 0.1 * eps)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.7), constrained_layout=True)

    # ---- A. convergence curves for the 3 highlighted scales (same K as panel C) ----
    axA = axes[0]
    for name, c in scales3.items():
        _, f_hist = budget_hist[name]
        resid = f_hist[:, valid].abs().mean(dim=1).cpu().numpy()
        axA.plot(np.arange(len(resid)), resid, color=colors3[name], lw=1.8,
                 label=f"{name.replace('gamma','γ')} (c={c:.1f})")
    axA.axhline(eps_geo, color="k", ls=":", lw=1)
    axA.text(K_ana * 0.5, eps_geo * 1.5, r"$\epsilon$", fontsize=10)
    axA.axvline(K_train, color="0.4", ls="--", lw=1)
    axA.text(K_train * 1.05, eps_geo * 1.5, r"$K_{\rm train}$", color="0.3", fontsize=8.5)
    axA.set_xscale("log"); axA.set_yscale("log")
    axA.set_xlabel("sphere-tracing iteration $k$")
    axA.set_ylabel(r"mean residual $|f(x_k)|$")
    axA.legend(loc="lower left")

    # ---- B. iterations-to-converge vs scale (HERO) ----
    axB = axes[1]
    axB.fill_between(c_sweep, nconv_q25, nconv_q75, color="tab:blue", alpha=0.18,
                     label="IQR")
    axB.plot(c_sweep, nconv_med, "o-", color="tab:blue", ms=3.5,
             label=r"median $N_{\rm conv}(c)$")
    cc = c_sweep[c_sweep >= 1.0]
    axB.plot(cc, N1 * cc, "--", color="k", lw=1.4,
             label=r"$c\,\cdot N_{\rm conv}(1)$  (unit-Lip law)")
    axB.axhline(K_train, color="tab:red", ls="-", lw=1.2)
    axB.text(c_sweep[0], K_train * 1.13, r"budget $K_{\rm train}$",
             color="tab:red", fontsize=8.5)
    cstar = K_train / max(N1, 1e-6)
    if c_sweep[0] < cstar < c_sweep[-1]:
        axB.axvspan(cstar, c_sweep[-1], color="tab:red", alpha=0.06)
    axB.set_xscale("log"); axB.set_yscale("log")
    axB.set_xlabel(r"scale $c$  ($f_c=f/c$)")
    axB.set_ylabel(r"iterations to converge $N_{\rm conv}$  ($|f|<\epsilon$)")
    axB.legend(loc="upper left")

    # ---- C. budget sweep: depth error vs K, red (c=gamma) converges at large K ----
    axC = axes[2]
    Ks = np.unique(np.geomspace(1, K_ana, 45).astype(int))
    for name, c in scales3.items():
        th, _ = budget_hist[name]
        errs = [max((th[k][valid] - vt_star).abs().median().item(), floor) for k in Ks]
        axC.plot(Ks, errs, "o-", color=colors3[name], ms=3,
                 label=f"{name.replace('gamma','γ')} (c={c:.1f})")
    axC.axvline(K_train, color="0.4", ls="--", lw=1)
    axC.text(K_train * 1.05, floor * 2, r"$K_{\rm train}$", color="0.3", fontsize=8.5)
    axC.set_xscale("log"); axC.set_yscale("log")
    axC.set_xlabel("step budget $K$")
    axC.set_ylabel(r"median depth error $|t_K - t^*|$")
    axC.legend(loc="lower left")

    out = args.out or (args.run / "diag" / f"scale_invariance_view{args.view}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")

    # ---- console summary ----
    def at(c):
        i = int(np.argmin(np.abs(c_sweep - c)))
        return nconv_med[i], hit_pure[i], hit_brkt[i], err_pure[i], err_brkt[i]
    print("\nscale          N_conv   hit(pure)  hit(+bracket)  err(pure)  err(+bracket)")
    for name, c in scales3.items():
        n, hp, hb, ep, eb = at(c)
        print(f"  {name:14s} c={c:6.2f}  {n:6.0f}    {hp:5.2f}      {hb:5.2f}"
              f"        {ep:.4f}     {eb:.4f}")


if __name__ == "__main__":
    main()
