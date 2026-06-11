#!/usr/bin/env python3
"""Spectral bias diagnostic for SDF regression on Ω = (-bound, bound)^3.

For each checkpoint in --run-dir, evaluate the learned SDF f_θ on a regular
N^3 cell-centred grid, compare to the ground-truth SDF f* (computed once and
cached), optionally near-surface-weight the error, take a 3D DCT-II, bin
coefficients into log-spaced Laplacian-frequency bands, and plot how the
band energies E_m(t) evolve during training.

Math (matches the request verbatim):
  Neumann eigenmodes of -Δ on Ω = (0,a)^3 are products of cosines with
      ν_k = (π/a) √(k1² + k2² + k3²).
  DCT-II on a cell-centred grid is the orthogonal projection onto these modes.
  With error e_t(x) = f_t(x) − f*(x) and weight w_τ(x) = exp(−|f*(x)|/τ),
      ê_k(t) = DCT3(w_τ · e_t)[k]
      E_m(t) = Σ_{k ∈ I_m} |ê_k(t)|²        I_m = { k : ρ_m ≤ ν_k < ρ_{m+1} }
      Ē_m(t) = E_m(t) / (E_m(t_0) + eps)

Outputs (in <run-dir>/spectral_diag/ unless --out-dir is given):
  spectrum_curves.png    Ē_m(t)               – one curve per band, viridis-coloured
  spectrum_heatmap.png   E_m(t) heat-map      – (step × ν), log-log axes, log colour
  spectrum_fraction.png  E_m(t)/‖f*‖²_m       – fraction of GT power *not yet* captured
  spectrum_final.png     E_m at t_0 vs t_end  – two-line power-spectrum comparison
  spectrum_l2.png        RMS error vs step    – sanity-check curve
  spectrum_data.npz      raw arrays for re-plotting

Example:
  python scripts/spectral_sdf_diagnostic.py \\
      --run-dir outputs/buddha_sweep/none_W256_D16_10M \\
      --mesh data/gt_meshes/happy_buddha_norm.ply \\
      --grid-res 128 --tau 0.03 --n-bands 24
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.fft import dctn

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lip_tracer.model import make_model
from fit_gt_sdf import load_mesh, signed_distance_repo_convention


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path,
                    default=Path("outputs/buddha_sweep/none_W256_D16_10M"))
    ap.add_argument("--mesh", type=Path,
                    default=Path("data/gt_meshes/happy_buddha_norm.ply"))
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default: <run-dir>/spectral_diag")
    ap.add_argument("--grid-res", type=int, default=128,
                    help="N for the N^3 evaluation grid (memory ~ 4·N^3 bytes)")
    ap.add_argument("--n-bands", type=int, default=24,
                    help="number of log-spaced Laplacian-frequency bands")
    ap.add_argument("--tau", type=float, default=0.03,
                    help="near-surface weight scale: w_τ(x)=exp(-|f*|/τ); 0 disables")
    ap.add_argument("--max-ckpts", type=int, default=0,
                    help="if >0, subsample checkpoints to ~this many (keeps first/last)")
    ap.add_argument("--chunk", type=int, default=131072,
                    help="model eval chunk size (points per forward)")
    ap.add_argument("--gt-chunk", type=int, default=200_000,
                    help="GT SDF chunk size")
    ap.add_argument("--sdf-backend", choices=["repo", "open3d"], default="repo",
                    help="backend for GT signed-distance computation. "
                         "'open3d' is ~50x faster (Embree); 'repo' is the "
                         "original cKDTree + closest_point implementation.")
    ap.add_argument("--device", type=str, default=None)
    return ap.parse_args()


def cell_centered_grid(bound: float, N: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Cell-centred grid on (-bound, bound)^3.  Returns (pts (N^3,3), shape, a)."""
    a = 2.0 * bound
    h = a / N
    xs = -bound + (np.arange(N) + 0.5) * h
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
    pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3).astype(np.float32)
    return pts, (N, N, N), a


def build_band_indices(N: int, a: float, n_bands: int):
    """Return (band_idx (N,N,N) int, edges (n_bands+1,), nu_centers (n_bands,))."""
    k = np.arange(N)
    K1, K2, K3 = np.meshgrid(k, k, k, indexing="ij")
    nu = (np.pi / a) * np.sqrt(K1.astype(np.float64) ** 2
                               + K2.astype(np.float64) ** 2
                               + K3.astype(np.float64) ** 2)
    nu_min = np.pi / a                       # smallest non-zero ν: k=(1,0,0)
    nu_max = (np.pi / a) * np.sqrt(3) * (N - 1)
    edges = np.geomspace(nu_min, nu_max, n_bands + 1)
    band_idx = np.digitize(nu, edges) - 1                       # -1 below nu_min (DC only)
    band_idx = np.clip(band_idx, 0, n_bands - 1)
    band_idx[(K1 == 0) & (K2 == 0) & (K3 == 0)] = -1            # drop DC
    nu_centers = np.sqrt(edges[:-1] * edges[1:])                # geometric centre
    return band_idx, edges, nu_centers, nu


def band_sum(power: np.ndarray, band_idx: np.ndarray, n_bands: int) -> np.ndarray:
    """Σ_{k ∈ I_m} power[k] for m=0..n_bands-1 (DC excluded since band_idx[DC]=-1)."""
    out = np.zeros(n_bands, dtype=np.float64)
    flat_p = power.ravel()
    flat_b = band_idx.ravel()
    mask = flat_b >= 0
    np.add.at(out, flat_b[mask], flat_p[mask])
    return out


def list_checkpoints(run_dir: Path, max_ckpts: int) -> list[Path]:
    ckpts = sorted(run_dir.glob("checkpoint_step*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"no checkpoint_step*.pt in {run_dir}")
    if max_ckpts and len(ckpts) > max_ckpts:
        idx = np.linspace(0, len(ckpts) - 1, max_ckpts).round().astype(int)
        idx = sorted(set(idx.tolist()))
        ckpts = [ckpts[i] for i in idx]
    return ckpts


def evaluate_grid(model: torch.nn.Module, pts_dev: torch.Tensor,
                  chunk: int, shape: tuple) -> np.ndarray:
    """Evaluate model on a (M,3) device tensor, return host (N,N,N) float32."""
    M = pts_dev.shape[0]
    out = np.empty(M, dtype=np.float32)
    with torch.no_grad():
        for i in range(0, M, chunk):
            out[i:i + chunk] = model(pts_dev[i:i + chunk]).float().cpu().numpy()
    return out.reshape(shape)


def signed_distance_open3d(mesh, pts: np.ndarray, chunk: int,
                           label: str = "") -> np.ndarray:
    """Embree-backed SDF via open3d.t.geometry.RaycastingScene.

    Roughly 50-100x faster than signed_distance_repo_convention on big grids.
    Sign convention: negative inside / positive outside, same as the repo
    convention -- verified on a sample at startup (see _verify_sdf_sign).
    """
    import open3d as o3d
    o3d_mesh = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(mesh.vertices.astype(np.float32)),
        o3d.core.Tensor(mesh.faces.astype(np.int64)),
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d_mesh)
    sdf = np.empty(len(pts), dtype=np.float32)
    n_chunks = (len(pts) + chunk - 1) // chunk
    t0 = time.time()
    for ci, i in enumerate(range(0, len(pts), chunk)):
        batch = pts[i:i + chunk].astype(np.float32)
        sdf[i:i + chunk] = scene.compute_signed_distance(
            o3d.core.Tensor(batch)).numpy()
        if (ci + 1) % max(1, n_chunks // 20) == 0 or ci + 1 == n_chunks:
            elapsed = time.time() - t0
            eta = elapsed / (ci + 1) * (n_chunks - ci - 1)
            print(f"  [{label}/o3d] chunk {ci+1}/{n_chunks}  "
                  f"elapsed {elapsed:.0f}s  ETA {eta:.0f}s", flush=True)
    return sdf


def _verify_sdf_sign(mesh, sample_pts: np.ndarray) -> float:
    """Compare open3d vs repo SDF on a sample.  Returns +1 if signs match the
    repo convention, -1 if open3d is flipped relative to it.  Aborts if the
    magnitudes disagree (then the two backends are computing different things,
    not just disagreeing on sign convention)."""
    repo = signed_distance_repo_convention(mesh, sample_pts, chunk=len(sample_pts),
                                           label="verify-repo")
    o3d_sd = signed_distance_open3d(mesh, sample_pts, chunk=len(sample_pts),
                                    label="verify-o3d")
    far = np.abs(repo) > 0.05
    if far.any():
        rel = np.abs(np.abs(repo[far]) - np.abs(o3d_sd[far])) / \
              np.abs(repo[far]).clip(min=1e-6)
        print(f"[verify] far-point magnitude rel-err  mean={rel.mean():.2e}  "
              f"p99={np.quantile(rel,0.99):.2e}", flush=True)
        if rel.mean() > 0.05:
            raise SystemExit("open3d and repo SDF magnitudes disagree too "
                             "much -- backends are computing different things")
    agree = (np.sign(repo) == np.sign(o3d_sd)).mean()
    disagree = (np.sign(repo) == -np.sign(o3d_sd)).mean()
    print(f"[verify] sign-match: {100*agree:.1f}%  sign-flip: {100*disagree:.1f}%",
          flush=True)
    if agree > 0.95:
        return 1.0
    if disagree > 0.95:
        print("[verify] open3d sign is flipped vs repo -- will negate output",
              flush=True)
        return -1.0
    raise SystemExit("open3d and repo disagree inconsistently (neither "
                     ">95% match nor >95% flip) -- mesh may be non-watertight")


def compute_or_load_gt(mesh_path: Path, pts: np.ndarray, shape: tuple,
                       cache_path: Path, gt_chunk: int,
                       backend: str = "repo") -> np.ndarray:
    if cache_path.exists():
        f_star = np.load(cache_path)
        if f_star.shape == shape:
            print(f"[gt] loaded cached f* from {cache_path}", flush=True)
            return f_star.astype(np.float32)
        print(f"[gt] cache shape {f_star.shape} ≠ {shape}; recomputing")
    print(f"[gt] computing f* on {shape} grid ({pts.shape[0]:,} pts) "
          f"using backend={backend} …", flush=True)
    mesh = load_mesh(mesh_path)
    parts = mesh.split(only_watertight=False)
    if len(parts) > 1:
        kept = max(parts, key=lambda m: m.area)
        print(f"[gt] mesh has {len(parts)} components; keeping largest "
              f"(area={kept.area:.4f}, verts={len(kept.vertices):,}, "
              f"faces={len(kept.faces):,})")
        mesh = kept
    t0 = time.time()
    if backend == "open3d":
        rng = np.random.default_rng(0)
        sample_pts = pts[rng.choice(len(pts), size=min(4096, len(pts)),
                                    replace=False)].astype(np.float32)
        sign = _verify_sdf_sign(mesh, sample_pts)
        f_star_flat = signed_distance_open3d(mesh, pts, chunk=gt_chunk, label="f*")
        if sign < 0:
            f_star_flat = -f_star_flat
    elif backend == "repo":
        f_star_flat = signed_distance_repo_convention(mesh, pts, chunk=gt_chunk, label="f*")
    else:
        raise ValueError(f"unknown backend {backend!r}; expected 'repo' or 'open3d'")
    f_star = f_star_flat.astype(np.float32).reshape(shape)
    np.save(cache_path, f_star)
    print(f"[gt] done in {time.time()-t0:.1f}s; cached → {cache_path}")
    return f_star


# ----------------------------------------------------------------------------
# Plotting — ICLR-quality figures
# ----------------------------------------------------------------------------

def _set_iclr_style() -> None:
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family":        "serif",
        "font.serif":         ["DejaVu Serif", "Times New Roman", "STIXGeneral"],
        "mathtext.fontset":   "dejavuserif",
        "font.size":          11,
        "axes.titlesize":     12,
        "axes.labelsize":     11,
        "legend.fontsize":    9,
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,
        "axes.linewidth":     0.8,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "figure.dpi":         150,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.02,
        "lines.linewidth":    1.3,
    })


def make_plots(out_dir: Path, steps: np.ndarray, E: np.ndarray,
               edges: np.ndarray, nu_centers: np.ndarray,
               f_star_power_bands: np.ndarray,
               L2_tot: np.ndarray, L2_w: np.ndarray,
               tau: float, N: int) -> None:
    _set_iclr_style()
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    _, n_bands = E.shape
    cmap = plt.cm.viridis
    eps = 1e-30
    # x-axis: protect against step=0 on log scale
    x_steps = np.maximum(steps.astype(float), 1.0)

    # ---------- 1. Per-band curves: Ē_m(t) = E_m(t)/E_m(t_0) ----------
    E_norm = E / (E[0:1] + eps)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for m in range(n_bands):
        color = cmap(m / max(n_bands - 1, 1))
        ax.plot(x_steps, np.maximum(E_norm[:, m], eps),
                color=color, lw=1.2, alpha=0.95)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("training step  $t$")
    ax.set_ylabel(r"$\bar{E}_m(t) \;=\; E_m(t) / E_m(t_0)$")
    ax.set_title(r"Spectral error relative to first checkpoint")
    ax.grid(True, which="both", alpha=0.25, lw=0.5)
    norm = mpl.colors.LogNorm(vmin=nu_centers[0], vmax=nu_centers[-1])
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.04)
    cb.set_label(r"band centre  $\nu_m$  (rad·unit$^{-1}$)")
    fig.tight_layout()
    fig.savefig(out_dir / "spectrum_curves.png")
    plt.close(fig)

    # ---------- 2. Heat-map E_m(t) on (step × ν) ----------
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    E_plot = np.maximum(E.T, eps)                                # (n_bands, n_steps)
    # use band edges for y and step midpoints for x (pcolormesh flat shading)
    # construct step edges geometrically
    if len(x_steps) >= 2:
        log_steps = np.log(x_steps)
        log_edges = np.empty(len(x_steps) + 1)
        log_edges[1:-1] = 0.5 * (log_steps[:-1] + log_steps[1:])
        log_edges[0]    = log_steps[0]  - 0.5 * (log_steps[1]  - log_steps[0])
        log_edges[-1]   = log_steps[-1] + 0.5 * (log_steps[-1] - log_steps[-2])
        step_edges = np.exp(log_edges)
    else:
        step_edges = np.array([x_steps[0] * 0.9, x_steps[0] * 1.1])
    pcm = ax.pcolormesh(step_edges, edges, E_plot,
                        norm=mpl.colors.LogNorm(vmin=E_plot.min(), vmax=E_plot.max()),
                        cmap="magma", shading="flat")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("training step  $t$")
    ax.set_ylabel(r"Laplacian frequency  $\nu$  (rad·unit$^{-1}$)")
    ax.set_title(r"$E_m(t) \;=\; \sum_{k \in I_m} |\hat e_k(t)|^2$")
    cb = fig.colorbar(pcm, ax=ax, pad=0.02, fraction=0.04)
    cb.set_label(r"band energy  $E_m(t)$")
    fig.tight_layout()
    fig.savefig(out_dir / "spectrum_heatmap.png")
    plt.close(fig)

    # ---------- 3. Fraction of GT band-power still un-captured ----------
    frac = E / (f_star_power_bands[None, :] + eps)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for m in range(n_bands):
        color = cmap(m / max(n_bands - 1, 1))
        ax.plot(x_steps, np.maximum(frac[:, m], eps),
                color=color, lw=1.2, alpha=0.95)
    ax.axhline(1.0, color="0.6", ls=":", lw=0.8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("training step  $t$")
    ax.set_ylabel(r"$E_m(t) \,/\, \|w_\tau f^*\|^2_m$")
    ax.set_title("Fraction of band-wise GT power still un-captured")
    ax.grid(True, which="both", alpha=0.25, lw=0.5)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.04)
    cb.set_label(r"band centre  $\nu_m$  (rad·unit$^{-1}$)")
    fig.tight_layout()
    fig.savefig(out_dir / "spectrum_fraction.png")
    plt.close(fig)

    # ---------- 4. First vs last spectrum + GT reference ----------
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(nu_centers, np.maximum(f_star_power_bands, eps),
            "^-", color="0.5", lw=1.4, ms=5,
            label=r"$\|w_\tau f^*\|^2_m$  (target)")
    ax.plot(nu_centers, np.maximum(E[0], eps),
            "o-", color="#3b528b", lw=1.4, ms=4,
            label=fr"$E_m$  at  $t={steps[0]:,}$")
    ax.plot(nu_centers, np.maximum(E[-1], eps),
            "s-", color="#d62728", lw=1.4, ms=4,
            label=fr"$E_m$  at  $t={steps[-1]:,}$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"Laplacian frequency  $\nu_m$")
    ax.set_ylabel("band energy")
    title = r"Power spectrum of $w_\tau (f_\theta - f^*)$" if tau > 0 else \
            r"Power spectrum of $(f_\theta - f^*)$"
    title += fr",  $\tau={tau:g}$,  $N={N}$"
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25, lw=0.5)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "spectrum_final.png")
    plt.close(fig)

    # ---------- 5. RMS error sanity curves ----------
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(x_steps, L2_tot, "o-", color="#3b528b", lw=1.3, ms=4,
            label=r"$\|f_\theta - f^*\|_{L^2(\Omega)} / \sqrt{|\Omega|}$")
    if tau > 0:
        ax.plot(x_steps, L2_w, "s-", color="#d62728", lw=1.3, ms=4,
                label=r"$\|w_\tau (f_\theta - f^*)\|_{L^2(\Omega)} / \sqrt{|\Omega|}$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("training step  $t$")
    ax.set_ylabel("grid RMS error")
    ax.set_title("Overall and weighted error over training")
    ax.grid(True, which="both", alpha=0.25, lw=0.5)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "spectrum_l2.png")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    if not args.run_dir.exists():
        raise FileNotFoundError(args.run_dir)
    cfg_path = args.run_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    cfg = json.loads(cfg_path.read_text())
    bound = float(cfg.get("bound", 1.1))

    out_dir = args.out_dir or (args.run_dir / "spectral_diag")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[cfg] run_dir={args.run_dir}  bound={bound}  out={out_dir}")

    # ---------- grid + bands ----------
    pts, shape, a = cell_centered_grid(bound, args.grid_res)
    band_idx, edges, nu_centers, _ = build_band_indices(args.grid_res, a, args.n_bands)
    print(f"[grid] N={args.grid_res}  a={a:.4f}  "
          f"ν ∈ [{edges[0]:.3f}, {edges[-1]:.3f}]  bands={args.n_bands}")

    # ---------- f* on grid (cached) ----------
    cache_path = out_dir / f"gt_sdf_grid_N{args.grid_res}.npy"
    f_star = compute_or_load_gt(args.mesh, pts, shape, cache_path,
                                args.gt_chunk, backend=args.sdf_backend)

    # Degenerate triangles near mesh holes give NaN closest-point sdf values
    # (see comment in fit_gt_sdf.sample_dataset). Mask these cells out: set
    # f*=0 and w=0 so they contribute nothing to the error or the DCT.
    bad = ~np.isfinite(f_star)
    n_bad = int(bad.sum())
    if n_bad:
        print(f"[gt] masking {n_bad} non-finite f* cells "
              f"({100*n_bad/f_star.size:.4f}% of grid)")
        f_star = np.where(bad, np.float32(0.0), f_star).astype(np.float32)

    # ---------- weight ----------
    if args.tau > 0:
        w = np.exp(-np.abs(f_star) / args.tau).astype(np.float32)
    else:
        w = np.ones_like(f_star)
    if n_bad:
        w = np.where(bad, np.float32(0.0), w)
    wf_star = w * f_star
    f_star_coef = dctn(wf_star, type=2, norm="ortho")
    f_star_power = (f_star_coef.astype(np.float64)) ** 2
    f_star_band  = band_sum(f_star_power, band_idx, args.n_bands)
    print(f"[gt] ‖w·f*‖² = {f_star_power.sum():.4e}  "
          f"(near-surface mass = w.mean()={w.mean():.4e})")

    # ---------- checkpoints ----------
    ckpts = list_checkpoints(args.run_dir, args.max_ckpts)
    print(f"[ckpts] {len(ckpts)} checkpoints "
          f"({ckpts[0].name} → {ckpts[-1].name})")

    # ---------- model ----------
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    sample = torch.load(ckpts[0], map_location="cpu", weights_only=False)
    model = make_model(
        hidden        = int(cfg.get("hidden", 256)),
        depth         = int(sample.get("depth",        cfg.get("depth", 16))),
        group_size    = int(sample.get("group_size",   cfg.get("group_size", 2))),
        activation    = str(sample.get("activation",   cfg.get("activation", "groupsort"))),
        input_encoding= str(sample.get("input_encoding", cfg.get("input_encoding", "identity"))),
        multires      = int(sample.get("multires",     cfg.get("multires", 0))),
        architecture  = str(sample.get("architecture", cfg.get("architecture", "cpl"))),
        lipschitz_mode= str(sample.get("lipschitz_mode", cfg.get("lipschitz_mode", "none"))),
    ).to(device).eval()
    print(f"[model] arch={model.architecture} hidden={cfg.get('hidden')} "
          f"depth={model.depth} enc={model.input_encoding}/{model.multires} "
          f"device={device}")

    pts_dev = torch.from_numpy(pts).to(device)
    w_dev_factor = w  # CPU multiply after eval; cheaper than a GPU copy for many ckpts
    ok_mask = np.isfinite(f_star) & (~bad if n_bad else np.ones_like(f_star, dtype=bool))
    ok_count = int(ok_mask.sum())

    # ---------- spectral sweep over checkpoints ----------
    n_ckpts = len(ckpts)
    steps  = np.zeros(n_ckpts, dtype=np.int64)
    E      = np.zeros((n_ckpts, args.n_bands), dtype=np.float64)
    L2_tot = np.zeros(n_ckpts, dtype=np.float64)
    L2_w   = np.zeros(n_ckpts, dtype=np.float64)

    t_start = time.time()
    for ti, ckpt_path in enumerate(ckpts):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["f"])
        steps[ti] = int(ckpt.get("step", 0))

        f_t = evaluate_grid(model, pts_dev, args.chunk, shape)
        err = f_t - f_star
        if n_bad:
            err = np.where(ok_mask, err, np.float32(0.0))
        L2_tot[ti] = float(np.sqrt(np.sum(err ** 2) / max(ok_count, 1)))
        err_w = (w_dev_factor * err).astype(np.float32)
        L2_w[ti]   = float(np.sqrt(np.sum(err_w ** 2) / max(ok_count, 1)))

        coef  = dctn(err_w, type=2, norm="ortho")
        power = (coef.astype(np.float64)) ** 2
        E[ti] = band_sum(power, band_idx, args.n_bands)

        dt = time.time() - t_start
        eta = dt / (ti + 1) * (n_ckpts - ti - 1)
        print(f"[{ti+1:>3d}/{n_ckpts}] step={steps[ti]:>10d}  "
              f"‖e‖={L2_tot[ti]:.4e}  ‖w·e‖={L2_w[ti]:.4e}  "
              f"ΣE={E[ti].sum():.4e}  elapsed={dt:.1f}s  eta={eta:.1f}s", flush=True)

    # ---------- save raw arrays ----------
    npz_path = out_dir / "spectrum_data.npz"
    np.savez(
        npz_path,
        steps             = steps,
        E                 = E,
        band_edges        = edges,
        nu_centers        = nu_centers,
        f_star_band_power = f_star_band,
        L2_tot            = L2_tot,
        L2_w              = L2_w,
        tau               = args.tau,
        bound             = bound,
        a                 = a,
        N                 = args.grid_res,
    )
    print(f"[save] raw arrays → {npz_path}")

    # ---------- figures ----------
    make_plots(out_dir, steps, E, edges, nu_centers, f_star_band,
               L2_tot, L2_w, args.tau, args.grid_res)
    print(f"[save] figures → {out_dir}/spectrum_*.png")


if __name__ == "__main__":
    main()
