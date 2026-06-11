"""Side-by-side |grad f| field comparison for three SDF checkpoints.

Loads NeuS, Geo-NeuS and our checkpoint, evaluates |grad f| and f on a 2D
slice of 3D space, and renders a 1x3 figure: |grad f| heatmap with the
zero level set overlaid in black.

Usage
-----
    python grad_compare.py \\
        --mine    outputs/.../checkpoint_best_photo.pt \\
        --neus    baselines/NeuS/exp/.../checkpoint.pth \\
        --geoneus baselines/Geo-Neus/exp/.../checkpoint.pth \\
        --neus_conf    baselines/NeuS/confs/wmask.conf \\
        --geoneus_conf baselines/Geo-Neus/confs/wmask.conf \\
        --axis xz --offset 0.0 --bound 1.0 --res 512 \\
        --out outputs/grad_compare.png
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "baselines" / "NeuS"))

from lip_tracer.model import make_model
from render_paper_marching import _checkpoint_model_kwargs


# ----------------------------------------------------------- loaders -------

# Each loader returns (sdf_fn, enc_fn, fwd_enc_fn):
#   sdf_fn(x)       : world coords -> SDF value
#   enc_fn(x)       : world coords -> positional-encoding vector  (None if no PE)
#   fwd_enc_fn(e)   : encoded vector -> SDF value  (post-encoding part of the net)
# This lets us measure either |df/dx| (world) or |df/dgamma| (post-PE).

def load_mine(ckpt_path: Path, device: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    f = make_model(**_checkpoint_model_kwargs(ckpt)).to(device).eval()
    f.load_state_dict(ckpt["f"], strict=False)

    def fwd_enc(e):
        h = torch.nn.functional.pad(e, (0, f.hidden - e.shape[-1]))
        h = f.net(h)
        if torch.is_grad_enabled():
            w = f.head_weight / torch.linalg.vector_norm(f.head_weight).clamp(min=1e-6)
        else:
            w = f._head_w_buf
        return (h * w).sum(-1) + f.head_bias.squeeze(-1)

    return (lambda x: f(x).squeeze(-1)), f.encoder, fwd_enc


def load_neus_like(ckpt_path: Path, conf_path: Path, device: str):
    """Loads NeuS / Geo-NeuS SDFNetwork from a checkpoint + conf file."""
    from pyhocon import ConfigFactory
    from models.fields import SDFNetwork
    conf = ConfigFactory.parse_file(str(conf_path))
    net = SDFNetwork(**conf["model.sdf_network"]).to(device).eval()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    net.load_state_dict(ckpt["sdf_network_fine"])

    enc = None
    if net.embed_fn_fine is not None:
        enc = lambda x: net.embed_fn_fine(x * net.scale)

    def fwd_enc(e):
        x = e
        for l in range(net.num_layers - 1):
            lin = getattr(net, "lin" + str(l))
            if l in net.skip_in:
                x = torch.cat([x, e], 1) / np.sqrt(2)
            x = lin(x)
            if l < net.num_layers - 2:
                x = net.activation(x)
        return x[:, 0] / net.scale

    return (lambda x: net.sdf(x).squeeze(-1)), enc, fwd_enc


# -------------------------------------------------------- slice eval -------

def make_slice_grid(axis: str, offset: float, bound: float, res: int):
    lin = np.linspace(-bound, bound, res, dtype=np.float32)
    a, b = np.meshgrid(lin, lin, indexing="ij")
    c = np.full_like(a, offset)
    if axis == "xy":   pts, xl, yl = np.stack([a, b, c], -1), "x", "y"
    elif axis == "xz": pts, xl, yl = np.stack([a, c, b], -1), "x", "z"
    else:              pts, xl, yl = np.stack([c, a, b], -1), "y", "z"
    return pts.reshape(-1, 3), xl, yl


def load_gt_slice(ply_path: Path, cameras_npz: Path, axis: str, offset: float,
                  slab: float, bound: float, res: int,
                  plane_mat: Path | None = None):
    """GT DTU point cloud -> 2D distance field on the figure grid.

    The cloud is in DTU world (mm); scale_mat_0 in the cameras file maps the
    normalised unit-sphere space back to world, so its inverse brings the GT
    into the same frame the networks (and the figure) live in.  Plane{N}.mat
    (when given) culls the support table exactly as the official DTU eval does.

    We return the 2-D distance-to-GT field so the GT surface can be drawn as a
    clean contour line — the same style as each model's predicted level set —
    instead of a fuzzy scatter band.
    """
    import trimesh
    from scipy.spatial import cKDTree
    scale_mat = np.load(cameras_npz)["scale_mat_0"].astype(np.float64)
    pc = trimesh.load(str(ply_path), process=False)
    V = np.asarray(pc.vertices, dtype=np.float64)               # world mm
    if plane_mat is not None:
        from scipy.io import loadmat
        P = loadmat(str(plane_mat))["P"].reshape(4).astype(np.float64)
        above = (np.concatenate([V, np.ones((len(V), 1))], 1) @ P) > 0
        print(f"GT plane cull: kept {int(above.sum())}/{len(V)} points "
              f"above the support plane", flush=True)
        V = V[above]
    Vh = np.concatenate([V, np.ones((len(V), 1))], axis=1)
    Vn = (np.linalg.inv(scale_mat) @ Vh.T).T[:, :3]             # normalised

    fixed = {"xy": 2, "xz": 1, "yz": 0}[axis]
    free  = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}[axis]
    m = np.abs(Vn[:, fixed] - offset) < slab
    pts2d = Vn[m][:, free]                                      # (N, 2)
    print(f"GT slice: {len(pts2d)} points in "
          f"|{['x','y','z'][fixed]}{offset:+.2f}|<{slab}", flush=True)

    lin = np.linspace(-bound, bound, res, dtype=np.float64)
    gx, gy = np.meshgrid(lin, lin, indexing="ij")
    dist, _ = cKDTree(pts2d).query(np.stack([gx.ravel(), gy.ravel()], -1), k=1)
    return dist.reshape(res, res)


def eval_sdf_and_grad(sdf_fn, enc_fn, fwd_enc_fn, pts: np.ndarray, device: str,
                      encoded: bool = False, chunk: int = 32768):
    """Returns (sdf, grad_norm). encoded=True measures |df/dgamma| (post-PE)."""
    if encoded and enc_fn is None:
        raise ValueError("encoded=True requested but model has no positional encoding")
    sdfs, gns = [], []
    for i in range(0, len(pts), chunk):
        xx = torch.from_numpy(pts[i:i + chunk]).to(device)
        with torch.enable_grad():
            x = xx.requires_grad_(True)
            y = sdf_fn(x)
            sdfs.append(y.detach().float().cpu().numpy())
            if encoded:
                e = enc_fn(xx).detach().requires_grad_(True)
                ye = fwd_enc_fn(e)
                g = torch.autograd.grad(ye.sum(), e, create_graph=False)[0]
            else:
                g = torch.autograd.grad(y.sum(), x, create_graph=False)[0]
        gns.append(g.norm(dim=-1).detach().float().cpu().numpy())
    return np.concatenate(sdfs), np.concatenate(gns)


# ----------------------------------------------------------- figure --------

def make_figure(panels, axis, offset, bound, res, vmin, vmax, out_path,
                cmap="viridis", contour_lw=1.6, fontsize=13, diverging=False,
                sym=r"|\nabla f|", suptitle=None, gt_field=None,
                gt_level=0.012):
    lin = np.linspace(-bound, bound, res)
    fixed = {"xy": "z", "xz": "y", "yz": "x"}[axis]
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n + 0.7, 4.6),
                             gridspec_kw={"wspace": 0.06}, squeeze=False)
    if diverging:
        # Centre the colour scale on the ideal |grad f| = 1 (eikonal target).
        norm = TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)
    last_im = None
    for k, (label, sdf, gn, xl, yl) in enumerate(panels):
        ax = axes[0, k]
        last_im = ax.imshow(gn.T, origin="lower",
                            extent=[-bound, bound, -bound, bound],
                            cmap=cmap, norm=norm,
                            interpolation="bilinear", aspect="equal")
        # GT surface (black, solid) — common reference in every panel …
        if gt_field is not None:
            ax.contour(lin, lin, gt_field.T, levels=[gt_level],
                       colors="black", linewidths=contour_lw,
                       linestyles="solid", zorder=3)
        # … and this model's own predicted surface (dashed when shown next to
        # the GT, solid when it is the only contour).
        ax.contour(lin, lin, sdf.T, levels=[0.0],
                   colors="black", linewidths=contour_lw,
                   linestyles="dashed" if gt_field is not None else "solid",
                   zorder=4)
        if k == 0 and gt_field is not None:
            handles = [
                Line2D([0], [0], color="black", linestyle="-",
                       lw=contour_lw, label="GT"),
                Line2D([0], [0], color="black", linestyle="--",
                       lw=contour_lw, label="prediction"),
            ]
            ax.legend(handles=handles, loc="upper right",
                      fontsize=fontsize - 4, framealpha=0.7, handlelength=1.8)
        ax.set_xlim(-bound, bound)
        ax.set_ylim(-bound, bound)
        ax.set_xlabel(xl, fontsize=fontsize - 1)
        # Prune the edge ticks so neighbouring panels' labels never collide.
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
        if k == 0:
            ax.set_ylabel(yl, fontsize=fontsize - 1)
        else:
            ax.set_yticklabels([])
        ax.set_title(f"{label}", fontsize=fontsize, fontweight="semibold")
        ax.tick_params(labelsize=fontsize - 3)

    fig.suptitle(suptitle if suptitle is not None
                 else rf"${sym}$  on  ${fixed}={offset:+.2f}$",
                 fontsize=fontsize + 1, y=1.02)
    fig.subplots_adjust(right=0.90)
    cax = fig.add_axes([0.915, 0.12, 0.015, 0.76])
    cb = fig.colorbar(last_im, cax=cax)
    cb.set_label(rf"${sym}$", fontsize=fontsize)
    cb.ax.tick_params(labelsize=fontsize - 3)

    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved → {out_path}", flush=True)


def make_figure_stacked(rows, axis, offset, bound, res, out_path,
                        contour_lw=1.6, fontsize=13):
    """Two-row figure sharing the same slice / columns.

    Row 0: world |grad f| (diverging around the eikonal target 1).
    Row 1: post-PE |df/dgamma| — the quantity our network bounds by construction.
    Each `row` dict carries: panels, cmap, norm, sym, cbar_label.
    """
    lin = np.linspace(-bound, bound, res)
    fixed = {"xy": "z", "xz": "y", "yz": "x"}[axis]
    n = len(rows[0]["panels"])
    fig, axes = plt.subplots(len(rows), n, figsize=(4.4 * n + 1.0, 4.55 * len(rows)),
                             gridspec_kw={"wspace": 0.06, "hspace": 0.10},
                             squeeze=False)
    for r, row in enumerate(rows):
        last_im = None
        for k, (label, sdf, gn, xl, yl) in enumerate(row["panels"]):
            ax = axes[r, k]
            last_im = ax.imshow(gn.T, origin="lower",
                                extent=[-bound, bound, -bound, bound],
                                cmap=row["cmap"], norm=row["norm"],
                                interpolation="bilinear", aspect="equal")
            ax.contour(lin, lin, sdf.T, levels=[0.0],
                       colors="black", linewidths=contour_lw)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
            ax.tick_params(labelsize=fontsize - 3)
            if r == 0:
                ax.set_title(label, fontsize=fontsize, fontweight="semibold")
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(xl, fontsize=fontsize - 1)
            if k == 0:
                ax.set_ylabel(rf"${row['sym']}$" + "\n" + f"{yl}",
                              fontsize=fontsize - 1)
            else:
                ax.set_yticklabels([])
        cb = fig.colorbar(last_im, ax=list(axes[r, :]), fraction=0.026, pad=0.015)
        cb.set_label(row["cbar_label"], fontsize=fontsize - 1)
        cb.ax.tick_params(labelsize=fontsize - 3)

    fig.suptitle(rf"Gradient field on ${fixed}={offset:+.2f}$",
                 fontsize=fontsize + 1, y=0.995)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved → {out_path}", flush=True)


# ---------------------------------------------------------------- CLI -----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mine",         type=Path, required=True)
    ap.add_argument("--neus",         type=Path, required=True)
    ap.add_argument("--geoneus",      type=Path, required=True)
    ap.add_argument("--neus_conf",    type=Path, required=True)
    ap.add_argument("--geoneus_conf", type=Path, required=True)
    ap.add_argument("--axis",   choices=["xy", "xz", "yz"], default="xz")
    ap.add_argument("--offset", type=float, default=0.0)
    ap.add_argument("--bound",  type=float, default=1.0)
    ap.add_argument("--res",    type=int,   default=512)
    ap.add_argument("--vmin",   type=float, default=None,
                    help="auto = 1st percentile across the three panels")
    ap.add_argument("--vmax",   type=float, default=None,
                    help="auto = 99th percentile across the three panels")
    ap.add_argument("--diverging", action="store_true",
                    help="diverging colormap centred on |grad f| = 1")
    ap.add_argument("--encoded", action="store_true",
                    help="measure |df/dgamma| (post positional-encoding) instead of |df/dx|")
    ap.add_argument("--add-encoded", dest="add_encoded", action="store_true",
                    help="two-row figure: world |grad f| on top, post-PE |df/dgamma| below")
    ap.add_argument("--ours-pe-col", dest="ours_pe_col", action="store_true",
                    help="diverging figure + an extra column: Ours post-PE |df/dgamma|")
    ap.add_argument("--cmap",   default="viridis")
    ap.add_argument("--gt_ply",     type=Path, default=None,
                    help="DTU GT point cloud (stlNNN_total.ply) for a common contour")
    ap.add_argument("--gt_cameras", type=Path, default=None,
                    help="cameras_sphere.npz holding scale_mat_0 (world<->normalised)")
    ap.add_argument("--gt_slab",    type=float, default=0.015,
                    help="half-thickness of the GT slab around the slice")
    ap.add_argument("--gt_plane",   type=Path, default=None,
                    help="DTU PlaneNNN.mat — culls the support table from the GT")
    ap.add_argument("--mine_cameras", type=Path, default=None,
                    help="cameras.npz our checkpoint was trained with (scale_mat_0)")
    ap.add_argument("--ref_cameras",  type=Path, default=None,
                    help="reference cameras.npz (NeuS/GT frame); with --mine_cameras, "
                         "our model is resampled into this common frame")
    ap.add_argument("--out",    type=Path, required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}", flush=True)

    models = {
        "Ours":     load_mine(args.mine, device),
        "NeuS":     load_neus_like(args.neus,    args.neus_conf,    device),
        "Geo-NeuS": load_neus_like(args.geoneus, args.geoneus_conf, device),
    }
    pts, xl, yl = make_slice_grid(args.axis, args.offset, args.bound, args.res)

    # Our checkpoint may live in a different DTU normalisation than NeuS/GT.
    # Resample it into the reference frame: a figure point x is fed to our
    # model as a*x + b, so f / |grad f| are evaluated at the right physical
    # location and the contour aligns with GT (grad stays in our native frame).
    pts_mine = pts
    if args.mine_cameras is not None and args.ref_cameras is not None:
        sm_m = np.load(args.mine_cameras)["scale_mat_0"].astype(np.float64)
        sm_r = np.load(args.ref_cameras)["scale_mat_0"].astype(np.float64)
        a = sm_r[0, 0] / sm_m[0, 0]
        b = (sm_r[:3, 3] - sm_m[:3, 3]) / sm_m[0, 0]
        pts_mine = (pts.astype(np.float64) * a + b).astype(np.float32)
        print(f"Ours frame align: a={a:.4f}  b={np.round(b, 4)}", flush=True)

    gt_field = None
    if args.gt_ply is not None:
        if args.gt_cameras is None:
            ap.error("--gt_ply requires --gt_cameras (for scale_mat_0)")
        gt_field = load_gt_slice(args.gt_ply, args.gt_cameras, args.axis,
                                 args.offset, args.gt_slab, args.bound,
                                 args.res, plane_mat=args.gt_plane)

    def eval_panels(encoded: bool):
        """Returns (panels, flat_grad) for the requested gradient quantity."""
        panels, all_gn = [], []
        tag = "|df/dgamma|" if encoded else "|grad f|"
        for name, (sdf_fn, enc_fn, fwd_enc_fn) in models.items():
            print(f"evaluating {name}  ({tag})…", flush=True)
            mpts = pts_mine if name == "Ours" else pts
            sdf, gn = eval_sdf_and_grad(sdf_fn, enc_fn, fwd_enc_fn, mpts, device,
                                        encoded=encoded)
            sdf = sdf.reshape(args.res, args.res)
            gn  = gn.reshape(args.res, args.res)
            panels.append((name, sdf, gn, xl, yl))
            all_gn.append(gn)
            p = np.percentile(gn, [1, 50, 90, 99])
            print(f"  {tag}  p1={p[0]:.3f}  p50={p[1]:.3f}  "
                  f"p90={p[2]:.3f}  p99={p[3]:.3f}  max={gn.max():.3f}", flush=True)
        return panels, np.concatenate([g.ravel() for g in all_gn])

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # ---- diverging figure + an extra "Ours post-PE" column -----------------
    if args.ours_pe_col:
        w_panels, w_flat = eval_panels(encoded=False)   # Ours, NeuS, Geo-NeuS
        sdf_fn, enc_fn, fwd_enc_fn = models["Ours"]
        print("evaluating Ours  (|df/dgamma|, post-PE)…", flush=True)
        e_sdf, e_gn = eval_sdf_and_grad(sdf_fn, enc_fn, fwd_enc_fn, pts_mine,
                                        device, encoded=True)
        e_sdf = e_sdf.reshape(args.res, args.res)
        e_gn  = e_gn.reshape(args.res, args.res)
        p = np.percentile(e_gn, [1, 50, 90, 99])
        print(f"  |df/dgamma|  p1={p[0]:.3f}  p50={p[1]:.3f}  "
              f"p90={p[2]:.3f}  p99={p[3]:.3f}  max={e_gn.max():.3f}", flush=True)
        ours_pe_panel = ("Ours\n" + r"$\|\nabla_\gamma f\|$ (post-PE)",
                         e_sdf, e_gn, w_panels[0][3], w_panels[0][4])

        # Colour scale fixed by the world panels — identical to the 3-panel fig.
        vmin = args.vmin if args.vmin is not None else float(np.percentile(w_flat, 1.0))
        vmax = args.vmax if args.vmax is not None else float(np.percentile(w_flat, 99.0))
        half = max(1.0 - vmin, vmax - 1.0)
        vmin, vmax = 1.0 - half, 1.0 + half
        print(f"vmin={vmin:.3f}  vmax={vmax:.3f}", flush=True)

        # Order: Ours | Ours post-PE | NeuS | Geo-NeuS
        panels = [w_panels[0], ours_pe_panel, w_panels[1], w_panels[2]]
        fixed = {"xy": "z", "xz": "y", "yz": "x"}[args.axis]
        suptitle = rf"$|\nabla f|$  on  ${fixed}={args.offset:+.2f}$"
        make_figure(panels, args.axis, args.offset, args.bound, args.res,
                    vmin, vmax, args.out, cmap="coolwarm", diverging=True,
                    sym=r"|\nabla f|", suptitle=suptitle, gt_field=gt_field)
        return

    # ---- two-row figure: world gradient + post-PE gradient -----------------
    if args.add_encoded:
        w_panels, w_flat = eval_panels(encoded=False)
        e_panels, _      = eval_panels(encoded=True)

        wmin = args.vmin if args.vmin is not None else float(np.percentile(w_flat, 1.0))
        wmax = args.vmax if args.vmax is not None else float(np.percentile(w_flat, 99.0))
        half = max(1.0 - wmin, wmax - 1.0)            # symmetric span around 1
        rows = [
            {"panels": w_panels, "cmap": "coolwarm",
             "norm": TwoSlopeNorm(vmin=1.0 - half, vcenter=1.0, vmax=1.0 + half),
             "sym": r"|\nabla f|",
             "cbar_label": r"$|\nabla f|$  (world, $=1$ ideal)"},
            {"panels": e_panels, "cmap": "viridis",
             "norm": Normalize(vmin=0.0, vmax=1.0),
             "sym": r"\|\partial f/\partial\gamma\|",
             "cbar_label": r"$\|\partial f/\partial\gamma\|$  (post-PE, $\leq 1$: 1-Lipschitz)"},
        ]
        make_figure_stacked(rows, args.axis, args.offset, args.bound, args.res,
                            args.out)
        return

    # ---- single-row figure -------------------------------------------------
    sym = r"\|\partial f / \partial \gamma\|" if args.encoded else r"|\nabla f|"
    panels, flat = eval_panels(encoded=args.encoded)
    vmax = args.vmax if args.vmax is not None else float(np.percentile(flat, 99.0))
    vmin = args.vmin if args.vmin is not None else float(np.percentile(flat, 1.0))
    if args.diverging:
        # Symmetric span around 1.0 so equal deviation either way reads equally.
        half = max(1.0 - vmin, vmax - 1.0)
        vmin, vmax = 1.0 - half, 1.0 + half
    print(f"vmin={vmin:.3f}  vmax={vmax:.3f}", flush=True)
    make_figure(panels, args.axis, args.offset, args.bound, args.res,
                vmin, vmax, args.out, cmap=args.cmap, diverging=args.diverging,
                sym=sym)


if __name__ == "__main__":
    main()
