#!/usr/bin/env python
"""Standalone diagnostic: compare two source-view selectors on DTU scenes.

This is a *no-training* analysis only. It does not touch the model, the loss,
or Chamfer evaluation. It loads camera centres and per-view sparse SfM points
that already ship with each DTU scene (``sfm_pairs.json`` + the per-view
``NNNNNN_sfm_points.txt`` files) and contrasts two ways of picking ``n_alt``
source views for every reference view:

  1. current      -- the ``n_alt`` nearest camera centres (Euclidean), exactly
                      what ``lip_tracer.data.precompute_alt_cameras`` does.

  2. neuralwarp    -- for every ref/src pair, intersect their visible sparse SfM
                      points, compute the triangulation (parallax) angle at every
                      common point, reject the pair if >75% of common points have
                      angle < 5 deg, then keep the ``n_alt`` survivors with the
                      most common points.

Outputs per scene (under ``--out-dir/<scene>/``):
  * pairs_<scene>.csv          -- one row per ordered ref/src pair.
  * thumbs_ref<ID>_<scene>.png -- ref image + two rows of n_alt source thumbnails.
  * cameras_ref<ID>_<scene>.png-- two-panel camera-layout (one panel per selector).
  * scatter_ref<ID>_<scene>.png-- median angle vs n_covis, Top-6 highlighted.
And across scenes (under ``--out-dir/``):
  * summary.csv                -- per-scene/per-selector aggregate stats.
  * summary_<metric>.png       -- grouped bar plots comparing the two selectors.

Camera centres come straight from ``sfm_pairs.json``'s ``camtoworld`` (verified
bit-identical to the ``cameras.npz``-derived centres the training loop uses), so
the nearest-centre row reproduces the live selector. Both selectors use the same
``n_alt`` (default 6).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection  # noqa: F401  (registers 3d)

ANGLE_THRESH_DEG = 5.0       # "small parallax" cutoff
REJECT_FRAC = 0.75           # reject pair if > this frac of common pts below thresh
QUANT_ATOL = 1e-5            # coordinate quantisation for cross-view point matching


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_scene(scene_dir: Path):
    """Return (centres (V,3), view_gids list[np.int64], coords (P,3), img_paths)."""
    meta = json.loads((scene_dir / "sfm_pairs.json").read_text())
    frames = meta["frames"]
    centres = np.stack(
        [np.asarray(fr["camtoworld"], dtype=np.float64)[:3, 3] for fr in frames]
    )

    # Build a global point id per quantised coordinate so the same SfM point seen
    # in several views maps to one id (the per-view txt files store only XYZ).
    key_to_gid: dict[tuple[int, int, int], int] = {}
    coords: list[np.ndarray] = []
    view_gids: list[np.ndarray] = []
    for fr in frames:
        pts = np.loadtxt(scene_dir / fr["sfm_sparse_points_view"], dtype=np.float64)
        if pts.ndim == 1:
            pts = pts[None]
        quant = np.round(pts / QUANT_ATOL).astype(np.int64)
        gids = np.empty(len(pts), dtype=np.int64)
        for i, (q, p) in enumerate(zip(map(tuple, quant), pts)):
            gid = key_to_gid.get(q)
            if gid is None:
                gid = len(coords)
                key_to_gid[q] = gid
                coords.append(p)
            gids[i] = gid
        view_gids.append(np.unique(gids))

    coords_arr = np.asarray(coords, dtype=np.float64)
    img_dir = scene_dir / "image"
    img_paths = sorted(p for p in img_dir.glob("*.png") if not p.name.startswith("._"))
    return centres, view_gids, coords_arr, img_paths


# --------------------------------------------------------------------------- #
# Pairwise statistics
# --------------------------------------------------------------------------- #
def triangulation_angles_deg(P: np.ndarray, c_ref: np.ndarray, c_src: np.ndarray):
    """Parallax angle (deg) at each 3D point P between the two camera rays."""
    a = c_ref[None, :] - P
    b = c_src[None, :] - P
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    denom = np.clip(na * nb, 1e-12, None)
    cos = np.clip((a * b).sum(1) / denom, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def compute_pairwise(centres, view_gids, coords):
    """Return dict[(ref,src)] -> stats and the per-ref selection sets."""
    V = len(centres)
    sets = [set(g.tolist()) for g in view_gids]
    stats: dict[tuple[int, int], dict] = {}
    for r in range(V):
        for s in range(V):
            if r == s:
                continue
            common = np.fromiter(sets[r] & sets[s], dtype=np.int64)
            n_covis = int(common.size)
            if n_covis == 0:
                med_ang = np.nan
                frac_below = np.nan
            else:
                ang = triangulation_angles_deg(coords[common], centres[r], centres[s])
                med_ang = float(np.median(ang))
                frac_below = float(np.mean(ang < ANGLE_THRESH_DEG))
            stats[(r, s)] = dict(
                camera_distance=float(np.linalg.norm(centres[r] - centres[s])),
                n_covis=n_covis,
                median_angle_deg=med_ang,
                frac_angle_below_5deg=frac_below,
            )
    return stats


def select_nearest(centres, ref, n_alt):
    d = np.linalg.norm(centres - centres[ref], axis=1)
    d[ref] = np.inf
    return list(np.argsort(d)[:n_alt])


def ranked_survivors(stats, ref, V):
    """Surviving (non-rejected, co-visible) sources for `ref`, ranked by n_covis."""
    survivors = []
    for s in range(V):
        if s == ref:
            continue
        st = stats[(ref, s)]
        if st["n_covis"] == 0:
            continue
        fb = st["frac_angle_below_5deg"]
        if not np.isnan(fb) and fb > REJECT_FRAC:
            continue
        survivors.append((s, st["n_covis"], st["median_angle_deg"]))
    survivors.sort(key=lambda x: x[1], reverse=True)
    return survivors


def write_pair_file(path, stats, V):
    """Write an MVSNet/NeuralWarp `pair.txt`.

    Layout::

        <V>                                  # number of reference views
        <ref_id>
        <num_surv> s0 n0 s1 n1 ...           # survivors ranked by n_covis desc

    The score field is n_covis (the number of co-visible sparse points), so the
    first ``n_alt`` ids per reference view are the ``n_alt`` with the most common
    points -- pick six later by just taking the first six. All survivors are
    kept (not truncated), so the choice of n_alt is deferred to training time.
    """
    lines = [str(V)]
    for r in range(V):
        surv = ranked_survivors(stats, r, V)
        lines.append(str(r))
        parts = [str(len(surv))]
        for s, n, _ang in surv:
            parts += [str(s), str(n)]
        lines.append(" ".join(parts))
    Path(path).write_text("\n".join(lines) + "\n")
    return [len(ranked_survivors(stats, r, V)) for r in range(V)]


def select_neuralwarp(stats, ref, V, n_alt):
    """Reject low-parallax pairs, then keep the n_alt with most common points."""
    survivors = []
    for s in range(V):
        if s == ref:
            continue
        st = stats[(ref, s)]
        if st["n_covis"] == 0:
            continue
        if st["frac_angle_below_5deg"] > REJECT_FRAC:
            continue
        survivors.append((s, st["n_covis"]))
    survivors.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in survivors[:n_alt]]


# --------------------------------------------------------------------------- #
# CSV
# --------------------------------------------------------------------------- #
def write_pairs_csv(path, stats, sel_near, sel_nw, V):
    import csv
    near_sets = {r: set(sel_near[r]) for r in range(V)}
    nw_sets = {r: set(sel_nw[r]) for r in range(V)}
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ref_id", "src_id", "camera_distance", "n_covis",
                    "median_angle_deg", "frac_angle_below_5deg",
                    "selected_nearest", "selected_neuralwarp6"])
        for r in range(V):
            for s in range(V):
                if r == s:
                    continue
                st = stats[(r, s)]
                w.writerow([
                    r, s,
                    f"{st['camera_distance']:.6f}",
                    st["n_covis"],
                    "" if np.isnan(st["median_angle_deg"]) else f"{st['median_angle_deg']:.4f}",
                    "" if np.isnan(st["frac_angle_below_5deg"]) else f"{st['frac_angle_below_5deg']:.4f}",
                    int(s in near_sets[r]),
                    int(s in nw_sets[r]),
                ])


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _load_thumb(img_paths, idx, max_px=256):
    import imageio.v2 as imageio
    img = imageio.imread(img_paths[idx])
    if img.ndim == 3 and img.shape[-1] >= 3:
        img = img[..., :3]
    h, w = img.shape[:2]
    step = max(1, int(max(h, w) / max_px))
    return img[::step, ::step]


def fig_thumbnails(out_path, ref, sel_near, sel_nw, stats, img_paths, scene, n_alt):
    rows = [("current  (nearest centre)", sel_near),
            ("NeuralWarp-style", sel_nw)]
    fig = plt.figure(figsize=(2.0 * n_alt + 2.4, 6.4))
    gs = fig.add_gridspec(2, n_alt + 1, width_ratios=[1.4] + [1] * n_alt,
                          hspace=0.35, wspace=0.08)

    # reference image spans both rows in the left column
    ax_ref = fig.add_subplot(gs[:, 0])
    ax_ref.imshow(_load_thumb(img_paths, ref))
    ax_ref.set_title(f"REF view {ref}", fontsize=11, fontweight="bold")
    ax_ref.axis("off")

    for ri, (label, sel) in enumerate(rows):
        for ci in range(n_alt):
            ax = fig.add_subplot(gs[ri, ci + 1])
            if ci < len(sel):
                s = sel[ci]
                ax.imshow(_load_thumb(img_paths, s))
                st = stats[(ref, s)]
                ang = st["median_angle_deg"]
                ax.set_title(
                    f"v{s}\n{ang:.1f}° | {st['n_covis']}",
                    fontsize=8)
            else:
                ax.text(0.5, 0.5, "(none)", ha="center", va="center")
            ax.set_xticks([]); ax.set_yticks([])
            if ci == 0:
                # row label as y-axis label (keeps the first thumbnail visible)
                ax.set_ylabel(label, fontsize=10, fontweight="bold")
                for sp in ax.spines.values():
                    sp.set_visible(False)
            else:
                ax.axis("off")
    fig.suptitle(f"{scene}  ref {ref}  -- thumbnail per source "
                 f"(id | median tri-angle | n_covis)", fontsize=11)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_cameras(out_path, ref, sel_near, sel_nw, centres, coords, scene, n_alt):
    sub = coords
    if len(sub) > 4000:
        sub = coords[np.random.default_rng(0).choice(len(coords), 4000, replace=False)]
    fig = plt.figure(figsize=(13, 6.2))
    for pi, (label, sel) in enumerate(
            [("current (nearest centre)", sel_near), ("NeuralWarp-style", sel_nw)]):
        ax = fig.add_subplot(1, 2, pi + 1, projection="3d")
        ax.scatter(sub[:, 0], sub[:, 1], sub[:, 2], s=1, c="0.7", alpha=0.25,
                   linewidths=0)
        ax.scatter(centres[:, 0], centres[:, 1], centres[:, 2], s=18,
                   c="steelblue", label="cameras")
        ax.scatter(*centres[ref], s=120, c="red", marker="*",
                   label=f"ref {ref}", depthshade=False)
        for s in sel:
            seg = np.stack([centres[ref], centres[s]])
            ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], c="crimson", lw=1.5)
            ax.scatter(*centres[s], s=45, c="orange", depthshade=False)
        ax.set_title(f"{label}\n{len(sel)} selected")
        ax.legend(loc="upper left", fontsize=8)
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    fig.suptitle(f"{scene}  ref {ref}  -- camera layout & selected edges",
                 fontsize=12)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_scatter(out_path, ref, sel_near, sel_nw, stats, V, scene):
    xs, ys, ids = [], [], []
    for s in range(V):
        if s == ref:
            continue
        st = stats[(ref, s)]
        if st["n_covis"] == 0 or np.isnan(st["median_angle_deg"]):
            continue
        xs.append(st["n_covis"]); ys.append(st["median_angle_deg"]); ids.append(s)
    xs, ys, ids = np.array(xs), np.array(ys), np.array(ids)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.scatter(xs, ys, s=22, c="0.6", label="all candidate src")
    near, nw = set(sel_near), set(sel_nw)
    mask_near = np.array([i in near for i in ids])
    mask_nw = np.array([i in nw for i in ids])
    ax.scatter(xs[mask_near], ys[mask_near], s=120, facecolors="none",
               edgecolors="tab:blue", linewidths=2, label="Top-6 current")
    ax.scatter(xs[mask_nw], ys[mask_nw], s=60, c="crimson", marker="x",
               label="Top-6 NeuralWarp")
    ax.axhline(ANGLE_THRESH_DEG, ls="--", c="k", lw=0.8,
               label=f"{ANGLE_THRESH_DEG:.0f}°")
    for i, x, y in zip(ids, xs, ys):
        if i in near or i in nw:
            ax.annotate(str(i), (x, y), fontsize=7,
                        textcoords="offset points", xytext=(3, 3))
    ax.set_xlabel("n_covis (common sparse points)")
    ax.set_ylabel("median triangulation angle (deg)")
    ax.set_title(f"{scene}  ref {ref}")
    ax.legend(fontsize=8)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Summary aggregation
# --------------------------------------------------------------------------- #
def aggregate(stats, sel, V):
    """Aggregate selected-pair stats across all reference views for one selector."""
    med_angles, n_covis, frac_flags = [], [], []
    for r in range(V):
        for s in sel[r]:
            st = stats[(r, s)]
            if not np.isnan(st["median_angle_deg"]):
                med_angles.append(st["median_angle_deg"])
                frac_flags.append(st["median_angle_deg"] < ANGLE_THRESH_DEG)
            n_covis.append(st["n_covis"])
    return dict(
        median_selected_angle=float(np.median(med_angles)) if med_angles else np.nan,
        frac_selected_below_5deg=float(np.mean(frac_flags)) if frac_flags else np.nan,
        mean_n_covis=float(np.mean(n_covis)) if n_covis else np.nan,
    )


def fig_summary_bars(out_dir, summary):
    scenes = [row["scene"] for row in summary if row["selector"] == "current"]
    metrics = [
        ("median_selected_angle", "median selected tri-angle (deg)"),
        ("frac_selected_below_5deg", "frac selected pairs median <5deg"),
        ("mean_n_covis", "mean n_covis of selected pairs"),
    ]
    by = {(row["scene"], row["selector"]): row for row in summary}
    x = np.arange(len(scenes)); w = 0.38
    for key, label in metrics:
        fig, ax = plt.subplots(figsize=(1.6 * len(scenes) + 2, 4.5))
        cur = [by[(sc, "current")][key] for sc in scenes]
        nw = [by[(sc, "neuralwarp")][key] for sc in scenes]
        b1 = ax.bar(x - w / 2, cur, w, label="current", color="tab:blue")
        b2 = ax.bar(x + w / 2, nw, w, label="neuralwarp", color="crimson")
        ax.bar_label(b1, fmt="%.2f", fontsize=8)
        ax.bar_label(b2, fmt="%.2f", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(scenes)
        ax.set_ylabel(label); ax.set_title(label)
        ax.legend()
        fig.savefig(out_dir / f"summary_{key}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def process_scene(scene_dir, out_dir, ref_views, n_alt, emit_pairs=False):
    scene = scene_dir.name
    print(f"\n=== {scene} ===")
    centres, view_gids, coords, img_paths = load_scene(scene_dir)
    V = len(centres)
    print(f"  {V} views, {len(coords)} unique sparse points, "
          f"per-view pts median={int(np.median([len(g) for g in view_gids]))}")

    stats = compute_pairwise(centres, view_gids, coords)
    sel_near = {r: select_nearest(centres, r, n_alt) for r in range(V)}
    sel_nw = {r: select_neuralwarp(stats, r, V, n_alt) for r in range(V)}

    sdir = out_dir / scene
    sdir.mkdir(parents=True, exist_ok=True)
    write_pairs_csv(sdir / f"pairs_{scene}.csv", stats, sel_near, sel_nw, V)

    if emit_pairs:
        pair_path = scene_dir / "pair.txt"
        n_surv = write_pair_file(pair_path, stats, V)
        print(f"  wrote {pair_path}  (survivors/ref: min={min(n_surv)} "
              f"max={max(n_surv)}; score field = n_covis, ranked desc)")

    refs = [r for r in ref_views if r < V]
    for r in refs:
        fig_thumbnails(sdir / f"thumbs_ref{r}_{scene}.png", r, sel_near[r],
                       sel_nw[r], stats, img_paths, scene, n_alt)
        fig_cameras(sdir / f"cameras_ref{r}_{scene}.png", r, sel_near[r],
                    sel_nw[r], centres, coords, scene, n_alt)
        fig_scatter(sdir / f"scatter_ref{r}_{scene}.png", r, sel_near[r],
                    sel_nw[r], stats, V, scene)
        n_surv = sum(1 for s in range(V) if s != r and stats[(r, s)]["n_covis"]
                     and stats[(r, s)]["frac_angle_below_5deg"] <= REJECT_FRAC)
        print(f"  ref {r:3d}: nearest={sel_near[r]}  nw={sel_nw[r]}  "
              f"(survivors={n_surv})")

    rows = []
    for name, sel in [("current", sel_near), ("neuralwarp", sel_nw)]:
        agg = aggregate(stats, sel, V)
        rows.append(dict(scene=scene, selector=name, **agg))
        print(f"  [{name:10s}] med_ang={agg['median_selected_angle']:.2f}deg  "
              f"frac<5deg={agg['frac_selected_below_5deg']:.3f}  "
              f"mean_covis={agg['mean_n_covis']:.0f}")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root",
                    default="/scratch/_projets_/willow/1-lip-tracer-new/data/dtu_idr")
    ap.add_argument("--scenes", nargs="+", default=["scan24", "scan65", "scan122"])
    ap.add_argument("--out-dir", default="outputs/view_selection_compare")
    ap.add_argument("--ref-views", nargs="+", type=int, default=[0, 24, 48],
                    help="reference views to render detailed figures for")
    ap.add_argument("--n-alt", type=int, default=6,
                    help="number of source views per reference (both selectors)")
    ap.add_argument("--emit-pairs", action="store_true",
                    help="write a NeuralWarp-style pair.txt into each scene's data "
                         "dir (survivors ranked by n_covis, score = n_covis)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)

    summary = []
    for sc in args.scenes:
        summary += process_scene(data_root / sc, out_dir, args.ref_views,
                                  args.n_alt, emit_pairs=args.emit_pairs)

    import csv
    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scene", "selector", "median_selected_angle",
                                          "frac_selected_below_5deg", "mean_n_covis"])
        w.writeheader()
        for row in summary:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                        for k, v in row.items()})
    fig_summary_bars(out_dir, summary)
    print(f"\nWrote outputs to {out_dir}/  (summary.csv + summary_*.png)")


if __name__ == "__main__":
    main()
