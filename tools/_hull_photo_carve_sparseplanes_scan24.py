"""Diagnostic: sparse-COLMAP-guided MULTI-PLANE photocarving of the scan24 hull.

A sibling of tools/_hull_photo_carve_scan24.py. Instead of per-pixel fronto-parallel
patch warps, it sweeps whole-image plane homographies for several plane families
— standard fronto-parallel planes PLUS parallel families aligned to the dominant
COLMAP/RANSAC surface normals — and scores each plane with a windowed ZNCC map.
The question is whether sparse-guided multi-plane carving *visibly* reveals the
recessed facades / roofs that the silhouette visual hull leaves filled in.

NOTHING here trains or touches the SDF MLP. Shared geometry/IO helpers are reused
from _hull_photo_carve_scan24 so this stays a minimal adaptation.

Run inside the training venv (GPU node recommended for the render):
    source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
    python _hull_photo_carve_sparseplanes_scan24.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter

from lip_tracer.data import load_views
from lip_tracer.visual_hull import carve

# reuse the shared helpers from the per-pixel diagnostic (minimal adaptation)
from _hull_photo_carve_scan24 import (
    occ_to_mesh_world, save_ply, render_mesh, default_ref_views,
    top6_sources, world_to_idx, bilinear_gray, ray_box_t,
)

# ------------------------------------------------------------------ config ---
# DOWN / PATCH / OUT_DIR are env-overridable so resolution variants can be
# launched without forking the file (e.g. DOWN=1 PATCH=5 OUT_DIR=... python ...).
import os

RUN     = Path("/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/"
               "outputs/run_20260603_093037_scan24_4962197")
OUT_DIR = Path(os.environ.get("OUT_DIR",
               "/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/"
               "outputs/hull_photo_carve_sparseplanes_scan24"))

PAIRS_PATH: Path | None = None             # None -> <scene>/pair.txt
REF_VIEWS: list[int] | None = None         # None -> the 4 init-viz debug views

DOWN          = int(os.environ.get("DOWN", 8))    # image downscale
N_FRONTO      = 32       # fronto-parallel planes (per reference view)
N_PER_NORMAL  = 16       # planes per COLMAP/RANSAC normal
PATCH         = int(os.environ.get("PATCH", 5))   # ZNCC window (PATCH x PATCH)
N_SRC         = 6        # source views per reference

TAU_NCC       = 0.4      # min best-plane score to accept a pixel
TAU_MARGIN    = 0.05     # min (best - second distinct-depth best) margin
MIN_SRC       = 2        # min valid source views
DELTA_VOX     = 2.0      # carve strictly in front of (t* - DELTA_VOX voxels)
VOTES_REQ     = 2        # independent reference-view votes to remove a voxel

# --- COLMAP filtering / RANSAC ---
TRACK_MIN     = 3        # min track length (per-view sfm-file membership)
MASK_CONSIST  = 0.8      # min fraction of in-frame projections inside fg mask
RANSAC_THRESH = 0.01     # plane inlier distance (world units; voxel ~0.012)
RANSAC_ITERS  = 600
RANSAC_MIN_IN = 150      # min inliers to accept a plane
NORMAL_MIN_ANG = 20.0    # deg; normals closer than this are not "distinct"
N_NORMALS     = 3        # keep top-3 distinct COLMAP normals

HULL_RES      = 256
BOUND         = 1.5


# ------------------------------------------------------------ COLMAP / RANSAC -

def load_filtered_colmap(scene: Path, sweep: dict, bound: float):
    """Filtered COLMAP points + diagnostics.

    Filters: track length >= TRACK_MIN (membership across per-view sfm files),
    projects inside the foreground mask in a consistent fraction of in-frame
    views (reprojection-quality proxy, since the .txt carries no error column),
    and lies inside [-bound, bound]^3. The sparse points are produced from the
    SAME cameras as the hull, so no similarity alignment is needed; we verify
    this and log the in-hull fraction instead.
    """
    pts = np.loadtxt(scene / "sparse_sfm_points.txt", dtype=np.float64)
    n_raw = len(pts)

    # track length via per-view sfm-file membership (quantised coordinate keys)
    atol = 1e-5
    def keyset(p):
        q = np.round(p / atol).astype(np.int64)
        return {tuple(r) for r in q}
    track = np.zeros(n_raw, dtype=np.int32)
    keys = np.round(pts / atol).astype(np.int64)
    for vf in sorted(scene.glob("[0-9]*_sfm_points.txt")):
        vp = np.loadtxt(vf, dtype=np.float64)
        if vp.ndim == 1:
            vp = vp[None]
        ks = keyset(vp)
        track += np.fromiter((tuple(r) in ks for r in keys),
                             dtype=bool, count=n_raw).astype(np.int32)
    keep = track >= TRACK_MIN

    # foreground-mask consistency across current cameras
    masks = sweep["mask"]; K = sweep["K"]; c2w = sweep["c2w"]
    Hh = masks[0].shape[0]; Ww = masks[0].shape[1]
    n_inframe = np.zeros(n_raw, dtype=np.int32)
    n_fg = np.zeros(n_raw, dtype=np.int32)
    for vi in range(len(c2w)):
        R = c2w[vi][:3, :3]; c = c2w[vi][:3, 3]
        xc = (pts - c[None]) @ R
        z = xc[:, 2]; front = z > 1e-6
        uvh = xc @ K[vi].T
        u = uvh[:, 0] / np.where(front, uvh[:, 2], 1.0)
        v = uvh[:, 1] / np.where(front, uvh[:, 2], 1.0)
        ui = np.round(u).astype(int); vj = np.round(v).astype(int)
        inb = front & (ui >= 0) & (ui < Ww) & (vj >= 0) & (vj < Hh)
        n_inframe += inb.astype(np.int32)
        if inb.any():
            fg = np.zeros(n_raw, dtype=bool)
            fg[inb] = masks[vi][vj[inb], ui[inb]]
            n_fg += fg.astype(np.int32)
    consist = n_fg / np.maximum(n_inframe, 1)
    keep &= (n_inframe >= TRACK_MIN) & (consist >= MASK_CONSIST)
    keep &= np.all(np.abs(pts) <= bound, axis=1)

    filt = pts[keep]
    info = {"raw": n_raw, "kept": int(len(filt)),
            "track_min": TRACK_MIN, "mask_consist": MASK_CONSIST}
    return filt, info


def fit_planes_ransac(pts: np.ndarray):
    """Sequential RANSAC plane fitting -> up to N_NORMALS distinct normals.

    Returns list of (normal(3), inlier_count). Normals refined by PCA on inliers;
    a normal within NORMAL_MIN_ANG of an already-kept normal is skipped (its
    inliers are still removed so later planes are found)."""
    rng = np.random.default_rng(0)
    remaining = pts.copy()
    kept: list[tuple[np.ndarray, int]] = []
    for _ in range(2 * N_NORMALS + 2):
        if len(remaining) < RANSAC_MIN_IN:
            break
        best_in = None; best_cnt = 0
        for _ in range(RANSAC_ITERS):
            idx = rng.choice(len(remaining), 3, replace=False)
            p0, p1, p2 = remaining[idx]
            n = np.cross(p1 - p0, p2 - p0)
            nn = np.linalg.norm(n)
            if nn < 1e-9:
                continue
            n = n / nn
            d = n @ p0
            dist = np.abs(remaining @ n - d)
            inl = dist < RANSAC_THRESH
            cnt = int(inl.sum())
            if cnt > best_cnt:
                best_cnt = cnt; best_in = inl
        if best_in is None or best_cnt < RANSAC_MIN_IN:
            break
        # refine normal via PCA on inliers (smallest-variance direction)
        P = remaining[best_in]
        c = P.mean(0)
        _, _, Vt = np.linalg.svd(P - c, full_matrices=False)
        normal = Vt[-1] / (np.linalg.norm(Vt[-1]) + 1e-12)
        distinct = all(
            np.degrees(np.arccos(min(1.0, abs(float(normal @ k))))) > NORMAL_MIN_ANG
            for k, _ in kept)
        if distinct:
            kept.append((normal, best_cnt))
        remaining = remaining[~best_in]
        if len(kept) >= N_NORMALS:
            break
    kept.sort(key=lambda kv: kv[1], reverse=True)
    return kept[:N_NORMALS]


# ------------------------------------------------------------- plane sweep ---

def view_rays(K: np.ndarray, c2w: np.ndarray, H: int, W: int):
    """(origin(3), dirs(H,W,3) normalised world) for a downscaled pinhole view."""
    xs, ys = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")  # (H,W) each
    xs = xs.astype(np.float64); ys = ys.astype(np.float64)
    dc = np.stack([(xs - K[0, 2]) / K[0, 0],
                   (ys - K[1, 2]) / K[1, 1],
                   np.ones_like(xs)], axis=-1)               # (H, W, 3)
    d = dc @ c2w[:3, :3].T
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    return c2w[:3, 3].astype(np.float64), d


def windowed_zncc(ref: np.ndarray, B: np.ndarray, valid: np.ndarray, k: int):
    """Per-pixel ZNCC over a k x k window; trusted only where the window is
    fully valid (returns -inf elsewhere)."""
    kf = float(k * k)
    cnt = uniform_filter(valid.astype(np.float64), size=k, mode="constant") * kf
    full = cnt > kf - 0.5
    Bz = np.where(valid, B, 0.0)
    mA = uniform_filter(ref, size=k, mode="constant")
    mB = uniform_filter(Bz, size=k, mode="constant")
    mAA = uniform_filter(ref * ref, size=k, mode="constant")
    mBB = uniform_filter(Bz * Bz, size=k, mode="constant")
    mAB = uniform_filter(ref * Bz, size=k, mode="constant")
    cov = mAB - mA * mB
    vA = mAA - mA * mA
    vB = mBB - mB * mB
    ok = full & (vA > 1e-6) & (vB > 1e-6)
    z = np.full_like(ref, -np.inf)
    denom = np.sqrt(np.maximum(vA * vB, 1e-12))
    z[ok] = (cov[ok] / denom[ok])
    return z, ok


def build_planes(ref_axis: np.ndarray, colmap_normals: list[np.ndarray],
                 occ_world: np.ndarray):
    """Candidate planes for one reference view.

    Returns list of (normal(3), offset, label) where plane is n·X = offset.
    label 0 = fronto-parallel (ref axis); 1..K = COLMAP normals."""
    planes = []
    for label, (normal, n_off) in enumerate(
            [(ref_axis, N_FRONTO)] + [(nv, N_PER_NORMAL) for nv in colmap_normals]):
        proj = occ_world @ normal
        lo, hi = float(proj.min()), float(proj.max())
        for off in np.linspace(lo, hi, n_off):
            planes.append((normal.astype(np.float64), float(off), label))
    return planes


def sweep_reference(ref: int, src_ids: list[int], sweep: dict, occ: np.ndarray,
                    occ_world: np.ndarray, colmap_normals: list[np.ndarray],
                    bound: float):
    """Full-image multi-plane ZNCC sweep for one reference view.

    Returns dict with per-pixel selected (depth, normal-label, best score,
    accepted mask), the view's free-vote grid, and counters."""
    res = occ.shape[0]
    voxel = 2 * bound / max(res - 1, 1)
    gray = sweep["gray"]; mask = sweep["mask"]; K = sweep["K"]; c2w = sweep["c2w"]
    H, W = gray[ref].shape
    o, dirs = view_rays(K[ref], c2w[ref], H, W)               # (3,), (H,W,3)
    ref_axis = c2w[ref][:3, 2].astype(np.float64)
    ref_axis /= np.linalg.norm(ref_axis)
    ref_gray = gray[ref]; ref_fg = mask[ref]

    src = [(c2w[s][:3, :3].astype(np.float64), c2w[s][:3, 3].astype(np.float64),
            K[s].astype(np.float64), gray[s], mask[s]) for s in src_ids]

    planes = build_planes(ref_axis, colmap_normals, occ_world)
    P = len(planes)
    score_stack = np.full((P, H, W), -np.inf, dtype=np.float32)
    depth_stack = np.zeros((P, H, W), dtype=np.float32)
    count_stack = np.zeros((P, H, W), dtype=np.int8)
    labels = np.array([p[2] for p in planes], dtype=np.int32)

    for pi, (normal, off, _lbl) in enumerate(planes):
        denom = dirs @ normal                                # (H,W)
        par = np.abs(denom) < 1e-6
        t = (off - o @ normal) / np.where(par, 1.0, denom)   # (H,W)
        X = o[None, None] + t[..., None] * dirs              # (H,W,3)
        iz, iy, ix, okv = world_to_idx(X.reshape(-1, 3), bound, res)
        inside = np.zeros(H * W, dtype=bool)
        inside[okv] = occ[iz[okv], iy[okv], ix[okv]]
        admissible = inside.reshape(H, W) & (t > 0) & ~par
        if not admissible.any():
            continue

        zs = np.full((len(src), H, W), -np.inf, dtype=np.float32)
        vs = np.zeros((len(src), H, W), dtype=bool)
        for si, (Rs, cs, Ks, gs, ms) in enumerate(src):
            xc = (X - cs[None, None]) @ Rs                   # (H,W,3) cam coords
            z = xc[..., 2]; front = z > 1e-6
            uvh = xc @ Ks.T
            zsafe = np.where(front, uvh[..., 2], 1.0)
            us = uvh[..., 0] / zsafe; vsr = uvh[..., 1] / zsafe
            warp, vin = bilinear_gray(gs, ms, us, vsr)
            valid = vin & front & ref_fg & admissible
            z_map, ok = windowed_zncc(ref_gray, warp, valid, PATCH)
            zs[si] = z_map.astype(np.float32)
            vs[si] = ok

        cnt = vs.sum(0)                                      # (H,W) valid sources
        zsort = np.sort(np.where(vs, zs, -np.inf), axis=0)
        top2 = 0.5 * (zsort[-1] + zsort[-2])                # mean best-2
        plane_score = np.where((cnt >= MIN_SRC) & admissible, top2, -np.inf)
        score_stack[pi] = plane_score.astype(np.float32)
        depth_stack[pi] = t.astype(np.float32)
        count_stack[pi] = np.minimum(cnt, 127).astype(np.int8)

    # ---- per-pixel selection ----
    best_p = np.argmax(score_stack, axis=0)                  # (H,W)
    ii, jj = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    best_score = score_stack[best_p, ii, jj]
    best_depth = depth_stack[best_p, ii, jj]
    best_count = count_stack[best_p, ii, jj]
    best_label = labels[best_p]

    # second distinct-depth best (planes >= DELTA_VOX voxels from the winner)
    far = np.abs(depth_stack - best_depth[None]) > (DELTA_VOX * voxel)
    second = np.where(far, score_stack, -np.inf).max(axis=0)
    margin = best_score - second

    accept = (np.isfinite(best_score) & ref_fg &
              (best_count >= MIN_SRC) &
              (best_score > TAU_NCC) & (margin > TAU_MARGIN))

    # ---- conservative free-space votes (this view) ----
    free_grid = np.zeros((res, res, res), dtype=bool)
    ay, ax = np.where(accept)
    for v0, u0 in zip(ay, ax):
        tb = ray_box_t(o, dirs[v0, u0], bound)
        if tb is None:
            continue
        t_near, _ = tb
        t_cut = best_depth[v0, u0] - DELTA_VOX * voxel
        if t_cut <= t_near:
            continue
        n_steps = max(int(np.ceil((t_cut - t_near) / (0.5 * voxel))), 1)
        tt = np.linspace(t_near, t_cut, n_steps)
        Pw = o[None] + tt[:, None] * dirs[v0, u0][None]
        jz, jy, jx, okw = world_to_idx(Pw, bound, res)
        sel = okw.copy(); sel[okw] &= occ[jz[okw], jy[okw], jx[okw]]
        free_grid[jz[sel], jy[sel], jx[sel]] = True

    diag = {
        "depth": np.where(accept, best_depth, np.nan),
        "label": np.where(accept, best_label, -1),
        "zncc":  np.where(np.isfinite(best_score), best_score, np.nan),
        "accept": accept,
    }
    return diag, free_grid, int(accept.sum()), int(ref_fg.sum()), P, best_score[accept], margin[accept]


# ---------------------------------------------------------- diagnostics png --

def save_view_diag(diag: dict, ref: int, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    d = diag["depth"]
    im0 = ax[0].imshow(d, cmap="viridis"); ax[0].set_title(f"v{ref} selected depth")
    plt.colorbar(im0, ax=ax[0], fraction=0.046)
    im1 = ax[1].imshow(diag["label"], cmap="tab10", vmin=-1, vmax=8)
    ax[1].set_title("plane-normal label (0=fronto,1..3=COLMAP)")
    plt.colorbar(im1, ax=ax[1], fraction=0.046)
    im2 = ax[2].imshow(diag["zncc"], cmap="magma", vmin=0, vmax=1)
    ax[2].set_title("best ZNCC"); plt.colorbar(im2, ax=ax[2], fraction=0.046)
    ax[3].imshow(diag["accept"], cmap="gray"); ax[3].set_title("accepted mask")
    for a in ax:
        a.axis("off")
    fig.tight_layout(); fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)


def gray_photo(views: dict, vi: int, H: int, W: int) -> np.ndarray:
    from PIL import Image as _PIL
    g = views["gray_full"][vi]
    g = np.array(_PIL.fromarray((g * 255).astype(np.uint8)).resize((W, H), _PIL.BILINEAR)) / 255.0
    return np.stack([g, g, g], -1)


# ------------------------------------------------------------------- main ----

def main() -> None:
    t0 = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((RUN / "config.json").read_text())
    scene = Path(cfg["scene"]); bound = float(cfg["eval"]["bound_dtu"])
    print(f"scene={scene}  bound={bound}")

    print(f"loading views (down={DOWN}) …")
    v_lo = load_views(scene, down=DOWN)
    v_hi = load_views(scene, down=1)
    V = v_lo["c2w"].shape[0]
    to_gray = lambda im: (0.299 * im[..., 0] + 0.587 * im[..., 1] + 0.114 * im[..., 2])
    sweep = {"gray": to_gray(v_lo["images"].numpy()).astype(np.float64),
             "mask": v_lo["masks"].numpy().astype(bool),
             "K": v_lo["K"].numpy().astype(np.float64),
             "c2w": v_lo["c2w"].numpy().astype(np.float64)}
    render_views = {"H": v_hi["H"], "W": v_hi["W"],
                    "K": v_hi["K"].numpy(), "c2w": v_hi["c2w"].numpy(),
                    "gray_full": to_gray(v_hi["images"].numpy())}

    ref_views = REF_VIEWS or default_ref_views(sweep["c2w"])
    pairs = top6_sources(scene, PAIRS_PATH, V)
    (OUT_DIR / "pairs_top6_effective.txt").write_text(
        "\n".join(f"{r}: {' '.join(map(str, pairs.get(r, [])))}" for r in ref_views))
    print(f"reference views: {ref_views}")

    # --- hull -> vh_original.ply -------------------------------------------
    print(f"\ncarving visual hull (res={HULL_RES}, border_aware=True) …")
    occ = carve(scene=scene, res=HULL_RES, bound=bound, border_aware=True)
    n_occ0 = int(occ.sum())
    print(f"  occupied voxels: {n_occ0} / {occ.size}")
    verts, faces = occ_to_mesh_world(occ, bound)
    save_ply(verts, faces, OUT_DIR / "vh_original.ply")

    occ_idx = np.argwhere(occ)                       # (M,3) order (iz,iy,ix)
    voxel = 2 * bound / max(HULL_RES - 1, 1)
    if len(occ_idx) > 200000:
        occ_idx = occ_idx[np.random.default_rng(0).choice(len(occ_idx), 200000, replace=False)]
    occ_world = np.stack([occ_idx[:, 2], occ_idx[:, 1], occ_idx[:, 0]], 1) * voxel - bound

    # --- COLMAP + RANSAC normals -------------------------------------------
    print("\nfiltering COLMAP sparse points …")
    colmap_pts, cinfo = load_filtered_colmap(scene, sweep, bound)
    in_hull = world_to_idx(colmap_pts, bound, HULL_RES)
    frac_in = float(occ[in_hull[0][in_hull[3]], in_hull[1][in_hull[3]],
                        in_hull[2][in_hull[3]]].mean()) if in_hull[3].any() else 0.0
    print(f"  kept {cinfo['kept']}/{cinfo['raw']} points  (in-hull frac={frac_in:.2f})")
    planes_ransac = fit_planes_ransac(colmap_pts)
    colmap_normals = [n for n, _ in planes_ransac]
    print("  RANSAC plane normals (inliers):")
    for n, c in planes_ransac:
        print(f"    n=[{n[0]:+.3f},{n[1]:+.3f},{n[2]:+.3f}]  inliers={c}")

    # --- multi-plane sweep + carve -----------------------------------------
    print(f"\nsweeping {len(ref_views)} reference views "
          f"({N_FRONTO} fronto + {len(colmap_normals)}x{N_PER_NORMAL} colmap planes) …")
    votes = np.zeros_like(occ, dtype=np.int32)
    n_fg_tot = n_acc_tot = n_planes_tot = 0
    all_sc: list[float] = []; all_mg: list[float] = []
    for r in ref_views:
        src_ids = pairs.get(r, [])
        if len(src_ids) < MIN_SRC:
            print(f"  ref {r}: only {len(src_ids)} sources, skipping"); continue
        diag, free_grid, n_acc, n_fg, P, sc, mg = sweep_reference(
            r, src_ids, sweep, occ, occ_world, colmap_normals, bound)
        votes += free_grid.astype(np.int32)
        n_fg_tot += n_fg; n_acc_tot += n_acc; n_planes_tot += P
        all_sc += list(sc); all_mg += list(mg[np.isfinite(mg)])
        save_view_diag(diag, r, OUT_DIR / f"diag_v{r}.png")
        print(f"  ref {r:2d}: planes={P}  fg_px={n_fg:6d}  accepted={n_acc:6d} "
              f"({100*n_acc/max(n_fg,1):4.1f}%)  free_voxels={int(free_grid.sum())}")

    remove = (votes >= VOTES_REQ) & occ
    carved = occ & ~remove
    n_removed = int(remove.sum())
    print(f"\n  removed voxels (>= {VOTES_REQ} votes): {n_removed} "
          f"({100*n_removed/max(n_occ0,1):.2f}% of hull)")
    cverts, cfaces = occ_to_mesh_world(carved, bound)
    save_ply(cverts, cfaces, OUT_DIR / "vh_photo_carved_sparseplanes.ply")

    # --- side-by-side render (init-viz settings) ---------------------------
    print("\nrendering original vs carved (init-viz Phong settings) …")
    imgs_o = render_mesh(verts, faces, render_views, ref_views)
    imgs_c = render_mesh(cverts, cfaces, render_views, ref_views)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Hc, Wc = imgs_o[0].shape[:2]
    fig, axes = plt.subplots(len(ref_views), 3, figsize=(12, 4 * len(ref_views)), squeeze=False)
    for row, vi in enumerate(ref_views):
        for col, (img, lbl) in enumerate([
                (gray_photo(render_views, vi, Hc, Wc), f"photo v{vi}"),
                (imgs_o[row], "vh_original"),
                (imgs_c[row], "vh_photo_carved_sparseplanes")]):
            axes[row][col].imshow(np.clip(img, 0, 1)); axes[row][col].axis("off")
            if row == 0:
                axes[row][col].set_title(lbl, fontsize=11)
    fig.suptitle(f"scan24 hull vs sparse-plane photo-carved "
                 f"(removed {100*n_removed/max(n_occ0,1):.1f}% of voxels)", fontsize=12)
    fig.tight_layout(); fig.savefig(OUT_DIR / "vh_compare.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("  saved vh_compare.png")

    # --- log ---------------------------------------------------------------
    runtime = time.perf_counter() - t0
    sc = np.array(all_sc); mg = np.array(all_mg)
    summary = {
        "scene": str(scene), "ref_views": ref_views,
        "sources_per_ref": {int(r): pairs.get(r, []) for r in ref_views},
        "settings": {"down": DOWN, "n_fronto": N_FRONTO, "n_per_normal": N_PER_NORMAL,
                     "patch": PATCH, "n_src": N_SRC, "tau_ncc": TAU_NCC,
                     "tau_margin": TAU_MARGIN, "delta_vox": DELTA_VOX,
                     "votes_req": VOTES_REQ, "hull_res": HULL_RES, "bound": bound},
        "colmap_points_raw": cinfo["raw"], "colmap_points_kept": cinfo["kept"],
        "colmap_in_hull_frac": frac_in,
        "ransac_normals": [{"normal": n.tolist(), "inliers": int(c)}
                           for n, c in planes_ransac],
        "planes_tested_total": n_planes_tot,
        "hull_voxels": n_occ0, "voxels_removed": n_removed,
        "pct_voxels_removed": 100 * n_removed / max(n_occ0, 1),
        "fg_pixels": n_fg_tot, "accepted_pixels": n_acc_tot,
        "frac_fg_accepted": n_acc_tot / max(n_fg_tot, 1),
        "ncc_mean": float(sc.mean()) if sc.size else None,
        "ncc_median": float(np.median(sc)) if sc.size else None,
        "margin_mean": float(mg.mean()) if mg.size else None,
        "margin_median": float(np.median(mg)) if mg.size else None,
        "runtime_sec": runtime,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n================ SUMMARY ================")
    print(f"  reference views     : {ref_views}")
    print(f"  COLMAP points       : {cinfo['kept']}/{cinfo['raw']} kept "
          f"(in-hull frac {frac_in:.2f})")
    print(f"  RANSAC normals      : "
          f"{[ (np.round(n,3).tolist(), int(c)) for n,c in planes_ransac ]}")
    print(f"  planes tested       : {n_planes_tot}")
    print(f"  hull voxels removed : {n_removed} / {n_occ0} "
          f"({summary['pct_voxels_removed']:.2f}%)")
    print(f"  fg pixels accepted  : {n_acc_tot} / {n_fg_tot} "
          f"({100*summary['frac_fg_accepted']:.2f}%)")
    if sc.size:
        print(f"  accepted NCC        : mean={sc.mean():.3f}  median={np.median(sc):.3f}")
    if mg.size:
        print(f"  peak margin         : mean={mg.mean():.3f}  median={np.median(mg):.3f}")
    print(f"  runtime             : {runtime:.1f}s")
    print(f"  outputs             : {OUT_DIR}")
    print("=========================================")


if __name__ == "__main__":
    main()
