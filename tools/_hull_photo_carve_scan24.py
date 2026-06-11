"""Diagnostic: cheap NCC plane-sweep carving of the scan24 visual hull.

Compares the CURRENT visual-hull initialisation with a conservatively photo-carved
version, to *visually* test whether a cheap ZNCC plane-sweep can recover the
recessed roofs / facades that the silhouette-only visual hull leaves filled in.

NOTHING here trains or touches the SDF MLP. It only:
  1. rebuilds the exact binary occupancy grid used for init (visual_hull.carve
     with the run's init params) and extracts vh_original.ply,
  2. plane-sweeps a few reference views against their 6 pair sources, scoring each
     candidate depth by the mean of the best-2 ZNCC source warps,
  3. conservatively carves hull voxels strictly in front of accepted depths
     (>=2 independent reference-view votes),
  4. extracts vh_photo_carved.ply,
  5. renders both with the SAME cameras + Phong settings as the init viz,
  6. logs carving / acceptance statistics.

Run inside the training venv:
    source /scratch/_projets_/willow/1-lip-tracer-new/.venv/bin/activate
    python _hull_photo_carve_scan24.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from skimage.measure import marching_cubes

from lip_tracer.data import load_views, load_pair_file
from lip_tracer.visual_hull import carve

# ------------------------------------------------------------------ config ---

RUN      = Path("/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/"
                "outputs/run_20260603_093037_scan24_4962197")
OUT_DIR  = Path("/home/glenain/Sphere-traced-1-Lipschitz-MLPs-/"
                "outputs/hull_photo_carve_scan24")

# A pairs_top6.txt source-view list (MVSNet pair.txt format) if present, else the
# scene's own pair.txt is used and the top-6 sources per reference are taken.
PAIRS_PATH: Path | None = None   # None -> <scene>/pair.txt

# Reference views to sweep. None -> the 4 views the init visualisation renders
# (3 evenly-spaced + the back view), i.e. the 4 already rendered for debugging.
REF_VIEWS: list[int] | None = None

# --- coarse plane-sweep settings (deliberately cheap for a first look) -------
DOWN        = 8      # image downscale
D_SAMPLES   = 32     # candidate depths per ray
PATCH       = 5      # ZNCC patch size (PATCH x PATCH)
N_SRC       = 6      # source views per reference (from pairs_top6)

TAU_NCC     = 0.4    # min best-depth score to accept a pixel
TAU_MARGIN  = 0.05   # min (best - second_best) peak margin to accept
MIN_SRC     = 2      # min valid source views for a candidate depth
DELTA_VOX   = 2.0    # carve only strictly in front of (t* - DELTA_VOX voxels)
VOTES_REQ   = 2      # independent reference-view votes to remove a hull voxel

HULL_RES    = 256
BOUND       = 1.5

# ----------------------------------------------------------------- helpers ---

def default_ref_views(c2w: np.ndarray) -> list[int]:
    """The same 4 views lip_tracer.train._render_poses renders for debugging."""
    V = len(c2w)
    ids = [int(round(i * (V - 1) / 2)) for i in range(3)]
    pos = c2w[:, :3, 3]
    dir0 = pos[ids[0]] / (np.linalg.norm(pos[ids[0]]) + 1e-6)
    dots = (pos / (np.linalg.norm(pos, axis=-1, keepdims=True) + 1e-6)) @ dir0
    ids.append(int(np.argmin(dots)))
    return ids


def top6_sources(scene: Path, pairs_path: Path | None, V: int) -> dict[int, list[int]]:
    """{ref_id: [up to N_SRC source ids]} from a MVSNet-format pair.txt."""
    path = pairs_path or (scene / "pair.txt")
    if not path.exists():
        path = scene / "pairs.txt"
    ranking = load_pair_file(path)
    out: dict[int, list[int]] = {}
    for r, ranked in ranking.items():
        seen: set[int] = set()
        srcs: list[int] = []
        for s in ranked:
            if s == r or s < 0 or s >= V or s in seen:
                continue
            seen.add(s); srcs.append(s)
            if len(srcs) == N_SRC:
                break
        out[r] = srcs
    return out


def occ_to_mesh_world(occ: np.ndarray, bound: float):
    """Marching cubes on the (z,y,x) occupancy grid -> (verts_xyz, faces)."""
    res = occ.shape[0]
    voxel = 2 * bound / max(res - 1, 1)
    verts, faces, *_ = marching_cubes(occ.astype(np.float32), level=0.5,
                                      spacing=(voxel,) * 3)
    # verts axes follow occ axes (z, y, x); reorder to world (x, y, z).
    verts_world = verts[:, [2, 1, 0]] - bound
    return verts_world.astype(np.float32), faces


def save_ply(verts: np.ndarray, faces: np.ndarray, path: Path) -> None:
    import trimesh
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.export(path)
    print(f"  saved {path.name}: {len(verts)} verts / {len(faces)} faces")


def ray_box_t(o: np.ndarray, d: np.ndarray, bound: float):
    """[t_near, t_far] of ray (o, d) through the cube [-bound, bound]^3, or None."""
    inv = 1.0 / np.where(np.abs(d) < 1e-9, 1e-9, d)
    t0 = (-bound - o) * inv
    t1 = (bound - o) * inv
    tmin = np.maximum.reduce(np.minimum(t0, t1))
    tmax = np.minimum.reduce(np.maximum(t0, t1))
    tmin = max(tmin, 1e-4)
    if tmax <= tmin:
        return None
    return tmin, tmax


def world_to_idx(p: np.ndarray, bound: float, res: int):
    """World (...,3) xyz -> integer (iz, iy, ix) into a (z,y,x) grid; in-bounds mask."""
    voxel = 2 * bound / max(res - 1, 1)
    g = np.round((p + bound) / voxel).astype(np.int64)   # (...,3) order (ix,iy,iz)
    ix, iy, iz = g[..., 0], g[..., 1], g[..., 2]
    ok = ((ix >= 0) & (ix < res) & (iy >= 0) & (iy < res) &
          (iz >= 0) & (iz < res))
    return iz, iy, ix, ok


def bilinear_gray(gray: np.ndarray, mask: np.ndarray, u: np.ndarray, v: np.ndarray):
    """Bilinear-sample gray & mask at (u, v) float pixel coords.

    Returns (values, valid) where valid requires all 4 taps in-frame and the
    bilinearly-interpolated mask > 0.5 (i.e. fully inside the foreground)."""
    H, W = gray.shape
    x0 = np.floor(u).astype(np.int64); y0 = np.floor(v).astype(np.int64)
    x1 = x0 + 1; y1 = y0 + 1
    inb = (x0 >= 0) & (y0 >= 0) & (x1 < W) & (y1 < H)
    xc0 = np.clip(x0, 0, W - 1); xc1 = np.clip(x1, 0, W - 1)
    yc0 = np.clip(y0, 0, H - 1); yc1 = np.clip(y1, 0, H - 1)
    wx = u - x0; wy = v - y0
    def samp(im):
        return ((1 - wx) * (1 - wy) * im[yc0, xc0] +
                wx * (1 - wy) * im[yc0, xc1] +
                (1 - wx) * wy * im[yc1, xc0] +
                wx * wy * im[yc1, xc1])
    val = samp(gray)
    mval = samp(mask.astype(np.float32))
    valid = inb & (mval > 0.5)
    return val, valid


# --------------------------------------------------------------- carving -----

def plane_sweep_view(ref: int, src_ids: list[int], views: dict, occ: np.ndarray,
                     bound: float):
    """ZNCC plane-sweep one reference view -> (accepted depths grid, stats).

    Returns:
      free_grid : bool (res,res,res), hull voxels this view votes free,
      accepts   : list of (score, margin) for accepted pixels,
      n_fg      : number of foreground pixels considered.
    """
    res = occ.shape[0]
    voxel = 2 * bound / max(res - 1, 1)
    half = PATCH // 2

    gray = views["gray"]; mask = views["mask"]
    K = views["K"]; c2w = views["c2w"]
    Href, Wref = gray[ref].shape

    o = c2w[ref][:3, 3].astype(np.float64)
    R_ref = c2w[ref][:3, :3].astype(np.float64)
    Kref = K[ref].astype(np.float64)
    Kref_inv = np.linalg.inv(Kref)
    n_plane = R_ref @ np.array([0.0, 0.0, 1.0])     # optical axis (world)
    n_plane /= np.linalg.norm(n_plane)

    # source camera arrays
    src = []
    for s in src_ids:
        Rs = c2w[s][:3, :3].astype(np.float64)
        cs = c2w[s][:3, 3].astype(np.float64)
        src.append((Rs.T, cs, K[s].astype(np.float64), gray[s], mask[s]))

    # patch pixel offsets (PATCH*PATCH, 2) as (du, dv)
    dd = np.arange(-half, half + 1)
    ou, ov = np.meshgrid(dd, dd, indexing="xy")
    off = np.stack([ou.ravel(), ov.ravel()], axis=-1).astype(np.float64)  # (P2, 2)
    P2 = off.shape[0]

    free_grid = np.zeros((res, res, res), dtype=bool)
    accepts: list[tuple[float, float]] = []

    fg_ys, fg_xs = np.where(mask[ref])
    n_fg = len(fg_xs)
    for u0, v0 in zip(fg_xs, fg_ys):
        # central ray direction (world, normalised)
        dc = Kref_inv @ np.array([u0 + 0.0, v0 + 0.0, 1.0])
        d_center = R_ref @ dc
        d_center /= np.linalg.norm(d_center)
        tb = ray_box_t(o, d_center, bound)
        if tb is None:
            continue
        t_near, t_far = tb
        ts = np.linspace(t_near, t_far, D_SAMPLES)          # (D,)
        X0 = o[None] + ts[:, None] * d_center[None]         # (D, 3)
        iz, iy, ix, okv = world_to_idx(X0, bound, res)
        occ_cand = np.zeros(D_SAMPLES, dtype=bool)
        occ_cand[okv] = occ[iz[okv], iy[okv], ix[okv]]
        if occ_cand.sum() < 1:
            continue

        # patch rays (world, unnormalised) for this pixel
        upx = u0 + off[:, 0]; vpx = v0 + off[:, 1]
        rays_cam = (Kref_inv @ np.stack([upx, vpx, np.ones(P2)], axis=0)).T   # (P2,3)
        r_world = rays_cam @ R_ref.T                                          # (P2,3)
        den = r_world @ n_plane                                               # (P2,)
        den = np.where(np.abs(den) < 1e-9, 1e-9, den)

        # reference patch values (constant across depth)
        ru = np.clip(np.round(vpx).astype(int), 0, Href - 1)
        rc = np.clip(np.round(upx).astype(int), 0, Wref - 1)
        ref_patch = gray[ref][ru, rc]                                        # (P2,)
        a_c = ref_patch - ref_patch.mean()
        a_norm = np.linalg.norm(a_c)
        if a_norm < 1e-4:
            continue                                # textureless ref patch -> skip

        # Vectorised over all occupied candidate depths (m) and patch pixels (P2)
        # per source. num = n·(X0-o) along the central ray for each candidate.
        cand = np.nonzero(occ_cand)[0]                            # (m,)
        m = cand.shape[0]
        num = X0[cand] @ n_plane - (o @ n_plane)                  # (m,)
        tprime = num[:, None] / den[None, :]                      # (m, P2)
        P = o[None, None] + tprime[..., None] * r_world[None]     # (m, P2, 3)
        zncc_all = np.full((D_SAMPLES, len(src)), np.nan)
        for si, (RsT, cs, Ks, gs, ms) in enumerate(src):
            xc = (P - cs[None, None]) @ RsT.T                     # (m, P2, 3) cam
            z = xc[..., 2]
            front = z > 1e-6
            uvh = xc @ Ks.T
            zsafe = np.where(front, uvh[..., 2], 1.0)
            us = uvh[..., 0] / zsafe
            vs = uvh[..., 1] / zsafe
            vals, valid = bilinear_gray(gs, ms, us, vs)           # (m, P2)
            valid &= front
            full = valid.all(axis=1)                              # (m,)
            b_c = vals - vals.mean(axis=1, keepdims=True)
            b_norm = np.linalg.norm(b_c, axis=1)
            good = full & (b_norm > 1e-4)
            z_s = (b_c @ a_c) / (a_norm * np.where(good, b_norm, 1.0))
            zncc_all[cand[good], si] = z_s[good]

        # score each candidate depth by the mean of its best-2 valid ZNCC
        scores = np.full(D_SAMPLES, -np.inf)
        for di in np.nonzero(occ_cand)[0]:
            vs = zncc_all[di][~np.isnan(zncc_all[di])]
            if len(vs) >= MIN_SRC:
                best2 = np.sort(vs)[-2:]
                scores[di] = float(best2.mean())
        if not np.isfinite(scores).any():
            continue
        order = np.argsort(scores)
        best_i = order[-1]
        best = scores[best_i]
        second = scores[order[-2]] if np.isfinite(scores[order[-2]]) else -np.inf
        margin = best - second
        if best <= TAU_NCC or margin <= TAU_MARGIN:
            continue

        accepts.append((best, margin))
        # carve strictly in front of (t* - DELTA): vote hull voxels free
        t_star = ts[best_i]
        t_cut = t_star - DELTA_VOX * voxel
        if t_cut <= t_near:
            continue
        n_steps = max(int(np.ceil((t_cut - t_near) / (0.5 * voxel))), 1)
        tt = np.linspace(t_near, t_cut, n_steps)
        Pw = o[None] + tt[:, None] * d_center[None]
        jz, jy, jx, okw = world_to_idx(Pw, bound, res)
        sel = okw.copy()
        sel[okw] &= occ[jz[okw], jy[okw], jx[okw]]
        free_grid[jz[sel], jy[sel], jx[sel]] = True

    return free_grid, accepts, n_fg


# --------------------------------------------------------------- render ------

def render_mesh(verts: np.ndarray, faces: np.ndarray, views: dict,
                view_ids: list[int], res: int = 400) -> list[np.ndarray]:
    """Ray-cast the mesh (pyembree) + Phong-shade with the init-viz constants.

    Renders the literal vh_*.ply geometry from the given cameras. Shading matches
    lip_tracer.train._render_poses: light=(0.577)^3, base=(0.72,0.72,0.85),
    shaded = (0.35 + 0.65·max(n·light,0))·base, white background.
    """
    import trimesh
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    inter = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)
    fnorm = mesh.face_normals
    light = np.array([0.577, 0.577, 0.577], dtype=np.float64)
    base  = np.array([0.72, 0.72, 0.85],    dtype=np.float64)
    H_full = views["H"]; W_full = views["W"]
    down = max(1, H_full // res)
    H, W = H_full // down, W_full // down
    imgs = []
    for vi in view_ids:
        K = views["K"][vi]; c2w = views["c2w"][vi]
        ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        xs_f = (xs + 0.5) * down - 0.5; ys_f = (ys + 0.5) * down - 0.5
        d_cam = np.stack([(xs_f - K[0, 2]) / K[0, 0],
                          (ys_f - K[1, 2]) / K[1, 1],
                          np.ones_like(xs_f)], axis=-1)
        d_w = d_cam @ c2w[:3, :3].T
        d_w /= np.linalg.norm(d_w, axis=-1, keepdims=True)
        d_w = d_w.reshape(-1, 3)
        o_w = np.broadcast_to(c2w[:3, 3], d_w.shape)
        tri = inter.intersects_first(ray_origins=o_w, ray_directions=d_w)  # (N,)
        hit = tri >= 0
        n = np.zeros_like(d_w)
        n[hit] = fnorm[tri[hit]]
        # orient normals toward the camera (oppose the view direction)
        flip = (n * d_w).sum(-1) > 0
        n[flip] *= -1.0
        diffuse = np.clip((n * light).sum(-1, keepdims=True), 0, 1)
        shaded = (0.35 + 0.65 * diffuse) * base
        img = np.where(hit[:, None], shaded, 1.0).reshape(H, W, 3)
        imgs.append(img)
    return imgs


def gray_photo(views: dict, vi: int, H: int, W: int) -> np.ndarray:
    from PIL import Image as _PIL
    g = views["gray_full"][vi]
    g = np.array(_PIL.fromarray((g * 255).astype(np.uint8)).resize((W, H), _PIL.BILINEAR)) / 255.0
    return np.stack([g, g, g], -1)


# ------------------------------------------------------------------- main ----

def main() -> None:
    t_start = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((RUN / "config.json").read_text())
    scene = Path(cfg["scene"])
    bound = float(cfg["eval"]["bound_dtu"])
    print(f"scene={scene}  bound={bound}")

    # --- views (downscaled for the sweep, full-res cameras for the render) ---
    print(f"loading views (down={DOWN}) …")
    v_lo = load_views(scene, down=DOWN)
    v_hi = load_views(scene, down=1)
    V = v_lo["c2w"].shape[0]
    to_gray = lambda im: (0.299 * im[..., 0] + 0.587 * im[..., 1] + 0.114 * im[..., 2])
    sweep = {
        "gray": to_gray(v_lo["images"].numpy()),      # (V, h, w)
        "mask": v_lo["masks"].numpy().astype(bool),
        "K":    v_lo["K"].numpy(),
        "c2w":  v_lo["c2w"].numpy(),
    }
    render_views = {
        "H": v_hi["H"], "W": v_hi["W"],
        "K": v_hi["K"].numpy(), "c2w": v_hi["c2w"].numpy(),
        "gray_full": to_gray(v_hi["images"].numpy()),
    }

    ref_views = REF_VIEWS or default_ref_views(sweep["c2w"])
    pairs = top6_sources(scene, PAIRS_PATH, V)
    (OUT_DIR / "pairs_top6_effective.txt").write_text(
        "\n".join(f"{r}: {' '.join(map(str, pairs.get(r, [])))}" for r in ref_views))
    print(f"reference views: {ref_views}")
    for r in ref_views:
        print(f"  ref {r:2d} sources -> {pairs.get(r, [])}")

    # --- 1. existing binary visual hull -> vh_original.ply -------------------
    print(f"\ncarving visual hull (res={HULL_RES}, border_aware=True) …")
    occ = carve(scene=scene, res=HULL_RES, bound=bound, border_aware=True)
    n_occ0 = int(occ.sum())
    print(f"  occupied voxels: {n_occ0} / {occ.size}")
    verts, faces = occ_to_mesh_world(occ, bound)
    save_ply(verts, faces, OUT_DIR / "vh_original.ply")

    # --- 2/3. plane-sweep + conservative carving ----------------------------
    print(f"\nplane-sweeping {len(ref_views)} reference views "
          f"(D={D_SAMPLES}, patch={PATCH}, src={N_SRC}) …")
    votes = np.zeros_like(occ, dtype=np.int32)
    all_scores: list[float] = []
    all_margins: list[float] = []
    n_fg_total = 0
    n_accept_total = 0
    for r in ref_views:
        src_ids = pairs.get(r, [])
        if len(src_ids) < MIN_SRC:
            print(f"  ref {r}: only {len(src_ids)} sources, skipping")
            continue
        free_grid, accepts, n_fg = plane_sweep_view(r, src_ids, sweep, occ, bound)
        votes += free_grid.astype(np.int32)
        n_fg_total += n_fg
        n_accept_total += len(accepts)
        all_scores += [a[0] for a in accepts]
        all_margins += [a[1] for a in accepts]
        print(f"  ref {r:2d}: fg_px={n_fg:5d}  accepted={len(accepts):5d} "
              f"({100*len(accepts)/max(n_fg,1):4.1f}%)  free_voxels={int(free_grid.sum())}")

    remove = (votes >= VOTES_REQ) & occ
    carved = occ & ~remove
    n_removed = int(remove.sum())
    print(f"\n  removed voxels (>= {VOTES_REQ} votes): {n_removed} "
          f"({100*n_removed/max(n_occ0,1):.2f}% of hull)")

    # --- 4. carved mesh -> vh_photo_carved.ply ------------------------------
    cverts, cfaces = occ_to_mesh_world(carved, bound)
    save_ply(cverts, cfaces, OUT_DIR / "vh_photo_carved.ply")

    # --- 5. side-by-side render (same cameras + Phong as init viz) ----------
    print("\nrendering original vs carved (init-viz Phong settings) …")
    imgs_o = render_mesh(verts, faces, render_views, ref_views)
    imgs_c = render_mesh(cverts, cfaces, render_views, ref_views)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Hc, Wc = imgs_o[0].shape[:2]
    fig, axes = plt.subplots(len(ref_views), 3, figsize=(12, 4 * len(ref_views)),
                             squeeze=False)
    for row, vi in enumerate(ref_views):
        photo = gray_photo(render_views, vi, Hc, Wc)
        for col, (img, lbl) in enumerate([
                (photo, f"photo v{vi}"),
                (imgs_o[row], "vh_original"),
                (imgs_c[row], "vh_photo_carved")]):
            axes[row][col].imshow(np.clip(img, 0, 1))
            axes[row][col].axis("off")
            if row == 0:
                axes[row][col].set_title(lbl, fontsize=11)
    fig.suptitle(f"scan24 hull vs photo-carved  "
                 f"(removed {100*n_removed/max(n_occ0,1):.1f}% of voxels)", fontsize=12)
    fig.tight_layout()
    cmp_path = OUT_DIR / "vh_compare.png"
    fig.savefig(cmp_path, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {cmp_path.name}")

    # --- 6. log ------------------------------------------------------------
    runtime = time.perf_counter() - t_start
    sc = np.array(all_scores); mg = np.array(all_margins)
    # margin is +inf for single-candidate rays (no competing depth — unambiguous);
    # exclude those from the margin stats so the mean is meaningful.
    mg_fin = mg[np.isfinite(mg)]
    n_single = int((~np.isfinite(mg)).sum())
    summary = {
        "scene": str(scene),
        "ref_views": ref_views,
        "sources_per_ref": {int(r): pairs.get(r, []) for r in ref_views},
        "settings": {"down": DOWN, "D": D_SAMPLES, "patch": PATCH, "n_src": N_SRC,
                     "tau_ncc": TAU_NCC, "tau_margin": TAU_MARGIN,
                     "delta_vox": DELTA_VOX, "votes_req": VOTES_REQ,
                     "hull_res": HULL_RES, "bound": bound},
        "hull_voxels": n_occ0,
        "voxels_removed": n_removed,
        "pct_voxels_removed": 100 * n_removed / max(n_occ0, 1),
        "fg_pixels": n_fg_total,
        "accepted_pixels": n_accept_total,
        "frac_fg_accepted": n_accept_total / max(n_fg_total, 1),
        "ncc_mean": float(sc.mean()) if sc.size else None,
        "ncc_median": float(np.median(sc)) if sc.size else None,
        "margin_mean": float(mg_fin.mean()) if mg_fin.size else None,
        "margin_median": float(np.median(mg_fin)) if mg_fin.size else None,
        "single_candidate_accepts": n_single,
        "runtime_sec": runtime,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n================ SUMMARY ================")
    print(f"  reference views     : {ref_views}")
    print(f"  sources per ref     : {[pairs.get(r, []) for r in ref_views]}")
    print(f"  hull voxels removed : {n_removed} / {n_occ0} "
          f"({summary['pct_voxels_removed']:.2f}%)")
    print(f"  fg pixels accepted  : {n_accept_total} / {n_fg_total} "
          f"({100*summary['frac_fg_accepted']:.2f}%)")
    if sc.size:
        print(f"  accepted NCC        : mean={sc.mean():.3f}  median={np.median(sc):.3f}")
    if mg_fin.size:
        print(f"  peak margin         : mean={mg_fin.mean():.3f}  median={np.median(mg_fin):.3f}"
              f"  ({n_single} single-candidate accepts excluded)")
    print(f"  runtime             : {runtime:.1f}s")
    print(f"  outputs             : {OUT_DIR}")
    print("=========================================")


if __name__ == "__main__":
    main()
