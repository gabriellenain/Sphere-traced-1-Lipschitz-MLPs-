#!/usr/bin/env python3
"""ICLR figure: distribution of the *supervised extent* over reference views.

The view-fixed ZNCC patch covers a world-space footprint
    extent = 2 * half_pix * z_ref / f_x          (calibrated to +/-half_pix px)
so the photometric low-pass scale *floats with the reference-view distance*.
This measures, per reference view, the supervised extent on the object surface
and contrasts the resulting distribution with the object-fixed
`--ncc-world-patch` constant. Everything is metric.

Two backends (same figure, same quantity 2*half_pix*z/f; only the surface-z
source differs):
  tnt : surface z from the official GT cloud projected per view (e.g. Ignatius
        face front-sheet). Needs scene pose/intrinsics + GT ply + _trans.txt.
  dtu : surface z from the sparse SFM points (z-buffered for visibility),
        poses/intrinsics/scale from meta_data.json.

Outputs <out>/supervised_extent_<scene>.{png,pdf} and prints summary stats.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


_PLY_T = {"char": "i1", "uchar": "u1", "uint8": "u1", "int8": "i1", "short": "i2",
          "ushort": "u2", "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
          "float": "f4", "float32": "f4", "double": "f8", "float64": "f8"}


def read_ply_xyz(path: Path) -> np.ndarray:
    """Header-aware binary-LE PLY vertex reader (x,y,z float/double, any extra
    scalar props); stops at the vertex block (ignores faces)."""
    with open(path, "rb") as fh:
        lines, raw = [], b""
        while b"end_header" not in raw:
            raw += fh.read(1)
        raw += fh.readline()                 # consume the newline after end_header
        for ln in raw.decode("ascii", "replace").splitlines():
            lines.append(ln.strip())
        nvert, props, in_vert = 0, [], False
        for ln in lines:
            if ln.startswith("element vertex"):
                nvert = int(ln.split()[-1]); in_vert = True
            elif ln.startswith("element"):
                in_vert = False
            elif ln.startswith("property") and in_vert:
                _, typ, name = ln.split()[:3]
                props.append((name, "<" + _PLY_T[typ]))
        d = np.frombuffer(fh.read(np.dtype(props).itemsize * nvert), dtype=props)
    return np.stack([d["x"].astype(np.float64), d["y"].astype(np.float64),
                     d["z"].astype(np.float64)], 1)


def pca_normals(P: np.ndarray, k: int = 24) -> np.ndarray:
    from scipy.spatial import cKDTree
    _, nb = cKDTree(P).query(P, k=k)
    Q = P[nb] - P[nb].mean(1, keepdims=True)
    _, v = np.linalg.eigh(np.einsum("nki,nkj->nij", Q, Q) / k)
    return v[:, :, 0]


# --------------------------------------------------------------------------- #
#  backends → (all_extent_mm, per_view_med_mm, per_view_dist_m, n_seen, n_tot,
#              f_x, m_per_unit, label)
# --------------------------------------------------------------------------- #
def compute_tnt(args):
    scene = Path(args.scene)
    K = np.loadtxt(scene / "intrinsics.txt"); f_x = float(K[0, 0])
    from PIL import Image
    Wimg, Himg = Image.open(sorted((scene / "rgb").glob("*.png"))[0]).size
    xyz = read_ply_xyz(Path(args.gt_ply))
    xyz = xyz[np.isfinite(xyz).all(1)]
    head = xyz[xyz[:, 2] > xyz[:, 2].max() - args.head_cm / 100.0]
    T = np.loadtxt(args.trans); Tinv = np.linalg.inv(T); s_m = float(np.linalg.norm(T[:3, 0]))
    poses = sorted((scene / "pose").glob("0_*.txt"))
    C = np.array([np.loadtxt(p).reshape(4, 4)[:3, 3] for p in poses])
    Cgt = (T @ np.c_[C, np.ones(len(C))].T).T[:, :3]
    hc = head.mean(0); front = (Cgt.mean(0) - hc); front[2] = 0; front /= np.linalg.norm(front)
    nrm = pca_normals(head); nrm[(nrm @ front) < 0] *= -1
    sel = (nrm @ front) > 0.4
    face_m = head[sel]; n_face = nrm[sel]
    face_w = (Tinv @ np.c_[face_m, np.ones(len(face_m))].T).T[:, :3]
    const = 2.0 * args.half_pix / f_x * s_m * 1000.0
    alle, pvm, pvd = [], [], []
    for i, p in enumerate(poses):
        c2w = np.loadtxt(p).reshape(4, 4); R = c2w[:3, :3]; t = c2w[:3, 3]
        Xc = (face_w - t) @ R; z = Xc[:, 2]
        u = f_x * Xc[:, 0] / np.clip(z, 1e-6, None) + K[0, 2]
        v = f_x * Xc[:, 1] / np.clip(z, 1e-6, None) + K[1, 2]
        vdir = Cgt[i] - face_m; vdir /= np.linalg.norm(vdir, axis=1, keepdims=True)
        vis = (np.einsum("ni,ni->n", n_face, vdir) > 0.15) & (z > 1e-3) \
            & (u >= 0) & (u < Wimg) & (v >= 0) & (v < Himg)
        if vis.sum() < args.min_pts:
            continue
        ext = const * z[vis]; alle.append(ext); pvm.append(np.median(ext))
        pvd.append(np.linalg.norm(C[i] - face_w.mean(0)) * s_m)
    return (np.concatenate(alle), np.array(pvm), np.array(pvd),
            len(pvm), len(poses), f_x, s_m, "GT face front-sheet")


def _rq(M):                                # M = K R  ->  K (upper-tri, +diag), R
    P = np.array([[0, 0, 1.], [0, 1, 0], [1, 0, 0]])
    Q, R = np.linalg.qr((P @ M).T); R = P @ R.T @ P; Q = P @ Q.T
    for i in range(3):
        if R[i, i] < 0:
            R[:, i] *= -1; Q[i, :] *= -1
    return R, Q


def compute_dtu(args):
    """Full-res DTU: world_mat/scale_mat from cameras.npz, surface z from sparse
    SFM points (z-buffered). Everything in the world(=mm) frame, so
    extent_mm = 2*half_pix*z_cam_mm / f_x directly."""
    scene = Path(args.scene)
    from PIL import Image
    imgs = [p for p in sorted((scene / "image").glob("*")) if not p.name.startswith("._")]
    Wimg, Himg = Image.open(imgs[0]).size
    z = np.load(scene / "cameras.npz")
    n = sum(1 for k in z.files if k.startswith("world_mat_") and "inv" not in k)
    s_mm = float(z["scale_mat_0"][0, 0]); s_m = s_mm / 1000.0
    Xn = np.loadtxt(scene / "sparse_sfm_points.txt")[:, :3]       # normalized frame
    alle, pvm, pvd, f_acc, cen = [], [], [], [], []
    for i in range(n):
        P = z[f"world_mat_{i}"][:3, :4]
        K, R = _rq(P[:, :3]); K = K / K[2, 2]; f_x = float(K[0, 0]); f_acc.append(f_x)
        t = np.linalg.solve(K, P[:, 3])                          # [R|t] = K^-1 P
        Cw = -R.T @ t                                            # camera centre (world mm)
        S = z[f"scale_mat_{i}"]; Xw = Xn * S[0, 0] + S[:3, 3]    # normalized -> world(mm)
        Xc = (R @ Xw.T).T + t; zc = Xc[:, 2]                     # camera-space depth (mm)
        u = f_x * Xc[:, 0] / np.clip(zc, 1e-6, None) + K[0, 2]
        v = f_x * Xc[:, 1] / np.clip(zc, 1e-6, None) + K[1, 2]
        infr = (zc > 1e-6) & (u >= 0) & (u < Wimg) & (v >= 0) & (v < Himg)
        if infr.sum() < args.min_pts:
            continue
        ui = u[infr].astype(int); vi = v[infr].astype(int); zi = zc[infr]
        pid = vi * Wimg + ui
        order = np.argsort(zi)
        _, first = np.unique(pid[order], return_index=True)      # nearest z per pixel
        zvis = zi[order][first]
        ext = (2.0 * args.half_pix / f_x) * zvis                 # mm (world frame is mm)
        alle.append(ext); pvm.append(np.median(ext)); cen.append(Cw)
    cen = np.array(cen); obj = (Xn * z["scale_mat_0"][0, 0] + z["scale_mat_0"][:3, 3]).mean(0)
    pvd = np.linalg.norm(cen - obj, axis=1) / 1000.0            # m
    return (np.concatenate(alle), np.array(pvm), np.array(pvd),
            len(pvm), n, float(np.mean(f_acc)), s_m, "sparse SFM (z-buffered, full-res)")


# --------------------------------------------------------------------------- #
def make_figure(data, args):
    alle, pvm, pvd, n_seen, n_tot, f_x, s_m, zsrc = data
    wp_mm = args.world_patch * s_m * 1000.0
    name = Path(args.scene).name
    print(f"[{name}] {n_seen}/{n_tot} ref views | extent mm: p10 {np.percentile(alle,10):.2f} "
          f"med {np.median(alle):.2f} p90 {np.percentile(alle,90):.2f} | "
          f"object-fixed {wp_mm:.2f} mm | z-src: {zsrc}")
    if plt is None:
        print("matplotlib unavailable; stats only"); return

    plt.rcParams.update({"font.size": 11, "axes.linewidth": 0.8})
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.2, 3.9),
                                   gridspec_kw={"wspace": 0.28})
    xmax = np.percentile(alle, 99.3) * 1.08
    axA.hist(alle, bins=42, color="#34495e", alpha=0.85, density=True,
             label="view-fixed patch  (2·half_pix·z/f)")
    axA.axvline(np.median(alle), color="#34495e", ls="--", lw=1.6,
                label=f"median {np.median(alle):.2f} mm")
    axA.axvline(wp_mm, color="#27ae60", lw=2.4,
                label=f"object-fixed {wp_mm:.2f} mm  (--ncc-world-patch {args.world_patch:g})")
    axA.set_xlabel("supervised extent  [mm]"); axA.set_ylabel("density")
    axA.set_xlim(0, xmax)
    axA.set_title("(a)  distribution over reference views × surface points", fontsize=10.5)
    axA.legend(fontsize=7.6, loc="upper right", framealpha=0.93)

    axB.scatter(pvd, pvm, s=16, c=pvm, cmap="viridis", edgecolor="none", alpha=0.85)
    axB.axhline(wp_mm, color="#27ae60", lw=2.0, label=f"object-fixed {wp_mm:.2f} mm")
    axB.set_xlabel("camera → object distance  [m]")
    axB.set_ylabel("supervised extent (per-view median)  [mm]")
    axB.set_title("(b)  extent floats with reference distance", fontsize=10.5)
    axB.legend(fontsize=8, loc="upper left", framealpha=0.92)
    axB.grid(alpha=0.25, lw=0.5)

    fig.suptitle(f"{name}: supervised patch extent  (half_pix={args.half_pix:g}, "
                 f"f≈{f_x:.0f} px)", fontsize=11.5, y=1.02)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    stem = out / f"supervised_extent_{name}"
    fig.savefig(f"{stem}.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    print(f"[saved] {stem}.png / .pdf")


def extent_sfm(scene_path, half_pix, min_pts=200):
    """Unified per-scene supervised-extent (mm) from sparse SFM pts, z-buffered.
    Works for DTU (cameras.npz + scale_mat, z already mm) and TnT (pose/ +
    intrinsics.txt, z normalized -> mm via GT-trans scale). Returns (all_mm,
    per_view_med_mm, per_view_dist_m, f_x)."""
    scene = Path(scene_path)
    Xn = np.loadtxt(scene / "sparse_sfm_points.txt")[:, :3]
    alle, pvm, cen = [], [], []
    if (scene / "cameras.npz").exists() and (scene / "image").exists():
        from PIL import Image
        imgs = [p for p in sorted((scene / "image").glob("*")) if not p.name.startswith("._")]
        Wimg, Himg = Image.open(imgs[0]).size
        z = np.load(scene / "cameras.npz")
        n = sum(1 for k in z.files if k.startswith("world_mat_") and "inv" not in k)
        S0 = z["scale_mat_0"]; Xw = Xn * S0[0, 0] + S0[:3, 3]; mm_per = 1.0
        views = []
        f_all = []
        for i in range(n):
            P = z[f"world_mat_{i}"][:3, :4]; K, R = _rq(P[:, :3]); K = K / K[2, 2]
            t = np.linalg.solve(K, P[:, 3]); views.append((float(K[0, 0]), R, t, K[0, 2], K[1, 2]))
            f_all.append(float(K[0, 0]))
        obj = Xw.mean(0)
    else:                                                      # TnT
        K = np.loadtxt(scene / "intrinsics.txt"); fx = float(K[0, 0])
        from PIL import Image
        Wimg, Himg = Image.open(sorted((scene / "rgb").glob("*.png"))[0]).size
        tr = list((Path("data/tnt_gt") / scene.name).glob("*trans*.txt"))
        mm_per = float(np.linalg.norm(np.loadtxt(tr[0])[:3, 0])) * 1000.0
        Xw = Xn
        poses = sorted((scene / "pose").glob("0_*.txt"))
        views = []
        for p in poses:
            c2w = np.loadtxt(p).reshape(4, 4)
            views.append((fx, c2w[:3, :3].T, -c2w[:3, :3].T @ c2w[:3, 3], K[0, 2], K[1, 2]))
        f_all = [fx]; obj = Xw.mean(0)
    for fx, R, t, cx, cy in views:
        Xc = (R @ Xw.T).T + t; zc = Xc[:, 2]
        u = fx * Xc[:, 0] / np.clip(zc, 1e-6, None) + cx
        v = fx * Xc[:, 1] / np.clip(zc, 1e-6, None) + cy
        infr = (zc > 1e-6) & (u >= 0) & (u < Wimg) & (v >= 0) & (v < Himg)
        if infr.sum() < min_pts:
            continue
        pid = (v[infr].astype(int)) * Wimg + u[infr].astype(int)
        zi = zc[infr]; order = np.argsort(zi)
        _, first = np.unique(pid[order], return_index=True)
        zvis = zi[order][first]
        ext = (2.0 * half_pix / fx) * zvis * mm_per
        alle.append(ext); pvm.append(np.median(ext))
        cw = -R.T @ t; cen.append(np.linalg.norm(cw - obj) * mm_per / 1000.0)
    return np.concatenate(alle), np.array(pvm), np.array(cen), float(np.mean(f_all))


def extent_gtmesh(scene_path, half_pix, world_unit_mm, gt_glob, min_pts=200):
    """Supervised extent from a GT mesh projected per view (z-buffered).
    cameras.npz (world_mat/scale_mat). GT ply auto-detected as normalized vs
    world frame. extent = 2*half_pix*z_world/f * world_unit_mm.  If
    world_unit_mm<=0 → return extent in % of object radius (non-metric scenes)."""
    scene = Path(scene_path)
    from PIL import Image
    imgs = [p for p in sorted((scene / "image").glob("*")) if not p.name.startswith("._")]
    Wimg, Himg = Image.open(imgs[0]).size
    cam = scene / "cameras.npz"
    z = np.load(cam if cam.exists() else scene / "cameras_sphere.npz")
    n = sum(1 for k in z.files if k.startswith("world_mat_") and "inv" not in k)
    S0 = z["scale_mat_0"]; s = float(S0[0, 0]); c0 = S0[:3, 3]
    pat = gt_glob.replace("{obj}", scene.name.replace("bmvs_", ""))
    gp = list(scene.glob(pat)) or list(Path().glob(pat))
    G = read_ply_xyz(gp[0])
    G = G[np.isfinite(G).all(1)]                       # drop NaN/inf verts (trimesh)
    # these trimesh GTs carry many degenerate verts at astronomical coords
    # (1e20–1e38). Drop by absolute magnitude — frame-agnostic (works whether the
    # real object is normalized ~1, DTU-mm ~1e2, or raw ~1e2), huge margin to 1e20.
    G = G[np.all(np.abs(G) < 1e5, axis=1)]
    if len(G) > 250000:
        G = G[np.random.default_rng(0).choice(len(G), 250000, replace=False)]
    # frame auto-detect: raw(world) vs normalized→world(scale_mat); pick the one
    # that projects in-frame for more cameras (robust to per-dataset GT convention).
    def _inframe(Xw):
        fr = 0.0
        for i in range(0, n, max(1, n // 5)):
            P = z[f"world_mat_{i}"][:3, :4]; K, R = _rq(P[:, :3]); K = K / K[2, 2]; fx = K[0, 0]
            Xc = (R @ Xw.T).T + np.linalg.solve(K, P[:, 3]); zc = Xc[:, 2]
            u = fx * Xc[:, 0] / np.clip(zc, 1e-6, None) + K[0, 2]
            v = fx * Xc[:, 1] / np.clip(zc, 1e-6, None) + K[1, 2]
            fr += np.mean((zc > 0) & (u >= 0) & (u < Wimg) & (v >= 0) & (v < Himg))
        return fr
    Xw = (G * s + c0) if _inframe(G * s + c0) >= _inframe(G) else G
    # Isolate the real object from scattered garbage verts: the object is in-frame
    # for most orbiting cameras; stray garbage is in-frame for only a few. Keep
    # verts seen by >=50% of sampled views (frame/scale-agnostic, robust to the
    # multi-scale degeneracies in these trimesh GTs).
    cams = list(range(0, n, max(1, n // 12)))
    cnt = np.zeros(len(Xw))
    for i in cams:
        P = z[f"world_mat_{i}"][:3, :4]; K, R = _rq(P[:, :3]); K = K / K[2, 2]; fx = K[0, 0]
        Xc = (R @ Xw.T).T + np.linalg.solve(K, P[:, 3]); zc = Xc[:, 2]
        u = fx * Xc[:, 0] / np.clip(zc, 1e-6, None) + K[0, 2]
        v = fx * Xc[:, 1] / np.clip(zc, 1e-6, None) + K[1, 2]
        cnt += (zc > 0) & (u >= 0) & (u < Wimg) & (v >= 0) & (v < Himg)
    Xw = Xw[cnt >= 0.5 * len(cams)]
    # metric: ext = 2*hp*z_world/f * world_unit_mm ; non-metric (<=0): % of object
    # radius = 2*hp*z_world/(f*r_obj)*100  (r_obj = object radius in the world frame).
    r_obj = float(np.median(np.linalg.norm(Xw - np.median(Xw, 0), axis=1)))
    unit_mm = world_unit_mm if world_unit_mm > 0 else (100.0 / r_obj)
    alle, pvm, cen = [], [], []
    for i in range(n):
        P = z[f"world_mat_{i}"][:3, :4]; K, R = _rq(P[:, :3]); K = K / K[2, 2]; fx = float(K[0, 0])
        t = np.linalg.solve(K, P[:, 3])
        Xc = (R @ Xw.T).T + t; zc = Xc[:, 2]
        u = fx * Xc[:, 0] / np.clip(zc, 1e-6, None) + K[0, 2]
        v = fx * Xc[:, 1] / np.clip(zc, 1e-6, None) + K[1, 2]
        infr = (zc > 1e-6) & (u >= 0) & (u < Wimg) & (v >= 0) & (v < Himg)
        if infr.sum() < min_pts:
            continue
        pid = (v[infr].astype(int)) * Wimg + u[infr].astype(int)
        zi = zc[infr]; order = np.argsort(zi)
        _, first = np.unique(pid[order], return_index=True)
        zvis = zi[order][first]
        ext = (2.0 * half_pix / fx) * zvis * unit_mm
        alle.append(ext); pvm.append(np.median(ext)); cen.append(np.linalg.norm(-R.T @ t))
    return np.concatenate(alle), np.array(pvm), np.array(cen), float(fx)


def make_multi(scenes, args):
    res = []
    for sc in scenes:
        try:
            if args.surface == "gtmesh":
                allmm, pvm, dist, fx = extent_gtmesh(sc, args.half_pix, args.world_unit_mm, args.gt_glob)
            else:
                allmm, pvm, dist, fx = extent_sfm(sc, args.half_pix)
        except Exception as e:
            print(f"[skip] {sc}: {e}"); continue
        res.append((Path(sc).name, allmm, np.median(allmm)))
        print(f"[{Path(sc).name}] extent mm: p10 {np.percentile(allmm,10):.2f} "
              f"med {np.median(allmm):.2f} p90 {np.percentile(allmm,90):.2f}  f≈{fx:.0f}px")
    if plt is None or not res:
        return
    res.sort(key=lambda r: r[2])
    names = [r[0] for r in res]; data = [r[1] for r in res]; meds = [r[2] for r in res]
    n = len(res)
    plt.rcParams.update({"font.size": 11, "axes.linewidth": 0.8})
    fig, ax = plt.subplots(figsize=(max(5.5, 0.62 * n + 2.6), 4.3))
    vp = ax.violinplot(data, showextrema=False, widths=0.82)
    for b in vp["bodies"]:
        b.set_facecolor("#34495e"); b.set_alpha(0.55); b.set_edgecolor("#2c3e50")
    xs = np.arange(1, n + 1)
    ax.scatter(xs, meds, color="#e67e22", zorder=5, s=26, label="median")
    if n <= 8:                                   # annotate only when uncluttered
        for x, m in zip(xs, meds):
            ax.annotate(f"{m:.2f}", (x, m), textcoords="offset points", xytext=(9, 0),
                        fontsize=8, color="#a85a0a", va="center")
    med_all = float(np.median(meds))
    ax.axhline(med_all, color="#e67e22", ls=":", lw=1.2, alpha=0.7,
               label=f"median of medians {med_all:.2f} mm")
    ax.set_xticks(xs); ax.set_xticklabels([nm.replace("scan", "") for nm in names],
                                          rotation=0 if n > 8 else 25,
                                          ha="center" if n > 8 else "right", fontsize=9)
    ax.set_xlabel("DTU scan" if "scan" in names[0] else "scene")
    ax.set_ylabel(args.ylabel)
    ax.set_ylim(0, max(np.percentile(d, 99) for d in data) * 1.12)
    if args.title:
        ax.set_title(args.title, fontsize=11.5)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.legend(fontsize=8.5, loc="upper left", framealpha=0.92, ncol=2)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    stem = out / f"supervised_extent_multi_{args.tag}"
    fig.savefig(f"{stem}.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    print(f"[saved] {stem}.png / .pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--multi", default="", help="comma list of scene dirs → one violin figure")
    ap.add_argument("--tag", default="dtu")
    ap.add_argument("--title", default="")
    ap.add_argument("--surface", choices=["sfm", "gtmesh"], default="sfm")
    ap.add_argument("--gt-glob", default="gt_mesh.ply", help="GT ply glob within each scene dir")
    ap.add_argument("--world-unit-mm", type=float, default=1.0,
                    help="mm per world unit (DTU/TnT=1, MVMannequin meters=1000; <=0 → %% of object radius)")
    ap.add_argument("--ylabel", default="supervised extent  [mm]")
    ap.add_argument("--mode", choices=["tnt", "dtu", "auto"], default="auto")
    ap.add_argument("--scene", default="data/tnt/Ignatius")
    ap.add_argument("--gt-ply", default="data/tnt_gt/Ignatius/Ignatius.ply")
    ap.add_argument("--trans", default="data/tnt_gt/Ignatius/Ignatius_trans.txt")
    ap.add_argument("--half-pix", type=float, default=2.0)
    ap.add_argument("--world-patch", type=float, default=0.009,
                    help="object-fixed footprint to mark (world units)")
    ap.add_argument("--head-cm", type=float, default=35.0, help="tnt: top-N cm = region")
    ap.add_argument("--min-pts", type=int, default=300)
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    if args.multi:
        make_multi([s for s in args.multi.split(",") if s], args)
        return
    mode = args.mode
    if mode == "auto":
        mode = "dtu" if (Path(args.scene) / "meta_data.json").exists() else "tnt"
    make_figure(compute_dtu(args) if mode == "dtu" else compute_tnt(args), args)


if __name__ == "__main__":
    main()
