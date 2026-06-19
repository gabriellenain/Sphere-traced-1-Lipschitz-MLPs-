"""Blender Cycles renderer for DTU meshes — follows the BlenderToolbox
demo_ambientOcclusion.py recipe verbatim (https://github.com/HTDerekLiu/BlenderToolbox),
swapping only the mesh path and the camera (DTU extrinsics instead of bt.setCamera
so the rendered window matches the training view).

All BlenderToolbox parameter values are kept identical to the demo so you have a
known-good starting point to tweak from.

Usage:
    blender --background --python render_blender.py -- \
        --mesh path/to/mesh.ply \
        --scene path/to/dtu/scan65 \
        --views 0,8,16,32,40 \
        --out_dir outputs/blender_render_scan65
"""
import argparse
import json
import os
import sys
from collections import Counter

import bpy
import numpy as np
from mathutils import Matrix

import blendertoolbox as bt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blender_studio

# ---- studio / hero-render mode (off by default; --studio turns it on) ----
STUDIO = False
STUDIO_SAMPLES = 256
STUDIO_FILL = 0.82
STUDIO_TRANSPARENT = False
STUDIO_BG = 0.21
STUDIO_STYLE = "marble"
STUDIO_ISOLATE = True
STUDIO_WHITE_BG = False
STUDIO_SOLIDIFY = 0.0
STUDIO_FRAME_PCT = 0.0


# ---------- DTU camera ----------
def rq(M):
    M = np.asarray(M, dtype=np.float64)
    P = np.flipud(np.eye(M.shape[0]))
    Q_, R_ = np.linalg.qr((P @ M).T)
    R = P @ R_.T @ P
    Q = P @ Q_.T
    return R, Q


def load_dtu_view(scene_dir, idx):
    cam_path = None
    for name in ("cameras_sphere.npz", "cameras.npz"):
        p = os.path.join(scene_dir, name)
        if os.path.exists(p):
            cam_path = p
            break
    if cam_path is None:
        raise FileNotFoundError(f"no cameras*.npz under {scene_dir}")
    cn = np.load(cam_path)
    P = cn[f"world_mat_{idx}"][:3, :4].astype(np.float64)
    M = P[:, :3]
    K, R = rq(M)
    sign = np.sign(np.diag(K))
    sign[sign == 0] = 1.0
    T = np.diag(sign)
    K = K @ T
    R = T @ R
    if np.linalg.det(R) < 0:
        K[:, 2] *= -1.0
        R[2, :] *= -1.0
    K = K / K[2, 2]
    t = np.linalg.solve(K, P[:, 3])
    cam_center = -R.T @ t
    S_inv = cn["scale_mat_inv_0"] if "scale_mat_inv_0" in cn.files \
        else np.linalg.inv(cn["scale_mat_0"])
    cam_center = (S_inv @ np.append(cam_center, 1.0))[:3]
    return K, R, cam_center, S_inv


def load_blender_view(scene_dir, idx, W, H):
    meta_path = os.path.join(scene_dir, "transforms_train.json")
    meta = json.load(open(meta_path))
    frames = meta["frames"]
    fr = frames[idx % len(frames)]
    c2w_nerf = np.asarray(fr["transform_matrix"], dtype=np.float64)
    c2w = c2w_nerf @ np.diag([1.0, -1.0, -1.0, 1.0])
    center = c2w[:3, 3]
    R_opencv = c2w[:3, :3].T
    fx = 0.5 * W / np.tan(0.5 * float(meta["camera_angle_x"]))
    K = np.array([[fx, 0.0, W / 2.0],
                  [0.0, fx, H / 2.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    return K, R_opencv, center, np.eye(4, dtype=np.float64)


def opencv_to_blender_pose(R_opencv, center):
    """OpenCV camera (x right, y down, z forward) -> Blender (x right, y up, z back)."""
    flip = np.diag([1.0, -1.0, -1.0])
    R_bl = R_opencv.T @ flip
    M = np.eye(4)
    M[:3, :3] = R_bl
    M[:3, 3] = center
    return M


def set_camera_intrinsics(cam, K, W, H):
    sensor_w = 36.0
    cam.data.sensor_width = sensor_w
    cam.data.sensor_height = sensor_w * H / W
    cam.data.sensor_fit = "HORIZONTAL"
    cam.data.lens = K[0, 0] * sensor_w / W
    cam.data.shift_x = -(K[0, 2] - W / 2.0) / W
    cam.data.shift_y = (K[1, 2] - H / 2.0) / W
    cam.data.clip_start = 0.01
    cam.data.clip_end = 100.0


# ---------- BlenderToolbox demo_ambientOcclusion.py — verbatim params ----------
SUBDIV_LEVEL = 0


def _make_rim_material(gray):
    mat = bpy.data.materials.new("MeshMaterial_RimClean")
    mat.use_nodes = True
    mat.diffuse_color = (gray, gray, gray, 1.0)
    nt = mat.node_tree
    for node in list(nt.nodes):
        nt.nodes.remove(node)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (gray, gray, gray, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.55
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def _apply_boundary_rim_cleanup(mesh_obj, gray=0.58, ring=1):
    """Assign a subdued material to faces touching open mesh boundaries."""
    me = mesh_obj.data
    edge_counts = Counter()
    edge_to_faces = {}
    for poly in me.polygons:
        for key in poly.edge_keys:
            edge_counts[key] += 1
            edge_to_faces.setdefault(key, []).append(poly.index)

    boundary_edges = {key for key, count in edge_counts.items() if count == 1}
    rim_faces = {fi for key in boundary_edges for fi in edge_to_faces[key]}
    if ring > 1:
        face_neighbors = {p.index: set() for p in me.polygons}
        for faces in edge_to_faces.values():
            if len(faces) == 2:
                a, b = faces
                face_neighbors[a].add(b)
                face_neighbors[b].add(a)
        frontier = set(rim_faces)
        for _ in range(ring - 1):
            nxt = {n for f in frontier for n in face_neighbors[f]} - rim_faces
            rim_faces |= nxt
            frontier = nxt
            if not frontier:
                break

    rim_mat = _make_rim_material(gray)
    me.materials.append(rim_mat)
    rim_slot = len(me.materials) - 1
    for poly in me.polygons:
        if poly.index in rim_faces:
            poly.material_index = rim_slot
    me.update()
    print(
        f"[rim-clean] {mesh_obj.name}: boundary_edges={len(boundary_edges):,} "
        f"rim_faces={len(rim_faces):,} gray={gray:g} ring={ring}",
        flush=True,
    )


def render_view(mesh_path, scene_dir, view_idx, out_path, res_x, res_y,
                cam_npz=None, normalized_mesh=False,
                rim_clean_boundary=False, rim_clean_gray=0.58, rim_clean_ring=1):
    # ------ initialize blender (demo values) ------
    numSamples = 100
    exposure = 1.5
    bt.blenderInit(res_x, res_y, numSamples, exposure)

    if os.environ.get("BLENDER_FORCE_CPU") == "1":
        bpy.context.scene.cycles.device = "CPU"
        print("[cpu] BLENDER_FORCE_CPU=1 -> Cycles device = CPU", flush=True)

    # ------ read mesh ------
    bpy.ops.wm.ply_import(filepath=mesh_path) if hasattr(bpy.ops.wm, "ply_import") \
        else bpy.ops.import_mesh.ply(filepath=mesh_path)
    mesh = bpy.context.selected_objects[0]

    if normalized_mesh:
        # mesh is already in the normalized (±0.5-ish) frame — no S_inv needed.
        verts_n = np.array([v.co[:] for v in mesh.data.vertices], dtype=np.float64)
        print(f"[debug] mesh AABB (normalized): {verts_n.min(0)} .. {verts_n.max(0)}",
              flush=True)
    else:
        # bring DTU/BMVS mesh into the normalized (scale_mat_inv) frame, ±0.5 box.
        # NeRF-synthetic scenes are already in the training/render frame.
        if os.path.exists(os.path.join(scene_dir, "transforms_train.json")):
            _, _, _, S_inv = load_blender_view(scene_dir, view_idx, res_x, res_y)
        else:
            _, _, _, S_inv = load_dtu_view(scene_dir, view_idx)
        verts = np.array([v.co[:] for v in mesh.data.vertices], dtype=np.float64)
        v_h = np.concatenate([verts, np.ones((len(verts), 1))], axis=1)
        verts_n = (v_h @ S_inv.T)[:, :3]
        for i, v in enumerate(mesh.data.vertices):
            v.co = verts_n[i].tolist()
        mesh.data.update()
        print(f"[debug] mesh AABB after S_inv: {verts_n.min(0)} .. {verts_n.max(0)}",
              flush=True)

    # ------ shading (demo values) ------
    bpy.ops.object.shade_smooth()

    # ------ subdivision (demo uses level=2; disabled here because a 230K-vert DTU
    # mesh blows past 32G RAM at that level; re-enable with --subdivision 1 or 2) ------
    if SUBDIV_LEVEL > 0:
        bt.subdivision(mesh, level=SUBDIV_LEVEL)

    # ------ camera: DTU extrinsics, or a custom pose from --cam-npz ------
    if cam_npz is not None:
        cd = np.load(cam_npz)
        K = cd["K"].astype(np.float64)
        R_opencv = cd["R"].astype(np.float64)     # world->cam rotation (OpenCV)
        center = cd["center"].astype(np.float64)  # cam centre, normalized frame
    elif os.path.exists(os.path.join(scene_dir, "transforms_train.json")):
        K, R_opencv, center, _ = load_blender_view(scene_dir, view_idx, res_x, res_y)
    else:
        K, R_opencv, center, _ = load_dtu_view(scene_dir, view_idx)
    bpy.ops.object.camera_add()
    cam = bpy.context.object
    cam.matrix_world = Matrix(opencv_to_blender_pose(R_opencv, center).tolist())
    set_camera_intrinsics(cam, K, res_x, res_y)
    cam.data.clip_end = 1000.0
    bpy.context.scene.camera = cam

    if STUDIO:
        blender_studio.apply(bpy.context.scene, mesh, verts_n, cam, dict(
            samples=STUDIO_SAMPLES, fill=STUDIO_FILL, transparent=STUDIO_TRANSPARENT,
            bg=STUDIO_BG, isolate=STUDIO_ISOLATE, style=STUDIO_STYLE,
            white_bg=STUDIO_WHITE_BG, solidify=STUDIO_SOLIDIFY,
            frame_pct=STUDIO_FRAME_PCT,
        ))
        if STUDIO_TRANSPARENT:
            bt.shadowThreshold(alphaThreshold=0.0, interpolationMode="CARDINAL")
    else:
        # ------ AO material (demo values) ------
        bt.setMat_ambient_occlusion(mesh, 10, 32)
        if rim_clean_boundary:
            _apply_boundary_rim_cleanup(mesh, rim_clean_gray, rim_clean_ring)

        # ------ invisible ground / shadow catcher (demo values) ------
        bt.invisibleGround(shadowBrightness=0.9)

        # ------ ambient light (demo values) ------
        bt.setLight_ambient(color=(0.8, 0.8, 0.8, 1))

        # ------ shadow threshold (demo values) ------
        bt.shadowThreshold(alphaThreshold=0.05, interpolationMode="CARDINAL")

    # ------ save .blend so you can open it in Blender GUI and tweak ------
    blend_path = os.path.splitext(out_path)[0] + ".blend"
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_mainfile(filepath=os.path.abspath(blend_path))
    print(f"[blend] saved scene to {blend_path}", flush=True)

    # ------ render ------
    bt.renderImage(out_path, cam)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--views", default=None)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--res_x", type=int, default=1600)
    ap.add_argument("--res_y", type=int, default=1200)
    ap.add_argument("--subdivision", type=int, default=0,
                    help="Catmull-Clark subdivision level; demo uses 2 but that needs ~32+GB")
    ap.add_argument("--cam-npz", default=None,
                    help="custom camera (K, R world->cam, center) in normalized frame; "
                         "overrides dataset --views extrinsics")
    ap.add_argument("--normalized-mesh", action="store_true",
                    help="mesh PLY is already in the normalized frame; skip S_inv")
    ap.add_argument("--name", default=None,
                    help="output basename when using --cam-npz (default: 'front')")
    ap.add_argument("--rim-clean-boundary", action="store_true",
                    help="use a subdued material on faces touching open mesh boundaries")
    ap.add_argument("--rim-clean-gray", type=float, default=0.58,
                    help="gray value for --rim-clean-boundary material")
    ap.add_argument("--rim-clean-ring", type=int, default=1,
                    help="number of face rings from open boundary to tint")
    ap.add_argument("--studio", action="store_true",
                    help="hero render: studio 3-point light, marble material, soft shadow, "
                         "AgX tonemap, denoise, tight auto-frame on the subject")
    ap.add_argument("--studio-samples", type=int, default=256)
    ap.add_argument("--studio-fill", type=float, default=0.82,
                    help="fraction of the frame the subject should fill (auto-zoom)")
    ap.add_argument("--studio-transparent", action="store_true",
                    help="render an alpha PNG (subject + soft shadow) for page compositing")
    ap.add_argument("--studio-bg", type=float, default=0.21,
                    help="neutral studio backdrop grey value (ignored if --studio-transparent)")
    ap.add_argument("--studio-style", choices=["marble", "clay"], default="marble",
                    help="marble (warm white) or clay (periwinkle sculpt-render look)")
    ap.add_argument("--studio-keep-floaters", action="store_true",
                    help="do NOT isolate the largest component (keep floor/bg geometry)")
    ap.add_argument("--studio-white-bg", action="store_true",
                    help="flat white backdrop instead of the grey studio gradient")
    ap.add_argument("--studio-solidify", type=float, default=0.0,
                    help="shell thickness (normalized units) so backdrop doesn't bleed "
                         "through an open single-view shell's silhouette cracks")
    ap.add_argument("--studio-frame-pct", type=float, default=0.0,
                    help=">0: auto-frame on the [pct,100-pct] projected percentile box so "
                         "kept floaters don't shrink/offset the subject (use with --studio-keep-floaters)")
    args = ap.parse_args(argv)
    global SUBDIV_LEVEL
    global STUDIO, STUDIO_SAMPLES, STUDIO_FILL, STUDIO_TRANSPARENT, STUDIO_BG, STUDIO_STYLE, STUDIO_ISOLATE, STUDIO_WHITE_BG, STUDIO_SOLIDIFY, STUDIO_FRAME_PCT
    SUBDIV_LEVEL = args.subdivision
    STUDIO = args.studio
    STUDIO_SAMPLES = args.studio_samples
    STUDIO_FILL = args.studio_fill
    STUDIO_TRANSPARENT = args.studio_transparent
    STUDIO_BG = args.studio_bg
    STUDIO_STYLE = args.studio_style
    STUDIO_ISOLATE = not args.studio_keep_floaters
    STUDIO_WHITE_BG = args.studio_white_bg
    STUDIO_SOLIDIFY = args.studio_solidify
    STUDIO_FRAME_PCT = args.studio_frame_pct

    os.makedirs(args.out_dir, exist_ok=True)
    if args.cam_npz is not None:
        name = args.name or "front"
        out = os.path.join(args.out_dir, f"{name}_shaded.png")
        print(f"[blender] rendering custom cam {args.cam_npz} -> {out}  (mesh={args.mesh})",
              flush=True)
        render_view(args.mesh, args.scene, 0, out, args.res_x, args.res_y,
                    cam_npz=args.cam_npz, normalized_mesh=args.normalized_mesh,
                    rim_clean_boundary=args.rim_clean_boundary,
                    rim_clean_gray=args.rim_clean_gray,
                    rim_clean_ring=args.rim_clean_ring)
        return
    for v in args.views.split(","):
        v = int(v.strip())
        out = os.path.join(args.out_dir, f"view{v:03d}_shaded.png")
        # per-view mesh: if --mesh contains a Python format slot, expand it.
        mesh_v = args.mesh.format(v=v, view=v) if ("{v" in args.mesh or "{view" in args.mesh) else args.mesh
        print(f"[blender] rendering view {v} -> {out}  (mesh={mesh_v})", flush=True)
        render_view(mesh_v, args.scene, v, out, args.res_x, args.res_y,
                    normalized_mesh=args.normalized_mesh,
                    rim_clean_boundary=args.rim_clean_boundary,
                    rim_clean_gray=args.rim_clean_gray,
                    rim_clean_ring=args.rim_clean_ring)


if __name__ == "__main__":
    main()
