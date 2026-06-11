"""Blender Cycles renderer for TnT meshes — mirrors tools/render_blender.py but
swaps the DTU camera loader (cameras_sphere.npz + scale_mat) for the NSVF-TnT
pose loader (pose/0_<seq>_<frame>.txt + intrinsics.txt + bbox.txt).

Saves a .blend alongside the rendered PNG so you can open it in the GUI.

Usage:
    blender --background --python render_blender_tnt.py -- \
        --mesh    outputs/.../pred_world_mesh_tnt.ply \
        --scene   data/tnt/Barn \
        --views   0,80,160,240 \
        --out_dir outputs/blender_render_tnt_barn
"""
import argparse
import glob
import os
import sys

import bpy
import numpy as np
from mathutils import Matrix

import blendertoolbox as bt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blender_studio


SUBDIV_LEVEL = 0
DECIMATE_RATIO = 1.0  # <1.0 applies a Decimate modifier *after* rendering, shrinking the saved .blend

# ---- studio / hero-render mode (off by default; --studio turns it on) ----
STUDIO = False
STUDIO_SAMPLES = 256
STUDIO_FILL = 0.82          # fraction of the frame the subject should occupy
STUDIO_TRANSPARENT = False  # True => alpha PNG (subject + soft shadow) for page compositing
STUDIO_MAT = (0.82, 0.80, 0.76)   # warm museum-marble base colour
STUDIO_BG = 0.21            # neutral studio-grey backdrop value (ignored if transparent)
STUDIO_ISOLATE = True       # keep only the largest connected component (drops floaters)
STUDIO_STYLE = "marble"     # hero material: "marble" (warm white) or "clay" (periwinkle)
STUDIO_FRAME_PCT = 0.0      # >0: percentile auto-frame robust to kept floaters
STUDIO_NO_FRAME = False     # True => render from the EXACT training camera (no auto-frame)
STUDIO_SOLIDIFY = 0.0       # >0: shell thickness+rim so open-shell silhouette cracks
                            #     (e.g. the arm) show a wall, not white backdrop bleed



def load_tnt_view(scene_dir, idx):
    """Returns (K, R_opencv, cam_center, S_inv) for view `idx` in an NSVF-TnT scene.

    Mesh is expected in *world* frame (un-normalised, matches data.py's
    inverse of `c2w[:3,3] = (c2w[:3,3] - center) / scale`). S_inv is the 4x4 that
    takes that world frame back into the unit-cube model frame used during
    training — same role scale_mat_inv plays for DTU.
    """
    pose_files = sorted(glob.glob(os.path.join(scene_dir, "pose", "0_*.txt")))
    if not pose_files:
        raise FileNotFoundError(f"no NSVF train poses under {scene_dir}/pose")
    pp = pose_files[idx]
    c2w = np.loadtxt(pp, dtype=np.float64).reshape(4, 4)

    # bbox.txt encodes the world->normalised affine: p_n = (p_w - center) / scale
    bbox_path = os.path.join(scene_dir, "bbox.txt")
    if not os.path.exists(bbox_path):  # NSVF layout keeps bbox under pose/ (matches data.py)
        bbox_path = os.path.join(scene_dir, "pose", "bbox.txt")
    bbox = np.loadtxt(bbox_path, dtype=np.float64)
    center = 0.5 * (bbox[:3] + bbox[3:6])
    scale  = float(np.max(0.5 * (bbox[3:6] - bbox[:3])))
    # NSVF normalises only the translation; rotation is unchanged. So un-normalise
    # the c2w translation to bring the camera into the world frame our mesh lives in.
    c2w_world = c2w.copy()
    c2w_world[:3, 3] = c2w[:3, 3] * scale + center

    R_w2c = c2w_world[:3, :3].T
    # The mesh is transformed into the normalized training frame before render,
    # so the camera center must live in that same frame. Rotation is unchanged
    # by the NSVF bbox normalization.
    cam_center = c2w[:3, 3]

    K = np.loadtxt(os.path.join(scene_dir, "intrinsics.txt"), dtype=np.float64)[:3, :3]

    # world -> normalised (mirror of DTU's S_inv)
    S_inv = np.eye(4, dtype=np.float64)
    S_inv[:3, :3] = np.eye(3) / scale
    S_inv[:3,  3] = -center / scale
    return K, R_w2c, cam_center, S_inv


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
    f_pix = K[0, 0]
    cam.data.lens = sensor_w * f_pix / W


def _render_view_studio(mesh, scene_dir, view_idx, out_path, res_x, res_y, verts_n):
    """Hero render via the shared studio recipe (lights/material/backdrop/auto-frame)."""
    scene = bpy.context.scene
    # camera first (same training viewpoint); centre already normalised by loader
    K, R_opencv, center, S_inv = load_tnt_view(scene_dir, view_idx)
    bpy.ops.object.camera_add()
    cam = bpy.context.object
    cam.matrix_world = Matrix(opencv_to_blender_pose(R_opencv, center).tolist())
    set_camera_intrinsics(cam, K, res_x, res_y)
    cam.data.clip_end = 1000.0
    scene.camera = cam

    blender_studio.apply(scene, mesh, verts_n, cam, dict(
        samples=STUDIO_SAMPLES, fill=STUDIO_FILL, transparent=STUDIO_TRANSPARENT,
        bg=STUDIO_BG, isolate=STUDIO_ISOLATE, style=STUDIO_STYLE,
        frame_pct=STUDIO_FRAME_PCT, no_frame=STUDIO_NO_FRAME,
        solidify=STUDIO_SOLIDIFY,
    ))

    if STUDIO_TRANSPARENT:
        bt.shadowThreshold(alphaThreshold=0.0, interpolationMode="CARDINAL")
    bt.renderImage(out_path, cam)


def render_view(mesh_path, scene_dir, view_idx, out_path, res_x, res_y):
    numSamples = 128
    exposure = 1.5
    bt.blenderInit(res_x, res_y, numSamples, exposure)

    # BLENDER_FORCE_CPU=1: render on CPU (blenderInit defaults to OptiX/GPU). Lets
    # this run on a crowded cluster's idle CPU partition instead of queueing for a
    # GPU — a few minutes slower, no queue wait. Setting scene.cycles.device is
    # enough; no GPU compute device is touched.
    if os.environ.get("BLENDER_FORCE_CPU") == "1":
        bpy.context.scene.cycles.device = "CPU"
        print("[cpu] BLENDER_FORCE_CPU=1 -> Cycles device = CPU", flush=True)

    # ------ read mesh ------
    bpy.ops.wm.ply_import(filepath=mesh_path) if hasattr(bpy.ops.wm, "ply_import") \
        else bpy.ops.import_mesh.ply(filepath=mesh_path)
    mesh = bpy.context.selected_objects[0]

    # bring TnT world-frame mesh into the normalised (±0.5-ish) frame so the demo
    # camera distances + AO scale match what tools/render_blender.py uses for DTU.
    _, _, _, S_inv = load_tnt_view(scene_dir, view_idx)
    verts = np.array([v.co[:] for v in mesh.data.vertices], dtype=np.float64)
    v_h = np.concatenate([verts, np.ones((len(verts), 1))], axis=1)
    verts_n = (v_h @ S_inv.T)[:, :3]
    for i, v in enumerate(mesh.data.vertices):
        v.co = verts_n[i].tolist()
    mesh.data.update()
    print(f"[debug] mesh AABB after S_inv: {verts_n.min(0)} .. {verts_n.max(0)}",
          flush=True)

    # ------ shading ------
    bpy.ops.object.shade_smooth()
    if SUBDIV_LEVEL > 0:
        bt.subdivision(mesh, level=SUBDIV_LEVEL)

    if STUDIO:
        _render_view_studio(mesh, scene_dir, view_idx, out_path, res_x, res_y, verts_n)
    else:
        # ------ AO material (same demo values as tools/render_blender.py) ------
        bt.setMat_ambient_occlusion(mesh, 10, 32)

        # ------ invisible ground / shadow catcher ------
        bt.invisibleGround(shadowBrightness=0.9)

        # ------ camera (centre already normalised by load_tnt_view) ------
        K, R_opencv, center, S_inv = load_tnt_view(scene_dir, view_idx)
        bpy.ops.object.camera_add()
        cam = bpy.context.object
        cam.matrix_world = Matrix(opencv_to_blender_pose(R_opencv, center).tolist())
        set_camera_intrinsics(cam, K, res_x, res_y)
        bpy.context.scene.camera = cam

        # ------ ambient light ------
        bt.setLight_ambient(color=(0.8, 0.8, 0.8, 1))

        # ------ shadow threshold ------
        bt.shadowThreshold(alphaThreshold=0.05, interpolationMode="CARDINAL")

        # ------ render (full mesh) ------
        bt.renderImage(out_path, cam)

    # ------ decimate the mesh for a small, downloadable .blend ------
    if DECIMATE_RATIO < 1.0:
        bpy.context.view_layer.objects.active = mesh
        mod = mesh.modifiers.new(name="DecimateForBlend", type="DECIMATE")
        mod.ratio = DECIMATE_RATIO
        bpy.ops.object.modifier_apply(modifier=mod.name)

    # ------ save .blend (gzipped, optionally decimated) ------
    blend_path = os.path.splitext(out_path)[0] + ".blend"
    bpy.ops.wm.save_mainfile(filepath=os.path.abspath(blend_path), compress=True)
    print(f"[blend] saved scene to {blend_path}", flush=True)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True, help="world-frame PLY (output of marching cubes after un-normalising)")
    ap.add_argument("--scene", required=True, help="NSVF-TnT scene dir, e.g. data/tnt/Barn")
    ap.add_argument("--views", required=True, help="comma-separated NSVF view indices, e.g. 0,80,160")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--res_x", type=int, default=1920)
    ap.add_argument("--res_y", type=int, default=1080)
    ap.add_argument("--subdivision", type=int, default=0)
    ap.add_argument("--decimate-ratio", type=float, default=1.0,
                    help="<1.0 decimates mesh after render so the saved .blend is small")
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
    ap.add_argument("--studio-keep-floaters", action="store_true",
                    help="do NOT isolate the largest component (keep floor/bg geometry)")
    ap.add_argument("--studio-style", choices=["marble", "clay"], default="marble",
                    help="hero material: marble (warm white) or clay (periwinkle)")
    ap.add_argument("--studio-frame-pct", type=float, default=0.0,
                    help=">0: auto-frame on the [pct,100-pct] projected percentile box so "
                         "kept floaters don't shrink/offset the subject (use with --studio-keep-floaters)")
    ap.add_argument("--studio-no-frame", action="store_true",
                    help="render from the EXACT training camera (skip auto-frame). Use for a "
                         "faithful single-view shell whose wide ground spills past the frame "
                         "when a tall subject drives the auto zoom.")
    ap.add_argument("--studio-solidify", type=float, default=0.0,
                    help=">0: give the open single-view shell this thickness (+filled rim) so "
                         "silhouette cracks at depth discontinuities (e.g. the arm) show a wall "
                         "instead of the backdrop bleeding through as bright white contours. "
                         "Units are the normalized render frame (~0.005-0.02 typical).")
    args = ap.parse_args(argv)
    global SUBDIV_LEVEL, DECIMATE_RATIO
    global STUDIO, STUDIO_SAMPLES, STUDIO_FILL, STUDIO_TRANSPARENT, STUDIO_BG, STUDIO_ISOLATE
    global STUDIO_STYLE, STUDIO_FRAME_PCT, STUDIO_NO_FRAME, STUDIO_SOLIDIFY
    SUBDIV_LEVEL = args.subdivision
    DECIMATE_RATIO = args.decimate_ratio
    STUDIO = args.studio
    STUDIO_SAMPLES = args.studio_samples
    STUDIO_FILL = args.studio_fill
    STUDIO_TRANSPARENT = args.studio_transparent
    STUDIO_BG = args.studio_bg
    STUDIO_ISOLATE = not args.studio_keep_floaters
    STUDIO_STYLE = args.studio_style
    STUDIO_FRAME_PCT = args.studio_frame_pct
    STUDIO_NO_FRAME = args.studio_no_frame
    STUDIO_SOLIDIFY = args.studio_solidify

    os.makedirs(args.out_dir, exist_ok=True)
    for v in args.views.split(","):
        v = int(v.strip())
        out = os.path.join(args.out_dir, f"view{v:03d}_shaded.png")
        mesh_v = args.mesh.format(v=v, view=v) if ("{v" in args.mesh or "{view" in args.mesh) else args.mesh
        print(f"[blender] rendering view {v} -> {out}  (mesh={mesh_v})", flush=True)
        render_view(mesh_v, args.scene, v, out, args.res_x, args.res_y)


if __name__ == "__main__":
    main()
