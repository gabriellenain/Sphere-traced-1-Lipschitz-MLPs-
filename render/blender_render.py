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
import os
import sys

import bpy
import numpy as np
from mathutils import Matrix

import blendertoolbox as bt


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


def render_view(mesh_path, scene_dir, view_idx, out_path, res_x, res_y):
    # ------ initialize blender (demo values) ------
    numSamples = 100
    exposure = 1.5
    bt.blenderInit(res_x, res_y, numSamples, exposure)

    # ------ read mesh ------
    bpy.ops.wm.ply_import(filepath=mesh_path) if hasattr(bpy.ops.wm, "ply_import") \
        else bpy.ops.import_mesh.ply(filepath=mesh_path)
    mesh = bpy.context.selected_objects[0]

    # bring DTU mesh into the normalized (scale_mat_inv) frame, ±0.5 box
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

    # ------ AO material (demo values) ------
    distance = 10
    samples = 32
    bt.setMat_ambient_occlusion(mesh, distance, samples)

    # ------ invisible ground / shadow catcher (demo values) ------
    bt.invisibleGround(shadowBrightness=0.9)

    # ------ camera: DTU extrinsics (only deviation from demo) ------
    K, R_opencv, center, _ = load_dtu_view(scene_dir, view_idx)
    bpy.ops.object.camera_add()
    cam = bpy.context.object
    cam.matrix_world = Matrix(opencv_to_blender_pose(R_opencv, center).tolist())
    set_camera_intrinsics(cam, K, res_x, res_y)
    bpy.context.scene.camera = cam

    # ------ ambient light (demo values) ------
    bt.setLight_ambient(color=(0.8, 0.8, 0.8, 1))

    # ------ shadow threshold (demo values) ------
    bt.shadowThreshold(alphaThreshold=0.05, interpolationMode="CARDINAL")

    # ------ save .blend so you can open it in Blender GUI and tweak ------
    blend_path = os.path.splitext(out_path)[0] + ".blend"
    bpy.ops.wm.save_mainfile(filepath=os.path.abspath(blend_path))
    print(f"[blend] saved scene to {blend_path}", flush=True)

    # ------ render ------
    bt.renderImage(out_path, cam)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--views", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--res_x", type=int, default=1600)
    ap.add_argument("--res_y", type=int, default=1200)
    ap.add_argument("--subdivision", type=int, default=0,
                    help="Catmull-Clark subdivision level; demo uses 2 but that needs ~32+GB")
    args = ap.parse_args(argv)
    global SUBDIV_LEVEL
    SUBDIV_LEVEL = args.subdivision

    os.makedirs(args.out_dir, exist_ok=True)
    for v in args.views.split(","):
        v = int(v.strip())
        out = os.path.join(args.out_dir, f"view{v:03d}_shaded.png")
        # per-view mesh: if --mesh contains a Python format slot, expand it.
        mesh_v = args.mesh.format(v=v, view=v) if ("{v" in args.mesh or "{view" in args.mesh) else args.mesh
        print(f"[blender] rendering view {v} -> {out}  (mesh={mesh_v})", flush=True)
        render_view(mesh_v, args.scene, v, out, args.res_x, args.res_y)


if __name__ == "__main__":
    main()
