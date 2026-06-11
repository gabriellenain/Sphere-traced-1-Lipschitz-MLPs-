"""Render a NeuS reconstruction mesh from the DTU camera of a specific view,
so the result matches the corresponding GT image pixel-for-pixel.

Run with Blender:
    /home/glenain/blender/blender -b -P render_neus_match_gt.py -- \
        --mesh scan122_paper_like_clean_largest_512.ply \
        --cameras baselines/NeuS/public_data/scan122/cameras_sphere.npz \
        --view 50 --width 1600 --height 1200 \
        --out neus_scan122_v50_render.png
"""
import sys
import argparse
from pathlib import Path

import numpy as np
import bpy
from mathutils import Matrix


def rq_decomp(A):
    """RQ decomposition of a square matrix: A = R Q, R upper-triangular, Q orthogonal."""
    P = np.fliplr(np.flipud(A))
    Q, R = np.linalg.qr(P.T)
    R = np.fliplr(np.flipud(R.T))
    Q = np.fliplr(np.flipud(Q.T))
    return R, Q


def decompose_P(P):
    """P is 3x4 = K [R | t]. Returns K (K[2,2]=1, positive diagonal), R (det +1), t."""
    M = P[:3, :3]
    K, R = rq_decomp(M)
    s = np.sign(np.diag(K))
    s[s == 0] = 1.0
    T = np.diag(s)
    K = K @ T
    R = T @ R
    if np.linalg.det(R) < 0:
        R = -R
        K = -K  # preserves K@R; restore positive diagonal below
    scale = K[2, 2]
    K = K / scale
    t = np.linalg.inv(K) @ (P[:, 3] / scale)
    return K, R, t


def set_camera_intrinsics(cam_obj, K, W, H, sensor_width=36.0):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    cam = cam_obj.data
    cam.type = 'PERSP'
    cam.lens_unit = 'MILLIMETERS'
    cam.sensor_fit = 'HORIZONTAL'
    cam.sensor_width = sensor_width
    cam.lens = fx * sensor_width / W
    cam.shift_x = (W * 0.5 - cx) / W
    cam.shift_y = (cy - H * 0.5) / W
    # pixel aspect
    bpy.context.scene.render.pixel_aspect_x = 1.0
    bpy.context.scene.render.pixel_aspect_y = fx / fy


def set_camera_extrinsics(cam_obj, R, t):
    """R, t are OpenCV world-to-camera. Blender wants camera-to-world with -Z forward."""
    R_w2c_blender = np.diag([1.0, -1.0, -1.0]) @ R
    t_w2c_blender = np.diag([1.0, -1.0, -1.0]) @ t
    R_c2w = R_w2c_blender.T
    t_c2w = -R_c2w @ t_w2c_blender
    M = np.eye(4)
    M[:3, :3] = R_c2w
    M[:3, 3] = t_c2w
    cam_obj.matrix_world = Matrix(M.tolist())


def clear_default_scene():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)


def make_clay_material(name="Clay", color=(0.72, 0.72, 0.72, 1.0)):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    diffuse = nt.nodes.new("ShaderNodeBsdfDiffuse")
    diffuse.inputs["Color"].default_value = color
    diffuse.inputs["Roughness"].default_value = 0.0
    nt.links.new(diffuse.outputs[0], out.inputs[0])
    return mat


def set_gray_world(scene, color=(0.85, 0.85, 0.85, 1.0)):
    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Color"].default_value = color
    bg.inputs["Strength"].default_value = 0.8
    out = nt.nodes.new("ShaderNodeOutputWorld")
    nt.links.new(bg.outputs[0], out.inputs[0])


def add_three_point_lights(cam_obj):
    """Lights anchored to the camera so shading is consistent with the view."""
    mw = np.array(cam_obj.matrix_world)
    cam_right = mw[:3, 0]
    cam_up = mw[:3, 1]
    cam_back = mw[:3, 2]  # +Z; camera looks at -Z
    cam_dir = -cam_back

    def add_sun(name, direction, energy):
        l = bpy.data.lights.new(name, type='SUN')
        l.energy = energy
        l.angle = np.deg2rad(20.0)
        obj = bpy.data.objects.new(name, l)
        bpy.context.collection.objects.link(obj)
        d = direction / (np.linalg.norm(direction) + 1e-9)
        z = -d
        helper = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(helper, z)) > 0.95:
            helper = np.array([0.0, 1.0, 0.0])
        x = np.cross(helper, z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        M = np.eye(4)
        M[:3, 0] = x; M[:3, 1] = y; M[:3, 2] = z
        obj.matrix_world = Matrix(M.tolist())
        return obj

    add_sun("KeyLight", cam_dir + 0.7 * cam_up - 0.2 * cam_right, 3.5)
    add_sun("FillLight", cam_dir - 0.4 * cam_up + 0.5 * cam_right, 0.6)


def import_ply_normalized(mesh_path: Path):
    bpy.ops.wm.ply_import(filepath=str(mesh_path))
    obj = bpy.context.selected_objects[0]
    obj.matrix_world = Matrix.Identity(4)
    return obj


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--cameras", required=True)
    ap.add_argument("--view", type=int, required=True)
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=1200)
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=64)
    args = ap.parse_args(argv)

    cams = np.load(args.cameras)
    world_mat = cams[f"world_mat_{args.view}"].astype(np.float64)
    scale_mat = cams[f"scale_mat_{args.view}"].astype(np.float64)

    # Combined projection: mesh stays in normalized [-1,1] space
    P = (world_mat @ scale_mat)[:3, :]
    K, R, t = decompose_P(P)
    print("K:\n", K)
    print("R:\n", R)
    print("t:", t)

    clear_default_scene()

    cam_data = bpy.data.cameras.new("DTUcam")
    cam_obj = bpy.data.objects.new("DTUcam", cam_data)
    bpy.context.collection.objects.link(cam_obj)

    scene = bpy.context.scene
    scene.camera = cam_obj
    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.image_settings.compression = 0

    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = True

    set_camera_intrinsics(cam_obj, K, args.width, args.height)
    set_camera_extrinsics(cam_obj, R, t)

    mesh_obj = import_ply_normalized(Path(args.mesh))

    cam_data.clip_start = 0.001
    cam_data.clip_end = 1000.0
    clay = make_clay_material()
    if mesh_obj.data.materials:
        mesh_obj.data.materials[0] = clay
    else:
        mesh_obj.data.materials.append(clay)

    set_gray_world(scene)
    add_three_point_lights(cam_obj)

    print("\n--- BLENDER CAMERA STATE ---")
    print("matrix_world:\n", np.array(cam_obj.matrix_world))
    print("location:", tuple(cam_obj.location))
    print("rotation_euler (deg):", tuple(np.degrees(cam_obj.rotation_euler)))
    print("lens (mm):", cam_data.lens)
    print("sensor_width / fit:", cam_data.sensor_width, cam_data.sensor_fit)
    print("shift_x/y:", cam_data.shift_x, cam_data.shift_y)
    print("clip start/end:", cam_data.clip_start, cam_data.clip_end)
    print("res:", scene.render.resolution_x, scene.render.resolution_y,
          "pixel aspect:", scene.render.pixel_aspect_x, scene.render.pixel_aspect_y)
    print("---\n")

    scene.render.filepath = str(Path(args.out).resolve())
    bpy.ops.render.render(write_still=True)
    print("wrote", scene.render.filepath)


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    main(argv)
