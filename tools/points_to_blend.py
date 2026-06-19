#!/usr/bin/env python3
"""Load a COLMAP/SfM point cloud and save a .blend you can open in the Blender GUI.

Bare vertices don't render in Cycles, so the points are turned into renderable
geometry via a Geometry-Nodes "Mesh to Points" + "Set Material" tree (one instanced
sphere per point, screen-cheap as render points). The .blend is the deliverable;
opening it drops you in front of the cloud with a camera + light already placed.

Run headless (no GPU needed to just build the file):
  ~/blender/blender -b --python tools/points_to_blend.py -- \
      --points data/dtu_idr/scan122/sparse_sfm_points.txt \
      --out outputs/scan122_points.blend [--radius 0.004] [--render out.png]

Accepts a whitespace .txt (N,>=3) or a .ply; extra columns past XYZ are ignored.
"""
import argparse
import glob
import os
import sys

import bpy
import numpy as np


def argv_after_ddash():
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def filter_min_views(pts, points_path, scene_dir, min_views):
    """Keep points observed by >= min_views of the per-view NNNNNN_sfm_points.txt
    files in scene_dir. View count = number of per-view files whose point set
    contains that point, matched on the EXACT coordinate strings of points_path
    (row-aligned with pts, so no float32 reformatting mismatch)."""
    from collections import Counter
    fs = sorted(f for f in glob.glob(os.path.join(scene_dir, "[0-9]" * 6 + "_sfm_points.txt")))
    if not fs:
        raise FileNotFoundError(f"no per-view NNNNNN_sfm_points.txt in {scene_dir}")
    cnt = Counter()
    for f in fs:
        cnt.update(set(line.strip() for line in open(f) if line.strip()))
    keys = [line.strip() for line in open(points_path) if line.strip()]
    if len(keys) != len(pts):
        raise ValueError(f"line/point mismatch: {len(keys)} lines vs {len(pts)} points")
    vc = np.array([cnt.get(k, 0) for k in keys])
    keep = vc >= min_views
    print(f"min_views={min_views}: kept {keep.sum():,}/{len(pts):,} over {len(fs)} views "
          f"(view-count min/median/max = {vc.min()}/{int(np.median(vc))}/{vc.max()})")
    return pts[keep]


def load_points(path):
    if path.lower().endswith(".ply"):
        bpy.ops.wm.ply_import(filepath=path) if hasattr(bpy.ops.wm, "ply_import") \
            else bpy.ops.import_mesh.ply(filepath=path)
        obj = bpy.context.selected_objects[0]
        pts = np.array([v.co for v in obj.data.vertices], dtype=np.float32)
        bpy.data.objects.remove(obj, do_unlink=True)
        return pts
    pts = np.loadtxt(path, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError(f"expected (N,>=3) point file, got {pts.shape} from {path}")
    return pts[:, :3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", required=True)
    ap.add_argument("--out", required=True, help="output .blend path")
    ap.add_argument("--radius", type=float, default=0.0,
                    help="point sphere radius (world units); 0 = auto from cloud extent")
    ap.add_argument("--min-views", type=int, default=0,
                    help="keep only points seen by >= this many per-view NNNNNN_sfm_points.txt "
                         "files; 0 = no filter. Requires --scene-dir (defaults to points' dir)")
    ap.add_argument("--scene-dir", default=None,
                    help="dir holding the per-view NNNNNN_sfm_points.txt (default: points' dir)")
    ap.add_argument("--color", default="0.85,0.55,0.2", help="r,g,b in 0..1")
    ap.add_argument("--render", default=None, help="also render a PNG to this path")
    ap.add_argument("--samples", type=int, default=64)
    args = ap.parse_args(argv_after_ddash())

    pts = load_points(args.points)
    if args.min_views > 0:
        scene_dir = args.scene_dir or os.path.dirname(os.path.abspath(args.points))
        pts = filter_min_views(pts, args.points, scene_dir, args.min_views)
    ctr = pts.mean(0)
    ext = float(np.linalg.norm(pts.max(0) - pts.min(0)))
    print(f"loaded {len(pts):,} points  centre={ctr.round(3)}  diag_extent={ext:.3f}")
    radius = args.radius if args.radius > 0 else 0.0025 * ext

    bpy.ops.wm.read_factory_settings(use_empty=True)

    mesh = bpy.data.meshes.new("colmap_points")
    mesh.from_pydata(pts.tolist(), [], [])
    mesh.update()
    obj = bpy.data.objects.new("colmap_points", mesh)
    bpy.context.collection.objects.link(obj)

    # material
    mat = bpy.data.materials.new("pt_mat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    r, g, b = (float(x) for x in args.color.split(","))
    bsdf.inputs["Base Color"].default_value = (r, g, b, 1.0)
    if "Roughness" in bsdf.inputs:
        bsdf.inputs["Roughness"].default_value = 0.5

    # geometry nodes: mesh -> points (instanced spheres) -> set material
    gn = obj.modifiers.new("pts", "NODES")
    ng = bpy.data.node_groups.new("pts_ng", "GeometryNodeTree")
    gn.node_group = ng
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    nin = ng.nodes.new("NodeGroupInput")
    nout = ng.nodes.new("NodeGroupOutput")
    m2p = ng.nodes.new("GeometryNodeMeshToPoints")
    m2p.inputs["Radius"].default_value = radius
    setmat = ng.nodes.new("GeometryNodeSetMaterial")
    setmat.inputs["Material"].default_value = mat
    ng.links.new(nin.outputs[0], m2p.inputs["Mesh"])
    ng.links.new(m2p.outputs["Points"], setmat.inputs["Geometry"])
    ng.links.new(setmat.outputs["Geometry"], nout.inputs[0])

    # camera + light framed on the cloud
    cam_data = bpy.data.cameras.new("cam")
    cam = bpy.data.objects.new("cam", cam_data)
    bpy.context.collection.objects.link(cam)
    cam.location = (ctr[0] + 1.6 * ext, ctr[1] - 1.6 * ext, ctr[2] + 1.1 * ext)
    direction = np.array(ctr) - np.array(cam.location)
    import mathutils
    cam.rotation_euler = mathutils.Vector(direction).to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = cam

    light_data = bpy.data.lights.new("key", "SUN")
    light_data.energy = 3.0
    light = bpy.data.objects.new("key", light_data)
    bpy.context.collection.objects.link(light)
    light.rotation_euler = (0.6, 0.2, 0.3)

    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.cycles.samples = args.samples
    bpy.context.scene.render.film_transparent = True

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    bpy.ops.wm.save_mainfile(filepath=os.path.abspath(args.out))
    print(f"saved .blend -> {args.out}  (point radius {radius:.4g})")

    if args.render:
        os.makedirs(os.path.dirname(os.path.abspath(args.render)), exist_ok=True)
        bpy.context.scene.render.filepath = os.path.abspath(args.render)
        bpy.ops.render.render(write_still=True)
        print(f"rendered -> {args.render}")


if __name__ == "__main__":
    main()
