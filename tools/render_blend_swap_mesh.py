"""Open an existing .blend, swap the mesh for a new one, render.

Usage:
    blender -b --python render_blend_swap_mesh.py -- \
        --blend outputs/blender_render_4907708/view050_shaded.blend \
        --new-mesh scan122_paper_like_clean_largest_512.ply \
        --already-normalized \
        --out neus_scan122_v50_blend_render.png
"""
import argparse
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector


def _import_normalize(path, S_inv, material):
    """Import a world-frame PLY, push it into the normalized frame (if S_inv given),
    smooth-shade it, and assign `material`. Returns the new object."""
    bpy.ops.wm.ply_import(filepath=path) if hasattr(bpy.ops.wm, "ply_import") \
        else bpy.ops.import_mesh.ply(filepath=path)
    obj = bpy.context.selected_objects[0]
    if S_inv is not None:
        verts = np.array([v.co[:] for v in obj.data.vertices], dtype=np.float64)
        vh = np.concatenate([verts, np.ones((len(verts), 1))], axis=1)
        vn = (vh @ S_inv.T)[:, :3]
        for i, v in enumerate(obj.data.vertices):
            v.co = vn[i].tolist()
        obj.data.update()
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.shade_smooth()
    if material is not None:
        if obj.data.materials:
            obj.data.materials[0] = material
        else:
            obj.data.materials.append(material)
    return obj


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--blend", required=True)
    ap.add_argument("--new-mesh", required=True)
    ap.add_argument("--scene-dir", default="baselines/NeuS/public_data/scan122",
                    help="Folder with cameras_sphere.npz (used only if --already-normalized is NOT set)")
    ap.add_argument("--already-normalized", action="store_true",
                    help="Set if new mesh is already in normalized DTU space")
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-blend", default=None,
                    help="path for the swapped .blend (default: --out with .blend suffix)")
    ap.add_argument("--solidify", type=float, default=0.0,
                    help="thickness (normalized units) for an open single-view shell so "
                         "white-backdrop bleed through silhouette cracks is hidden")
    ap.add_argument("--extra-mesh", action="append", default=[],
                    help="neighbour-view mesh(es) to depth-composite behind the primary: "
                         "loaded into the same scene and pushed --back-offset away from the "
                         "camera, so they only show through the primary's contour gaps. Repeatable.")
    ap.add_argument("--back-offset", type=float, default=0.004,
                    help="distance (normalized units) to push --extra-mesh away from the camera "
                         "so the primary wins overlaps and neighbours only fill its gaps")
    args = ap.parse_args(argv)

    bpy.ops.wm.open_mainfile(filepath=args.blend)

    old_mesh = None
    old_material = None
    for o in bpy.data.objects:
        if o.type == 'MESH' and o.name != 'Plane':
            old_mesh = o
            if o.data.materials:
                old_material = o.data.materials[0]
            break
    if old_mesh is None:
        raise RuntimeError("could not find non-Plane mesh in .blend")

    bpy.data.objects.remove(old_mesh, do_unlink=True)

    S_inv = None
    if not args.already_normalized:
        cams = np.load(Path(args.scene_dir) / "cameras_sphere.npz")
        S_inv = cams.get("scale_mat_inv_0")
        if S_inv is None:
            S_inv = np.linalg.inv(cams["scale_mat_0"])

    new = _import_normalize(args.new_mesh, S_inv, old_material)

    # ---- depth-composite neighbour views: push them slightly behind the primary
    # along the camera view ray so the z-buffer keeps the primary everywhere it has
    # surface, and a neighbour only shows through the primary's contour gaps.
    if args.extra_mesh:
        cam = bpy.context.scene.camera
        view_dir = (cam.matrix_world.to_3x3() @ Vector((0.0, 0.0, -1.0))).normalized()
        for ep in args.extra_mesh:
            ex = _import_normalize(ep, S_inv, old_material)
            ex.location = view_dir * args.back_offset
            print(f"[extra mesh] {ep}  verts={len(ex.data.vertices)}  pushed back {args.back_offset}")

    if args.solidify > 0.0:
        # give an open single-view shell some thickness with a filled rim, so the
        # thin cracks between cut pieces show a clay wall instead of the white
        # backdrop bleeding through as bright contour lines.
        bpy.context.view_layer.objects.active = new
        mod = new.modifiers.new("Solidify", type="SOLIDIFY")
        mod.thickness = args.solidify
        mod.offset = -1.0          # extrude away from the camera-facing normals
        mod.use_rim = True
        mod.use_rim_only = False

    bbox = np.array([list(v.co) for v in new.data.vertices])
    print(f"[new mesh] verts={len(new.data.vertices)}  bbox={bbox.min(0)} .. {bbox.max(0)}")

    # save the swapped scene so the .blend (camera/lights/material preserved, mesh
    # replaced) is itself a deliverable — open + F12 reproduces the render.
    blend_out = args.save_blend or str(Path(args.out).with_suffix(".blend"))
    bpy.ops.wm.save_mainfile(filepath=str(Path(blend_out).resolve()), compress=True)
    print("saved blend", blend_out)

    bpy.context.scene.render.filepath = str(Path(args.out).resolve())
    bpy.ops.render.render(write_still=True)
    print("wrote", bpy.context.scene.render.filepath)


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    main(argv)
