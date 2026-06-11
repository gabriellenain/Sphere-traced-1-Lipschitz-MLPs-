#!/usr/bin/env python3
"""Import a normalized PLY mesh into Blender and save a compact .blend file."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy


def _parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--name", default="visual_hull_sdf_target")
    return parser.parse_args(argv)


def main() -> None:
    args = _parse_args()
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    mesh_path = str(args.mesh.resolve())
    if hasattr(bpy.ops.wm, "ply_import"):
        bpy.ops.wm.ply_import(filepath=mesh_path)
    else:
        bpy.ops.import_mesh.ply(filepath=mesh_path)
    obj = bpy.context.selected_objects[0]
    obj.name = args.name
    obj.data.name = f"{args.name}_mesh"

    material = bpy.data.materials.new(name="visual_hull_target_gray")
    material.diffuse_color = (0.65, 0.65, 0.65, 1.0)
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    if principled is not None:
        principled.inputs["Base Color"].default_value = (0.65, 0.65, 0.65, 1.0)
        principled.inputs["Roughness"].default_value = 0.7
    obj.data.materials.append(material)

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.shade_smooth()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.out.resolve()))
    print(f"saved -> {args.out.resolve()}")


if __name__ == "__main__":
    main()
