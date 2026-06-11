"""Save a copy of a sphere-traced Blender scene with subdued open-boundary rims.

This is intended for screen-space sphere-traced meshes: keep the front surface
and camera untouched, but assign a less-bright material to triangles adjacent to
one-sided mesh edges so Blender's AO render does not produce tiny white rims.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import bpy


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--object", default=None, help="mesh object name; default = largest mesh")
    ap.add_argument("--gray", type=float, default=0.58, help="rim material base color")
    ap.add_argument("--ring", type=int, default=1, help="face rings from boundary to tint")
    return ap.parse_args(argv)


def largest_mesh_object(name: str | None) -> bpy.types.Object:
    if name:
        obj = bpy.data.objects.get(name)
        if obj is None or obj.type != "MESH":
            raise SystemExit(f"mesh object not found: {name}")
        return obj
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        raise SystemExit("no mesh objects in scene")
    return max(meshes, key=lambda o: len(o.data.polygons))


def make_rim_material(base: bpy.types.Material | None, gray: float) -> bpy.types.Material:
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
    if base is not None:
        mat.use_screen_refraction = getattr(base, "use_screen_refraction", False)
    return mat


def boundary_face_indices(mesh: bpy.types.Mesh, ring: int) -> set[int]:
    edge_counts: Counter[tuple[int, int]] = Counter()
    edge_to_faces: dict[tuple[int, int], list[int]] = {}
    for poly in mesh.polygons:
        for key in poly.edge_keys:
            edge_counts[key] += 1
            edge_to_faces.setdefault(key, []).append(poly.index)

    boundary_edges = {key for key, count in edge_counts.items() if count == 1}
    selected = {face for key in boundary_edges for face in edge_to_faces[key]}
    if ring <= 1:
        return selected

    face_neighbors: dict[int, set[int]] = {p.index: set() for p in mesh.polygons}
    for faces in edge_to_faces.values():
        if len(faces) == 2:
            a, b = faces
            face_neighbors[a].add(b)
            face_neighbors[b].add(a)

    frontier = set(selected)
    for _ in range(ring - 1):
        nxt = {n for f in frontier for n in face_neighbors[f]}
        nxt -= selected
        selected |= nxt
        frontier = nxt
        if not frontier:
            break
    return selected


def main() -> None:
    args = parse_args()
    obj = largest_mesh_object(args.object)
    mesh = obj.data

    base_mat = obj.data.materials[0] if obj.data.materials else None
    rim_mat = make_rim_material(base_mat, args.gray)
    obj.data.materials.append(rim_mat)
    rim_slot = len(obj.data.materials) - 1

    faces = boundary_face_indices(mesh, max(args.ring, 1))
    for poly in mesh.polygons:
        if poly.index in faces:
            poly.material_index = rim_slot
    mesh.update()

    bpy.context.scene["rimclean_note"] = (
        f"Assigned {len(faces)} open-boundary-adjacent faces on {obj.name} "
        f"to {rim_mat.name}; main surface/camera unchanged."
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.out), compress=True)
    print(
        f"[rimclean] object={obj.name} boundary_faces={len(faces)} "
        f"rim_material={rim_mat.name} out={args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
