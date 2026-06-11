#!/usr/bin/env python3
"""Keep the largest face-connected component of a mesh."""
from __future__ import annotations

import argparse
from pathlib import Path

import trimesh


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mesh", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    mesh = trimesh.load(str(args.mesh), force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    components = mesh.split(only_watertight=False)
    if not components:
        raise RuntimeError(f"no connected components found in {args.mesh}")
    largest = max(components, key=lambda part: len(part.faces))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    largest.export(str(args.out))
    print(
        f"[largest-cc] {len(components)} -> 1 component  "
        f"faces={len(largest.faces):,}/{len(mesh.faces):,}  "
        f"verts={len(largest.vertices):,}/{len(mesh.vertices):,}  "
        f"saved {args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
