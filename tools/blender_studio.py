"""Shared 'studio / hero render' setup for the Blender mesh renderers
(tools/render_blender.py for DTU/BMVS, tools/render_blender_tnt.py for TnT).

Turns the flat demo-AO look into a CVPR-grade scene: a marble material with a
subtle AO multiply, a soft three-point light rig, a low-contrast gradient
backdrop, AgX tonemapping + denoising, and a tight auto-frame that re-aims the
camera at the subject centroid so it sits centred and fills the frame.

Everything is built in the *camera's* frame, so it works for any scene up-axis
(TnT is +Y up, DTU/BMVS differ) without per-scene tuning. Call `apply(...)`
after the per-script camera has been created and positioned; it returns the
(possibly isolated) mesh object. The caller still saves the .blend — which is
the artifact that matters: open it in the GUI and hit F12.
"""
from __future__ import annotations

import bpy
import numpy as np
import mathutils


DEFAULTS = dict(
    samples=256,
    fill=0.82,            # fraction of the frame the subject should fill
    transparent=False,    # alpha PNG (subject + soft shadow) vs solid gradient backdrop
    bg=0.21,              # backdrop grey value (solid mode)
    isolate=True,         # keep only the largest connected component (drop floaters)
    style="marble",       # "marble" (warm white) or "clay" (periwinkle matcap look)
    mat=None,             # base colour override (defaults per style)
    white_bg=False,       # flat white backdrop instead of the grey gradient
    solidify=0.0,         # shell thickness so backdrop doesn't bleed through cracks
    frame_pct=0.0,        # >0: auto-frame on the [pct, 100-pct] projected percentile
                          # box instead of strict min/max, so kept floaters don't
                          # shrink/offset the subject (use with isolate=False).
    no_frame=False,       # skip auto_frame entirely: render from the EXACT training
                          # camera the caller positioned. Use for a faithful
                          # single-view shell whose wide ground spills past the
                          # frame when the tall subject drives the auto zoom.
)

STYLE_COLOR = {
    "marble": (0.82, 0.80, 0.76),   # warm museum white
    "clay":   (0.42, 0.43, 0.92),   # periwinkle clay (sculpt-render look)
}


def _aim_at(obj, target):
    """Point obj's -Z at target (lights/cameras emit down -Z), keeping +Y up."""
    d = mathutils.Vector(target) - mathutils.Vector(obj.location)
    obj.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()


def configure_engine(scene, samples, transparent, style="marble"):
    scene.cycles.samples = samples
    scene.cycles.max_bounces = 8
    # clay wants a flatter, more saturated matcap look (lower exposure, gentler look);
    # marble can take a touch more contrast/exposure.
    scene.cycles.film_exposure = 0.6 if style == "clay" else 1.0
    scene.cycles.use_denoising = True
    try:
        scene.cycles.denoiser = "OPENIMAGEDENOISE"
    except Exception:
        pass
    scene.render.film_transparent = transparent
    try:
        scene.view_settings.view_transform = "AgX"
        scene.view_settings.look = "AgX - Base Contrast" if style == "clay" \
            else "AgX - Medium High Contrast"
    except Exception:
        scene.view_settings.view_transform = "Filmic"


def keep_largest_component(mesh):
    """Split into loose parts, keep only the most-poly one (drops floor/bg floaters)."""
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = mesh
    mesh.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.separate(type="LOOSE")
    bpy.ops.object.mode_set(mode="OBJECT")
    parts = [o for o in bpy.context.selected_objects if o.type == "MESH"]
    if not parts:
        return mesh
    parts.sort(key=lambda o: len(o.data.polygons), reverse=True)
    keep = parts[0]
    for o in parts[1:]:
        bpy.data.objects.remove(o, do_unlink=True)
    print(f"[studio] kept largest component: {len(keep.data.polygons):,} faces "
          f"(dropped {len(parts) - 1} floaters)", flush=True)
    return keep


def set_material(mesh, style, base, ao_dist=0.35, ao_strength=0.5):
    """Matte sculpt material with a subtle AO multiply so crevices read without the
    blown-out pure-AO look. `style` picks marble (warm white, slight subsurface) or
    clay (periwinkle matcap-style, opaque, a touch rougher)."""
    rough = 0.42 if style == "marble" else 0.55
    spec = 0.35 if style == "marble" else 0.2
    mat = bpy.data.materials.new(style.capitalize())
    mesh.data.materials.clear()
    mesh.data.materials.append(mat)
    mesh.active_material = mat
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Specular IOR Level"].default_value = spec
    bsdf.inputs["IOR"].default_value = 1.45
    bsdf.inputs["Coat Roughness"].default_value = 0.0
    if style == "marble":
        if "Subsurface Weight" in bsdf.inputs:
            bsdf.inputs["Subsurface Weight"].default_value = 0.10
        if "Subsurface Radius" in bsdf.inputs:
            bsdf.inputs["Subsurface Radius"].default_value = (0.25, 0.18, 0.13)
    ao = nt.nodes.new("ShaderNodeAmbientOcclusion")
    ao.inputs["Distance"].default_value = ao_dist
    ao.samples = 16
    rgb = nt.nodes.new("ShaderNodeRGB")
    rgb.outputs[0].default_value = (*base, 1.0)
    mix = nt.nodes.new("ShaderNodeMixRGB")
    mix.blend_type = "MULTIPLY"
    mix.inputs["Fac"].default_value = ao_strength
    nt.links.new(rgb.outputs[0], mix.inputs["Color1"])
    nt.links.new(ao.outputs["Color"], mix.inputs["Color2"])
    nt.links.new(mix.outputs["Color"], bsdf.inputs["Base Color"])


def white_backdrop(cam, centroid, radius):
    """A large pure-white card behind the subject, facing the camera. Real geometry,
    so the subject's silhouette anti-aliases cleanly against it (no compositor
    fringe) and any thin gaps in an open single-view shell show white-through-white
    (invisible) instead of a bright bleed. Emission is pushed past the AgX white
    point; camera-only ray visibility means it adds NO light to the subject."""
    _, _, toward = _camera_basis(cam)             # subject -> camera direction
    loc = mathutils.Vector(centroid) - toward * (radius * 4.0)   # behind subject
    bpy.ops.mesh.primitive_plane_add(location=loc, size=radius * 60.0)
    plane = bpy.context.object
    plane.rotation_euler = toward.to_track_quat("Z", "Y").to_euler()   # face camera
    mat = bpy.data.materials.new("WhiteBackdrop")
    plane.data.materials.append(mat)
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    em.inputs["Strength"].default_value = 20.0     # past AgX white point -> pure white
    nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
    for attr in ("visible_diffuse", "visible_glossy", "visible_transmission",
                 "visible_volume_scatter", "visible_shadow"):
        if hasattr(plane, attr):
            setattr(plane, attr, False)            # camera-only: no light onto subject


def studio_world(gray, transparent):
    """Soft neutral fill; subtle vertical screen-space gradient backdrop when solid."""
    world = bpy.data.scenes[0].world
    world.use_nodes = True
    nt = world.node_tree
    bg = nt.nodes["Background"]
    if transparent:
        bg.inputs["Color"].default_value = (0.12, 0.12, 0.12, 1.0)
        bg.inputs["Strength"].default_value = 0.6
        return
    tex = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    g0, g1 = gray * 1.15, gray * 0.6   # top a touch lighter, bottom darker
    ramp.color_ramp.elements[0].color = (g1, g1, g1 * 1.02, 1.0)
    ramp.color_ramp.elements[1].color = (g0, g0, g0 * 1.02, 1.0)
    nt.links.new(tex.outputs["Window"], sep.inputs["Vector"])
    nt.links.new(sep.outputs["Y"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = 1.0


def _camera_basis(cam):
    """World-space right / up / toward-camera unit vectors of the camera."""
    R = cam.matrix_world.to_3x3()
    right = (R @ mathutils.Vector((1.0, 0.0, 0.0))).normalized()
    up = (R @ mathutils.Vector((0.0, 1.0, 0.0))).normalized()
    toward = (R @ mathutils.Vector((0.0, 0.0, 1.0))).normalized()  # +Z points back at camera
    return right, up, toward


def studio_lights(cam, centroid, radius, scale=1.0):
    """Soft three-point rig (key + fill + rim) built in the camera's frame and
    aimed at the subject — so 'above/left' is relative to the view, for any up-axis.
    `scale` dims the rig (clay wants ~half power so the colour stays saturated)."""
    right, up, toward = _camera_basis(cam)
    c = mathutils.Vector(centroid)
    power = radius * radius * scale

    def place(name, r, u, t, size, watts):
        light = bpy.data.lights.new(name, type="AREA")
        light.size = size * radius
        light.energy = watts * power
        obj = bpy.data.objects.new(name, light)
        obj.location = c + (right * r + up * u + toward * t) * radius
        bpy.context.collection.objects.link(obj)
        _aim_at(obj, centroid)

    place("Key",  -1.1, 1.0,  1.1, size=2.5, watts=320.0)
    place("Fill",  1.3, 0.3,  0.6, size=4.0, watts=110.0)
    place("Rim",   0.3, 1.2, -1.3, size=2.0, watts=260.0)


def studio_ground(cam, centroid, radius):
    """Shadow-catcher beneath the subject (camera-down) for the transparent look."""
    _, up, _ = _camera_basis(cam)
    loc = mathutils.Vector(centroid) - up * (radius * 1.05)
    bpy.ops.mesh.primitive_plane_add(location=loc, size=40.0 * radius)
    ground = bpy.context.object
    ground.rotation_euler = up.to_track_quat("Z", "Y").to_euler()
    try:
        ground.is_shadow_catcher = True
    except Exception:
        ground.cycles.is_shadow_catcher = True


def auto_frame(cam, mesh, centroid, fill, iters=4, pct=0.0):
    """Re-aim the camera at the centroid (preserving its roll) and zoom the lens
    until the subject fills `fill` of the frame — centred, upright, same viewpoint.

    pct>0 frames on the [pct, 100-pct] percentile of the projected silhouette
    instead of strict min/max, so a few stray floaters (kept via isolate=False)
    can't inflate the bbox and shrink/offset the subject."""
    from bpy_extras.object_utils import world_to_camera_view
    scene = bpy.context.scene
    loc = mathutils.Vector(cam.location)
    _, old_up, _ = _camera_basis(cam)
    back = (loc - mathutils.Vector(centroid)).normalized()    # camera +Z (view dir is -Z)
    right = old_up.cross(back).normalized()
    up = back.cross(right).normalized()
    M = mathutils.Matrix((
        (right.x, up.x, back.x, loc.x),
        (right.y, up.y, back.y, loc.y),
        (right.z, up.z, back.z, loc.z),
        (0.0, 0.0, 0.0, 1.0),
    ))
    cam.matrix_world = M
    bpy.context.view_layer.update()
    pts = [mesh.matrix_world @ v.co for v in mesh.data.vertices]

    def proj_bbox():
        xs, ys = [], []
        for co in pts:
            uvz = world_to_camera_view(scene, cam, co)
            if uvz.z <= 0.0:
                continue
            xs.append(uvz.x)
            ys.append(uvz.y)
        return xs, ys

    def bounds(vals):
        # robust [lo, hi] of a projected axis: percentile box if pct>0, else min/max.
        if pct > 0.0:
            return float(np.percentile(vals, pct)), float(np.percentile(vals, 100.0 - pct))
        return min(vals), max(vals)

    # zoom so the projected silhouette fills `fill` of the frame
    for _ in range(iters):
        xs, ys = proj_bbox()
        if not xs:
            break
        x0, x1 = bounds(xs)
        y0, y1 = bounds(ys)
        span = max(x1 - x0, y1 - y0)
        if span <= 1e-6:
            break
        cam.data.lens *= fill / span
        bpy.context.view_layer.update()

    # final 2-D recentre via sensor shift — robust to an asymmetric single-view
    # shell where the 3-D centroid doesn't sit at the visual centre. Measured
    # signs: du/dshift_x = -1, dv/dshift_y = -W/H (horizontal sensor fit).
    aspect = scene.render.resolution_y / scene.render.resolution_x
    for _ in range(3):
        xs, ys = proj_bbox()
        if not xs:
            break
        x0, x1 = bounds(xs)
        y0, y1 = bounds(ys)
        ucx = 0.5 * (x0 + x1)
        ucy = 0.5 * (y0 + y1)
        cam.data.shift_x += (ucx - 0.5)
        cam.data.shift_y += (ucy - 0.5) * aspect
        bpy.context.view_layer.update()


def apply(scene, mesh, verts_n, cam, opts):
    """Full studio setup on an already-positioned camera. Returns the (isolated) mesh.

    `verts_n` is the mesh's vertices in the render frame (after any normalization
    the caller applied). The camera must already carry the training-view pose.
    """
    o = {**DEFAULTS, **(opts or {})}
    style = o.get("style", "marble")
    configure_engine(scene, o["samples"], o["transparent"], style)

    if o["isolate"]:
        mesh = keep_largest_component(mesh)
        bpy.ops.object.shade_smooth()
        verts_n = np.array([v.co[:] for v in mesh.data.vertices], dtype=np.float64)

    base = o.get("mat") or STYLE_COLOR.get(style, STYLE_COLOR["marble"])
    set_material(mesh, style, base)

    if o.get("solidify", 0.0) > 0.0:
        # thickness + filled rim so an open single-view shell's silhouette cracks
        # show a clay wall, not the backdrop bleeding through as bright contours.
        bpy.context.view_layer.objects.active = mesh
        mod = mesh.modifiers.new("Solidify", type="SOLIDIFY")
        mod.thickness = o["solidify"]
        mod.offset = -1.0
        mod.use_rim = True

    centroid = 0.5 * (verts_n.min(0) + verts_n.max(0))
    radius = max(0.05, float(np.linalg.norm(verts_n.max(0) - verts_n.min(0))) * 0.5)

    # frame first so the light rig uses the final camera orientation. no_frame
    # keeps the caller's exact training-view camera (auto_frame re-aims+zooms,
    # which clips a wide single-view ground shell when a tall subject drives the
    # zoom — the lost geometry is in f_theta+mesh, only the reframe drops it).
    if not o.get("no_frame", False):
        auto_frame(cam, mesh, centroid.tolist(), o["fill"], pct=o.get("frame_pct", 0.0))
    studio_lights(cam, centroid.tolist(), radius, scale=0.5 if style == "clay" else 1.0)
    studio_world(o["bg"], o["transparent"])
    if o["transparent"]:
        studio_ground(cam, centroid.tolist(), radius)
    if o.get("white_bg") and not o["transparent"]:
        white_backdrop(cam, centroid.tolist(), radius)
    return mesh
