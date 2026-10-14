"""Per-task object template meshes for the Track 1 export.

The released Track 1 data carries no object meshes (the organisers hold the reference meshes),
so CHROME submits a simple, metrically sized, closed template per object class, built here from
trimesh primitives. Dimensions are typical catalogue sizes for each object class, chosen by eye
from the task videos and the object prompts in meta/episodes_metadata.jsonl -- they are NOT
measured from the reference meshes. Every template is built in a z-up object frame (metres),
centred on its bounding-box centre (the point the OWL-ViT box centroid tracks), and is a union
of closed (watertight) parts, so the scorer's winding-number inside test is well defined.
Overlapping parts are fine for PEN (|winding| > 0.5 inside either part) and for CD-O (surface
sampling only). Run ``python -m chrome.object_templates`` (or scripts/build_object_templates.py)
to regenerate chrome/assets/objects/*.obj.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh
from trimesh.creation import annulus, box, cylinder, icosphere, torus

ASSET_DIR = Path(__file__).resolve().parent / "assets" / "objects"


def _T(x=0.0, y=0.0, z=0.0):
    return trimesh.transformations.translation_matrix([x, y, z])


def _Rx(a):
    return trimesh.transformations.rotation_matrix(a, [1, 0, 0])


def _Ry(a):
    return trimesh.transformations.rotation_matrix(a, [0, 1, 0])


def hula_hoop():
    # 'shiny skinny hula hoop': ~80 cm outer diameter, ~1.6 cm tube
    return torus(major_radius=0.40, minor_radius=0.008, major_sections=96, minor_sections=10)


def big_red_bowl():
    # large mixing bowl: 36 cm rim diameter, 15 cm deep, 8 mm wall, closed shell
    R, H, t = 0.18, 0.15, 0.008
    n = 14
    ang = np.linspace(0, np.pi / 2, n)
    outer = np.stack([R * np.sin(ang), H - H * np.cos(ang)], 1)[::-1]          # rim -> bottom centre
    inner = np.stack([(R - t) * np.sin(ang), t + (H - t) * (1 - np.cos(ang))], 1)
    prof = np.vstack([[[R, H]], outer[1:-1], [[1e-4, 0.0]], [[1e-4, t]], inner[1:-1], [[R - t, H]]])
    m = trimesh.creation.revolve(np.vstack([prof, prof[:1]]), sections=48)
    m.apply_translation([0, 0, -H / 2])
    return m


def iron():
    # clothes iron: 26 x 11 cm pointed soleplate, 7 cm body, handle on top (total ~14 cm tall)
    sole = np.array([(-0.13, -0.055), (0.05, -0.055), (0.13, 0.0), (0.05, 0.055), (-0.13, 0.055)])
    pts = np.vstack([np.c_[sole, np.zeros(5)], np.c_[sole * [1.0, 0.8], np.full(5, 0.07)]])
    body = trimesh.convex.convex_hull(pts)  # tapered pointed prism
    handle = box([0.16, 0.03, 0.03], _T(-0.02, 0, 0.115))
    post_f = box([0.03, 0.03, 0.05], _T(0.05, 0, 0.085))
    post_b = box([0.03, 0.03, 0.05], _T(-0.09, 0, 0.085))
    m = trimesh.util.concatenate([body, handle, post_f, post_b])
    m.apply_translation([0, 0, -0.065])
    return m


def white_desk():
    # small white desk: 100 x 50 cm top, 3 cm thick, 73 cm tall, 4 cm square legs
    L, W, H, t, g = 1.00, 0.50, 0.73, 0.03, 0.04
    parts = [box([L, W, t], _T(0, 0, H - t / 2))]
    for sx in (-1, 1):
        for sy in (-1, 1):
            parts.append(box([g, g, H - t], _T(sx * (L / 2 - g), sy * (W / 2 - g), (H - t) / 2)))
    m = trimesh.util.concatenate(parts)
    m.apply_translation([0, 0, -H / 2])
    return m


def black_pan():
    # frying pan: 26 cm diameter, 5 cm deep, 20 cm handle
    R, H, t = 0.13, 0.05, 0.004
    prof = np.array([[R, H], [R - 0.02, 0.0], [1e-4, 0.0], [1e-4, t], [R - 0.02 - t, t], [R - t, H]])
    pan = trimesh.creation.revolve(np.vstack([prof, prof[:1]]), sections=48)
    handle = box([0.20, 0.03, 0.015], _T(R + 0.095, 0, H - 0.01))
    m = trimesh.util.concatenate([pan, handle])
    b = m.bounds
    m.apply_translation(-0.5 * (b[0] + b[1]))
    return m


def foam_grass_block():
    # foam block with grass top: 30 x 20 x 15 cm
    return box([0.30, 0.20, 0.15])


def paint_roller():
    # 23 cm x 6 cm roller cover on a wire frame with a 25 cm handle
    roller = cylinder(radius=0.03, height=0.23, sections=24, transform=_Ry(np.pi / 2))
    arm = box([0.012, 0.012, 0.10], _T(0.12, 0, 0.05))
    bar = box([0.12, 0.012, 0.012], _T(0.065, 0, 0.10))
    handle = cylinder(radius=0.015, height=0.25, sections=16, transform=_T(0.0, 0, 0.225))
    m = trimesh.util.concatenate([roller, arm, bar, handle])
    b = m.bounds
    m.apply_translation(-0.5 * (b[0] + b[1]))
    return m


def pink_foam_roll():
    # foam roller: 15 cm diameter, 45 cm long (lying along x)
    return cylinder(radius=0.075, height=0.45, sections=40, transform=_Ry(np.pi / 2))


def short_wood_stool():
    # short stool: 30 cm round seat, 3 cm thick, 45 cm tall, 4 splayed-ish legs
    H, R, t = 0.45, 0.15, 0.03
    parts = [cylinder(radius=R, height=t, sections=40, transform=_T(0, 0, H - t / 2))]
    for k in range(4):
        a = np.pi / 4 + k * np.pi / 2
        parts.append(cylinder(radius=0.015, height=H - t, sections=12,
                              transform=_T(0.10 * np.cos(a), 0.10 * np.sin(a), (H - t) / 2)))
    m = trimesh.util.concatenate(parts)
    m.apply_translation([0, 0, -H / 2])
    return m


def white_laptop_cart():
    # laptop cart: 55 x 45 cm base on castors, 5 cm column, 60 x 45 cm tray at 90 cm
    parts = [box([0.55, 0.45, 0.05], _T(0, 0, 0.05)),
             box([0.06, 0.06, 0.80], _T(0, 0, 0.475)),
             box([0.60, 0.45, 0.03], _T(0, 0, 0.885))]
    for sx in (-1, 1):
        for sy in (-1, 1):
            parts.append(cylinder(radius=0.025, height=0.05, sections=12, transform=_T(sx * 0.24, sy * 0.19, 0.025)))
    m = trimesh.util.concatenate(parts)
    m.apply_translation([0, 0, -0.45])
    return m


TEMPLATES = {
    "hula_hoop": hula_hoop, "big_red_bowl": big_red_bowl, "iron": iron, "white_desk": white_desk,
    "black_pan": black_pan, "foam_grass_block": foam_grass_block, "paint_roller": paint_roller,
    "pink_foam_roll": pink_foam_roll, "short_wood_stool": short_wood_stool,
    "white_laptop_cart": white_laptop_cart,
}


def build_all(out_dir: Path = ASSET_DIR) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for name, fn in TEMPLATES.items():
        m = fn()
        m.merge_vertices()
        m.export(out_dir / f"{name}.obj")
        parts = m.split(only_watertight=False)
        report[name] = {"vertices": int(len(m.vertices)), "faces": int(len(m.faces)),
                        "extents_m": [round(float(x), 3) for x in m.extents],
                        "parts": int(len(parts)), "all_parts_watertight": bool(all(p.is_watertight for p in parts)),
                        "area_m2": round(float(m.area), 4)}
    (out_dir / "templates.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


def load_template(name: str) -> trimesh.Trimesh:
    return trimesh.load(ASSET_DIR / f"{name}.obj", force="mesh", process=False)


if __name__ == "__main__":
    print(json.dumps(build_all(), indent=1))
