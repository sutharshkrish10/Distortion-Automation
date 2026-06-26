"""
phase5_measure.py  --  Stage C, steps 4-5: distortion angles + overhang length.

From the fitted reference planes (phase 4):
    * Distortion Angle Leg 1 = interior angle between the overhang-surface plane
      and Leg 1's inner-wall plane (nominally 90 deg).
    * Distortion Angle Leg 2 = same for Leg 2.
    * distortion delta = angle - Nominal angle for that same corner.
    * Overhang Length = inner-wall-to-inner-wall span across the slot, measured
      along the slot-width axis at the overhang level.

Both legs are measured independently so asymmetric distortion is captured.

"""

from __future__ import annotations

import numpy as np

import config as C
from common import LOG, angle_between
from phase4_segment import segment_legs_overhang


def _interior_angle(overhang_plane, wall_plane) -> float:
    a = angle_between(overhang_plane.normal, wall_plane.normal)
    return a if a <= 90.0 else 180.0 - a


def _span(seg) -> float:
    """Inner-wall-to-inner-wall distance along the slot-width axis."""
    p1, p2 = seg.planes["leg1_wall"], seg.planes["leg2_wall"]
    if p1 is None or p2 is None:
        return float("nan")
    return float(abs(np.dot(p2.centroid - p1.centroid, seg.W_dir)))


def _overhang_is_horizontal(oh, V) -> bool:

    if oh is None:
        return False
    a = angle_between(oh.normal, V)
    return min(a, 180.0 - a) <= C.OVERHANG_MAX_TILT_DEG


def _signed_wall_tilt(nom_normal, src_normal, D) -> float:
  
    D = np.asarray(D, float)
    D = D / (np.linalg.norm(D) + 1e-12)

    def proj(n):
        n = np.asarray(n, float)
        n = n - np.dot(n, D) * D
        nn = np.linalg.norm(n)
        return n / nn if nn > 1e-9 else n

    a, b = proj(nom_normal), proj(src_normal)
    s = float(np.dot(np.cross(a, b), D))
    c = float(np.dot(a, b))
    return float(np.degrees(np.arctan2(s, c)))


def _interwall_angle(seg) -> float:
 
    w1, w2 = seg.planes.get("leg1_wall"), seg.planes.get("leg2_wall")
    if w1 is None or w2 is None:
        return float("nan")
    D = np.cross(seg.V_dir, seg.W_dir)
    nn = np.linalg.norm(D)
    if nn < 1e-9:
        return float("nan")
    return _signed_wall_tilt(w1.normal, w2.normal, D / nn)


def _leg_lean(pts, V_dir, W_dir) -> float:
  
    if pts is None or len(pts) < 50:
        return float("nan")
    V = pts @ V_dir
    W = pts @ W_dir
    if float(np.std(V)) < 1e-6:
        return float("nan")
    slope = float(np.polyfit(V, W, 1)[0])
    return float(np.degrees(np.arctan(slope)))


def measure_distortion_and_span(seg) -> dict:
  
    oh = seg.planes.get("overhang")
    w1 = seg.planes.get("leg1_wall")
    w2 = seg.planes.get("leg2_wall")
    res = {"ok": seg.ok and None not in (w1, w2), "note": seg.note}
    oh_ok = _overhang_is_horizontal(oh, seg.V_dir)
    res["angle_leg1"] = _interior_angle(oh, w1) if (oh_ok and w1) else float("nan")
    res["angle_leg2"] = _interior_angle(oh, w2) if (oh_ok and w2) else float("nan")
    res["lean_leg1"] = _leg_lean(seg.masks.get("_inner1_pts"), seg.V_dir, seg.W_dir)
    res["lean_leg2"] = _leg_lean(seg.masks.get("_inner2_pts"), seg.V_dir, seg.W_dir)
    res["interwall"] = _interwall_angle(seg)
    res["overhang_length"] = _span(seg)
    return res


def measure_part(part_id, aligned, nominal_measure=None):
    rows = []
    segs = {}
    order = [s for s in ("nominal", "ct", "zephyr") if s in aligned]
    base = nominal_measure
    for src in order:
        try:
            seg = segment_legs_overhang(aligned[src].cloud)
            segs[src] = seg
            m = measure_distortion_and_span(seg)
        except Exception as e:
            LOG.warning("[%s] measurement failed for %s: %s", part_id, src, e)
            continue

        if src == "nominal":
            base = m
           
        if base:
            d1 = m["lean_leg1"] - base["lean_leg1"]
            d2 = -(m["lean_leg2"] - base["lean_leg2"])
        else:
            d1 = d2 = float("nan")
   
        perleg_reliable = src != "zephyr"
        sc = (base["interwall"] - m["interwall"]) if base else float("nan")  # + = closing
        dL = (m["overhang_length"] - base["overhang_length"]) if base else float("nan")
        
        leg1_angle = C.NOMINAL_LEG_ANGLE_DEG - d1
        leg2_angle = C.NOMINAL_LEG_ANGLE_DEG - d2

        rows.append({
            "part": part_id, "source": src,
            "angle_leg1_deg": round(m["angle_leg1"], 3),
            "angle_leg2_deg": round(m["angle_leg2"], 3),
            "leg1_angle_vs_span_deg": round(leg1_angle, 2),
            "leg2_angle_vs_span_deg": round(leg2_angle, 2),
            "distortion_leg1_deg": round(d1, 3),
            "distortion_leg2_deg": round(d2, 3),
            "perleg_reliable": perleg_reliable,
            "slot_closure_deg": round(sc, 3),
            "overhang_span_mm": round(m["overhang_length"], 4),
            "overhang_span_delta_mm": round(dL, 4),
            "segmentation_ok": m["ok"],
        })
        LOG.info("[%s] %-6s  leg-dist L1=%+.2f L2=%+.2f%s  slot-closure=%+.2f deg  "
                 "(overhang angle L1=%.2f L2=%.2f)  span=%.3f mm", part_id, src,
                 d1, d2, "" if perleg_reliable else " [low-conf]", sc,
                 m["angle_leg1"], m["angle_leg2"], m["overhang_length"])
    return rows, base, segs


if __name__ == "__main__":
    from phase0_match import match_parts
    from phase1_normalize import load_and_normalize
    from phase2_register import register_part
    for pid, srcs in match_parts().items():
        aligned = register_part(pid, load_and_normalize(pid, srcs))
        measure_part(pid, aligned)
