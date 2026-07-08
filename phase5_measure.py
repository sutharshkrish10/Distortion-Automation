"""Stage C, steps 4-5: distortion angles and overhang length.

From the fitted reference planes (phase 4):
    - Distortion angle Leg 1 = interior angle between the overhang-surface plane
      and Leg 1's inner-wall plane (nominally 90 deg).
    - Distortion angle Leg 2 = same for Leg 2.
    - distortion delta = angle - nominal angle for that same corner.
    - Overhang length = inner-wall-to-inner-wall span across the slot, measured
      along the slot-width axis at the overhang level.

Both legs are measured independently so asymmetric distortion is captured.

measure_distortion_and_span(segmentation) returns a dict.
measure_part(part_id, aligned, nominal_measure=None) returns a list of row dicts.
"""

from __future__ import annotations

import numpy as np

import config as C
from common import LOG, angle_between
from phase4_segment import inner_face_positions, segment_legs_overhang


def _interior_angle(overhang_plane, wall_plane) -> float:
    """Interior corner angle (deg) between the overhang surface and an inner wall.

    For an ideal right-angle corner the plane normals are perpendicular, giving
    90 deg. We report the acute-folded angle between normals, which equals the
    interior dihedral for these near-orthogonal faces and stays continuous as
    the wall tilts (distortion)."""
    a = angle_between(overhang_plane.normal, wall_plane.normal)
    return a if a <= 90.0 else 180.0 - a


def _span(seg) -> float:
    """Overhang span = the true slot width along the slot-width axis W.

    Measured between the two inner-wall faces (leg1's slot-facing high-W edge and
    leg2's slot-facing low-W edge), so it returns the actual
    inner-face-to-inner-face clearance, and the nominal parts recover their
    design slot of 2 / 4 / 6 mm. (This previously used the wall-band centroids,
    which sit a little inside each leg and overshoot the true width by ~0.3-1 mm
    by an amount that varies with leg width, so absolute values weren't
    comparable to the design.)"""
    f1, f2 = inner_face_positions(seg)
    if f1 is None or f2 is None:
        return float("nan")
    return float(abs(f2 - f1))


def _overhang_is_horizontal(oh, V) -> bool:
    """True if the fitted overhang plane really is the (near-horizontal) slot
    ceiling, i.e. its normal is close to the build axis V. Photogrammetry that
    never saw the ceiling fits a near-vertical plane here, which we reject."""
    if oh is None:
        return False
    a = angle_between(oh.normal, V)
    return min(a, 180.0 - a) <= C.OVERHANG_MAX_TILT_DEG


def _signed_wall_tilt(nom_normal, src_normal, D) -> float:
    """Signed rotation (deg) from the nominal inner-wall normal to the source
    inner-wall normal, measured about the slot-depth axis D (the leg-bending
    axis). Both normals are projected into the plane perpendicular to D, so only
    the in-slot lean is measured; the sign follows the right-hand rule about +D."""
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
    """Signed angle between the two inner-wall normals, measured about a depth
    axis built right-handed from (V, W) so its orientation is stable. This is
    rigid-invariant (independent of how the cloud is registered), so it gives a
    reproducible 'slot closure' even when registration is weak (e.g. Zephyr).
    Reported relative to nominal by the caller; sign: + = slot closing."""
    w1, w2 = seg.planes.get("leg1_wall"), seg.planes.get("leg2_wall")
    if w1 is None or w2 is None:
        return float("nan")
    D = np.cross(seg.V_dir, seg.W_dir)
    nn = np.linalg.norm(D)
    if nn < 1e-9:
        return float("nan")
    return _signed_wall_tilt(w1.normal, w2.normal, D / nn)


def _leg_lean(pts, V_dir, W_dir) -> float:
    """Lean of an inner wall measured from its own base: fit the wall's
    slot-width position W as a straight line in height V and return atan(dW/dV)
    in degrees (0 = perfectly upright). This is intrinsic to the cloud (no
    registration to nominal), and the line fit averages many points, so the slope
    is far more precise than the per-point scatter (SE ~0.03 deg here). The caller
    subtracts the nominal lean to get the leg's distortion (deflection from
    base)."""
    if pts is None or len(pts) < 50:
        return float("nan")
    V = pts @ V_dir
    W = pts @ W_dir
    if float(np.std(V)) < 1e-6:
        return float("nan")
    slope = float(np.polyfit(V, W, 1)[0])
    return float(np.degrees(np.arctan(slope)))


def measure_distortion_and_span(seg) -> dict:
    """Measure leg deflection + overhang span for one segmented source.

    Per-leg deflection uses the intrinsic leg-lean (each inner wall's lean from
    its base; the caller subtracts the nominal lean), which is registration-free
    and precise. slot_closure (inter-wall angle vs nominal) is also
    registration-free and is the only per-distortion metric trustworthy for
    Zephyr. The legacy overhang-vs-wall angle is returned too, but is NaN when the
    overhang ceiling wasn't captured (see _overhang_is_horizontal)."""
    oh = seg.planes.get("overhang")
    w1 = seg.planes.get("leg1_wall")
    w2 = seg.planes.get("leg2_wall")
    res = {"ok": seg.ok and None not in (w1, w2), "note": seg.note}

    # legacy overhang-vs-wall angle (only where the ceiling was actually captured)
    oh_ok = _overhang_is_horizontal(oh, seg.V_dir)
    res["angle_leg1"] = _interior_angle(oh, w1) if (oh_ok and w1) else float("nan")
    res["angle_leg2"] = _interior_angle(oh, w2) if (oh_ok and w2) else float("nan")

    # per-leg deflection: each inner wall's lean from base (intrinsic, reg-free)
    res["lean_leg1"] = _leg_lean(seg.masks.get("_inner1_pts"), seg.V_dir, seg.W_dir)
    res["lean_leg2"] = _leg_lean(seg.masks.get("_inner2_pts"), seg.V_dir, seg.W_dir)

    # inter-wall slot closure (registration-independent; valid for Zephyr too).
    # Raw per-source value; the caller subtracts the nominal baseline.
    res["interwall"] = _interwall_angle(seg)
    res["overhang_length"] = _span(seg)
    return res


def measure_part(part_id, aligned, nominal_measure=None):
    """Measure each available source; distortion delta is vs the Nominal source.

    Returns (rows, nominal_measure) so the caller can reuse the Nominal baseline.
    """
    rows = []
    segs = {}
    # ensure Nominal is measured first: its inner-wall planes are the reference
    # for the wall-vs-nominal leg distortion of CT/Zephyr.
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
        # per-leg deflection = leg lean minus nominal lean; flip leg2 so + = leaned
        # toward the slot (closing) for both legs.
        if base:
            d1 = m["lean_leg1"] - base["lean_leg1"]
            d2 = -(m["lean_leg2"] - base["lean_leg2"])
        else:
            d1 = d2 = float("nan")
        # Per-leg reliability is by SOURCE, not alignment fitness: only Zephyr
        # photogrammetry physically under-reconstructs an occluded inner wall.
        # CT/nominal see both walls, and per-leg lean is registration-invariant,
        # so CT's lower ICP fitness (genuine part distortion) is not a concern.
        perleg_reliable = src != "zephyr"
        sc = (base["interwall"] - m["interwall"]) if base else float("nan")  # + = closing
        dL = (m["overhang_length"] - base["overhang_length"]) if base else float("nan")
        # leg-vs-span angle expressed as 90 + magnitude of deflection, so it always
        # reads >= 90 (e.g. 91.2, 92.6) regardless of lean direction. The signed
        # in/out direction is preserved in distortion_leg1/2_deg and slot_closure_deg.
        leg1_angle = C.NOMINAL_LEG_ANGLE_DEG + abs(d1)
        leg2_angle = C.NOMINAL_LEG_ANGLE_DEG + abs(d2)

        rows.append({
            "part": part_id, "source": src,
            "angle_leg1_deg": round(m["angle_leg1"], 3),
            "angle_leg2_deg": round(m["angle_leg2"], 3),
            "leg1_angle_vs_span_deg": round(leg1_angle, 2),
            "leg2_angle_vs_span_deg": round(leg2_angle, 2),
            "distortion_leg1_deg": round(d1, 3),
            "distortion_leg2_deg": round(d2, 3),
            "perleg_reliable": perleg_reliable,
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
