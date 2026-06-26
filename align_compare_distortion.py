"""
align_compare_distortion.py
===========================
Overhang-specimen distortion pipeline -- main orchestrator.

Brings three representations of each U-channel overhang specimen (Nominal CAD,
CT surface mesh, Zephyr photogrammetry cloud) into a common leg-based frame,
computes surface deviations, and measures the per-leg distortion angles and the
overhang length. See README.md and the reference slides
(Data Set/Leg and Overhang/Slide1.JPG, Slide2.JPG).

The work is split into independent, individually runnable phases:
    phase0_match      discover + match files into parts (by ID regex)
    phase1_normalize  load + unit/scale normalization
    phase2_register   Stage A: coarse RANSAC -> fine ICP -> leg-datum ICP
    phase3_deviation  Stage B: signed surface deviations + heat-maps/histograms
    phase4_segment    Stage C 1-3: axis detection, segmentation, plane fits
    phase5_measure    Stage C 4-5: distortion angles + overhang length
    phase6_report     aggregate CSVs, segment clouds, annotated geometry plots

Each phase is a plain module you can import or run standalone; this file just
loops over discovered parts and chains them, never crashing the batch.

Usage:
    python align_compare_distortion.py                # full batch
    python align_compare_distortion.py --parts 2PR    # subset
    python align_compare_distortion.py --visualize     # open3d windows
    python align_compare_distortion.py --skip-deviation
"""

from __future__ import annotations

import os
# open3d's RANSAC (global registration + plane fitting) is multithreaded and not
# reproducible across threads even when its RNG is seeded; pin OpenMP to a single
# thread so the whole pipeline is bit-for-bit deterministic. numpy/MKL use a
# separate thread pool, so the streaming CT sampler is unaffected. Must be set
# before open3d is imported (via common below).
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import traceback

import config as C
from common import LOG
from phase0_match import match_parts
from phase1_normalize import load_and_normalize
from phase2_register import register_part
from phase3_deviation import deviations_for_part, signed_deviation, _heatmap_cloud
from phase4_segment import segment_legs_overhang
from phase5_measure import measure_distortion_and_span, measure_part
from phase6_report import (registration_rows, save_segment_colored_cloud,
                           segment_cloud_3d_png, segment_cloud_png, write_reports)


def _visualize_alignment(part_id, aligned):
    import open3d as o3d
    geoms = []
    palette = {"nominal": (0.6, 0.6, 0.6), "ct": (0.9, 0.3, 0.3),
               "zephyr": (0.3, 0.5, 0.9)}
    for src, a in aligned.items():
        c = o3d.geometry.PointCloud(a.cloud)
        c.paint_uniform_color(palette.get(src, (0, 0, 0)))
        geoms.append(c)
    o3d.visualization.draw_geometries(geoms, window_name=f"{part_id} aligned (all 3)")


def _visualize_pairs(part_id, aligned):
    """Dedicated interactive windows per pair: Nominal-vs-CT and Nominal-vs-Zephyr.

    For each pair, two windows open in sequence (close one to advance):
      1. overlay   -- Nominal (grey) + as-built (coloured), to eyeball sag/offset
      2. heat-map  -- as-built coloured by signed deviation from Nominal
                      (blue = under/inside, white ~ 0, red = proud/outside),
                      clipped to +/-config.HEATMAP_CLIP mm.
    """
    import open3d as o3d
    nominal = aligned.get("nominal")
    if nominal is None:
        return
    nom_grey = o3d.geometry.PointCloud(nominal.cloud)
    nom_grey.paint_uniform_color((0.72, 0.72, 0.72))
    color = {"ct": (0.90, 0.30, 0.30), "zephyr": (0.25, 0.50, 0.90)}

    for src in ("ct", "zephyr"):
        a = aligned.get(src)
        if a is None:
            continue
        # 1) overlay
        mov = o3d.geometry.PointCloud(a.cloud)
        mov.paint_uniform_color(color[src])
        LOG.info("[%s] viewer: Nominal(grey) vs %s overlay -- close window to continue",
                 part_id, src)
        o3d.visualization.draw_geometries(
            [nom_grey, mov], window_name=f"{part_id}: Nominal (grey) vs {src.upper()}")
        # 2) deviation heat-map
        try:
            dist, _ = signed_deviation(a.cloud, nominal.cloud)
            hm = _heatmap_cloud(a.cloud, dist)
            LOG.info("[%s] viewer: %s vs Nominal deviation heat-map "
                     "(blue=under, red=proud, +/-%.2f mm)", part_id, src, C.HEATMAP_CLIP)
            o3d.visualization.draw_geometries(
                [hm], window_name=f"{part_id}: {src.upper()} vs Nominal "
                                  f"deviation (+/-{C.HEATMAP_CLIP} mm)")
        except Exception as e:
            LOG.warning("[%s] heat-map view failed for %s: %s", part_id, src, e)


def process_part(part_id, srcs, args):
    LOG.info("=" * 70)
    LOG.info("PART %s", part_id)
    for s, p in srcs.items():
        LOG.info("    %-8s : %s", s, p.name if p else "** MISSING **")

    reg_rows, comp_rows, dist_rows = [], [], []

    # Phase 1
    loaded = load_and_normalize(part_id, srcs)
    if "nominal" not in loaded:
        LOG.warning("[%s] no nominal reference -- skipping part", part_id)
        return reg_rows, comp_rows, dist_rows

    # Phase 2  (Stage A)
    aligned = register_part(part_id, loaded)
    reg_rows += registration_rows(part_id, aligned)
    if args.visualize:
        _visualize_alignment(part_id, aligned)        # all three overlaid
        _visualize_pairs(part_id, aligned)            # Nominal-vs-CT, Nominal-vs-Zephyr

    # Phase 3  (Stage B)
    if not args.skip_deviation:
        comp_rows += deviations_for_part(part_id, aligned)

    # Phase 4 + 5  (Stage C) -- measure Nominal first so its per-leg lean is the
    # baseline the other sources' deflections are taken relative to.
    base = None
    for src in (s for s in C.SOURCES if s in aligned):
        try:
            seg = segment_legs_overhang(aligned[src].cloud)
            m = measure_distortion_and_span(seg)
            if src == "nominal":
                base = m
            # per-leg deflection = each leg's intrinsic lean minus the nominal
            # leg's lean (registration-free); flip leg2 so + = leaned toward the
            # slot (closing) for both legs. slot_closure = inter-wall angle vs
            # nominal (registration-independent -> valid for Zephyr too).
            if base:
                d1 = m["lean_leg1"] - base["lean_leg1"]
                d2 = -(m["lean_leg2"] - base["lean_leg2"])
            else:
                d1 = d2 = float("nan")
            # per-leg is occlusion-limited only for Zephyr photogrammetry, which
            # reconstructs the two inner walls unequally; keep its raw values but
            # flag them low-confidence (trust slot_closure). CT/nominal see both
            # walls and per-leg lean is registration-invariant, so CT's lower ICP
            # fitness (from genuine part distortion) does not make it unreliable.
            perleg_reliable = src != "zephyr"
            # sign so + = slot closing, consistent with the per-leg deflection
            sc = base["interwall"] - m["interwall"] if base else float("nan")
            dL = m["overhang_length"] - base["overhang_length"] if base else float("nan")
            # absolute interior leg-vs-span angle
            # (nominal 90; > 90 = leg leaned outward / slot opening)
            leg1_angle = C.NOMINAL_LEG_ANGLE_DEG - d1
            leg2_angle = C.NOMINAL_LEG_ANGLE_DEG - d2
            row = {
                "part": part_id, "source": src,
                "angle_leg1_deg": round(m["angle_leg1"], 3),   # overhang-vs-wall; NaN if ceiling absent
                "angle_leg2_deg": round(m["angle_leg2"], 3),
                "leg1_angle_vs_span_deg": round(leg1_angle, 2),  # absolute interior angle
                "leg2_angle_vs_span_deg": round(leg2_angle, 2),
                "distortion_leg1_deg": round(d1, 3),
                "distortion_leg2_deg": round(d2, 3),
                "perleg_reliable": perleg_reliable,
                "slot_closure_deg": round(sc, 3),
                "overhang_span_mm": round(m["overhang_length"], 4),
                "overhang_span_delta_mm": round(dL, 4),
                "segmentation_ok": m["ok"],
            }
            dist_rows.append(row)
            LOG.info("[%s] %-6s leg-angle L1=%.2f L2=%.2f deg (vs-span)%s | "
                     "leg-dist L1=%+.2f L2=%+.2f  slot-closure=%+.2f deg  span=%.3f mm",
                     part_id, src, leg1_angle, leg2_angle,
                     "" if perleg_reliable else " [low-conf]",
                     d1, d2, sc, m["overhang_length"])

            # Phase 6 per-source artifacts
            save_segment_colored_cloud(part_id, src, seg)
            segment_cloud_png(part_id, src, seg, row)
            segment_cloud_3d_png(part_id, src, seg, row)
            if args.visualize:
                _visualize_segmentation(part_id, src, seg)
        except Exception:
            LOG.warning("[%s] Stage C failed for %s:\n%s", part_id, src,
                        traceback.format_exc())
    return reg_rows, comp_rows, dist_rows


def _visualize_segmentation(part_id, src, seg):
    import numpy as np
    import open3d as o3d
    pts = seg.points
    colors = np.tile(np.array(C.SEG_COLORS["other"]), (len(pts), 1))
    for label in ("overhang_surface", "leg_1", "leg_2"):
        m = seg.masks.get(label)
        if m is not None and getattr(m, "dtype", None) == bool:
            colors[m] = C.SEG_COLORS[label]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd], window_name=f"{part_id}/{src} segments")


def main():
    ap = argparse.ArgumentParser(description="Overhang distortion pipeline")
    ap.add_argument("--parts", nargs="*", default=None,
                    help="subset of part IDs (e.g. 2PR 4P2); default = all")
    ap.add_argument("--visualize", action="store_true",
                    help="open3d windows for alignment + segmentation")
    ap.add_argument("--skip-deviation", action="store_true",
                    help="skip Stage B (faster)")
    args = ap.parse_args()

    # Seed open3d's global RNG up front so the whole pipeline is reproducible:
    # this covers the nominal Poisson-disk sampling (the registration target) as
    # well as the RANSAC global registration. Without it, a fresh nominal cloud
    # each run perturbs every downstream alignment and metric.
    import open3d as o3d
    o3d.utility.random.seed(C.RANSAC_SEED)

    parts = match_parts()
    if args.parts:
        parts = {k: v for k, v in parts.items() if k in set(args.parts)}
    if not parts:
        LOG.error("no parts found under %s", C.DATA_ROOT)
        return

    reg_all, comp_all, dist_all = [], [], []
    for pid, srcs in parts.items():
        try:
            r, c, d = process_part(pid, srcs, args)
            reg_all += r
            comp_all += c
            dist_all += d
        except Exception:
            LOG.warning("[%s] part failed entirely:\n%s", pid, traceback.format_exc())

    write_reports(reg_all, comp_all, dist_all)
    LOG.info("=" * 70)
    LOG.info("DONE. Outputs under %s", C.OUTPUT_ROOT)


if __name__ == "__main__":
    main()
