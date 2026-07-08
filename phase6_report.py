"""Aggregate CSV reports, segment-coloured clouds, and the annotated geometry
plot per part/source (mirroring Slide1.JPG / Slide2.JPG).

Main entry points:
    write_reports(reg_rows, comp_rows, dist_rows)
    segment_cloud_png(part_id, source, seg)        # annotated cross-section plot
    segment_cloud_3d_png(part_id, source, seg)     # 3D isometric companion
    save_segment_colored_cloud(part_id, source, seg)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
from common import LOG, make_pcd, save_cloud
from phase4_segment import inner_face_positions


# CSV reports
def _safe_to_csv(rows, path):
    """Write a CSV, but don't let a locked file (e.g. open in Excel) abort the
    whole batch: log a warning and carry on so the other reports and artifacts
    still get written. The locked file keeps its previous contents until it is
    closed."""
    if not rows:
        return
    try:
        pd.DataFrame(rows).to_csv(path, index=False)
        LOG.info("wrote %s", path.name)
    except PermissionError:
        LOG.warning("could NOT write %s -- file is locked (open in Excel?); "
                    "left unchanged. Close it and re-run to refresh.", path.name)
    except OSError as e:
        LOG.warning("could NOT write %s -- %s", path.name, e)


def write_reports(reg_rows, comp_rows, dist_rows):
    C.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    _safe_to_csv(reg_rows, C.REGISTRATION_CSV)
    _safe_to_csv(comp_rows, C.COMPARISON_CSV)
    _safe_to_csv(dist_rows, C.DISTORTION_CSV)


def registration_rows(part_id, aligned) -> list[dict]:
    rows = []
    for src, a in aligned.items():
        if src == "nominal":
            continue
        rows.append({
            "part": part_id, "source": src,
            "applied_scale": a.info.get("scale", 1.0),
            "n_points": a.info.get("n_points", len(a.cloud.points)),
            "coarse_fitness": getattr(a.coarse, "fitness", None),
            "coarse_rmse": getattr(a.coarse, "inlier_rmse", None),
            "fine_fitness": getattr(a.fine, "fitness", None),
            "fine_rmse": getattr(a.fine, "inlier_rmse", None),
            "leg_fitness": getattr(a.leg, "fitness", None),
            "leg_rmse": getattr(a.leg, "inlier_rmse", None),
        })
    return rows


# Segment-coloured cloud
def save_segment_colored_cloud(part_id, source, seg):
    pts = seg.points
    colors = np.tile(np.array(C.SEG_COLORS["other"]), (len(pts), 1))
    for label in ("overhang_surface", "leg_1", "leg_2"):
        m = seg.masks.get(label)
        if m is not None and m.dtype == bool:
            colors[m] = C.SEG_COLORS[label]
    pcd = make_pcd(pts)
    import open3d as o3d
    pcd.colors = o3d.utility.Vector3dVector(colors)
    save_cloud(pcd, C.OUTPUT_ROOT / part_id / f"segments_{source}.ply")


# Annotated geometry plot (Slide1/Slide2 style), in the W-V cross-section
def segment_cloud_png(part_id, source, seg, measure: dict | None = None):
    pts = seg.points
    w = pts @ seg.W_dir
    v = pts @ seg.V_dir

    fig, ax = plt.subplots(figsize=(7, 6))
    # background points
    ax.scatter(w, v, s=1, c="#cfcfcf", linewidths=0)
    for label, col in (("leg_1", C.SEG_COLORS["leg_1"]),
                       ("leg_2", C.SEG_COLORS["leg_2"]),
                       ("overhang_surface", C.SEG_COLORS["overhang_surface"])):
        m = seg.masks.get(label)
        if m is not None and getattr(m, "dtype", None) == bool:
            ax.scatter(w[m], v[m], s=2, c=[col], linewidths=0, label=label)

    # Fitted inner-wall and overhang lines. Inner walls are drawn at their slot
    # face (w_override), so the span arrow between them equals the reported span.
    def draw_plane_line(plane, color, length, vertical, w_override=None):
        if plane is None:
            return
        c = plane.centroid
        cw = float(w_override) if w_override is not None else float(np.dot(c, seg.W_dir))
        cv = float(np.dot(c, seg.V_dir))
        if vertical:                         # inner wall ~ constant W
            ax.plot([cw, cw], [cv - length, cv + length], color=color, lw=2)
        else:                                # overhang ~ constant V
            ax.plot([cw - length, cw + length], [cv, cv], color=color, lw=2)
        return cw, cv

    span_v = (v.max() - v.min()) * 0.25
    span_w = (w.max() - w.min()) * 0.25
    f1, f2 = inner_face_positions(seg)       # inner-wall FACE positions along W
    c1 = draw_plane_line(seg.planes.get("leg1_wall"), "#a01010", span_v, True, f1)
    c2 = draw_plane_line(seg.planes.get("leg2_wall"), "#1030a0", span_v, True, f2)
    co = draw_plane_line(seg.planes.get("overhang"), "#108010", span_w, False)

    # mark the two distortion-angle corners + overhang-length span
    if co is not None:
        v_oh = co[1]
        if c1 is not None:
            ax.plot(c1[0], v_oh, "ko", ms=6)
        if c2 is not None:
            ax.plot(c2[0], v_oh, "ko", ms=6)
        if c1 is not None and c2 is not None:
            ax.annotate("", xy=(c2[0], v_oh), xytext=(c1[0], v_oh),
                        arrowprops=dict(arrowstyle="<->", color="k", lw=1.2))
            ax.text((c1[0] + c2[0]) / 2, v_oh + span_v * 0.15,
                    "Overhang Length", ha="center", fontsize=9)

    title = f"{part_id} / {source}"
    if measure:
        la1 = measure.get("leg1_angle_vs_span_deg")
        la2 = measure.get("leg2_angle_vs_span_deg")
        title += (f"   leg-angle L1={la1:.2f} L2={la2:.2f} deg  "
                  f"span={measure['overhang_span_mm']:.2f}mm")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("slot-width axis W (mm)")
    ax.set_ylabel("build axis V (mm)")
    ax.set_aspect("equal", "box")
    ax.legend(loc="upper right", fontsize=8, markerscale=4)
    fig.tight_layout()
    out = C.OUTPUT_ROOT / part_id / f"annotated_{source}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)


# 3D segment screenshot (companion to the 2D annotated_<src>.png)
def segment_cloud_3d_png(part_id, source, seg, measure: dict | None = None):
    """Isometric 3D screenshot of the segmented cloud, saved as
    annotated_3d_<source>.png. It shows the depth dimension the 2D
    annotated_<source>.png flattens. Colours match SEG_COLORS (leg 1 red,
    leg 2 blue, overhang green, other grey). Rendered off-screen with PyVista;
    a missing dependency or render failure is logged and skipped so it does not
    stop the batch."""
    try:
        import pyvista as pv
    except Exception as e:
        LOG.warning("[%s] 3D segment view skipped (pyvista unavailable: %s)",
                    part_id, e)
        return
    try:
        pts = seg.points
        labelled = np.zeros(len(pts), dtype=bool)
        for lbl in ("leg_1", "leg_2", "overhang_surface"):
            m = seg.masks.get(lbl)
            if m is not None and getattr(m, "dtype", None) == bool:
                labelled |= m

        p = pv.Plotter(off_screen=True, window_size=(1000, 900))
        p.set_background("white")
        for lbl in ("leg_1", "leg_2", "overhang_surface", "other"):
            mask = ~labelled if lbl == "other" else seg.masks.get(lbl)
            if mask is None or getattr(mask, "dtype", None) != bool or not mask.any():
                continue
            p.add_mesh(pv.PolyData(pts[mask]), color=tuple(C.SEG_COLORS[lbl]),
                       point_size=3, render_points_as_spheres=False,
                       label=lbl.replace("_", " "))

        title = f"{part_id} / {source}  (3D segments)"
        if measure:
            la1 = measure.get("leg1_angle_vs_span_deg")
            la2 = measure.get("leg2_angle_vs_span_deg")
            title += (f"   leg-angle L1={la1:.2f} L2={la2:.2f} deg   "
                      f"span={measure['overhang_span_mm']:.2f}mm")
        p.add_text(title, font_size=11, color="black")
        p.add_legend(bcolor="white", size=(0.22, 0.18))
        p.add_axes(color="black")
        p.view_isometric()
        out = C.OUTPUT_ROOT / part_id / f"annotated_3d_{source}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        p.screenshot(str(out))
        p.close()
    except Exception as e:
        LOG.warning("[%s] 3D segment view failed for %s: %s", part_id, source, e)
