"""
phase6_report.py  --  aggregate CSV reports, segment-coloured clouds, and the
annotated geometry plot per part/source (mirroring Slide1.JPG / Slide2.JPG).

"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
from common import LOG, make_pcd, save_cloud

# CSV reports

def write_reports(reg_rows, comp_rows, dist_rows):
    C.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if reg_rows:
        pd.DataFrame(reg_rows).to_csv(C.REGISTRATION_CSV, index=False)
        LOG.info("wrote %s", C.REGISTRATION_CSV.name)
    if comp_rows:
        pd.DataFrame(comp_rows).to_csv(C.COMPARISON_CSV, index=False)
        LOG.info("wrote %s", C.COMPARISON_CSV.name)
    if dist_rows:
        pd.DataFrame(dist_rows).to_csv(C.DISTORTION_CSV, index=False)
        LOG.info("wrote %s", C.DISTORTION_CSV.name)


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

# segment-coloured cloud
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



# annotated geometry plot (Slide1/Slide2 style), in the W-V cross-section

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

    # fitted inner-wall + overhang lines through their centroids
    def draw_plane_line(plane, color, length, vertical):
        if plane is None:
            return
        c = plane.centroid
        cw, cv = np.dot(c, seg.W_dir), np.dot(c, seg.V_dir)
        if vertical:                         # inner wall ~ constant W
            ax.plot([cw, cw], [cv - length, cv + length], color=color, lw=2)
        else:                                # overhang ~ constant V
            ax.plot([cw - length, cw + length], [cv, cv], color=color, lw=2)
        return cw, cv

    span_v = (v.max() - v.min()) * 0.25
    span_w = (w.max() - w.min()) * 0.25
    c1 = draw_plane_line(seg.planes.get("leg1_wall"), "#a01010", span_v, True)
    c2 = draw_plane_line(seg.planes.get("leg2_wall"), "#1030a0", span_v, True)
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
        d1, d2 = measure.get("distortion_leg1_deg"), measure.get("distortion_leg2_deg")
        sc = measure.get("slot_closure_deg")
        title += (f"   leg-dist L1={d1:+.2f} L2={d2:+.2f}  "
                  f"slot-closure={sc:+.2f} deg   "
                  f"span={measure['overhang_span_mm']:.2f}mm")
        a1, a2 = measure.get("angle_leg1_deg"), measure.get("angle_leg2_deg")
        if a1 == a1:                      
            title += f"\noverhang angle L1={a1:.1f} L2={a2:.1f} deg"
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
    """Clean isometric 3D screenshot of the segmented cloud, saved as
    annotated_3d_<source>.png -- shows the depth dimension the 2D
    annotated_<source>.png flattens. Colours match SEG_COLORS (leg 1 red,
    leg 2 blue, overhang green, other grey). Best-effort via PyVista off-screen:
    a missing dependency or render failure is logged and skipped, never crashing
    the batch."""
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
            sc = measure.get("slot_closure_deg")
            title += (f"   slot-closure={sc:+.2f} deg   "
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
