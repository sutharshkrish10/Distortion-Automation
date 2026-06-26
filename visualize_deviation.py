"""
visualize_deviation.py  --  interactive 3D overlay + signed-deviation heat-map.

Superimposes the Nominal CAD mesh against each as-built test mesh (CT surface
mesh and Zephyr photogrammetry mesh) and colours the test surface by its signed
distance to the Nominal surface, so distorted regions stand out.

Registration
------------
PRE-ALIGNED. The test mesh is placed into the Nominal frame using the alignment
the main pipeline already computed and saved in Output/<part>/transforms.json
(scale + 4x4), so the picture matches the numbers in distortion_report.csv. The
saved transform maps the *scaled* source cloud -> Nominal, where phase 1 scaled
about the sampled cloud's centre; we reproduce that exact scale + centre on the
raw mesh before applying the transform. Pass --icp to instead refine with a
fresh point-to-plane ICP (seeded from the saved transform, or identity).

Signed distance
---------------
Per test-mesh vertex, the signed distance to the Nominal surface
(vtkImplicitPolyDataDistance): + = test surface proud of / outside Nominal
(extra material), - = under / inside. Same sign convention as the phase 3
deviation heat-map.

Output
------
An interactive window per (part, source) pair plus a saved screenshot
Output/<part>/deviation_<source>.png. Use --off-screen for headless batch.

Usage
-----
    python visualize_deviation.py                         # all parts, both pairs
    python visualize_deviation.py --part 4P2 --pair ct
    python visualize_deviation.py --clip 1.0 --threshold 0.5
    python visualize_deviation.py --off-screen            # save PNGs, no window
    python visualize_deviation.py --part 6P1 --full       # don't decimate CT
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pyvista as pv

import config as C
from common import (LOG, is_binary_stl, load_mesh, mesh_to_cloud, stl_cloud)
from phase0_match import match_parts

# Diverging map: blue = under/inside Nominal, white ~ 0, red = proud/outside.
CMAP = "coolwarm"
# Above this triangle count a CT display copy is decimated so interaction stays
# smooth (the CT meshes are 10-12 M triangles). Disable with --full.
DECIMATE_ABOVE = 2_000_000
DECIMATE_TARGET = 1_500_000


# --------------------------------------------------------------------------
# Mesh I/O
# --------------------------------------------------------------------------
def _read_mesh_pv(path) -> pv.PolyData:
    """Read an STL as PyVista PolyData. VTK's reader auto-detects binary/ASCII;
    if it yields nothing (the 'solid'-prefixed binary STLs can trip parsers) we
    fall back to trimesh and rebuild the PolyData."""
    mesh = pv.read(str(path))
    if mesh.n_cells == 0 or mesh.n_points == 0:
        import trimesh
        tm = trimesh.load(str(path), force="mesh")
        faces = np.hstack(
            [np.full((len(tm.faces), 1), 3, dtype=np.int64),
             np.asarray(tm.faces, dtype=np.int64)]).ravel()
        mesh = pv.PolyData(np.asarray(tm.vertices, dtype=float), faces)
    return mesh.triangulate().clean()


def _sampled_centre(path, source) -> np.ndarray:
    """Reproduce phase 1's scaling centre: the mean of the sampled cloud the
    pipeline scaled about (deterministic -- same sampler, same seed)."""
    if is_binary_stl(path):
        cloud, _ = stl_cloud(path, C.POISSON_SAMPLES[source])
    else:
        cloud = mesh_to_cloud(load_mesh(path), C.POISSON_SAMPLES[source])
    return np.asarray(cloud.get_center(), dtype=float)


# --------------------------------------------------------------------------
# Alignment
# --------------------------------------------------------------------------
def _load_transform(part_id, source):
    """Return (scale, T 4x4) from Output/<part>/transforms.json, or None."""
    tf_path = C.OUTPUT_ROOT / part_id / "transforms.json"
    if not tf_path.exists():
        return None
    data = json.loads(tf_path.read_text())
    if source not in data:
        return None
    rec = data[source]
    return float(rec["scale"]), np.asarray(rec["transform"], dtype=float)


def _apply_alignment(mesh: pv.PolyData, path, source, scale, T) -> pv.PolyData:
    """Scale about the pipeline's sampled-cloud centre, then apply the 4x4."""
    pts = np.asarray(mesh.points, dtype=float)
    if abs(scale - 1.0) > 1e-9:
        centre = _sampled_centre(path, source)
        pts = centre + scale * (pts - centre)
    pts = pts @ T[:3, :3].T + T[:3, 3]
    out = mesh.copy()
    out.points = pts
    return out


def _fit_bbox(test: pv.PolyData, nominal: pv.PolyData) -> pv.PolyData:
    """Display-only anisotropic fit: map the test's axis-aligned bounding box
    onto the nominal's, per axis. This makes the base (W x D footprint) match the
    nominal and AMPLIFIES the short build height to fill it -- so a partial Zephyr
    scan (which is proportionally short in height) overlays the nominal instead of
    looking smaller. NB this stretches the captured surface to span material that
    was never scanned, so the deviation map in the stretched axis is no longer a
    true distance -- it is a visualization aid, not a measurement."""
    nb = np.asarray(nominal.bounds).reshape(3, 2)     # [[xmin,xmax],[ymin,ymax],[zmin,zmax]]
    tb = np.asarray(test.bounds).reshape(3, 2)
    pts = np.asarray(test.points, dtype=float).copy()
    for a in range(3):
        span = tb[a, 1] - tb[a, 0]
        if span > 1e-9:
            pts[:, a] = nb[a, 0] + (pts[:, a] - tb[a, 0]) / span * (nb[a, 1] - nb[a, 0])
    out = test.copy()
    out.points = pts
    return out


def _icp_refine(test: pv.PolyData, nominal: pv.PolyData, init=np.eye(4)):
    """Fresh point-to-plane ICP on the meshes' points (open3d), seeded from
    `init`. Returns the 4x4 mapping test -> nominal."""
    import open3d as o3d
    from common import estimate_normals
    src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(test.points)))
    tgt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(nominal.points)))
    estimate_normals(src, C.VOXEL_SIZE)
    estimate_normals(tgt, C.VOXEL_SIZE)
    icp = o3d.pipelines.registration.registration_icp(
        src, tgt, C.ICP_DIST_FACTOR * C.VOXEL_SIZE, init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=C.ICP_MAX_ITER))
    LOG.info("    ICP refine: fitness=%.3f rmse=%.3f", icp.fitness, icp.inlier_rmse)
    return np.asarray(icp.transformation, dtype=float)


# --------------------------------------------------------------------------
# Deviation + rendering
# --------------------------------------------------------------------------
def _signed_deviation(test: pv.PolyData, nominal: pv.PolyData) -> pv.PolyData:
    """Per test-vertex signed distance to the Nominal surface (+ = proud)."""
    out = test.compute_implicit_distance(nominal)
    out["deviation_mm"] = np.asarray(out["implicit_distance"], dtype=float)
    out["abs_dev_mm"] = np.abs(out["deviation_mm"])
    return out


def _stats(dev: np.ndarray, thresh: float) -> str:
    a = np.abs(dev)
    return (f"deviation (mm):  mean {dev.mean():+.3f}   rms {np.sqrt((dev**2).mean()):.3f}\n"
            f"p95 |dev| {np.percentile(a, 95):.3f}   max |dev| {a.max():.3f}\n"
            f"|dev| > {thresh:.2f} mm:  {100.0 * (a > thresh).mean():.1f}% of surface")


def visualize_pair(part_id, source, srcs, args):
    nom_path, test_path = srcs.get("nominal"), srcs.get(source)
    if nom_path is None or test_path is None:
        LOG.warning("[%s] %s: missing nominal or %s file -- skipped",
                    part_id, source, source)
        return

    LOG.info("[%s] %s: reading meshes", part_id, source)
    nominal = _read_mesh_pv(nom_path)
    test = _read_mesh_pv(test_path)

    # --- align test -> nominal -----------------------------------------
    # Prefer the pipeline's saved transform (consistent with the report); --icp
    # refines on top of it. If no transform is saved, ICP runs from identity as
    # a best-effort fallback (may fail for scale-mismatched Zephyr).
    saved = _load_transform(part_id, source)
    if saved is not None:
        scale, T = saved
        test = _apply_alignment(test, test_path, source, scale, T)
        LOG.info("[%s] %s: pre-aligned (scale x%.4f, saved transform)",
                 part_id, source, scale)
    elif not args.icp:
        LOG.warning("[%s] %s: no saved transform -- forcing ICP fallback",
                    part_id, source)
    if args.icp or saved is None:
        T_icp = _icp_refine(test, nominal)        # test already roughly aligned -> init = I
        test = test.transform(T_icp, inplace=False)

    fitted = False
    if args.fit_bbox:
        test = _fit_bbox(test, nominal)
        fitted = True

    # --- decimate heavy CT display copy --------------------------------
    # Use decimate_pro (vtkDecimatePro: incremental edge-collapse, low memory),
    # NOT decimate (vtkQuadricDecimation), which builds per-vertex quadric
    # matrices over the whole 12M-tri CT mesh and OOMs on a tight machine.
    if not args.full and test.n_cells > DECIMATE_ABOVE:
        red = 1.0 - DECIMATE_TARGET / test.n_cells
        LOG.info("[%s] %s: decimating %d -> ~%d tris for display "
                 "(--full to disable)", part_id, source, test.n_cells, DECIMATE_TARGET)
        try:
            test = test.decimate_pro(red, preserve_topology=False)
        except Exception as e:
            # last resort: render the vertices as a point cloud (signed distance is
            # per-point, so the heat-map still works; just no triangle surface).
            LOG.warning("[%s] %s: mesh decimation failed (%s); falling back to a "
                        "subsampled point cloud", part_id, source, e)
            pts = np.asarray(test.points)
            k = min(len(pts), DECIMATE_TARGET)
            idx = np.random.default_rng(0).choice(len(pts), k, replace=False)
            test = pv.PolyData(pts[idx])

    # --- signed deviation ----------------------------------------------
    test = _signed_deviation(test, nominal)
    dev = test["deviation_mm"]
    clip = args.clip
    thresh = args.threshold
    LOG.info("[%s] %s: %s", part_id, source, _stats(dev, thresh).replace("\n", "  "))

    # --- render ---------------------------------------------------------
    # bbox-fitted views go to a separate file so they never clobber the honest
    # (uniform-scale) deviation screenshot.
    suffix = "_fitbbox" if fitted else ""
    out_png = C.OUTPUT_ROOT / part_id / f"deviation_{source}{suffix}.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)

    p = pv.Plotter(off_screen=args.off_screen, window_size=(1280, 960))
    p.set_background("white")
    title = f"{part_id}  |  Nominal vs {source.upper()}"
    if fitted:
        title += "   [bbox-fitted: base matched, height amplified -- display only]"
    p.add_text(title, font_size=11, color="black")
    p.add_text(_stats(dev, thresh), position="lower_left", font_size=9, color="black")

    # nominal reference: translucent grey
    p.add_mesh(nominal, color="lightgrey", opacity=0.25, name="nominal",
               label="Nominal (reference)")

    # test surface coloured by signed deviation
    p.add_mesh(test, scalars="deviation_mm", cmap=CMAP, clim=(-clip, clip),
               name="test", label=f"{source.upper()} (deviation)",
               scalar_bar_args=dict(title="Signed deviation (mm)", n_labels=5,
                                    fmt="%+.2f", color="black",
                                    title_font_size=16, label_font_size=12))

    # distinct highlight for |dev| > threshold, updated by a slider
    def _show_over(value):
        over = test.threshold(value=value, scalars="abs_dev_mm")
        p.remove_actor("over", reset_camera=False)
        if over.n_points:
            p.add_mesh(over, color="#111111", style="wireframe", line_width=2,
                       name="over", reset_camera=False,
                       label=f"|dev| > {value:.2f} mm")
        p.add_text(f"threshold = {value:.2f} mm", position="upper_right",
                   font_size=10, color="black", name="thr_txt")

    _show_over(thresh)
    if not args.off_screen:
        p.add_slider_widget(_show_over, rng=(0.0, max(clip, float(np.abs(dev).max()))),
                            value=thresh, title="Threshold (mm)",
                            pointa=(0.62, 0.92), pointb=(0.92, 0.92),
                            style="modern")
    p.add_legend(bcolor="white")
    p.add_axes(color="black")

    if args.off_screen:
        p.screenshot(str(out_png))
        p.close()
    else:
        p.show(screenshot=str(out_png), auto_close=True)
    LOG.info("[%s] %s: screenshot -> %s", part_id, source, out_png)


def main():
    ap = argparse.ArgumentParser(description="Nominal-vs-as-built deviation viewer")
    ap.add_argument("--part", nargs="*", default=None,
                    help="part IDs (2PR 4P2 6P1); default all")
    ap.add_argument("--pair", choices=("ct", "zephyr", "both"), default="both",
                    help="which comparison(s) to show")
    ap.add_argument("--clip", type=float, default=C.HEATMAP_CLIP,
                    help="+/- range (mm) for the deviation colour map")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="highlight |deviation| above this (mm)")
    ap.add_argument("--icp", action="store_true",
                    help="refine alignment with a fresh ICP instead of trusting the saved transform")
    ap.add_argument("--full", action="store_true",
                    help="do not decimate the CT mesh for display (heavy)")
    ap.add_argument("--fit-bbox", action="store_true",
                    help="anisotropically fit the test bbox to the nominal's (base "
                         "matched, short height amplified); display only, not a true distance")
    ap.add_argument("--off-screen", action="store_true",
                    help="headless: save screenshots without opening a window")
    args = ap.parse_args()

    parts = match_parts()
    if args.part:
        parts = {k: v for k, v in parts.items() if k in set(args.part)}
    if not parts:
        LOG.error("no parts found under %s", C.DATA_ROOT)
        return

    sources = ("ct", "zephyr") if args.pair == "both" else (args.pair,)
    for pid, srcs in parts.items():
        for src in sources:
            try:
                visualize_pair(pid, src, srcs, args)
            except Exception:
                import traceback
                LOG.warning("[%s] %s: visualization failed:\n%s",
                            pid, src, traceback.format_exc())


if __name__ == "__main__":
    main()
