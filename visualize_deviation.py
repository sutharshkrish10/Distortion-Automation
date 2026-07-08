"""Interactive 3D viewers for nominal vs as-built.

There are two modes (--mode), both reading what the main pipeline already
aligned and saved under Output/<part>/, so the pictures match
distortion_report.csv:

  overlay (default): a 3-way superimposition of Open3D point clouds. Nominal
      (grey), CT (orange) and Zephyr (blue) at once, in the shared nominal frame.
      Toggle layers to compare Nominal vs CT or Nominal vs Zephyr. Reads
      Output/<part>/{nominal_sampled,aligned_ct,aligned_zephyr}.ply and the
      baked-colour deviation clouds for the heat-map sub-mode. This is fast.

  deviation: a signed-deviation heat-map of PyVista meshes. One pair per window:
      the Nominal CAD mesh (translucent grey) plus the test mesh coloured by
      signed distance to nominal (+ = proud/outside, - = inside), a live
      threshold slider, and a saved screenshot at
      Output/<part>/deviation_<source>.png.

Both modes loop over every matched part by default (one window per part; close
it to get the next) and honour --part / --pair to narrow the selection.

In deviation mode the test is pre-aligned from Output/<part>/transforms.json
(scale plus 4x4). Phase 1 scaled the source about its sampled cloud's centre,
which we reproduce on the raw mesh before applying the transform. --icp refines
on top with a fresh point-to-plane ICP.

Usage:
    python visualize_deviation.py                         # overlay (points), all parts
    python visualize_deviation.py --solid                 # overlay as solid surfaces
    python visualize_deviation.py --part 4P2              # overlay, just 4P2
    python visualize_deviation.py --pair ct               # overlay, hide Zephyr
    python visualize_deviation.py --mode deviation        # heat-map, all pairs
    python visualize_deviation.py --mode deviation --part 6P1 --pair ct --full
    python visualize_deviation.py --off-screen            # save PNGs, no window
"""

from __future__ import annotations

import argparse
import json

import numpy as np

import config as C
from common import LOG

# Mode 1: 3-way superimposed overlay (Open3D point clouds).
# Solid per-source colours: nominal grey, CT orange, Zephyr blue.
COL_NOMINAL = (0.60, 0.60, 0.60)
COL_CT      = (1.00, 0.50, 0.10)
COL_ZEPHYR  = (0.15, 0.40, 0.90)

OVERLAY_HELP = """
=== Overlay viewer (points) ===================================
  nominal = grey | CT = orange | Zephyr = blue
  1/2/3  toggle nominal / CT / Zephyr
  D      solid colours  <->  deviation heat-map (blue under, red proud)
  [ / ]  point size      R  reset view      H  this help
  mouse: drag rotate, scroll zoom, ctrl-drag pan
===============================================================""".rstrip()

SOLID_HELP = """
=== Overlay viewer (solid surfaces) ===========================
  nominal = grey (translucent) | CT = orange | Zephyr = blue
  1/2/3  toggle nominal / CT / Zephyr
  d      solid colours  <->  deviation heat-map (blue under, red proud)
  r      reset view
  mouse: drag rotate, scroll zoom
===============================================================""".rstrip()


def _o3d_load(path, colour):
    import open3d as o3d
    if not path.exists():
        return None
    pc = o3d.io.read_point_cloud(str(path))
    if pc.is_empty():
        return None
    if colour is not None:
        pc.paint_uniform_color(colour)
    return pc


def overlay_part(part_id, sources, args):
    """Open one interactive window superimposing nominal + the requested test
    clouds (CT/Zephyr) for `part_id`, all already in the nominal frame."""
    import open3d as o3d

    d = C.OUTPUT_ROOT / part_id
    if not d.is_dir():
        LOG.warning("[%s] overlay: no Output folder (%s) -- run the pipeline first", part_id, d)
        return

    nominal = _o3d_load(d / "nominal_sampled.ply", COL_NOMINAL)
    if nominal is None:
        LOG.warning("[%s] overlay: nominal_sampled.ply missing/empty -- skipped", part_id)
        return

    want_ct = "ct" in sources
    want_ze = "zephyr" in sources
    # solid + deviation(heat-map) representations of each test surface
    ct_solid = _o3d_load(d / "aligned_ct.ply", COL_CT) if want_ct else None
    ze_solid = _o3d_load(d / "aligned_zephyr.ply", COL_ZEPHYR) if want_ze else None
    ct_dev = _o3d_load(d / "deviation_ct_vs_nominal.ply", None) if want_ct else None
    ze_dev = _o3d_load(d / "deviation_zephyr_vs_nominal.ply", None) if want_ze else None

    state = {"nominal": True, "ct": ct_solid is not None,
             "zephyr": ze_solid is not None, "mode": "solid"}

    def active_geom(layer):
        if layer == "nominal":
            return nominal                       # always the grey reference
        if layer == "ct":
            return ct_dev if state["mode"] == "dev" else ct_solid
        return ze_dev if state["mode"] == "dev" else ze_solid

    added = {"nominal": None, "ct": None, "zephyr": None}

    def sync(vis):
        for layer in ("nominal", "ct", "zephyr"):
            want = active_geom(layer) if state[layer] else None
            cur = added[layer]
            if cur is want:
                continue
            if cur is not None:
                vis.remove_geometry(cur, reset_bounding_box=False)
            if want is not None:
                vis.add_geometry(want, reset_bounding_box=False)
            added[layer] = want
        vis.update_renderer()

    def toggle(layer):
        def cb(vis):
            state[layer] = not state[layer]
            print(f"  {layer:8s} -> {'on' if state[layer] else 'off'}")
            sync(vis)
            return False
        return cb

    def toggle_mode(vis):
        state["mode"] = "dev" if state["mode"] == "solid" else "solid"
        print(f"  mode -> {'deviation heat-map' if state['mode'] == 'dev' else 'solid colours'}")
        sync(vis)
        return False

    def point_size(delta):
        def cb(vis):
            ro = vis.get_render_option()
            ro.point_size = float(np.clip(ro.point_size + delta, 1.0, 12.0))
            print(f"  point size -> {ro.point_size:.0f}")
            return False
        return cb

    def reset_view(vis):
        vis.reset_view_point(True)
        return False

    def show_help(vis):
        print(OVERLAY_HELP)
        return False

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=f"Overlay  {part_id}  -  nominal(grey)/CT(orange)/Zephyr(blue)",
                      width=1280, height=860, visible=not args.off_screen)
    ro = vis.get_render_option()
    ro.point_size = 2.5
    ro.background_color = np.array([0.04, 0.04, 0.06])

    for layer in ("nominal", "ct", "zephyr"):
        g = active_geom(layer)
        if state[layer] and g is not None:
            vis.add_geometry(g)               # first add sets the camera
            added[layer] = g

    vis.register_key_callback(ord("1"), toggle("nominal"))
    vis.register_key_callback(ord("2"), toggle("ct"))
    vis.register_key_callback(ord("3"), toggle("zephyr"))
    vis.register_key_callback(ord("D"), toggle_mode)
    vis.register_key_callback(ord("["), point_size(-1.0))
    vis.register_key_callback(ord("]"), point_size(+1.0))
    vis.register_key_callback(ord("R"), reset_view)
    vis.register_key_callback(ord("H"), show_help)

    LOG.info("[%s] overlay: nominal%s%s", part_id,
             "" if ct_solid is None else " + CT",
             "" if ze_solid is None else " + Zephyr")

    if args.off_screen:
        # headless: render once and save a screenshot instead of blocking.
        vis.poll_events()
        vis.update_renderer()
        out_png = d / "overlay.png"
        vis.capture_screen_image(str(out_png), do_render=True)
        vis.destroy_window()
        LOG.info("[%s] overlay: screenshot -> %s", part_id, out_png)
    else:
        print(OVERLAY_HELP)
        vis.run()
        vis.destroy_window()


# Overlay, solid variant (--solid): the same 3-way superimposition but rendering
# the actual surface meshes instead of sampled points. There are no saved aligned
# meshes (CT/Zephyr are streamed as points), so we load the source meshes and
# apply the pipeline's saved transform, exactly like the deviation mode, then
# show them translucent so all layers stay visible. This is heavier than the
# point view (the CT mesh is decimated for display).
SOLID_COLOR = {"ct": "orange", "zephyr": "royalblue"}
SOLID_OPACITY = {"nominal": 0.25, "ct": 0.55, "zephyr": 0.60}


def solid_overlay_part(part_id, srcs, sources, args):
    import pyvista as pv

    nom_path = srcs.get("nominal")
    if nom_path is None:
        LOG.warning("[%s] solid overlay: no nominal mesh -- skipped", part_id)
        return
    LOG.info("[%s] solid overlay: reading nominal mesh", part_id)
    nominal = _read_mesh_pv(nom_path)

    # load + align each requested test mesh (and precompute its signed deviation)
    tests = {}
    for src in sources:
        tp = srcs.get(src)
        if tp is None:
            continue
        saved = _load_transform(part_id, src)
        if saved is None:
            LOG.warning("[%s] %s: no saved transform -- skipped in solid overlay",
                        part_id, src)
            continue
        LOG.info("[%s] %s: reading + aligning mesh", part_id, src)
        m = _read_mesh_pv(tp)
        scale, T = saved
        m = _apply_alignment(m, tp, src, scale, T)
        if not args.full and m.n_cells > DECIMATE_ABOVE:
            red = 1.0 - DECIMATE_TARGET / m.n_cells
            LOG.info("[%s] %s: decimating %d -> ~%d tris (--full to disable)",
                     part_id, src, m.n_cells, DECIMATE_TARGET)
            try:
                m = m.decimate_pro(red, preserve_topology=False)
            except Exception as e:
                LOG.warning("[%s] %s: decimation failed (%s); using full mesh",
                            part_id, src, e)
        tests[src] = _signed_deviation(m, nominal)

    state = {"nominal": True, "ct": "ct" in tests, "zephyr": "zephyr" in tests,
             "mode": "solid"}
    clip = args.clip

    p = pv.Plotter(off_screen=args.off_screen, window_size=(1280, 960))
    p.set_background("white")
    p.add_text(f"{part_id}  |  Nominal(grey) / CT(orange) / Zephyr(blue)  -  solid",
               font_size=11, color="black")
    p.add_text("1/2/3 toggle layers   d solid<->deviation   r reset",
               position="lower_left", font_size=9, color="black")
    p.add_axes(color="black")

    actors = {}
    bar_owner = {"src": None}     # only one scalar bar (shared range) to avoid clutter

    def add_layer(src):
        if src == "nominal":
            actors["nominal"] = p.add_mesh(nominal, color="lightgrey",
                                           opacity=SOLID_OPACITY["nominal"], name="nominal")
            return
        m = tests[src]
        if state["mode"] == "dev":
            show_bar = bar_owner["src"] in (None, src)
            if show_bar:
                bar_owner["src"] = src
            kw = dict(scalars="deviation_mm", cmap=CMAP, clim=(-clip, clip),
                      name=src, show_scalar_bar=show_bar)
            if show_bar:
                kw["scalar_bar_args"] = dict(title="Signed deviation (mm)", color="black",
                                             n_labels=5, fmt="%+.2f")
            actors[src] = p.add_mesh(m, **kw)
        else:
            actors[src] = p.add_mesh(m, color=SOLID_COLOR[src],
                                     opacity=SOLID_OPACITY[src], name=src)

    def rebuild():
        for s in list(actors):
            p.remove_actor(s, reset_camera=False)
        actors.clear()
        bar_owner["src"] = None
        for s in ("nominal", "ct", "zephyr"):
            if state[s] and (s == "nominal" or s in tests):
                add_layer(s)
        p.render()

    def toggle(src):
        def cb():
            state[src] = not state[src]
            rebuild()
        return cb

    def toggle_mode():
        state["mode"] = "dev" if state["mode"] == "solid" else "solid"
        rebuild()

    for s in ("nominal", "ct", "zephyr"):
        if state[s] and (s == "nominal" or s in tests):
            add_layer(s)

    LOG.info("[%s] solid overlay: nominal%s%s", part_id,
             " + CT" if "ct" in tests else "", " + Zephyr" if "zephyr" in tests else "")

    if args.off_screen:
        out_png = C.OUTPUT_ROOT / part_id / "overlay_solid.png"
        p.screenshot(str(out_png))
        p.close()
        LOG.info("[%s] solid overlay: screenshot -> %s", part_id, out_png)
    else:
        p.add_key_event("1", toggle("nominal"))
        p.add_key_event("2", toggle("ct"))
        p.add_key_event("3", toggle("zephyr"))
        p.add_key_event("d", toggle_mode)
        print(SOLID_HELP)
        p.show()


# Mode 2: signed-deviation heat-map (PyVista meshes).
# Diverging map: blue = under/inside Nominal, white ~ 0, red = proud/outside.
CMAP = "coolwarm"
# Above this triangle count a CT display copy is decimated so interaction stays
# smooth (the CT meshes are 10-12 M triangles). Disable with --full.
DECIMATE_ABOVE = 2_000_000
DECIMATE_TARGET = 1_500_000


def _read_mesh_pv(path):
    """Read an STL as PyVista PolyData. VTK's reader auto-detects binary/ASCII;
    if it yields nothing (the 'solid'-prefixed binary STLs can trip parsers) we
    fall back to trimesh and rebuild the PolyData."""
    import pyvista as pv
    mesh = pv.read(str(path))
    if mesh.n_cells == 0 or mesh.n_points == 0:
        import trimesh
        tm = trimesh.load(str(path), force="mesh")
        faces = np.hstack(
            [np.full((len(tm.faces), 1), 3, dtype=np.int64),
             np.asarray(tm.faces, dtype=np.int64)]).ravel()
        mesh = pv.PolyData(np.asarray(tm.vertices, dtype=float), faces)
    return mesh.triangulate().clean()


def _sampled_centre(path, source):
    """Reproduce phase 1's scaling centre: the mean of the sampled cloud the
    pipeline scaled about (deterministic, since same sampler and same seed)."""
    from common import is_binary_stl, load_mesh, mesh_to_cloud, stl_cloud
    if is_binary_stl(path):
        cloud, _ = stl_cloud(path, C.POISSON_SAMPLES[source])
    else:
        cloud = mesh_to_cloud(load_mesh(path), C.POISSON_SAMPLES[source])
    return np.asarray(cloud.get_center(), dtype=float)


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


def _apply_alignment(mesh, path, source, scale, T):
    """Scale about the pipeline's sampled-cloud centre, then apply the 4x4."""
    pts = np.asarray(mesh.points, dtype=float)
    if abs(scale - 1.0) > 1e-9:
        centre = _sampled_centre(path, source)
        pts = centre + scale * (pts - centre)
    pts = pts @ T[:3, :3].T + T[:3, 3]
    out = mesh.copy()
    out.points = pts
    return out


def _fit_bbox(test, nominal):
    """Display-only anisotropic fit: map the test's axis-aligned bounding box
    onto the nominal's, per axis. This makes the base (W x D footprint) match the
    nominal and amplifies the short build height to fill it, so a partial Zephyr
    scan (which is proportionally short in height) overlays the nominal instead of
    looking smaller. Note this stretches the captured surface to span material
    that was never scanned, so the deviation map in the stretched axis is no
    longer a true distance; it is a visualization aid, not a measurement."""
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


def _icp_refine(test, nominal, init=None):
    """Fresh point-to-plane ICP on the meshes' points (open3d), seeded from
    `init`. Returns the 4x4 mapping test into nominal."""
    import open3d as o3d
    from common import estimate_normals
    if init is None:
        init = np.eye(4)
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


def _signed_deviation(test, nominal):
    """Per test-vertex signed distance to the Nominal surface (+ = proud)."""
    out = test.compute_implicit_distance(nominal)
    out["deviation_mm"] = np.asarray(out["implicit_distance"], dtype=float)
    out["abs_dev_mm"] = np.abs(out["deviation_mm"])
    return out


def _stats(dev, thresh):
    a = np.abs(dev)
    return (f"deviation (mm):  mean {dev.mean():+.3f}   rms {np.sqrt((dev**2).mean()):.3f}\n"
            f"p95 |dev| {np.percentile(a, 95):.3f}   max |dev| {a.max():.3f}\n"
            f"|dev| > {thresh:.2f} mm:  {100.0 * (a > thresh).mean():.1f}% of surface")


def deviation_pair(part_id, source, srcs, args):
    import pyvista as pv
    nom_path, test_path = srcs.get("nominal"), srcs.get(source)
    if nom_path is None or test_path is None:
        LOG.warning("[%s] %s: missing nominal or %s file -- skipped",
                    part_id, source, source)
        return

    LOG.info("[%s] %s: reading meshes", part_id, source)
    nominal = _read_mesh_pv(nom_path)
    test = _read_mesh_pv(test_path)

    # Align test to nominal.
    # Prefer the pipeline's saved transform (consistent with the report); --icp
    # refines on top of it. If no transform is saved, ICP runs from identity as
    # a fallback (which may fail for scale-mismatched Zephyr).
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
        T_icp = _icp_refine(test, nominal)        # test already roughly aligned, init = I
        test = test.transform(T_icp, inplace=False)

    fitted = False
    if args.fit_bbox:
        test = _fit_bbox(test, nominal)
        fitted = True

    # Decimate the heavy CT display copy.
    # Use decimate_pro (vtkDecimatePro: incremental edge-collapse, low memory),
    # not decimate (vtkQuadricDecimation), which builds per-vertex quadric
    # matrices over the whole 12M-tri CT mesh and runs out of memory on a tight
    # machine.
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

    # Signed deviation.
    test = _signed_deviation(test, nominal)
    dev = test["deviation_mm"]
    clip = args.clip
    thresh = args.threshold
    LOG.info("[%s] %s: %s", part_id, source, _stats(dev, thresh).replace("\n", "  "))

    # Render.
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
    ap = argparse.ArgumentParser(description="Nominal-vs-as-built 3D viewers")
    ap.add_argument("--mode", choices=("overlay", "deviation"), default="overlay",
                    help="overlay = 3-way superimposition (default); "
                         "deviation = per-pair signed-deviation heat-map")
    ap.add_argument("--part", nargs="*", default=None,
                    help="part IDs (2PR 4P2 6P1); default all")
    ap.add_argument("--pair", choices=("ct", "zephyr", "both"), default="both",
                    help="which test surface(s) to include")
    ap.add_argument("--solid", action="store_true",
                    help="[overlay] render solid surface MESHES instead of sampled "
                         "points (translucent; CT decimated; slower to open)")
    ap.add_argument("--clip", type=float, default=C.HEATMAP_CLIP,
                    help="[deviation] +/- range (mm) for the colour map")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="[deviation] highlight |deviation| above this (mm)")
    ap.add_argument("--icp", action="store_true",
                    help="[deviation] refine alignment with a fresh ICP instead of the saved transform")
    ap.add_argument("--full", action="store_true",
                    help="[deviation] do not decimate the CT mesh for display (heavy)")
    ap.add_argument("--fit-bbox", action="store_true",
                    help="[deviation] anisotropically fit the test bbox to the nominal's "
                         "(base matched, short height amplified); display only")
    ap.add_argument("--off-screen", action="store_true",
                    help="headless: save screenshots without opening a window")
    args = ap.parse_args()

    from phase0_match import match_parts
    parts = match_parts()
    if args.part:
        want = set(args.part)
        parts = {k: v for k, v in parts.items() if k in want}
    if not parts:
        LOG.error("no parts found under %s", C.DATA_ROOT)
        return

    sources = ("ct", "zephyr") if args.pair == "both" else (args.pair,)

    if args.mode == "overlay":
        for pid, srcs in parts.items():
            try:
                if args.solid:
                    solid_overlay_part(pid, srcs, sources, args)
                else:
                    overlay_part(pid, sources, args)
            except Exception:
                import traceback
                LOG.warning("[%s] overlay failed:\n%s", pid, traceback.format_exc())
    else:
        for pid, srcs in parts.items():
            for src in sources:
                try:
                    deviation_pair(pid, src, srcs, args)
                except Exception:
                    import traceback
                    LOG.warning("[%s] %s: visualization failed:\n%s",
                                pid, src, traceback.format_exc())


if __name__ == "__main__":
    main()
