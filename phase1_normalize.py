"""Load each source and normalize its units/scale.

The Nominal STL is sampled to a Poisson-disk cloud and defines the reference
frame; the CT and Zephyr STLs are sampled the same way.

Photogrammetry (Zephyr) clouds can carry an arbitrary scale, so we compare each
source's bbox-diagonal to the Nominal's and apply a uniform scale factor when
they disagree by more than SCALE_AUTODETECT_TOL (this can be overridden in
config). Registration in phase 2 cleans up the residual.

load_and_normalize(part_id, srcs) returns {source: Loaded}.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import open3d as o3d

import config as C
from common import (LOG, bbox_diagonal, estimate_normals, is_binary_stl,
                    load_mesh, mesh_to_cloud, stl_cloud)


@dataclass
class Loaded:
    source: str
    path: Path
    cloud: o3d.geometry.PointCloud          # normals estimated, unit-scaled
    mesh: o3d.geometry.TriangleMesh | None  # present for nominal/ct (scaled)
    scale: float = 1.0
    n_points: int = 0
    info: dict = field(default_factory=dict)
    # Untrimmed copy (stamp kept) used only for the saved aligned overlay;
    # measurement and registration always use `cloud`. None unless the CT stamp
    # was kept.
    cloud_full: o3d.geometry.PointCloud | None = None


def _apply_scale(geom, factor: float, center=(0, 0, 0)):
    if abs(factor - 1.0) > 1e-9:
        geom.scale(factor, center=np.asarray(center, dtype=float))
    return geom


def _trim_ct_stamp(cloud, design_height: float):
    """Drop the raised ID stamp that sits above the design build-height.

    The stamp is extra material on top of the part (present in CT, absent from
    the clean nominal), so CT reads taller than the design. Working in CT's own
    (roughly axis-aligned) frame, the build axis V is the largest-extent axis.
    The base end is the denser of the two extreme slabs (full footprint), and the
    stamp is the material beyond design_height + margin from that base. Returns
    (trimmed_cloud, n_dropped, v_axis), and does nothing when CT is not
    over-height."""
    pts = np.asarray(cloud.points)
    ext = pts.max(0) - pts.min(0)
    v = int(np.argmax(ext))
    cut = design_height + C.CT_STAMP_TRIM_MARGIN_MM
    if ext[v] <= cut:
        return cloud, 0, v
    vmin, vmax = float(pts[:, v].min()), float(pts[:, v].max())
    slab = 1.0
    n_lo = int((pts[:, v] < vmin + slab).sum())
    n_hi = int((pts[:, v] > vmax - slab).sum())
    keep = (pts[:, v] <= vmin + cut) if n_lo >= n_hi else (pts[:, v] >= vmax - cut)
    trimmed = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts[keep]))
    return trimmed, int((~keep).sum()), v


def _resolve_scale(part_id: str, source: str, diag_src: float, diag_nom: float) -> float:
    override = C.SCALE_OVERRIDE.get((part_id, source))
    if override is not None:
        return float(override)
    if diag_src <= 0:
        return 1.0
    ratio = diag_nom / diag_src
    if abs(ratio - 1.0) > C.SCALE_AUTODETECT_TOL:
        return ratio
    return 1.0


def load_and_normalize(part_id: str,
                       srcs: dict[str, Path | None]) -> dict[str, Loaded]:
    out: dict[str, Loaded] = {}

    # Nominal first: it defines the reference scale.
    nom_path = srcs.get("nominal")
    if nom_path is None:
        LOG.warning("[%s] no Nominal STL -- cannot build reference frame; "
                    "skipping part", part_id)
        return out
    nom_mesh = load_mesh(nom_path)
    diag_nom = bbox_diagonal(nom_mesh)
    nom_cloud = mesh_to_cloud(nom_mesh, C.POISSON_SAMPLES["nominal"])
    estimate_normals(nom_cloud, C.VOXEL_SIZE)
    out["nominal"] = Loaded("nominal", nom_path, nom_cloud, nom_mesh, 1.0,
                            len(nom_cloud.points),
                            {"bbox_diag": round(diag_nom, 3)})
    LOG.info("[%s] nominal : %d pts, bbox diag %.2f mm",
             part_id, len(nom_cloud.points), diag_nom)

    # CT mesh.
    # CT surface meshes can be very large (hundreds of M triangles, tens of GB),
    # and loading the full mesh runs the machine out of memory. Binary STLs are
    # stream-sampled off disk (no mesh in RAM, so ct_mesh stays None and only the
    # aligned-cloud PLY is saved downstream, not an aligned STL). Smaller or
    # non-binary CT files use the in-memory mesh loader instead.
    if srcs.get("ct"):
        try:
            ct_path = srcs["ct"]
            if is_binary_stl(ct_path):
                ct_cloud, diag_ct = stl_cloud(ct_path, C.POISSON_SAMPLES["ct"])
                ct_mesh = None
            else:
                ct_mesh = load_mesh(ct_path)
                diag_ct = bbox_diagonal(ct_mesh)
                ct_cloud = mesh_to_cloud(ct_mesh, C.POISSON_SAMPLES["ct"])
            scale = _resolve_scale(part_id, "ct", diag_ct, diag_nom)
            # CT is metric: never bbox-auto-scale it (the raised ID stamp inflates
            # its diagonal). An explicit override still takes precedence.
            if C.CT_TRUE_SCALE and (part_id, "ct") not in C.SCALE_OVERRIDE:
                scale = 1.0
            if ct_mesh is not None:
                _apply_scale(ct_mesh, scale, ct_mesh.get_center())
            _apply_scale(ct_cloud, scale, ct_cloud.get_center())
            # Remove the raised ID stamp (above design height) before
            # registration and deviation. Optionally keep the untrimmed cloud so
            # it can be saved as the aligned-CT overlay, so the stamp shows up in
            # renders but never affects alignment or the reported numbers.
            ct_full = None
            if C.CT_STAMP_TRIM:
                design_h = float(max(nom_mesh.get_axis_aligned_bounding_box().get_extent()))
                trimmed, n_drop, vax = _trim_ct_stamp(ct_cloud, design_h)
                if n_drop:
                    LOG.info("[%s] ct      : trimmed %d stamp pts >%.1f+%.1f mm "
                             "(build axis %d)", part_id, n_drop, design_h,
                             C.CT_STAMP_TRIM_MARGIN_MM, vax)
                    if C.CT_KEEP_STAMP_IN_OUTPUT:
                        ct_full = ct_cloud            # untrimmed scaled cloud
                        estimate_normals(ct_full, C.VOXEL_SIZE)
                ct_cloud = trimmed
            estimate_normals(ct_cloud, C.VOXEL_SIZE)
            out["ct"] = Loaded("ct", ct_path, ct_cloud, ct_mesh, scale,
                               len(ct_cloud.points),
                               {"bbox_diag": round(diag_ct * scale, 3)},
                               cloud_full=ct_full)
            LOG.info("[%s] ct      : %d pts, scale x%.4f", part_id,
                     len(ct_cloud.points), scale)
        except Exception as e:                       # keep the batch going
            LOG.warning("[%s] CT load failed: %s", part_id, e)

    # Zephyr mesh.
    # Zephyr ships cropped surface STL meshes (part only), sampled to a cloud
    # just like CT. The binary STLs carry an ASCII-looking "solid" header that
    # can trip open3d's parser, so we prefer the size-based binary stream sampler
    # (is_binary_stl / stl_cloud) and fall back to the mesh loader.
    if srcs.get("zephyr"):
        try:
            zep_path = srcs["zephyr"]
            if is_binary_stl(zep_path):
                zep, diag_zep = stl_cloud(zep_path, C.POISSON_SAMPLES["zephyr"])
            else:
                zep_mesh = load_mesh(zep_path)
                diag_zep = bbox_diagonal(zep_mesh)
                zep = mesh_to_cloud(zep_mesh, C.POISSON_SAMPLES["zephyr"])
            scale = _resolve_scale(part_id, "zephyr", diag_zep, diag_nom)
            # Metric Zephyr export: pin true scale like CT (the photogrammetry
            # bbox diagonal inflates the auto-scale and shrinks the part and slot).
            # An explicit override still takes precedence.
            if C.ZEPHYR_TRUE_SCALE and (part_id, "zephyr") not in C.SCALE_OVERRIDE:
                scale = 1.0
            _apply_scale(zep, scale, zep.get_center())
            estimate_normals(zep, C.VOXEL_SIZE)
            out["zephyr"] = Loaded("zephyr", zep_path, zep, None, scale,
                                   len(zep.points),
                                   {"bbox_diag": round(diag_zep * scale, 3)})
            LOG.info("[%s] zephyr  : %d pts, scale x%.4f", part_id,
                     len(zep.points), scale)
        except Exception as e:
            LOG.warning("[%s] Zephyr load failed: %s", part_id, e)

    return out


if __name__ == "__main__":
    from phase0_match import match_parts
    for pid, srcs in match_parts().items():
        load_and_normalize(pid, srcs)
