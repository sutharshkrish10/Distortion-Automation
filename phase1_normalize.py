"""
phase1_normalize.py  --  load each source and normalize units/scale.

* Nominal STL  -> mesh -> Poisson-disk cloud (this defines the reference frame).
* CT STL       -> mesh -> Poisson-disk cloud.
* Zephyr STL   -> cropped surface mesh -> Poisson-disk cloud.

Scale handling: photogrammetry (Zephyr) clouds carry an arbitrary scale, so we
compare each source's bbox-diagonal to the Nominal's and apply a uniform scale
factor when they disagree by more than SCALE_AUTODETECT_TOL (overridable in
config). Registration (phase 2) cleans up the residual.

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
    cloud: o3d.geometry.PointCloud          
    mesh: o3d.geometry.TriangleMesh | None  
    scale: float = 1.0
    n_points: int = 0
    info: dict = field(default_factory=dict)


def _apply_scale(geom, factor: float, center=(0, 0, 0)):
    if abs(factor - 1.0) > 1e-9:
        geom.scale(factor, center=np.asarray(center, dtype=float))
    return geom


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
            if ct_mesh is not None:
                _apply_scale(ct_mesh, scale, ct_mesh.get_center())
            _apply_scale(ct_cloud, scale, ct_cloud.get_center())
            estimate_normals(ct_cloud, C.VOXEL_SIZE)
            out["ct"] = Loaded("ct", ct_path, ct_cloud, ct_mesh, scale,
                               len(ct_cloud.points),
                               {"bbox_diag": round(diag_ct * scale, 3)})
            LOG.info("[%s] ct      : %d pts, scale x%.4f", part_id,
                     len(ct_cloud.points), scale)
        except Exception as e:                       # never crash the batch
            LOG.warning("[%s] CT load failed: %s", part_id, e)

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
