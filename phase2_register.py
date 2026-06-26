"""
phase2_register.py  --  Stage A: bring each as-built source into the Nominal frame.

Pipeline per source (Nominal is the fixed reference):
    3. coarse : voxel-downsample + FPFH + RANSAC global registration
                (no pre-alignment assumed)
    4. fine   : point-to-plane ICP seeded from the coarse transform
    5. leg-datum refinement : a second ICP using ONLY the segmented leg regions,
       so the distorted overhang does not pull the alignment

The Nominal cloud/mesh stays put (identity transform); CT and Zephyr are moved.

Public API:
    register(source_pcd, target_pcd, voxel_size) -> RegResult
    register_part(part_id, loaded) -> dict[source] -> Aligned   (+ saves outputs)
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

import config as C
from common import LOG, angle_between, estimate_normals, save_cloud, save_mesh
from phase4_segment import detect_axes, leg_points, segment_legs_overhang


def _coverage(ref_pts: np.ndarray, aligned_pts: np.ndarray, tol: float) -> float:
    tree = cKDTree(aligned_pts)
    d, _ = tree.query(ref_pts, k=1, workers=-1)
    return float((d < tol).mean())


@dataclass
class RegResult:
    transform: np.ndarray
    fitness: float
    inlier_rmse: float
    stage: str = ""


@dataclass
class Aligned:
    source: str
    transform: np.ndarray             
    coarse: RegResult | None
    fine: RegResult | None
    leg: RegResult | None
    cloud: o3d.geometry.PointCloud       
    mesh: o3d.geometry.TriangleMesh | None
    info: dict = field(default_factory=dict)

def _downsample(pcd, voxel):
    d = pcd.voxel_down_sample(voxel)
    estimate_normals(d, voxel)
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        d, o3d.geometry.KDTreeSearchParamHybrid(
            radius=C.FPFH_RADIUS_FACTOR * voxel, max_nn=100))
    return d, fpfh


def register(source_pcd, target_pcd, voxel_size) -> tuple[RegResult, RegResult]:
    src_d, src_f = _downsample(source_pcd, voxel_size)
    tgt_d, tgt_f = _downsample(target_pcd, voxel_size)
    dist = C.RANSAC_DIST_FACTOR * voxel_size
    ref_pts = np.asarray(target_pcd.points)
    src_pts = np.asarray(source_pcd.points)

    best = None  
    for seed in C.RANSAC_SEEDS:
        o3d.utility.random.seed(seed)
        coarse = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            src_d, tgt_d, src_f, tgt_f, True, dist,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            C.RANSAC_N,
            [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
             o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist)],
            o3d.pipelines.registration.RANSACConvergenceCriteria(
                C.RANSAC_MAX_ITER, C.RANSAC_CONFIDENCE))

        icp = o3d.pipelines.registration.registration_icp(
            source_pcd, target_pcd, C.ICP_DIST_FACTOR * voxel_size, coarse.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=C.ICP_MAX_ITER))

        T = icp.transformation
        aligned_pts = src_pts @ T[:3, :3].T + T[:3, 3]
        cov = _coverage(ref_pts, aligned_pts, C.REG_COVERAGE_TOL)
        if best is None or cov > best[0]:
            best = (cov,
                    RegResult(coarse.transformation.copy(), coarse.fitness,
                              coarse.inlier_rmse, "coarse"),
                    RegResult(icp.transformation.copy(), icp.fitness,
                              icp.inlier_rmse, "fine"))

    return best[1], best[2]


def _build_axis(pts: np.ndarray, ref: np.ndarray) -> np.ndarray:
    _, evecs = np.linalg.eigh(np.cov((pts - c).T))
    k = int(np.argmax([abs(evecs[:, j] @ ref) for j in range(3)]))
    b = evecs[:, k]
    return b if np.dot(b, ref) >= 0 else -b


def _min_rot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
  
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = float(np.dot(a, b))
    if s < 1e-9:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))


def _upright_refine(aligned_pcd, nominal_pcd, init_T) -> tuple[np.ndarray, float]:

    al = copy.deepcopy(aligned_pcd).transform(init_T)
    pts = np.asarray(al.points)
    V_world, W_world, _, _, _, _, _ = detect_axes(pts)
    b = _build_axis(pts, V_world)
    roll = angle_between(b, V_world)
    if not (C.UPRIGHT_MAX_ROLL_DEG < roll <= C.UPRIGHT_MAX_APPLY_DEG):
        return init_T, 0.0

    c = pts.mean(0)
    R1 = _min_rot(b, V_world)                       
    R2 = np.eye(3)
    try:
        al2 = copy.deepcopy(al)
        al2.rotate(R1, center=c)
        seg = segment_legs_overhang(al2)
        s = seg.planes["leg2_wall"].centroid - seg.planes["leg1_wall"].centroid
        s = s - np.dot(s, V_world) * V_world
        if np.linalg.norm(s) > 1e-6 and angle_between(s, W_world) <= 20.0:
            R2 = _min_rot(s, W_world)
    except Exception as e:
        LOG.warning("    upright step 2 (wall vector) skipped: %s", e)

    R = R2 @ R1
    T_corr = np.eye(4)
    T_corr[:3, :3] = R
    T_corr[:3, 3] = c - R @ c                       
    return T_corr @ init_T, roll


def _leg_refine(source_pcd, target_pcd, init_T, voxel) -> RegResult | None:
    try:
        src_aligned = copy.deepcopy(source_pcd).transform(init_T)
        src_legs_pts = leg_points(src_aligned)
        tgt_legs_pts = leg_points(target_pcd)
    except Exception as e:
        LOG.warning("    leg-datum refinement skipped (segmentation failed: %s)", e)
        return None
    if len(src_legs_pts) < 50 or len(tgt_legs_pts) < 50:
        LOG.warning("    leg-datum refinement skipped (too few leg points)")
        return None

    src_legs = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src_legs_pts))
    tgt_legs = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(tgt_legs_pts))
    estimate_normals(src_legs, voxel)
    estimate_normals(tgt_legs, voxel)

    icp = o3d.pipelines.registration.registration_icp(
        src_legs, tgt_legs, C.LEG_REFINE_DIST_FACTOR * voxel, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=C.ICP_MAX_ITER))
    total = icp.transformation @ init_T
    return RegResult(total, icp.fitness, icp.inlier_rmse, "leg")


def register_part(part_id, loaded) -> dict:
    out_dir = C.OUTPUT_ROOT / part_id
    nominal = loaded.get("nominal")
    if nominal is None:
        return {}

    save_cloud(nominal.cloud, out_dir / "nominal_sampled.ply")
    results = {"nominal": Aligned("nominal", np.eye(4), None, None, None,
                                  nominal.cloud, nominal.mesh)}

    transforms_log = {}
    for src in ("ct", "zephyr"):
        ld = loaded.get(src)
        if ld is None:
            continue
        try:
            coarse, fine = register(ld.cloud, nominal.cloud, C.VOXEL_SIZE)
            best = fine
            leg = None
            if C.LEG_REFINE:
                leg = _leg_refine(ld.cloud, nominal.cloud, fine.transform, C.VOXEL_SIZE)
                if leg is not None and leg.fitness >= 0.1:
                    best = leg
            roll = 0.0
            if C.UPRIGHT_REFINE:
                up_T, roll = _upright_refine(ld.cloud, nominal.cloud, best.transform)
                if roll > 0.0:
                    best = RegResult(up_T, best.fitness, best.inlier_rmse,
                                     best.stage + "+upright")
            LOG.info("[%s] %-6s coarse fit=%.3f rmse=%.3f | fine fit=%.3f rmse=%.3f"
                     "%s%s", part_id, src, coarse.fitness, coarse.inlier_rmse,
                     fine.fitness, fine.inlier_rmse,
                     f" | leg fit={leg.fitness:.3f} rmse={leg.inlier_rmse:.3f}"
                     if leg else "",
                     f" | upright -{roll:.1f}deg" if roll > 0.0 else "")

            aligned_cloud = copy.deepcopy(ld.cloud).transform(best.transform)
            aligned_mesh = None
            if ld.mesh is not None:
                aligned_mesh = copy.deepcopy(ld.mesh).transform(best.transform)
                save_mesh(aligned_mesh, out_dir / f"aligned_{src}.stl")
            save_cloud(aligned_cloud, out_dir / f"aligned_{src}.ply")

            results[src] = Aligned(src, best.transform, coarse, fine, leg,
                                   aligned_cloud, aligned_mesh,
                                   {"scale": ld.scale, "n_points": ld.n_points})
            transforms_log[src] = {
                "scale": ld.scale,
                "transform": best.transform.tolist(),
                "coarse_fitness": coarse.fitness, "coarse_rmse": coarse.inlier_rmse,
                "fine_fitness": fine.fitness, "fine_rmse": fine.inlier_rmse,
                "leg_fitness": (leg.fitness if leg else None),
                "leg_rmse": (leg.inlier_rmse if leg else None),
                "stage_used": best.stage,
            }
        except Exception as e:
            LOG.warning("[%s] registration failed for %s: %s", part_id, src, e)

    if transforms_log:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "transforms.json").write_text(json.dumps(transforms_log, indent=2))
    return results


if __name__ == "__main__":
    from phase0_match import match_parts
    from phase1_normalize import load_and_normalize
    for pid, srcs in match_parts().items():
        register_part(pid, load_and_normalize(pid, srcs))
