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
    """Fraction of reference (nominal) points with an aligned as-built point
    within `tol`. Flip-safe registration score: a wrong basin leaves much of the
    nominal surface uncovered even when its inlier rmse looks fine."""
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
    transform: np.ndarray                 # 4x4 mapping source -> nominal frame
    coarse: RegResult | None
    fine: RegResult | None
    leg: RegResult | None
    cloud: o3d.geometry.PointCloud        # aligned, full-res
    mesh: o3d.geometry.TriangleMesh | None
    info: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
def _downsample(pcd, voxel):
    d = pcd.voxel_down_sample(voxel)
    estimate_normals(d, voxel)
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        d, o3d.geometry.KDTreeSearchParamHybrid(
            radius=C.FPFH_RADIUS_FACTOR * voxel, max_nn=100))
    return d, fpfh


def register(source_pcd, target_pcd, voxel_size) -> tuple[RegResult, RegResult]:
    """Multi-start coarse RANSAC + point-to-plane ICP, kept by best coverage.

    The coarse RANSAC is run from each seed in config.RANSAC_SEEDS, each refined
    by ICP; the candidate whose aligned cloud best COVERS the nominal surface
    wins (see _coverage). Fixed seed list => deterministic. Returns the winning
    (coarse, fine) RegResults; transforms map source -> target frame."""
    src_d, src_f = _downsample(source_pcd, voxel_size)
    tgt_d, tgt_f = _downsample(target_pcd, voxel_size)
    dist = C.RANSAC_DIST_FACTOR * voxel_size
    ref_pts = np.asarray(target_pcd.points)
    src_pts = np.asarray(source_pcd.points)

    best = None  # (coverage, coarse_res, fine_res)
    for seed in C.RANSAC_SEEDS:
        # seed open3d's global RNG so each RANSAC start is deterministic
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
    """The part's build (vertical) direction as the PCA eigenvector CLOSEST to
    `ref` -- the bbox-based world build axis from detect_axes, which is robust to
    a partial/biased cloud. Picking the globally-longest eigenvector instead
    fails on the wider 4mm/6mm parts (build is not the longest extent) and on
    partial Zephyr clouds (PCA weighted by which faces were captured), which sent
    the correction off by tens of degrees. The eigenvector nearest the known
    build axis is the small residual roll we actually want to remove."""
    c = pts.mean(0)
    _, evecs = np.linalg.eigh(np.cov((pts - c).T))
    k = int(np.argmax([abs(evecs[:, j] @ ref) for j in range(3)]))
    b = evecs[:, k]
    return b if np.dot(b, ref) >= 0 else -b


def _min_rot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minimal rotation matrix taking unit vector a onto unit vector b (Rodrigues).
    Used only for the few-degree upright correction, so a and b are never
    anti-parallel."""
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
    """De-roll an under-constrained body onto the nominal frame, returning
    (transform, roll_removed). roll_removed = 0 means no correction was applied.

    Two flip-free steps, applied only for a plausible residual roll
    (UPRIGHT_MAX_ROLL_DEG < roll <= UPRIGHT_MAX_APPLY_DEG):
      1. rotate the build axis (the PCA eigenvector nearest the bbox vertical)
         onto world vertical, snapping the part upright -- this is the tilt seen
         in the V-W plots;
      2. about that axis, rotate the segmented wall-centroid vector (leg1->leg2)
         onto world W, so the legs sit left/right like the nominal.
    The magnitude cap rejects the large bogus rolls PCA produces on sparse /
    partial clouds (e.g. the 4mm Zephyr, already upright but reading ~35 deg)."""
    al = copy.deepcopy(aligned_pcd).transform(init_T)
    pts = np.asarray(al.points)
    # bbox-based world axes (robust to partial clouds); the part is already
    # roughly aligned to them, so these are the upright targets.
    V_world, W_world, _, _, _, _, _ = detect_axes(pts)
    b = _build_axis(pts, V_world)
    roll = angle_between(b, V_world)
    # apply only for a plausible residual roll; below threshold = already upright,
    # above the cap = a PCA artifact on a sparse/partial cloud (not a real roll).
    if not (C.UPRIGHT_MAX_ROLL_DEG < roll <= C.UPRIGHT_MAX_APPLY_DEG):
        return init_T, 0.0

    c = pts.mean(0)
    R1 = _min_rot(b, V_world)                       # snap build axis -> world vertical
    # step 2: about the vertical, rotate the leg1->leg2 wall vector onto world W,
    # so the legs sit left/right like the nominal. Skipped if it would be a large
    # (flip-like) rotation -- registration already fixes the gross left/right.
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
    T_corr[:3, 3] = c - R @ c                        # rotate about the body centroid
    return T_corr @ init_T, roll


def _leg_refine(source_pcd, target_pcd, init_T, voxel) -> RegResult | None:
    """Second ICP restricted to the leg regions of both clouds (Stage A step 5)."""
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
    # compose: legs ICP refines the already-aligned cloud, so total = icp * init
    total = icp.transformation @ init_T
    return RegResult(total, icp.fitness, icp.inlier_rmse, "leg")


def register_part(part_id, loaded) -> dict:
    """Register CT and Zephyr to Nominal; save aligned clouds/meshes + transforms."""
    out_dir = C.OUTPUT_ROOT / part_id
    nominal = loaded.get("nominal")
    if nominal is None:
        return {}

    # nominal stays in place; persist its sampled cloud for downstream phases
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
            # upright datum: de-roll an under-constrained (rolled) body onto the
            # nominal frame; no-op when the residual roll is below threshold.
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
