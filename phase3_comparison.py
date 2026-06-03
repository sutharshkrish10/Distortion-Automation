"""
phase3_comparison.py — Nominal/actual distortion comparison using Open3D.
Aligns actual surface (CT or Zephyr) to nominal CAD STL via ICP,
then computes per-point signed surface deviations.
"""

import json
import numpy as np
import open3d as o3d
import trimesh
from pathlib import Path
from typing import Optional
import config


def load_as_point_cloud(path: Path, voxel_size: float = 0.05) -> o3d.geometry.PointCloud:
    """
    Load STL/PLY as an Open3D point cloud.
    - PLY point clouds are loaded directly.
    - STL / PLY meshes are surface-sampled to 200 k points.
    """
    suffix = path.suffix.lower()

    # Try reading as a point cloud first (works for .ply without faces)
    if suffix == ".ply":
        pcd = o3d.io.read_point_cloud(str(path))
        if len(pcd.points) > 0:
            pcd = pcd.voxel_down_sample(voxel_size)
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=30)
            )
            return pcd

    # For STL or PLY meshes: sample the surface
    mesh = trimesh.load(str(path), force="mesh")
    if not hasattr(mesh, "faces") or len(mesh.faces) == 0:
        # Last resort: use vertices directly
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(mesh.vertices, dtype=np.float64))
    else:
        pts = trimesh.sample.sample_surface(mesh, count=200_000)[0]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))

    pcd = pcd.voxel_down_sample(voxel_size)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=30)
    )
    return pcd


def coarse_align(source: o3d.geometry.PointCloud,
                 target: o3d.geometry.PointCloud,
                 voxel_size: float) -> np.ndarray:
    """FPFH-based global registration for initial coarse alignment."""
    def compute_fpfh(pcd, vs):
        pcd_down = pcd.voxel_down_sample(vs * 5)
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=vs * 10, max_nn=30)
        )
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=vs * 25, max_nn=100),
        )
        return pcd_down, fpfh

    src_down, src_fpfh = compute_fpfh(source, voxel_size)
    tgt_down, tgt_fpfh = compute_fpfh(target, voxel_size)

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down, tgt_down, src_fpfh, tgt_fpfh,
        mutual_filter=True,
        max_correspondence_distance=voxel_size * 15,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * 15),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4_000_000, 500),
    )
    return result.transformation


def icp_refine(source: o3d.geometry.PointCloud,
               target: o3d.geometry.PointCloud,
               init_transform: np.ndarray,
               threshold: float) -> np.ndarray:
    """Point-to-plane ICP refinement."""
    result = o3d.pipelines.registration.registration_icp(
        source, target,
        max_correspondence_distance=threshold,
        init=init_transform,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=config.ICP_MAX_ITERATIONS
        ),
    )
    print(f"  ICP fitness: {result.fitness:.4f}  RMSE: {result.inlier_rmse:.4f} mm")
    return result.transformation


def compute_deviations(
    actual_pcd: o3d.geometry.PointCloud,
    nominal_pcd: o3d.geometry.PointCloud,
) -> np.ndarray:
    """
    For each point in actual, find the nearest nominal point.
    Returns signed distances: positive = outside nominal, negative = inside.
    Sign is approximated from normal direction.
    """
    actual_pts  = np.asarray(actual_pcd.points)
    nominal_pts = np.asarray(nominal_pcd.points)
    nominal_norms = np.asarray(nominal_pcd.normals)

    tree = o3d.geometry.KDTreeFlann(nominal_pcd)
    deviations = np.zeros(len(actual_pts))

    for i, pt in enumerate(actual_pts):
        _, idx, _ = tree.search_knn_vector_3d(pt, 1)
        diff = pt - nominal_pts[idx[0]]
        normal = nominal_norms[idx[0]]
        sign = np.sign(np.dot(diff, normal))
        deviations[i] = sign * np.linalg.norm(diff)

    return deviations


def compute_statistics(deviations: np.ndarray, tolerance: float) -> dict:
    return {
        "n_points":       int(len(deviations)),
        "max_dev_mm":     float(np.max(deviations)),
        "min_dev_mm":     float(np.min(deviations)),
        "mean_dev_mm":    float(np.mean(deviations)),
        "abs_mean_mm":    float(np.mean(np.abs(deviations))),
        "rms_mm":         float(np.sqrt(np.mean(deviations**2))),
        "std_mm":         float(np.std(deviations)),
        "pct_in_tol":     float(np.mean(np.abs(deviations) <= tolerance) * 100),
        "tolerance_mm":   tolerance,
    }


def color_point_cloud(pcd: o3d.geometry.PointCloud,
                      deviations: np.ndarray,
                      vmin: float = -1.0,
                      vmax: float =  1.0) -> o3d.geometry.PointCloud:
    """Map deviations to red-green-blue colormap on the point cloud."""
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("RdYlGn_r")
    norm = np.clip((deviations - vmin) / (vmax - vmin), 0, 1)
    colors = cmap(norm)[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def compare(
    name: str,
    actual_path: Path,
    nominal_path: Path,
    output_dir: Path,
) -> Optional[dict]:
    """
    Full comparison pipeline for one pair (actual vs. nominal).
    Returns stats dict.
    """
    out_ply   = output_dir / f"{name}_deviation_cloud.ply"
    out_json  = output_dir / f"{name}_stats.json"

    if out_json.exists():
        print(f"  [SKIP] {name}: comparison already done")
        return json.loads(out_json.read_text())

    if not actual_path.exists():
        print(f"  [SKIP] {name}: actual mesh not found — {actual_path}")
        return None
    if not nominal_path.exists():
        print(f"  [SKIP] {name}: nominal STL not found — {nominal_path}")
        return None

    print(f"\n  Comparing '{name}'")
    print(f"    Actual  : {actual_path.name}")
    print(f"    Nominal : {nominal_path.name}")

    vs = config.VOXEL_DOWNSAMPLE

    print("  Loading point clouds...")
    actual_pcd  = load_as_point_cloud(actual_path,  voxel_size=vs)
    nominal_pcd = load_as_point_cloud(nominal_path, voxel_size=vs)
    print(f"  Actual: {len(actual_pcd.points):,} pts  "
          f"Nominal: {len(nominal_pcd.points):,} pts")

    print("  Coarse alignment (RANSAC)...")
    try:
        T_coarse = coarse_align(actual_pcd, nominal_pcd, voxel_size=vs)
    except Exception as e:
        print(f"  [WARN] Coarse alignment failed ({e}), using identity")
        T_coarse = np.eye(4)

    print("  ICP fine registration...")
    T_icp = icp_refine(actual_pcd, nominal_pcd, T_coarse, threshold=config.ICP_THRESHOLD)
    actual_pcd.transform(T_icp)

    print("  Computing per-point deviations...")
    deviations = compute_deviations(actual_pcd, nominal_pcd)

    stats = compute_statistics(deviations, config.TOLERANCE_MM)
    stats["name"] = name
    out_json.write_text(json.dumps(stats, indent=2))

    # Color-coded point cloud
    pcd_colored = color_point_cloud(actual_pcd, deviations,
                                    vmin=-config.TOLERANCE_MM * 2,
                                    vmax= config.TOLERANCE_MM * 2)
    o3d.io.write_point_cloud(str(out_ply), pcd_colored)

    # Also save raw deviations as .npy
    np.save(str(output_dir / f"{name}_deviations.npy"), deviations)

    _print_stats(stats)
    return stats


def _print_stats(s: dict):
    print(f"    max       : {s['max_dev_mm']:+.3f} mm")
    print(f"    min       : {s['min_dev_mm']:+.3f} mm")
    print(f"    mean      : {s['mean_dev_mm']:+.3f} mm")
    print(f"    RMS       : {s['rms_mm']:.3f} mm")
    print(f"    std       : {s['std_mm']:.3f} mm")
    print(f"    in ±{s['tolerance_mm']:.1f}mm  : {s['pct_in_tol']:.1f}%")


def run_all(
    zephyr_meshes: dict[str, Path],
    ct_meshes: dict[str, Path],
    output_dir: Path | None = None,
) -> dict[str, dict]:
    """
    Compare every available actual surface against the appropriate nominal STL.
    """
    output_dir = output_dir or config.OUT_COMPARISON
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n-- Phase 3: Nominal/Actual Comparison (Open3D) --")

    all_stats: dict[str, dict] = {}

    # Zephyr meshes → compare against 6P1 nominal (adjust per project if needed)
    for name, path in zephyr_meshes.items():
        nominal = config.STL_NOMINAL.get("6P1", list(config.STL_NOMINAL.values())[0])
        stats = compare(f"zephyr_{name}", path, nominal, output_dir)
        if stats:
            all_stats[f"zephyr_{name}"] = stats

    # CT meshes → compare against 6P1 nominal
    for name, path in ct_meshes.items():
        nominal = config.STL_NOMINAL.get("6P1", list(config.STL_NOMINAL.values())[0])
        stats = compare(f"ct_{name}", path, nominal, output_dir)
        if stats:
            all_stats[f"ct_{name}"] = stats

    print(f"\n  Comparisons complete: {len(all_stats)} result(s)")
    return all_stats


if __name__ == "__main__":
    config.print_config()
    # Quick standalone test with the existing PLY
    if config.EXISTING_PLY.exists():
        nominal = config.STL_NOMINAL.get("6P1")
        if nominal and nominal.exists():
            run_all(
                zephyr_meshes={"existing": config.EXISTING_PLY},
                ct_meshes={},
            )
