"""
phase3_deviation.py  --  Stage B: signed surface-deviation between co-registered
sources.

For each configured pair (moving, reference) we compute the signed point-to-
surface distance of the moving cloud against the reference surface. Sign is set
by the reference surface normal (positive = outside / away from material).

Outputs per pair:
    * deviation-coloured heat-map PLY  (clipped to +/-HEATMAP_CLIP)
    * histogram PNG
    * summary stats (mean, std, rms, min, max, p95) for comparison_report.csv

Public API:
    signed_deviation(moving_pcd, ref_pcd) -> (distances, stats)
    deviations_for_part(part_id, aligned) -> list[stat dict]
"""

from __future__ import annotations

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
from common import LOG, estimate_normals, save_cloud


def _stats(d: np.ndarray) -> dict:
    return {
        "n": int(d.size),
        "mean": float(np.mean(d)),
        "std": float(np.std(d)),
        "rms": float(np.sqrt(np.mean(d ** 2))),
        "min": float(np.min(d)),
        "max": float(np.max(d)),
        "abs_p95": float(np.percentile(np.abs(d), 95)),
    }


def signed_deviation(moving: o3d.geometry.PointCloud,
                     reference: o3d.geometry.PointCloud):
    """Signed nearest-surface distance of `moving` w.r.t. `reference`.

    Sign comes from the reference normal at the closest point: a moving point
    sitting outside the reference material (along +normal) is positive."""
    ref = reference
    if not ref.has_normals():
        estimate_normals(ref, C.VOXEL_SIZE)
    ref_pts = np.asarray(ref.points)
    ref_nrm = np.asarray(ref.normals)
    mov_pts = np.asarray(moving.points)

    # vectorized nearest-neighbour query, then sign from the reference normal
    tree = cKDTree(ref_pts)
    d, idx = tree.query(mov_pts, k=1, workers=-1)
    vec = mov_pts - ref_pts[idx]
    sign = np.sign(np.einsum("ij,ij->i", vec, ref_nrm[idx]) + 1e-12)
    dist = sign * d
    return dist, _stats(dist)


def _heatmap_cloud(moving, dist) -> o3d.geometry.PointCloud:
    clip = C.HEATMAP_CLIP
    t = np.clip((dist + clip) / (2 * clip), 0, 1)         # 0..1
    # blue(neg) - white(0) - red(pos)
    colors = np.empty((len(t), 3))
    colors[:, 0] = np.clip(2 * t, 0, 1)                   # R
    colors[:, 2] = np.clip(2 * (1 - t), 0, 1)             # B
    colors[:, 1] = 1 - np.abs(2 * t - 1)                  # G
    out = o3d.geometry.PointCloud(moving.points)
    out.colors = o3d.utility.Vector3dVector(colors)
    return out


def _histogram(dist, title, png_path):
    plt.figure(figsize=(6, 4))
    plt.hist(dist, bins=C.HIST_BINS, color="#3070b0", edgecolor="none")
    plt.axvline(0, color="k", lw=0.8)
    plt.xlabel("signed deviation (mm)")
    plt.ylabel("points")
    plt.title(title)
    plt.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(png_path, dpi=120)
    plt.close()


def deviations_for_part(part_id, aligned) -> list[dict]:
    out_dir = C.OUTPUT_ROOT / part_id
    rows = []
    for mov, ref in C.DEVIATION_PAIRS:
        if mov not in aligned or ref not in aligned:
            continue
        try:
            dist, stats = signed_deviation(aligned[mov].cloud, aligned[ref].cloud)
            pair = f"{mov}_vs_{ref}"
            save_cloud(_heatmap_cloud(aligned[mov].cloud, dist),
                       out_dir / f"deviation_{pair}.ply")
            _histogram(dist, f"{part_id}: {mov} vs {ref}",
                       out_dir / f"deviation_{pair}_hist.png")
            row = {"part": part_id, "pair": pair, **stats}
            rows.append(row)
            LOG.info("[%s] dev %-16s mean=%+.3f rms=%.3f p95=%.3f (mm)",
                     part_id, pair, stats["mean"], stats["rms"], stats["abs_p95"])
        except Exception as e:
            LOG.warning("[%s] deviation %s vs %s failed: %s", part_id, mov, ref, e)
    return rows


if __name__ == "__main__":
    from phase0_match import match_parts
    from phase1_normalize import load_and_normalize
    from phase2_register import register_part
    for pid, srcs in match_parts().items():
        aligned = register_part(pid, load_and_normalize(pid, srcs))
        deviations_for_part(pid, aligned)
