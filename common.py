"""
common.py
=========
Shared helpers used across every phase: logging, robust I/O (mesh / cloud),
mesh->cloud sampling, normal estimation, and small geometry utilities.

Keeping these here means the phase modules stay focused on their stage logic.
"""

from __future__ import annotations

import logging
import struct
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh

import config as C


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
def get_logger(name: str = "distortion") -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s",
                                          datefmt="%H:%M:%S"))
        log.addHandler(h)
        log.setLevel(logging.INFO)
    return log


LOG = get_logger()


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------
def load_mesh(path: Path) -> o3d.geometry.TriangleMesh:
    """Load a triangle mesh (STL) via open3d, falling back to trimesh."""
    mesh = o3d.io.read_triangle_mesh(str(path))
    if len(mesh.triangles) == 0:                       # open3d failed -> trimesh
        tm = trimesh.load(str(path), force="mesh")
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(np.asarray(tm.vertices)),
            o3d.utility.Vector3iVector(np.asarray(tm.faces)),
        )
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.compute_vertex_normals()
    return mesh


# --- Streaming sampler for huge binary STL meshes --------------------------
# The CT surface meshes are full-resolution reconstructions (hundreds of
# millions of triangles, tens of GB). open3d.read_triangle_mesh loads the whole
# thing into RAM -> the process is OS-killed before any phase runs. Since the
# pipeline only ever needs a ~1e5-point sample of each CT surface, we read the
# STL straight off disk in chunks and area-sample it, never holding more than
# one chunk (+ the surviving sample) in memory.

# On-disk binary-STL layout: 80-byte header, uint32 triangle count, then 50
# bytes per triangle (face normal + 3 vertices + 2-byte attribute).
_STL_TRI_DTYPE = np.dtype([
    ("normal", "<f4", (3,)),
    ("v0", "<f4", (3,)),
    ("v1", "<f4", (3,)),
    ("v2", "<f4", (3,)),
    ("attr", "<u2"),
])


def is_binary_stl(path: Path) -> bool:
    """True iff `path` is a binary STL (size == 84 + triangle_count * 50).
    ASCII STLs and other formats fail this and fall back to the mesh loader."""
    size = path.stat().st_size
    if size < 84 or path.suffix.lower() != ".stl":
        return False
    with open(path, "rb") as f:
        f.seek(80)
        n_tris = struct.unpack("<I", f.read(4))[0]
    return size == 84 + n_tris * 50


def sample_binary_stl(path: Path, n_samples: int,
                      chunk_tris: int = 1_000_000, seed: int = 0):
    """Area-weighted surface sampling read straight from a binary STL in chunks.

    Returns (points Nx3, normals Nx3, bbox_min, bbox_max). Peak memory is
    O(chunk_tris + n_samples), independent of mesh size, so a 22 GB / 440 M-tri
    CT mesh samples without ever materializing the full mesh.

    Sampling uses Efraimidis-Spirakis weighted reservoir keys (key = u**(1/area)
    per triangle): keeping the top-`n_samples` keys draws triangles with
    probability proportional to area, i.e. an area-uniform surface sample -- the
    same distribution mesh.sample_points_uniformly produces. Because
    n_samples << triangles here, at most one point lands on any triangle, so
    sampling triangles without replacement matches sampling the surface.
    """
    n = int(n_samples)
    rng = np.random.default_rng(seed)

    keep_keys = np.empty(0, dtype=np.float64)
    keep_pts = np.empty((0, 3), dtype=np.float64)
    keep_nrm = np.empty((0, 3), dtype=np.float64)
    bmin = np.full(3, np.inf)
    bmax = np.full(3, -np.inf)

    with open(path, "rb") as f:
        f.seek(80)
        n_tris = struct.unpack("<I", f.read(4))[0]       # data begins at byte 84
        LOG.info("    streaming %s (%d triangles, %.1f GB) -> %d-point sample",
                 path.name, n_tris, path.stat().st_size / 1e9, n)

        remaining = n_tris
        while remaining > 0:
            m = min(chunk_tris, remaining)
            buf = f.read(m * 50)
            m = len(buf) // 50                           # truncated-file guard
            if m == 0:
                break
            remaining -= m
            tri = np.frombuffer(buf, dtype=_STL_TRI_DTYPE, count=m)

            v0 = tri["v0"].astype(np.float64)
            e1 = tri["v1"].astype(np.float64) - v0
            e2 = tri["v2"].astype(np.float64) - v0

            # running bounding box over the chunk's vertices
            bmin = np.minimum(bmin, np.minimum.reduce(
                [v0.min(0), (v0 + e1).min(0), (v0 + e2).min(0)]))
            bmax = np.maximum(bmax, np.maximum.reduce(
                [v0.max(0), (v0 + e1).max(0), (v0 + e2).max(0)]))

            cross = np.cross(e1, e2)
            area = 0.5 * np.linalg.norm(cross, axis=1)
            good = area > 1e-20
            ng = int(good.sum())
            if ng == 0:
                continue

            keys = np.zeros(m)
            keys[good] = rng.random(ng) ** (1.0 / area[good])

            # only the chunk's strongest keys can enter the global top-N
            cand = np.nonzero(good)[0]
            k = min(n, ng)
            if cand.size > k:
                cand = cand[np.argpartition(keys[cand], cand.size - k)[cand.size - k:]]

            # uniform barycentric point inside each surviving triangle
            r1 = rng.random(cand.size)
            r2 = rng.random(cand.size)
            flip = (r1 + r2) > 1.0
            r1[flip], r2[flip] = 1.0 - r1[flip], 1.0 - r2[flip]
            pts = v0[cand] + r1[:, None] * e1[cand] + r2[:, None] * e2[cand]
            nrm = cross[cand] / (2.0 * area[cand][:, None])   # unit face normal

            keep_keys = np.concatenate([keep_keys, keys[cand]])
            keep_pts = np.concatenate([keep_pts, pts])
            keep_nrm = np.concatenate([keep_nrm, nrm])
            if keep_keys.size > n:
                sel = np.argpartition(keep_keys, keep_keys.size - n)[keep_keys.size - n:]
                keep_keys, keep_pts, keep_nrm = keep_keys[sel], keep_pts[sel], keep_nrm[sel]

    return keep_pts, keep_nrm, bmin, bmax


def stl_cloud(path: Path, n_samples: int):
    """Sample a binary STL into a normal-bearing point cloud without loading the
    mesh. Returns (PointCloud, bbox_diagonal)."""
    pts, nrm, bmin, bmax = sample_binary_stl(path, n_samples)
    pcd = make_pcd(pts)
    pcd.normals = o3d.utility.Vector3dVector(nrm)
    return pcd, float(np.linalg.norm(bmax - bmin))


def load_cloud(path: Path) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(str(path))
    if len(pcd.points) == 0:
        raise ValueError(f"empty / unreadable point cloud: {path}")
    return pcd


def save_cloud(pcd: o3d.geometry.PointCloud, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(path), pcd)


def save_mesh(mesh: o3d.geometry.TriangleMesh, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(path), mesh)


# --------------------------------------------------------------------------
# Mesh -> cloud + normals
# --------------------------------------------------------------------------
def mesh_to_cloud(mesh: o3d.geometry.TriangleMesh, n_samples: int) -> o3d.geometry.PointCloud:
    """Sample a mesh into a point cloud (Poisson-disk = even spacing, slower;
    uniform = fast). Controlled by config.SAMPLE_METHOD.

    The seed is reset on every call: open3d's samplers draw from the global RNG,
    so without this a part's sampled cloud would depend on how much randomness
    earlier parts consumed (batch position), making registration -- and the
    borderline CT alignments downstream -- non-reproducible. Re-seeding makes
    each part's nominal/CT cloud identical regardless of batch order. The CT
    stream sampler is independently seeded; RANSAC re-seeds in phase 2."""
    n = int(n_samples)
    o3d.utility.random.seed(C.RANSAC_SEED)
    if C.SAMPLE_METHOD == "uniform":
        return mesh.sample_points_uniformly(number_of_points=n)
    return mesh.sample_points_poisson_disk(number_of_points=n)


def estimate_normals(pcd: o3d.geometry.PointCloud, voxel: float) -> o3d.geometry.PointCloud:
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=C.NORMAL_RADIUS_FACTOR * voxel, max_nn=C.NORMAL_MAX_NN))
    pcd.normalize_normals()
    return pcd


# --------------------------------------------------------------------------
# Geometry utilities
# --------------------------------------------------------------------------
def bbox_diagonal(geom) -> float:
    ext = geom.get_max_bound() - geom.get_min_bound()
    return float(np.linalg.norm(ext))


def points_of(geom) -> np.ndarray:
    """Return Nx3 vertex/point array for a mesh or point cloud."""
    if isinstance(geom, o3d.geometry.TriangleMesh):
        return np.asarray(geom.vertices)
    return np.asarray(geom.points)


def transform_copy(geom, T: np.ndarray):
    g = geom.__class__(geom)        # cheap copy via copy-ctor
    g.transform(T)
    return g


def angle_between(u: np.ndarray, v: np.ndarray) -> float:
    """Unsigned angle (deg) between two vectors, in [0, 180]."""
    u = u / (np.linalg.norm(u) + 1e-12)
    v = v / (np.linalg.norm(v) + 1e-12)
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v), -1.0, 1.0))))


def make_pcd(points: np.ndarray) -> o3d.geometry.PointCloud:
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=float))
    return p
