"""
phase4_segment.py  --  Stage C, steps 1-3.

Auto-detect the part axes, segment leg_1 / leg_2 / overhang_surface, identify
each leg's INNER wall (the face toward the slot), and fit RANSAC reference
planes to the two inner walls and the overhang surface.

Operates in the aligned Nominal frame (everything is co-registered there), so
the world axes coincide with the part axes; PCA is used only as a fallback when
the bbox extents are ambiguous.

Geometry, established from the nominal solids:
    * vertical (build) axis  = the largest bbox extent.
    * width  axis (Leg1->Leg2) = the horizontal axis whose end-slab splits into
      two clusters (the legs flanking the slot).
    * depth  axis = the remaining axis.
    * the "open end" along the vertical axis is where the two legs live; the
      other end is the solid connecting block. The overhang surface is the
      horizontal face bridging the legs at the inner end of the slot.

Public API:
    segment_legs_overhang(geometry) -> Segmentation
"""

from __future__ import annotations

import os
# sklearn/joblib core-detection fails on some Windows setups (slow subprocess +
# warning); pin it before sklearn is imported.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

from dataclasses import dataclass, field

import numpy as np
import open3d as o3d
from sklearn.cluster import KMeans

import config as C
from common import LOG, angle_between, make_pcd, points_of


@dataclass
class Plane:
    normal: np.ndarray            # unit normal
    d: float                      # plane offset (n.x + d = 0)
    centroid: np.ndarray          # centroid of inlier points
    n_inliers: int


@dataclass
class Segmentation:
    points: np.ndarray                       # Nx3 source points
    V_dir: np.ndarray                        # vertical/build axis (unit)
    W_dir: np.ndarray                        # slot-width axis Leg1->Leg2 (unit)
    D_dir: np.ndarray                        # slot-depth axis (unit)
    W_mid: float                             # slot centre along W
    slot_halfwidth: float
    v_overhang: float                        # overhang level along V
    open_end: str                            # 'lo' or 'hi' along V
    masks: dict = field(default_factory=dict)   # label -> bool mask into points
    planes: dict = field(default_factory=dict)  # 'overhang'/'leg1_wall'/'leg2_wall'
    ok: bool = True
    note: str = ""


# --------------------------------------------------------------------------
# axis detection
# --------------------------------------------------------------------------
def _slot_gap(coords: np.ndarray):
    """KMeans(2) split of 1-D coords -> (gap, centre, halfwidth) for a slot.

    gap = clear distance between the inner edges of the two clusters
    (negative if the two clusters overlap, i.e. no slot here)."""
    if len(coords) < 10:
        return -1e9, 0.0, 0.0
    km = KMeans(n_clusters=2, n_init=4, random_state=0).fit(coords.reshape(-1, 1))
    a = coords[km.labels_ == 0]
    b = coords[km.labels_ == 1]
    if len(a) == 0 or len(b) == 0:
        return -1e9, 0.0, 0.0
    if a.mean() > b.mean():
        a, b = b, a                       # a = lower cluster
    gap = b.min() - a.max()
    centre = 0.5 * (a.max() + b.min())
    return float(gap), float(centre), float(gap / 2.0)


def detect_axes(points: np.ndarray):
    """Return (V_dir, W_dir, D_dir, W_mid, slot_halfwidth, v_overhang, open_end)."""
    mn, mx = points.min(0), points.max(0)
    ext = mx - mn
    basis = np.eye(3)

    v_ax = int(np.argmax(ext))            # vertical = tallest extent
    horiz = [a for a in range(3) if a != v_ax]

    best = None                            # (gap, h_ax, end, centre, halfwidth, v_oh)
    for end in ("lo", "hi"):
        if end == "lo":
            sel = points[:, v_ax] < mn[v_ax] + C.END_SLICE_FRAC * ext[v_ax]
        else:
            sel = points[:, v_ax] > mx[v_ax] - C.END_SLICE_FRAC * ext[v_ax]
        slab = points[sel]
        if len(slab) < 20:
            continue
        for h in horiz:
            gap, centre, half = _slot_gap(slab[:, h])
            if best is None or gap > best[0]:
                best = (gap, h, end, centre, half)

    if best is None or best[0] <= 0:
        LOG.warning("    slot not clearly detected; using bbox-centre fallback")
        h = horiz[int(np.argmin([ext[horiz[0]], ext[horiz[1]]]))]
        centre = 0.5 * (mn[h] + mx[h])
        half = 0.1 * ext[h]
        open_end = "lo"
    else:
        _, h, open_end, centre, half = best

    w_ax = h
    d_ax = [a for a in horiz if a != w_ax][0]

    # Overhang level along V: the notch CEILING -- the horizontal surface that
    # bridges the legs at the inner end of the slot. Inside the slot (narrow W
    # band) the only solid faces are the ceiling and the part's far outer face;
    # the ceiling is the one nearest the OPEN end, so we 2-cluster the in-slot V
    # values and take the cluster centre closest to the open-end extreme.
    in_slot = np.abs(points[:, w_ax] - centre) < max(half * C.OVERHANG_W_FRAC, 1e-6)
    vals = points[in_slot, v_ax]
    open_pos = mn[v_ax] if open_end == "lo" else mx[v_ax]
    if len(vals) > 20:
        km = KMeans(n_clusters=2, n_init=4, random_state=0).fit(vals.reshape(-1, 1))
        centers = km.cluster_centers_.ravel()
        v_oh = float(centers[np.argmin(np.abs(centers - open_pos))])
    elif len(vals) > 0:
        v_oh = float(np.median(vals))
    else:
        v_oh = float(0.5 * (mn[v_ax] + mx[v_ax]))

    return (basis[v_ax], basis[w_ax], basis[d_ax],
            float(centre), float(abs(half)), v_oh, open_end)


# --------------------------------------------------------------------------
# plane fitting (RANSAC via open3d)
# --------------------------------------------------------------------------
def _fit_plane(pts: np.ndarray, expect_normal: np.ndarray | None = None) -> Plane | None:
    if len(pts) < max(C.PLANE_RANSAC_N, 10):
        return None
    pcd = make_pcd(pts)
    model, inliers = pcd.segment_plane(C.PLANE_DIST_THRESH, C.PLANE_RANSAC_N,
                                       C.PLANE_NUM_ITER)
    a, b, c, d = model
    n = np.array([a, b, c], float)
    nn = np.linalg.norm(n)
    if nn < 1e-9:
        return None
    n /= nn
    d /= nn
    if expect_normal is not None and np.dot(n, expect_normal) < 0:
        n, d = -n, -d                      # orient consistently
    inl = pts[inliers]
    return Plane(n, float(d), inl.mean(0), len(inliers))


def _fit_plane_pca(pts: np.ndarray, expect_normal: np.ndarray | None = None) -> Plane | None:
    """Least-squares (PCA) plane through `pts`: normal = smallest-variance axis.
    Used for the inner walls, where the slab is overwhelmingly the wall itself --
    PCA is stable and captures the wall's lean, whereas RANSAC on a thin slab can
    latch onto a competing sub-plane (e.g. a depth end-face) when inliers are
    few (seen on 4P2)."""
    if len(pts) < max(C.PLANE_RANSAC_N, 10):
        return None
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    n = vt[2]
    nn = np.linalg.norm(n)
    if nn < 1e-9:
        return None
    n = n / nn
    if expect_normal is not None and np.dot(n, expect_normal) < 0:
        n = -n
    return Plane(n, float(-np.dot(n, c)), c, len(pts))


# --------------------------------------------------------------------------
# main segmentation
# --------------------------------------------------------------------------
def segment_legs_overhang(geometry) -> Segmentation:
    pts = points_of(geometry)
    V, W, D, W_mid, half, v_oh, open_end = detect_axes(pts)

    w = pts @ W
    v = pts @ V

    # legs sit on the open-end side of the overhang level along V
    if open_end == "lo":
        leg_side = v < v_oh + 0.05 * (pts @ V).ptp()
    else:
        leg_side = v > v_oh - 0.05 * (pts @ V).ptp()

    # Bound each leg at its slot EDGE (W_mid -/+ half), not the slot centre.
    # Leg material lies outside the slot; using the centre let a wide slot
    # (4P2/6P1, half=2-3) pull the overhang-ceiling strip (which spans the slot
    # at V~v_oh) into the leg, so the inner-wall fit latched onto that horizontal
    # sliver instead of the vertical wall. (For 2PR's narrow slot edge~centre.)
    leg1_mask = leg_side & (w < W_mid - half)      # lower-W leg  = Leg 1
    leg2_mask = leg_side & (w > W_mid + half)      # higher-W leg = Leg 2
    # overhang ceiling: inside the slot in W AND near the overhang level in V,
    # so we don't pick up the block's outer top face (also normal +/-V).
    v_extent = v.max() - v.min()
    overhang_mask = ((np.abs(w - W_mid) < max(half * C.OVERHANG_W_FRAC, 1e-6)) &
                     (np.abs(v - v_oh) < max(C.OVERHANG_V_FRAC * v_extent,
                                             3 * C.PLANE_DIST_THRESH)))

    seg = Segmentation(pts, V, W, D, W_mid, half, v_oh, open_end,
                       masks={"leg_1": leg1_mask, "leg_2": leg2_mask,
                              "overhang_surface": overhang_mask})

    if leg1_mask.sum() < 20 or leg2_mask.sum() < 20 or overhang_mask.sum() < 20:
        seg.ok = False
        seg.note = "insufficient points in one or more segments"
        LOG.warning("    segmentation weak: leg1=%d leg2=%d overhang=%d",
                    leg1_mask.sum(), leg2_mask.sum(), overhang_mask.sum())

    # --- inner walls: slab of each leg nearest the slot centre ----------
    def inner_wall(mask, toward_high):
        lp = pts[mask]
        if len(lp) < 20:
            return None, None
        lw = lp @ W
        width = lw.max() - lw.min()
        band = max(width * C.WALL_BAND_FRAC, C.PLANE_DIST_THRESH * 2)
        if toward_high:                    # inner edge is the high-W side
            sub = lp[lw > lw.max() - band]
        else:
            sub = lp[lw < lw.min() + band]
        return sub, _fit_plane_pca(sub, expect_normal=W)

    w1_pts, p1 = inner_wall(leg1_mask, toward_high=True)    # leg1 inner = +W face
    w2_pts, p2 = inner_wall(leg2_mask, toward_high=False)   # leg2 inner = -W face
    p_oh = _fit_plane(pts[overhang_mask], expect_normal=V)

    seg.masks["leg_1_inner"] = (w1_pts is not None)
    seg.planes = {"leg1_wall": p1, "leg2_wall": p2, "overhang": p_oh}
    seg.masks["_inner1_pts"] = w1_pts
    seg.masks["_inner2_pts"] = w2_pts

    if p1 is None or p2 is None or p_oh is None:
        seg.ok = False
        seg.note = (seg.note + "; plane fit failed").strip("; ")

    LOG.info("    axes V/W/D=%s/%s/%s  slot@%.2f half=%.2f  open=%s | "
             "leg1=%d leg2=%d oh=%d",
             "XYZ"[int(np.argmax(np.abs(V)))], "XYZ"[int(np.argmax(np.abs(W)))],
             "XYZ"[int(np.argmax(np.abs(D)))], W_mid, half, open_end,
             leg1_mask.sum(), leg2_mask.sum(), overhang_mask.sum())
    return seg


def leg_points(geometry) -> np.ndarray:
    """Convenience for Stage-A leg-datum refinement: just the two leg regions."""
    seg = segment_legs_overhang(geometry)
    m = seg.masks["leg_1"] | seg.masks["leg_2"]
    return seg.points[m]


if __name__ == "__main__":
    from phase0_match import match_parts
    from phase1_normalize import load_and_normalize
    parts = match_parts()
    pid = next(iter(parts))
    loaded = load_and_normalize(pid, parts[pid])
    for s, ld in loaded.items():
        LOG.info("segmenting %s / %s", pid, s)
        segment_legs_overhang(ld.cloud)
