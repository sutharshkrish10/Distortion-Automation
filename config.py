"""Central configuration for the overhang distortion pipeline.

All tunable values live here so no magic numbers sit inside the phase modules.
Paths are resolved relative to this file, so the project folder can be moved
without breaking anything.

Geometry (see Data Set/Leg and Overhang/Slide1.JPG and Slide2.JPG):
the U-shaped overhang artifact is a solid block with a slot.
  - Leg 1 / Leg 2: the two upstanding walls flanking the slot.
  - Overhang surface: the horizontal face of the slot between the legs.
  - Distortion angle: interior angle between the overhang plane and each leg's
    inner wall, nominally 90 deg; distortion = angle - nominal angle.
  - Overhang length: inner-wall-to-inner-wall span across the slot.
"""

import os
from pathlib import Path

# Paths
PROJECT_DIR = Path(__file__).resolve().parent
DATA_ROOT   = PROJECT_DIR.parent / "Data Set"
OUTPUT_ROOT = PROJECT_DIR / "Output"

# Sibling source folders inside DATA_ROOT (one file per part in each).
SOURCE_DIRS = {
    "nominal": DATA_ROOT / "Nominal STL Files",
    # Clean part-only CT surface meshes (1.8-12 M tris) whose proportions match
    # the nominal (auto-scale ~1.0). The "CT Surface mesh" folder held the raw
    # full-resolution reconstructions (15-22 GB, oversized and mis-shaped) which
    # broke 4P2/6P1 registration, so we switched away from those.
    "ct":      DATA_ROOT / "actual CT scan",
    # Zephyr photogrammetry is now delivered as cropped surface STL meshes (part
    # only, turntable removed) rather than raw fused dense point clouds. They are
    # sampled to a cloud in phase 1 just like the CT/nominal meshes.
    "zephyr":  DATA_ROOT / "Zephyr STL",
}

# Reference slides (for documentation / annotated-plot styling reference).
SLIDE_DIR = DATA_ROOT / "Leg and Overhang" / "Leg and Overhang"

# Part matching
# Parts are matched by the ID embedded in the filename. Filenames are messy
# ("2mm", "4 mm", "6mm", "Mesh from 2PR.stl"), so we never hard-code spacing.
# A part is identified by its overhang size in mm; we capture the leading digit
# and map it to the canonical part ID used in the Nominal STL names.
PART_SIZE_RE = r"(?<!\d)([246])\s*(?:mm|P)"   # 2/4/6 followed by 'mm' or 'P...'
SIZE_TO_PARTID = {"2": "2PR", "4": "4P2", "6": "6P1"}   # canonical IDs

# Phase 1: load and unit normalization
# If an as-built source's bbox-diagonal differs from the Nominal's by more than
# this tolerance, a scale factor is applied (diag_nominal / diag_source),
# unless an explicit override is given below.
SCALE_AUTODETECT_TOL = 0.05          # 5 % bbox-diagonal mismatch triggers scaling
SCALE_OVERRIDE = {                   # {(part_id, source): factor}; [] = auto
    # ("2PR", "zephyr"): 0.71,
}
# CT is metric (real millimetres), so it must NOT be bbox-auto-scaled: the parts
# carry a RAISED ID STAMP ("2PR"/"4P2"/"6P1") that is present in the CT scan but
# absent from the clean nominal design. The stamp inflates CT's bbox diagonal,
# which fooled the auto-scaler into squashing CT to fit (breaking registration
# and erasing real deviation). Pin CT to true scale; the stamp then shows only as
# a small localized bump. An explicit SCALE_OVERRIDE still wins if ever needed.
CT_TRUE_SCALE = True
# The raised ID stamp sits ABOVE the design build-height. Trim CT material more
# than this margin above the nominal's height (in CT's own frame) before
# registration: the stamp then neither throws the coarse fit into a bad basin nor
# inflates the deviation stats (the trimmed cloud is what phase 3 measures). Set
# False to keep the stamp.
CT_STAMP_TRIM = True
CT_STAMP_TRIM_MARGIN_MM = 0.5        # keep up to design_height + this, drop above
# Keep the trimmed ID stamp in the SAVED aligned-CT overlay (aligned_ct.ply) so
# renders/overlays show the real printed part, while alignment, segmentation and
# every reported number still use the stamp-trimmed cloud. This is display only
# and changes no measurement. (The saved .ply is read only by the viewer, not by
# the segmentation/deviation phases, which use the in-memory trimmed cloud.)
CT_KEEP_STAMP_IN_OUTPUT = True

# Zephyr photogrammetry meshes exported WITH a real-world scale reference (scale
# bar / known reference distance at capture) are already metric, like CT. Pin
# them to true scale instead of the bbox-diagonal auto-scale: photogrammetry
# edge-fuzz + the tall/wide bbox inflate the diagonal, so the auto-scaler shrinks
# the part ~10% (e.g. 4mm Zephyr x0.888), which undersizes the slot (true 4.21 mm
# read as 3.77) and pushes the sparse cloud into a bad registration basin. With
# true scale the slot matches a direct in-Zephyr measurement + CT. REQUIRES the
# Zephyr export to be metric; set False for an unscaled/arbitrary-scale export (it
# falls back to auto-scale). An explicit SCALE_OVERRIDE still wins.
ZEPHYR_TRUE_SCALE = True

# Mesh sampling. Poisson-disk gives even spacing (per spec) but is slow
# on dense meshes (e.g. the 6M-vertex CT); "uniform" is much faster and fine for
# registration/deviation. Switch here if runs are too slow.
SAMPLE_METHOD = os.environ.get("SAMPLE_METHOD", "poisson")   # "poisson" | "uniform"
POISSON_SAMPLES = {
    "nominal": 60000,
    "ct":      120000,
    "zephyr":  120000,
}
NORMAL_RADIUS_FACTOR = 3.0           # normal-estimation radius = factor * voxel
NORMAL_MAX_NN = 30

# Phase 2: registration (Nominal is the fixed reference frame)
VOXEL_SIZE = 0.4                     # mm; downsample size for coarse stage
FPFH_RADIUS_FACTOR = 5.0             # FPFH feature radius = factor * voxel
RANSAC_DIST_FACTOR = 1.5             # global-registration inlier dist = factor*voxel
RANSAC_N = 4                         # points per RANSAC sample
RANSAC_MAX_ITER = 4_000_000
RANSAC_CONFIDENCE = 0.999
# open3d's RANSAC reads a global RNG; pin it so registration (and every metric
# derived from the alignment) is reproducible run to run. RANSAC_SEED also seeds
# the mesh-to-cloud Poisson sampling (common.mesh_to_cloud) so each part's cloud is
# independent of batch position.
RANSAC_SEED = 0

# Coarse RANSAC occasionally lands a near-symmetric part in a flipped/rotated
# basin that ICP cannot escape; the sparsest as-built clouds (e.g. the 4 mm
# Zephyr STL) are the most prone to it. We therefore run the coarse stage from
# several fixed seeds and keep the one with the best nominal-to-aligned coverage
# (fraction of nominal points with an as-built point within COVERAGE_TOL mm).
# Coverage is used, not ICP fitness/rmse, because a flipped fit can score a
# deceptively good rmse while leaving most of the nominal surface uncovered.
# Fixed seed list => still fully deterministic.
RANSAC_SEEDS = (0, 1, 2, 3)
REG_COVERAGE_TOL = 0.5               # mm; nominal point counts as covered within this

ICP_DIST_FACTOR = 2.0                # fine ICP max corr. dist = factor * voxel
ICP_MAX_ITER = 100

# Stage A step 5: a second ICP using ONLY the segmented leg regions, so the
# distorted overhang does not bias the datum. Run on the full-resolution clouds.
LEG_REFINE = True
LEG_REFINE_DIST_FACTOR = 1.5

# Stage A step 5b: a gentle final full-cloud ICP for ZEPHYR only. Partial
# photogrammetry clouds settle a few degrees tilted after the leg datum (the
# overhang/base are under-captured, so the leg ICP can't fully pin the
# orientation). This small refine pulls that residual out. Guarded HARD so it can
# never become the bogus ~10 deg de-roll: applied only if the move is below
# ZEPHYR_FINAL_ICP_MAX_DEG AND it does not reduce nominal coverage.
ZEPHYR_FINAL_ICP = True
ZEPHYR_FINAL_ICP_MAX_DEG = 6.0       # reject corrections larger than this (a real
                                     # residual is ~2-3 deg; bigger = bad basin)
ZEPHYR_FINAL_ICP_MIN_COVERAGE_GAIN = -0.005  # must not drop coverage (small slack)
# A ~3 deg tilt displaces the part top by ~0.8 mm, so the fine ICP's tight 0.8 mm
# window can't see it and converges to a different (still-tilted) pose. Use a
# wider correspondence distance here so the refine reaches the fully-aligned pose.
ZEPHYR_FINAL_ICP_DIST_MM = 1.5

# Stage A step 6: "upright datum". A partial, near-symmetric as-built surface
# (the cropped Zephyr mesh) is rotationally under-constrained, so RANSAC/ICP can
# settle with the part body ROLLED several degrees about the depth axis vs the
# nominal, which makes the overlay look tilted and inflates the deviation map.
# After registration we re-roll each source's principal-axis frame onto the
# nominal's (~world) frame with a minimal, flip-free rotation, but ONLY when the
# residual body roll exceeds UPRIGHT_MAX_ROLL_DEG. Well-aligned sources (CT,
# nominal) fall under the threshold and are left exactly as-is, so their
# authoritative per-leg readings do not move; slot_closure and span are
# rotation-invariant either way.
UPRIGHT_REFINE = True
UPRIGHT_MAX_ROLL_DEG = 3.0
# Only apply the correction for a plausible residual roll. A genuine residual
# from coarse-RANSAC + ICP + leg-datum refinement is small (single digits, e.g.
# 2PR Zephyr ~9 deg); a much larger "roll" means the PCA build-axis estimate was
# thrown off by a sparse/partial cloud (e.g. the 4mm Zephyr, which reads ~35 deg
# yet is already upright); those are rejected, leaving the registration as-is.
UPRIGHT_MAX_APPLY_DEG = 20.0
# The magnitude cap alone is not enough: on the full-height 4mm Zephyr re-crop the
# PCA build axis reads ~12 deg (inside the cap) yet the part is already upright, so
# the "correction" TILTS it and inflates the deviation/overlay. Therefore accept a
# de-roll only if it does not REDUCE nominal coverage (a genuine de-roll snaps more
# of the surface onto nominal; a bogus one pulls it away). Small negative slack
# tolerates measurement noise on a correct, coverage-neutral de-roll.
UPRIGHT_MIN_COVERAGE_GAIN = -0.01

# Phase 3: surface deviation
DEVIATION_PAIRS = [                  # (moving/source, reference); sign by ref surface
    ("ct", "nominal"),
    ("zephyr", "nominal"),
    ("ct", "zephyr"),
]
HEATMAP_CLIP = 1.0                   # mm; +/- range for the deviation colour map
HIST_BINS = 80

# Phase 4: segmentation (legs + overhang) in the aligned Nominal frame
END_SLICE_FRAC = 0.15                # fraction of vertical extent used as an end slab
BIMODAL_BINS = 24                    # histogram bins for slot-gap detection
WALL_BAND_FRAC = 0.30                # inner-wall slab thickness, as frac of leg width
OVERHANG_W_FRAC = 0.80               # keep |W-centre| < frac*half_slot for overhang fit
OVERHANG_V_FRAC = 0.12               # keep |V-overhang_level| < frac*V_extent (avoid block top)
DBSCAN_EPS_FACTOR = 3.0              # DBSCAN eps = factor * voxel (fallback cleanup)
DBSCAN_MIN_POINTS = 20

# RANSAC plane fits (open3d segment_plane)
PLANE_DIST_THRESH = 0.20             # mm
PLANE_RANSAC_N = 3
PLANE_NUM_ITER = 2000

# Phase 5: measurement
# Leg distortion is reported as the rotation of each leg's inner-wall plane
# relative to the SAME leg's nominal inner wall (in the co-registered frame),
# signed so + = leg leaned inward / slot closing. This needs only the walls, so
# Zephyr (which never captures the recessed overhang ceiling) is still measurable.
#
# The legacy overhang-vs-wall angle is also reported, but only where the overhang
# ceiling was actually captured: if the fitted overhang plane's normal tilts more
# than this from the build axis V, that angle is treated as unmeasured (NaN)
# rather than reported as a misleading number (e.g. Zephyr's ~81 deg-off plane).
OVERHANG_MAX_TILT_DEG = 30.0

# Per-leg result is ALSO reported as an absolute interior "leg-vs-span" angle:
# nominal corner = 90 deg, and a leg leaning OUTWARD (slot opening) reads > 90.
# Since distortion_leg is signed + = leaned inward/closing, the absolute angle is
# NOMINAL_LEG_ANGLE_DEG - distortion.
NOMINAL_LEG_ANGLE_DEG = 90.0

# Overhang span = the true slot width, measured between the two inner-wall FACES
# (the slot-facing edges of each leg's inner band), not the band centroids. A
# robust percentile of each band's W is used as its face so stray edge points
# don't skew it; the nominal parts then recover their design slot of 2 / 4 / 6 mm.
SPAN_FACE_PCT = 98.0

# Per-leg leg-lean distortion is flagged reliable by SOURCE, not by alignment
# fitness. Photogrammetry (Zephyr) physically cannot see the recessed/grazing
# inner wall, so its per-leg values are still REPORTED but tagged
# perleg_reliable=False (trust slot_closure for Zephyr). CT/nominal see both
# walls, and per-leg lean + slot_closure are registration-invariant anyway, so
# CT's lower ICP fitness (~0.65-0.77, from genuine ~1mm part distortion) does NOT
# make its per-leg readings unreliable.

# Phase 6: reporting
REGISTRATION_CSV = OUTPUT_ROOT / "registration_report.csv"
COMPARISON_CSV   = OUTPUT_ROOT / "comparison_report.csv"
DISTORTION_CSV   = OUTPUT_ROOT / "distortion_report.csv"

# Colours for segment-coloured clouds / annotated plots
SEG_COLORS = {
    "leg_1":            (0.85, 0.20, 0.20),
    "leg_2":            (0.20, 0.40, 0.85),
    "overhang_surface": (0.20, 0.75, 0.30),
    "other":            (0.70, 0.70, 0.70),
}

SOURCES = ("nominal", "ct", "zephyr")
