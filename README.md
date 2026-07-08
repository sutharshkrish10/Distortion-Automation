# Overhang Specimen Distortion Pipeline

Brings three representations of each U-channel overhang specimen into a common,
**leg-based** coordinate frame, computes surface deviations between them, and
measures the per-leg **distortion angles** and the **overhang length**.

Three sources per part:

| Source   | Folder (`Data Set/`)   | Type            | Role                         |
|----------|------------------------|-----------------|------------------------------|
| Nominal  | `Nominal STL Files`    | STL mesh        | reference CAD / master frame |
| CT       | `actual CT scan`       | STL mesh        | as-built (CT reconstruction) |
| Zephyr   | `Zephyr STL`           | PLY surface mesh | as-built (photogrammetry)   |

Parts are matched by their **ID in the filename** with a regex (`2 mm`, `4 mm`,
`6 mm` map to `2PR`, `4P2`, `6P1`), so new parts can be added without any code
change. The filenames are inconsistent (`2mm`, `4 mm`, `Mesh from 2PR.stl`), so
the matching never assumes a fixed spacing.

## Geometry and measurement definitions

See `Data Set/Leg and Overhang/Leg and Overhang/Slide1.JPG` and `Slide2.JPG`.

Each part is a solid block with a central slot. The two upstanding walls
flanking the slot are **Leg 1** (lower side of the slot-width axis) and **Leg 2**
(upper side). The horizontal face bridging the legs at the inner end of the slot
is the **overhang surface**.

- **Distortion Angle Leg 1**: interior angle between the overhang-surface plane
  and Leg 1's **inner** vertical wall (the face toward the slot). Nominally 90°.
- **Distortion Angle Leg 2**: same, for Leg 2. Measured independently so
  asymmetric distortion is captured.
- **distortion Δ**: `angle − nominal angle` for that same corner.
- **Overhang Length**: inner-wall-to-inner-wall span across the slot, along the
  slot-width axis, at the overhang level (Slide2).

## Phases (each file is importable and can be run on its own)

| File                  | Stage | Does                                                       |
|-----------------------|-------|------------------------------------------------------------|
| `config.py`           |       | **all tunables** (voxel/sample/RANSAC/ICP/segmentation)    |
| `common.py`           |       | logging, I/O, sampling, normals, geometry helpers          |
| `phase0_match.py`     |       | discover and match files into parts (`match_parts()`)      |
| `phase1_normalize.py` | A1-2  | load and unit/scale normalization (`load_and_normalize()`) |
| `phase2_register.py`  | A3-6  | coarse RANSAC, fine ICP, then **leg-datum ICP** (`register()`) |
| `phase3_deviation.py` | B     | signed deviations, heat-maps, histograms (`signed_deviation()`) |
| `phase4_segment.py`   | C1-3  | axis detect, segment legs/overhang, plane fits (`segment_legs_overhang()`) |
| `phase5_measure.py`   | C4-5  | distortion angles and span (`measure_distortion_and_span()`) |
| `phase6_report.py`    |       | aggregate CSVs, segment clouds, annotated plots            |
| `align_compare_distortion.py` | | **main orchestrator** looping over parts and sources   |

Because each phase is its own module, you can tweak or rerun a single stage
(e.g. re-tune segmentation in `config.py` and run `python phase4_segment.py`)
without touching the rest.

### Stage A: leg-based alignment (Nominal is the fixed reference)
1. Load Nominal/CT/Zephyr; detect a unit/scale mismatch by the bbox-diagonal
   ratio and apply a (configurable) scale factor.
2. Poisson-disk sample the meshes to clouds and estimate normals.
3. **Coarse**: voxel-downsample, FPFH features, RANSAC global registration (no
   pre-alignment assumed).
4. **Fine**: point-to-plane ICP seeded from the coarse transform.
5. **Leg-datum refinement**: a second ICP using only the segmented leg regions,
   so the distorted overhang does not bias the datum.
6. Save the aligned CT (STL) and aligned Zephyr (PLY) to `Output/<PartID>/`, plus
   each 4×4 transform and its fitness / inlier-RMSE (`transforms.json`).

### Stage B: surface deviation
Signed point-to-surface distances for CT vs Nominal, Zephyr vs Nominal, and CT vs
Zephyr (sign from the reference surface normal). Reports mean/std/RMS/min/max/p95
and saves deviation-coloured heat-map PLYs and histogram PNGs.

### Stage C: segmentation and measurement
Axis auto-detection (vertical = largest extent; slot-width = the horizontal axis
whose end-slab splits into two leg clusters; depth = the remainder), segmentation
into `leg_1`/`leg_2`/`overhang_surface`, RANSAC plane fits to both inner walls and
the overhang, then the two distortion angles, their Δ vs Nominal, and the overhang
length.

## Outputs (`Output/`)

- `registration_report.csv`: per part and source, the fitness, inlier RMSE,
  applied scale, and point counts.
- `comparison_report.csv`: per part and pair, the mean/std/RMS/min/max/p95
  deviation.
- `distortion_report.csv`: per part and source, the Distortion Angle Leg 1/Leg 2,
  Δ vs Nominal, and Overhang Length (with its Δ).
- `Output/<PartID>/`: aligned clouds/meshes, `transforms.json`, deviation
  heat-maps and histograms, `segments_<src>.ply`, and `annotated_<src>.png`
  (segmented legs, overhang, fitted inner-wall planes, marked corners, span),
  mirroring Slide1/Slide2.

## Install

```
pip install open3d trimesh numpy scipy scikit-learn pandas matplotlib pyvista
```

(`pyvista` is only needed for the 3D screenshots and `visualize_deviation.py`; the
core pipeline runs without it.)

## Usage

```
python align_compare_distortion.py                  # full batch, all parts
python align_compare_distortion.py --parts 2PR      # one part
python align_compare_distortion.py --skip-deviation # skip Stage B (faster)
python align_compare_distortion.py --visualize      # interactive open3d windows
```

With `--visualize`, each part opens interactive open3d windows (drag to rotate,
scroll to zoom): the 3-way overlay (Nominal grey / CT red / Zephyr blue), then a
pair of windows for each comparison (Nominal vs CT and Nominal vs Zephyr): first
an overlay (Nominal grey plus the as-built coloured), then a signed-deviation
heat-map (blue = under/inside, white ~ 0, red = proud, clipped to `HEATMAP_CLIP`
mm), followed by the per-source segmentation view. Close each window to advance
to the next.

Individual phases:

```
python phase0_match.py        # show matched files
python phase4_segment.py      # segment the first matched part
```

## Notes and caveats

- The Zephyr surface is photogrammetry and can arrive at an arbitrary scale and
  orientation; the scale is auto-estimated from the bbox-diagonal ratio to the
  Nominal, then registration refines it. If a Zephyr capture includes
  turntable/background geometry, crop it first or set a per-part factor in
  `config.SCALE_OVERRIDE`.
- All thresholds live in `config.py`. The defaults are tuned for roughly 10-16 mm
  parts; re-tune `VOXEL_SIZE`, `PLANE_DIST_THRESH`, and the segmentation fractions
  if the geometry scale changes.
- The batch does not stop on a bad or missing file; it logs a warning and skips
  that source.
