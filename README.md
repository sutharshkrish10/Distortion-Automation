# Overhang Specimen Distortion Pipeline

Brings three representations of each U-channel overhang specimen into a common,
**leg-based** coordinate frame, computes surface deviations between them, and
measures the per-leg **distortion angles** and the **overhang length**.

Three sources per part:

| Source   | Folder (`Data Set/`)   | Type            | Role                         |
|----------|------------------------|-----------------|------------------------------|
| Nominal  | `Nominal STL Files`    | STL mesh        | reference CAD / master frame |
| CT       | `CT Surface mesh`      | STL mesh        | as-built (CT reconstruction) |
| Zephyr   | `Dense Point Cloud`    | PLY point cloud | as-built (photogrammetry)    |

Parts are matched by **ID in the filename** with a regex (`2 mm`/`4 mm`/`6 mm`
-> `2PR`/`4P2`/`6P1`), so new parts drop in with no code change. Filenames are
intentionally inconsistent (`2mm`, `4 mm`, `Mesh from 2PR.stl`) — never hard-code
spacing.

## Geometry & measurement definitions

See `Data Set/Leg and Overhang/Leg and Overhang/Slide1.JPG` and `Slide2.JPG`.

Each part is a solid block with a central slot/notch. The two upstanding walls
flanking the slot are **Leg 1** (lower side of the slot-width axis) and **Leg 2**
(upper side). The horizontal face bridging the legs at the inner end of the slot
is the **overhang surface**.

- **Distortion Angle Leg 1** — interior angle between the overhang-surface plane
  and Leg 1's **inner** vertical wall (the face toward the slot). Nominally 90°.
- **Distortion Angle Leg 2** — same, for Leg 2. Measured independently so
  asymmetric distortion is captured.
- **distortion Δ** — `angle − Nominal angle` for that same corner.
- **Overhang Length** — inner-wall-to-inner-wall span across the slot, along the
  slot-width axis, at the overhang level (Slide2).

## Phases (each file is importable and individually runnable)

| File                  | Stage | Does                                                       |
|-----------------------|-------|------------------------------------------------------------|
| `config.py`           | —     | **all tunables** (voxel/sample/RANSAC/ICP/segmentation…)   |
| `common.py`           | —     | logging, I/O, sampling, normals, geometry helpers          |
| `phase0_match.py`     | —     | discover + match files into parts (`match_parts()`)        |
| `phase1_normalize.py` | A1-2  | load + unit/scale normalization (`load_and_normalize()`)   |
| `phase2_register.py`  | A3-6  | coarse RANSAC → fine ICP → **leg-datum ICP** (`register()`)|
| `phase3_deviation.py` | B     | signed deviations + heat-maps + histograms (`signed_deviation()`) |
| `phase4_segment.py`   | C1-3  | axis detect, segment legs/overhang, plane fits (`segment_legs_overhang()`) |
| `phase5_measure.py`   | C4-5  | distortion angles + span (`measure_distortion_and_span()`) |
| `phase6_report.py`    | —     | aggregate CSVs, segment clouds, annotated plots            |
| `align_compare_distortion.py` | — | **main orchestrator** looping parts × sources          |

Because each phase is its own module, you can tweak or rerun a single stage
(e.g. re-tune segmentation in `config.py` and run `python phase4_segment.py`)
without touching the rest.

### Stage A — leg-based alignment (Nominal is the fixed reference)
1. Load Nominal/CT/Zephyr; detect unit/scale mismatch by bbox-diagonal ratio and
   apply a (configurable) scale factor.
2. Poisson-disk sample the meshes to clouds; estimate normals (Zephyr used directly).
3. **Coarse**: voxel-downsample + FPFH + RANSAC global registration (no
   pre-alignment assumed).
4. **Fine**: point-to-plane ICP seeded from the coarse transform.
5. **Leg-datum refinement**: a second ICP using only the segmented leg regions,
   so the distorted overhang doesn't bias the datum.
6. Save aligned CT (STL) + aligned Zephyr (PLY) to `Output/<PartID>/`, plus each
   4×4 transform and its fitness / inlier-RMSE (`transforms.json`).

### Stage B — surface deviation
Signed point-to-surface distances for CT↔Nominal, Zephyr↔Nominal, CT↔Zephyr
(sign from the reference surface normal). Reports mean/std/RMS/min/max/p95 and
saves deviation-coloured heat-map PLYs + histogram PNGs.

### Stage C — segmentation + measurement
Axis auto-detection (vertical = largest extent; slot-width = the horizontal axis
whose end-slab splits into two leg clusters; depth = remainder), segmentation
into `leg_1`/`leg_2`/`overhang_surface`, RANSAC plane fits to both inner walls
and the overhang, then the two distortion angles, their Δ vs Nominal, and the
overhang length.

## Outputs (`Output/`)

- `registration_report.csv` — per part & source: fitness, inlier RMSE, applied
  scale, point counts.
- `comparison_report.csv` — per part & pair: mean/std/RMS/min/max/p95 deviation.
- `distortion_report.csv` — per part & source: Distortion Angle Leg 1/Leg 2,
  Δ vs Nominal, Overhang Length (+Δ).
- `Output/<PartID>/`: aligned clouds/meshes, `transforms.json`, deviation
  heat-maps + histograms, `segments_<src>.ply`, and `annotated_<src>.png`
  (segmented legs, overhang, fitted inner-wall planes, marked corners, span —
  mirroring Slide1/Slide2).

## Install

```
pip install open3d trimesh numpy scipy scikit-learn pandas matplotlib
```

## Usage

```
python align_compare_distortion.py                 # full batch, all parts
python align_compare_distortion.py --parts 2PR     # one part
python align_compare_distortion.py --skip-deviation # skip Stage B (faster)
python align_compare_distortion.py --visualize      # interactive open3d windows
```

With `--visualize`, per part you get interactive open3d windows (drag=rotate,
scroll=zoom): the 3-way overlay (Nominal grey / CT red / Zephyr blue), then a
**dedicated pair of windows for each comparison** — Nominal-vs-CT and
Nominal-vs-Zephyr — first an *overlay* (Nominal grey + as-built coloured), then a
*signed-deviation heat-map* (blue = under/inside, white ~ 0, red = proud, clipped
to `HEATMAP_CLIP` mm), followed by the per-source segmentation view. Close each
window to advance to the next.

Individual phases:

```
python phase0_match.py        # show matched files
python phase4_segment.py      # segment the first matched part
```
- All thresholds live in `config.py`. The defaults are tuned for ~10–16 mm parts;
  re-tune `VOXEL_SIZE`, `PLANE_DIST_THRESH`, and the segmentation fractions if
  geometry scale changes.
- The batch never crashes on a bad/missing file — it logs a warning and skips.
