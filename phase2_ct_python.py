"""
phase2_ct_python.py — Laptop path: extract surface mesh from CT TIFF stack.
Uses scikit-image (Otsu threshold + marching cubes) as a VGStudio substitute.
Output is an STL mesh comparable to the nominal CAD model.
"""

import numpy as np
from pathlib import Path
from typing import Optional

import tifffile
from skimage.filters import threshold_otsu
from skimage.measure import marching_cubes
from skimage.morphology import binary_closing, ball
import trimesh

import config


def load_tiff_stack(tiff_dir: Path, max_slices: int = 0, step: int = 1) -> np.ndarray:
    """
    Load sorted .tif slices from a directory into a 3D numpy array (Z, Y, X).
    max_slices=0 means load all; step>1 subsamples for speed.
    """
    tif_files = sorted(tiff_dir.glob(config.CT_TIFF_PATTERN))
    if not tif_files:
        raise FileNotFoundError(f"No TIF files found in {tiff_dir}")

    if max_slices > 0:
        tif_files = tif_files[:max_slices]
    tif_files = tif_files[::step]

    print(f"  Loading {len(tif_files)} TIF slices from {tiff_dir.name}...")

    sample = tifffile.imread(str(tif_files[0]))
    dtype = sample.dtype
    volume = np.zeros((len(tif_files), *sample.shape), dtype=dtype)

    for i, f in enumerate(tif_files):
        volume[i] = tifffile.imread(str(f))
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(tif_files)} slices loaded")

    print(f"  Volume shape: {volume.shape}, dtype: {dtype}, "
          f"range: [{volume.min()}, {volume.max()}]")
    return volume


def extract_surface(volume: np.ndarray, voxel_size_mm: float = 0.00983) -> trimesh.Trimesh:
    """
    Threshold the CT volume (Otsu) then run marching cubes to extract the
    part surface as a triangle mesh.
    """
    print("  Computing Otsu threshold...")
    # Downsample slightly for threshold computation to save RAM
    sample = volume[::2, ::2, ::2]
    thresh = threshold_otsu(sample)
    print(f"  Otsu threshold: {thresh:.1f}")

    print("  Binarising volume...")
    binary = volume > thresh

    print("  Morphological closing (radius=1) to fill small gaps...")
    binary = binary_closing(binary, ball(1))

    print("  Running marching cubes...")
    verts, faces, normals, _ = marching_cubes(
        binary,
        level=0.5,
        spacing=(voxel_size_mm, voxel_size_mm, voxel_size_mm),
    )

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    print(f"  Surface mesh: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces")
    return mesh


def process_ct_dataset(
    name: str,
    tiff_dir: Path,
    output_dir: Path,
    voxel_size_mm: float = 0.00983,
    subsample_step: int = 2,
    max_slices: int = 0,
) -> Optional[Path]:
    """
    Full pipeline: load TIFFs → extract surface → save STL.
    subsample_step=2 skips every other slice to halve RAM usage on a laptop.
    """
    out_path = output_dir / f"{name}_ct_surface.stl"

    if out_path.exists():
        print(f"  [SKIP] {name}: output already exists — {out_path.name}")
        return out_path

    if not tiff_dir.exists():
        print(f"  [SKIP] {name}: TIFF dir not found — {tiff_dir}")
        return None

    tif_count = len(list(tiff_dir.glob(config.CT_TIFF_PATTERN)))
    print(f"\n  Processing CT dataset '{name}': {tif_count} slices in {tiff_dir.name}")

    try:
        volume = load_tiff_stack(tiff_dir, max_slices=max_slices, step=subsample_step)
        effective_voxel = voxel_size_mm * subsample_step
        mesh = extract_surface(volume, voxel_size_mm=effective_voxel)

        del volume   # free RAM

        mesh.export(str(out_path))
        size_mb = out_path.stat().st_size / 1e6
        print(f"  [OK]   Saved CT surface -> {out_path.name} ({size_mb:.1f} MB)")
        return out_path

    except MemoryError:
        print(f"  [ERROR] Out of memory for '{name}'. Try increasing subsample_step.")
        return None
    except Exception as e:
        print(f"  [ERROR] CT processing failed for '{name}': {e}")
        return None


def run_all(output_dir: Path | None = None) -> dict[str, Path]:
    """
    Process all configured CT TIFF directories.
    For the large 6mm stack (1935 slices × 14 MB each ≈ heavy),
    we use step=4 on the laptop to keep RAM under ~4 GB.
    """
    output_dir = output_dir or config.OUT_CT
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n-- Phase 2A: CT Surface Extraction (Python / laptop) --")

    # 6mm Image Stack: 1935 slices @ ~9.8 µm voxel
    # step=4 → process every 4th slice → ~484 effective slices (fast & lean)
    results: dict[str, Path] = {}

    # 6mm Image Stack: 1935 reconstructed volume slices (1769×1346 uint16 = 4.76 MB each)
    # step=8 → ~242 slices → ~1.15 GB RAM (safe on 16 GB laptop with ~3 GB free)
    #
    # 3218: TIFs are raw CT projections (2400×3072), NOT reconstructed volume slices.
    # They cannot be fed directly into marching cubes. The .vol file (8.7 GB) would
    # require VGStudio MAX or a CBCT reconstruction library.
    # → Skip 3218 in laptop mode; VGStudio handles it at the university PC.
    dataset_settings = {
        "6mm": {
            "tiff_dir":      config.CT_TIFF_DIRS["6mm"],
            "voxel_size_mm": 0.00983,
            "subsample_step": 8,
        },
    }

    for name, settings in dataset_settings.items():
        path = process_ct_dataset(name, output_dir=output_dir, **settings)
        if path:
            results[name] = path

    print(f"\n  CT extraction complete: {len(results)} dataset(s)")
    return results


if __name__ == "__main__":
    config.print_config()
    run_all()
