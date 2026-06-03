"""
config.py — Central configuration for distortion analysis pipeline.
Auto-detects whether VGStudio MAX is installed (university PC) or not (laptop).
"""

import os
import shutil
from pathlib import Path

# -- Root paths --
BASE_DIR        = Path("D:/Automation")
SCRIPT_DIR      = Path(__file__).parent
OUTPUT_DIR      = SCRIPT_DIR / "output"

# -- Input data paths --
ZEPHYR_PROJECTS = {
    "6mm": BASE_DIR / "6mm.zep",
    "2mm": BASE_DIR / "2mm 3D.zep",
}

CT_TIFF_DIRS = {
    "6mm": BASE_DIR / "6 mm Image Stack",
    "3218": BASE_DIR / "3218",
}

VGL_PROJECTS = {
    "6mm_sutharsh": BASE_DIR / "3218_Sutharsh_6P1.vgl",
    "3218_deeshka":  BASE_DIR / "3218" / "3218_Deeshka_6P1.vgl",
}

STL_NOMINAL = {
    "6P1": BASE_DIR / "6P1.stl",
    "2PR": BASE_DIR / "2PR.stl",
}

EXISTING_PLY = BASE_DIR / "Dense point cloud 1.ply"

# -- Software paths --
ZEPHYR_EXE = Path("C:/Program Files/3DF Zephyr/3DF Zephyr.exe")

# VGStudio MAX — typical install locations to probe
_VGSTUDIO_CANDIDATES = [
    Path("C:/Program Files/Volume Graphics/VGStudio MAX 2026.1/bin/vgstudiomax.exe"),
    Path("C:/Program Files/Volume Graphics/VGStudio MAX 2025/bin/vgstudiomax.exe"),
    Path("C:/Program Files/Volume Graphics/VGStudio MAX 2024/bin/vgstudiomax.exe"),
    Path("C:/Program Files (x86)/Volume Graphics/VGStudio MAX 2026.1/bin/vgstudiomax.exe"),
]

def _find_vgstudio() -> Path | None:
    for p in _VGSTUDIO_CANDIDATES:
        if p.exists():
            return p
    found = shutil.which("vgstudiomax")
    return Path(found) if found else None

VGSTUDIO_EXE: Path | None = _find_vgstudio()

# -- Mode detection --
MODE = "university" if VGSTUDIO_EXE else "laptop"

# -- Analysis settings --
TOLERANCE_MM        = 0.5    # deviation threshold for pass/fail (mm)
ICP_MAX_ITERATIONS  = 100
ICP_THRESHOLD       = 2.0    # coarse ICP distance threshold (mm)
VOXEL_DOWNSAMPLE    = 0.05   # point cloud downsample voxel size (mm)
CT_TIFF_PATTERN     = "*.tif"

# -- Output paths --
OUT_ZEPHYR     = OUTPUT_DIR / "zephyr"
OUT_CT         = OUTPUT_DIR / "ct"
OUT_VGSTUDIO   = OUTPUT_DIR / "vgstudio"
OUT_COMPARISON = OUTPUT_DIR / "comparison"
OUT_REPORT     = OUTPUT_DIR / "report"

# -- Sanity report --
def print_config():
    print("=" * 60)
    print(f"  MODE          : {MODE.upper()}")
    print(f"  3D Zephyr     : {'FOUND' if ZEPHYR_EXE.exists() else 'NOT FOUND'} — {ZEPHYR_EXE}")
    print(f"  VGStudio MAX  : {'FOUND — ' + str(VGSTUDIO_EXE) if VGSTUDIO_EXE else 'NOT INSTALLED (laptop mode)'}")
    print(f"  Base data dir : {BASE_DIR}")
    print(f"  Output dir    : {OUTPUT_DIR}")
    print("-" * 60)
    for name, path in ZEPHYR_PROJECTS.items():
        print(f"  ZEP [{name:4s}]   : {'OK' if path.exists() else 'MISSING'} — {path.name}")
    for name, path in STL_NOMINAL.items():
        print(f"  STL [{name:4s}]   : {'OK' if path.exists() else 'MISSING'} — {path.name}")
    for name, d in CT_TIFF_DIRS.items():
        count = len(list(d.glob(CT_TIFF_PATTERN))) if d.exists() else 0
        print(f"  CT  [{name:4s}]   : {count} TIFs in {d.name}")
    print(f"  PLY (Zephyr)  : {'OK' if EXISTING_PLY.exists() else 'MISSING'}")
    print("=" * 60)
