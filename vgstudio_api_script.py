"""
vgstudio_api_script.py — Runs INSIDE VGStudio MAX via its built-in Python interpreter.
Called by phase2_vgstudio.py on the university PC.

Usage (invoked automatically by phase2_vgstudio.py):
    vgstudiomax.exe --script path/to/vgstudio_api_script.py

VGStudio MAX Python API reference:
  vgl.openProject()   vgl.getActiveDataset()   vgl.computeSurfaceDetermination()
  vgl.computeNominalActualComparison()          vgl.exportReport()
"""

import os
import sys
import json
from pathlib import Path

# These modules are only available inside VGStudio MAX
try:
    import vgl
    import vglAlgorithms
    INSIDE_VGSTUDIO = True
except ImportError:
    INSIDE_VGSTUDIO = False

# -- Paths passed via environment variables set by phase2_vgstudio.py --
VGL_PROJECT = Path(os.environ.get("VGA_VGL_PROJECT", ""))
STL_NOMINAL = Path(os.environ.get("VGA_STL_NOMINAL", ""))
OUTPUT_DIR  = Path(os.environ.get("VGA_OUTPUT_DIR", ""))
DATASET_NAME = os.environ.get("VGA_DATASET_NAME", "dataset")


def run_analysis():
    """Full VGStudio MAX automation: open project → compare → export."""

    if not INSIDE_VGSTUDIO:
        print("[ERROR] This script must run inside VGStudio MAX.")
        print("        It is launched automatically by phase2_vgstudio.py")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    # 1. Open the VGL project (already has CT volume + CAD loaded)
    print(f"[VGStudio] Opening project: {VGL_PROJECT}")
    vgl.openProject(str(VGL_PROJECT))

    dataset = vgl.getActiveDataset()
    if dataset is None:
        print("[ERROR] No active dataset found in project.")
        sys.exit(1)

    # 2. Surface determination (find the CT surface boundary)
    print("[VGStudio] Running surface determination...")
    surf_params = vglAlgorithms.SurfaceDeterminationParameters()
    surf_params.setThresholdMode(vglAlgorithms.ThresholdMode.AUTOMATIC)
    vglAlgorithms.computeSurfaceDetermination(dataset, surf_params)
    print("[VGStudio] Surface determination complete.")

    # 3. Load / verify nominal CAD object
    print(f"[VGStudio] Loading nominal STL: {STL_NOMINAL}")
    cad_object = None
    for obj in vgl.getObjects():
        if obj.getType() == vgl.ObjectType.CAD_MESH:
            cad_object = obj
            break

    if cad_object is None and STL_NOMINAL.exists():
        cad_object = vgl.importCADMesh(str(STL_NOMINAL))

    if cad_object is None:
        print("[ERROR] No CAD nominal model found or importable.")
        sys.exit(1)

    # 4. Register actual (CT) surface to nominal CAD (best-fit)
    print("[VGStudio] Registering CT surface to nominal CAD...")
    reg_params = vglAlgorithms.RegistrationParameters()
    reg_params.setMode(vglAlgorithms.RegistrationMode.BEST_FIT)
    vglAlgorithms.computeRegistration(dataset, cad_object, reg_params)
    print("[VGStudio] Registration complete.")

    # 5. Nominal / actual comparison
    print("[VGStudio] Running nominal/actual comparison...")
    comp_params = vglAlgorithms.NominalActualComparisonParameters()
    comp_params.setTolerance(0.5)           # ±0.5 mm tolerance band
    comp_params.setDirection(vglAlgorithms.ComparisonDirection.BOTH)
    comparison = vglAlgorithms.computeNominalActualComparison(
        dataset, cad_object, comp_params
    )
    print("[VGStudio] Comparison complete.")

    # 6. Export deviation data as CSV
    csv_path = OUTPUT_DIR / f"{DATASET_NAME}_deviation.csv"
    print(f"[VGStudio] Exporting deviation CSV → {csv_path}")
    comparison.exportToCSV(str(csv_path))
    results["csv"] = str(csv_path)

    # 7. Export color-coded deviation screenshot
    img_path = OUTPUT_DIR / f"{DATASET_NAME}_deviation_map.png"
    print(f"[VGStudio] Exporting deviation map → {img_path}")
    vgl.exportScreenshot(str(img_path), width=1920, height=1080)
    results["image"] = str(img_path)

    # 8. Export summary statistics
    stats = {
        "dataset":        DATASET_NAME,
        "max_deviation":  comparison.getMaxDeviation(),
        "min_deviation":  comparison.getMinDeviation(),
        "mean_deviation": comparison.getMeanDeviation(),
        "rms_deviation":  comparison.getRMSDeviation(),
        "pct_in_tolerance": comparison.getPercentageInTolerance(),
    }
    stats_path = OUTPUT_DIR / f"{DATASET_NAME}_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    results["stats"] = str(stats_path)

    # 9. Save updated project
    vgl.saveProject(str(VGL_PROJECT))

    print(f"\n[VGStudio] All done. Results written to: {OUTPUT_DIR}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    run_analysis()
