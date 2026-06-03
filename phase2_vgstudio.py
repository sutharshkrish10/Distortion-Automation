"""
phase2_vgstudio.py — University PC path: drive VGStudio MAX via subprocess.
Sets environment variables that vgstudio_api_script.py reads, then launches
VGStudio MAX with that script as the --script argument.
"""

import os
import json
import subprocess
from pathlib import Path
from typing import Optional

import config


def run_vgstudio_analysis(
    dataset_name: str,
    vgl_project: Path,
    stl_nominal: Path,
    output_dir: Path,
) -> Optional[dict]:
    """
    Launch VGStudio MAX with the automation script for one dataset.
    Returns the stats dict if successful, None otherwise.
    """
    script_path = Path(__file__).parent / "vgstudio_api_script.py"
    stats_path  = output_dir / f"{dataset_name}_stats.json"

    if stats_path.exists():
        print(f"  [SKIP] {dataset_name}: results already exist")
        return json.loads(stats_path.read_text())

    if not config.VGSTUDIO_EXE or not config.VGSTUDIO_EXE.exists():
        print(f"  [SKIP] VGStudio MAX not found — cannot run {dataset_name}")
        return None

    if not vgl_project.exists():
        print(f"  [SKIP] {dataset_name}: VGL project not found — {vgl_project}")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update({
        "VGA_VGL_PROJECT":  str(vgl_project),
        "VGA_STL_NOMINAL":  str(stl_nominal),
        "VGA_OUTPUT_DIR":   str(output_dir),
        "VGA_DATASET_NAME": dataset_name,
    })

    cmd = [
        str(config.VGSTUDIO_EXE),
        "--script", str(script_path),
        "--no-gui",
    ]

    print(f"\n  [VGSTUDIO] Launching analysis for '{dataset_name}'")
    print(f"             CMD: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            env=env,
            timeout=1800,     # 30 min max
            capture_output=True,
            text=True,
        )

        print(result.stdout[-2000:] if result.stdout else "  (no stdout)")

        if result.returncode != 0:
            print(f"  [WARN] VGStudio exited with code {result.returncode}")
            if result.stderr:
                print(f"  stderr: {result.stderr[:500]}")

        if stats_path.exists():
            stats = json.loads(stats_path.read_text())
            print(f"  [OK] {dataset_name} analysis complete")
            _print_stats(stats)
            return stats
        else:
            print(f"  [WARN] Stats file not created for {dataset_name}")
            return None

    except subprocess.TimeoutExpired:
        print(f"  [ERROR] VGStudio timed out for {dataset_name}")
        return None
    except Exception as e:
        print(f"  [ERROR] {dataset_name}: {e}")
        return None


def _print_stats(stats: dict):
    print(f"    max deviation  : {stats.get('max_deviation', 'N/A'):.3f} mm")
    print(f"    min deviation  : {stats.get('min_deviation', 'N/A'):.3f} mm")
    print(f"    mean deviation : {stats.get('mean_deviation', 'N/A'):.3f} mm")
    print(f"    RMS deviation  : {stats.get('rms_deviation', 'N/A'):.3f} mm")
    print(f"    % in tolerance : {stats.get('pct_in_tolerance', 'N/A'):.1f}%")


def run_all(output_dir: Path | None = None) -> dict[str, dict]:
    """Run VGStudio analysis for all configured VGL projects."""
    output_dir = output_dir or config.OUT_VGSTUDIO
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n-- Phase 2B: VGStudio MAX Analysis (university PC) --")

    if not config.VGSTUDIO_EXE:
        print("  [INFO] VGStudio MAX not detected — skipping this phase.")
        print("         Re-run this script on the university PC to use VGStudio.")
        return {}

    datasets = {
        "6mm_sutharsh": {
            "vgl_project": config.VGL_PROJECTS["6mm_sutharsh"],
            "stl_nominal": config.STL_NOMINAL["6P1"],
        },
        "3218_deeshka": {
            "vgl_project": config.VGL_PROJECTS["3218_deeshka"],
            "stl_nominal": config.STL_NOMINAL["6P1"],
        },
    }

    results: dict[str, dict] = {}
    for name, cfg in datasets.items():
        stats = run_vgstudio_analysis(name, output_dir=output_dir, **cfg)
        if stats:
            results[name] = stats

    print(f"\n  VGStudio analysis complete: {len(results)} dataset(s)")
    return results


if __name__ == "__main__":
    config.print_config()
    run_all()
