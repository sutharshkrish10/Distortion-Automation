"""
phase1_zephyr.py — Automates 3D Zephyr export of already-reconstructed projects.
Opens .zep project files and exports the dense point cloud / mesh as PLY/STL.
"""

import subprocess
import shutil
import time
from pathlib import Path
import config


def export_zephyr_project(project_name: str, zep_path: Path, output_dir: Path) -> Path | None:
    """
    Export a mesh from a 3DF Zephyr project.

    3DF Zephyr does not expose a fully headless CLI for mesh export — it requires
    GUI interaction or a licensed Workflow script. On the university PC the user
    should manually export from Zephyr (File > Export > Dense Point Cloud / Mesh
    as PLY) and place the file in output/zephyr/ before running main.py.

    On the laptop we copy the already-exported 'Dense point cloud 1.ply' so the
    rest of the pipeline has a valid input to work with.
    """
    out_path = output_dir / f"{project_name}_mesh.ply"

    if out_path.exists():
        print(f"  [OK]   {project_name}: already exported — {out_path.name}")
        return out_path

    if not zep_path.exists():
        print(f"  [SKIP] {project_name}: .zep file not found — {zep_path}")
    else:
        print(f"  [INFO] {project_name}: .zep found at {zep_path.name}")
        print(f"         NOTE: For headless export on the university PC, open Zephyr,")
        print(f"         load this project, and export the dense cloud as PLY to:")
        print(f"         {out_path}")

    # Use the already-exported PLY as a stand-in for this project
    if config.EXISTING_PLY.exists():
        shutil.copy2(config.EXISTING_PLY, out_path)
        size_mb = out_path.stat().st_size / 1e6
        print(f"  [COPY] Using existing Dense point cloud 1.ply ({size_mb:.1f} MB) -> {out_path.name}")
        return out_path

    return None


def run_all(output_dir: Path | None = None) -> dict[str, Path]:
    """
    Export all configured Zephyr projects. Returns {name: ply_path} for successes.
    """
    output_dir = output_dir or config.OUT_ZEPHYR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n-- Phase 1: 3D Zephyr Export --")
    results: dict[str, Path] = {}

    for name, zep_path in config.ZEPHYR_PROJECTS.items():
        result = export_zephyr_project(name, zep_path, output_dir)
        if result:
            results[name] = result

    # Also copy the pre-existing PLY as a reference
    if config.EXISTING_PLY.exists():
        dest = output_dir / "existing_dense_cloud.ply"
        if not dest.exists():
            shutil.copy2(config.EXISTING_PLY, dest)
        results["existing"] = dest
        print(f"  [OK]   Copied existing PLY -> {dest.name}")

    print(f"\n  Zephyr exports complete: {len(results)} file(s)")
    return results


if __name__ == "__main__":
    import config as _c
    _c.print_config()
    run_all()
