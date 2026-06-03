"""
main.py — Entry point for the 3D printed part distortion analysis pipeline.

Usage:
    python main.py                  # run full pipeline
    python main.py --skip-ct        # skip CT processing (fast, uses existing PLY only)
    python main.py --skip-zephyr    # skip 3D Zephyr export
    python main.py --only-report    # regenerate report from existing comparison results
"""

import sys
import json
import argparse
import time
from pathlib import Path

import config
import phase1_zephyr
import phase2_ct_python
import phase2_vgstudio
import phase3_comparison
import phase3b_angle_measurement
import phase4_report


def parse_args():
    p = argparse.ArgumentParser(description="3D Print Distortion Analysis Pipeline")
    p.add_argument("--skip-zephyr",  action="store_true", help="Skip Zephyr export")
    p.add_argument("--skip-ct",      action="store_true", help="Skip CT processing")
    p.add_argument("--skip-vgstudio",action="store_true", help="Skip VGStudio (university mode only)")
    p.add_argument("--only-report",  action="store_true", help="Only regenerate report")
    return p.parse_args()


def collect_existing_comparison_results() -> dict[str, dict]:
    """Load any already-completed comparison stats from disk."""
    results = {}
    for f in sorted(config.OUT_COMPARISON.glob("*_stats.json")):
        name = f.stem.replace("_stats", "")
        results[name] = json.loads(f.read_text())
    return results


def _collect_existing_angle_results() -> dict[str, dict]:
    """Load any already-completed angle results from disk."""
    results = {}
    for f in sorted(config.OUT_COMPARISON.glob("*_angles.json")):
        name = f.stem.replace("_angles", "")
        results[name] = json.loads(f.read_text())
    return results


def main():
    t0 = time.time()
    args = parse_args()

    print("\n" + "=" * 60)
    print("  3D PRINTED PART — DISTORTION ANALYSIS PIPELINE")
    print("=" * 60)
    config.print_config()

    zephyr_meshes: dict[str, Path] = {}
    ct_meshes:     dict[str, Path] = {}
    all_stats:     dict[str, dict] = {}

    if args.only_report:
        all_stats = collect_existing_comparison_results()
        if not all_stats:
            print("[ERROR] No existing comparison results found. Run without --only-report first.")
            sys.exit(1)
        angle_results = _collect_existing_angle_results()
        phase4_report.run_all(all_stats, angle_results=angle_results)
        print(f"\nDone in {time.time()-t0:.0f}s")
        return

    # -- Phase 1: 3D Zephyr Export --
    if not args.skip_zephyr:
        zephyr_meshes = phase1_zephyr.run_all()
    else:
        print("\n[INFO] Skipping Phase 1 (Zephyr export)")
        # Still pick up existing PLY
        existing = config.OUT_ZEPHYR / "existing_dense_cloud.ply"
        if existing.exists():
            zephyr_meshes["existing"] = existing

    # -- Phase 2A/B: CT Processing --
    if not args.skip_ct:
        if config.MODE == "university" and not args.skip_vgstudio:
            # University PC: use VGStudio MAX
            vgstudio_stats = phase2_vgstudio.run_all()
            # VGStudio exports its own CSV-based stats; we re-use them in phase 3
            all_stats.update(vgstudio_stats)
            # VGStudio deviation maps are already computed; skip Open3D comparison
            # for those datasets (unless you want a cross-check)
            print("\n[INFO] VGStudio results collected; skipping Open3D re-comparison for those datasets.")
        else:
            # Laptop: Python CT surface extraction
            ct_meshes = phase2_ct_python.run_all()
    else:
        print("\n[INFO] Skipping Phase 2 (CT processing)")

    # -- Phase 3: Open3D Nominal/Actual Comparison --
    # Always run Open3D comparison on Zephyr meshes.
    # On laptop, also run on CT-extracted meshes.
    comparison_stats = phase3_comparison.run_all(
        zephyr_meshes=zephyr_meshes,
        ct_meshes=ct_meshes,
    )
    all_stats.update(comparison_stats)

    if not all_stats:
        # Last resort: load any results already on disk
        all_stats = collect_existing_comparison_results()

    if not all_stats:
        print("\n[WARN] No results to report. Check that input files exist.")
        sys.exit(0)

    # -- Phase 3B: Leg Angle Measurement --
    angle_results = phase3b_angle_measurement.run_all(
        zephyr_meshes=zephyr_meshes,
        ct_meshes=ct_meshes,
    )

    # -- Phase 4: Report --
    report_paths = phase4_report.run_all(all_stats, angle_results=angle_results)

    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print(f"  PIPELINE COMPLETE  ({elapsed/60:.1f} min)")
    print("=" * 60)
    if report_paths.get("html"):
        print(f"  HTML report : {report_paths['html']}")
    if report_paths.get("pdf"):
        print(f"  PDF report  : {report_paths['pdf']}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
