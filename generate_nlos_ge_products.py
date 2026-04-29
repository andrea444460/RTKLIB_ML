#!/usr/bin/env python3
"""
Generate NLOS plot/CSV and Google Earth KMZ products.

For each device folder, this script expects:
  - <prefix>_rtk_nlos_ml_on.pos
  - <prefix>_rtk_nlos_ml_on.trace
  - <prefix>_rtk_nlos_ml_off.pos

It then runs:
  1) plot_nlos_trace.py (from ML-ON trace) -> PNG + CSV
  2) pos_to_google_earth.py with 2 tracks:
     - ML ON   : solid blue
     - ML OFF  : nlos_count (colored by CSV)
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import List, Tuple


DEFAULT_DATASET_ROOT = Path(
    r"C:\Users\Andrea\Desktop\Dataset_sample_2\Dataset_Smartphones_20250917"
)

ML_ON_SUFFIX = "_rtk_nlos_ml_on"
ML_OFF_SUFFIX = "_rtk_nlos_ml_off"

# KML AABBGGRR
COLOR_BLUE = "ffff0000"


def run_cmd(cmd: List[str], cwd: Path | None = None, dry_run: bool = False) -> int:
    pretty = " ".join(f'"{c}"' if " " in c else c for c in cmd)
    print(f"[CMD] {pretty}")
    if dry_run:
        return 0
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    return proc.returncode


def find_jobs(dataset_root: Path) -> List[Tuple[Path, str]]:
    jobs: List[Tuple[Path, str]] = []
    for ml_on_pos in sorted(dataset_root.rglob(f"*{ML_ON_SUFFIX}.pos")):
        folder = ml_on_pos.parent
        prefix = ml_on_pos.name[: -len(f"{ML_ON_SUFFIX}.pos")]
        ml_on_trace = folder / f"{prefix}{ML_ON_SUFFIX}.trace"
        ml_off_pos = folder / f"{prefix}{ML_OFF_SUFFIX}.pos"
        if not ml_on_trace.exists():
            print(f"[SKIP] Missing trace: {ml_on_trace}")
            continue
        if not ml_off_pos.exists():
            print(f"[SKIP] Missing ML-OFF pos: {ml_off_pos}")
            continue
        jobs.append((folder, prefix))
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate plot_nlos_trace outputs and KMZ comparison files."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--python",
        type=str,
        default="python",
        help="Python executable used to launch helper scripts.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without executing."
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    plot_script = script_dir / "plot_nlos_trace.py"
    ge_script = script_dir / "pos_to_google_earth.py"
    if not plot_script.exists() or not ge_script.exists():
        raise SystemExit("[ERR] Required scripts not found next to this launcher.")
    if not args.dataset_root.exists():
        raise SystemExit(f"[ERR] Dataset root not found: {args.dataset_root}")

    jobs = find_jobs(args.dataset_root)
    if not jobs:
        raise SystemExit("[ERR] No valid jobs found (ml_on/ml_off/trace/GT set incomplete).")

    print(f"[OK] Jobs found: {len(jobs)}")
    ok = 0
    failed = 0

    for folder, prefix in jobs:
        print(f"\n=== JOB: {folder} ===")
        ml_on_pos = folder / f"{prefix}{ML_ON_SUFFIX}.pos"
        ml_on_trace = folder / f"{prefix}{ML_ON_SUFFIX}.trace"
        ml_off_pos = folder / f"{prefix}{ML_OFF_SUFFIX}.pos"

        nlos_plot = folder / f"{prefix}_nlos_trace_plot.png"
        nlos_csv = folder / f"{prefix}_nlos_output.csv"
        ge_out_base = folder / f"{prefix}_comparison_ml_blue_off_nlos"

        cmd_plot = [
            args.python,
            str(plot_script),
            str(ml_on_trace),
            "--source",
            "epochsat",
            "--type",
            "all",
            "--out",
            str(nlos_plot),
            "--csv",
            str(nlos_csv),
        ]
        rc_plot = run_cmd(cmd_plot, dry_run=args.dry_run)
        if rc_plot != 0:
            failed += 1
            print(f"[FAIL] plot_nlos_trace failed for {folder}")
            continue

        cmd_ge = [
            args.python,
            str(ge_script),
            str(ml_on_pos),
            str(ml_off_pos),
            "--format",
            "kmz",
            "--output",
            str(ge_out_base),
            "--track-modes",
            "solid,nlos_count",
            "--track-colors",
            f"{COLOR_BLUE},ff0000ff",
            "--nlos-csv",
            str(nlos_csv),
            "--nlos-threshold",
            "0.5",
            "--nlos-time-bin",
            "1.0",
            "--no-open",
        ]
        rc_ge = run_cmd(cmd_ge, dry_run=args.dry_run)
        if rc_ge != 0:
            failed += 1
            print(f"[FAIL] pos_to_google_earth failed for {folder}")
            continue

        ok += 1
        print(f"[OK] Done: {folder}")

    print("\n=== SUMMARY ===")
    print(f"jobs={len(jobs)} ok={ok} failed={failed} dry_run={args.dry_run}")
    if failed > 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

