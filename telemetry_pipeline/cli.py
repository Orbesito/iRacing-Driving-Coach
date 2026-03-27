from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .derived_channels import add_derived_units
from .io_mu_csv import load_mu_csv
from .persistence import save_session_bundle


def main() -> None:
    """CLI entrypoint for loading and inspecting a Mu telemetry session."""
    parser = argparse.ArgumentParser(
        description="Load and validate a Mu-exported iRacing CSV session file."
    )
    parser.add_argument(
        "csv_file",
        type=str,
        help="Path to the Mu-exported CSV file",
    )
    parser.add_argument(
        "--preview-rows",
        type=int,
        default=5,
        help="Number of sample rows to preview (default: 5).",
    )
    parser.add_argument(
        "--no-derived",
        action="store_true",
        help="Skip derived channels (Speed_kmh, SteeringWheelAngle_deg, YawRate_deg_s).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help=(
            "Optional directory to save processed session artifacts "
            "(telemetry CSV + metadata/units/report JSON)."
        ),
    )
    args = parser.parse_args()

    if args.preview_rows < 1:
        raise ValueError("--preview-rows must be >= 1")

    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")

    print(f"\nPython interpreter: {sys.executable}")
    print(f"Loading Mu CSV: {csv_path}")
    metadata, units, df, parse_report = load_mu_csv(csv_path)

    print("\nParse summary:")
    for key, value in parse_report.items():
        print(f"  {key}: {value}")

    if args.no_derived:
        print("\nSkipping derived channels (--no-derived enabled).")
    else:
        print("\nApplying derived channel conversions...")
        df, derived_units = add_derived_units(df)
        units.update(derived_units)

    print("\nSession metadata:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")

    print(f"\nLoaded dataframe shape: {df.shape}")
    preview_cols = [
        col
        for col in [
            "SessionTime",
            "Lap",
            "LapDistPct",
            "Speed",
            "Speed_kmh",
            "Brake",
            "Throttle",
            "SteeringWheelAngle_deg",
            "YawRate_deg_s",
        ]
        if col in df.columns
    ]

    if preview_cols:
        print(f"\nSelected columns preview (first {args.preview_rows} rows):")
        print(df[preview_cols].head(args.preview_rows))
    else:
        print("\nNo preview columns found in this session.")

    print("\nUnits for selected columns:")
    for col in preview_cols:
        print(f"  {col}: {units.get(col, 'unknown')}")

    required_for_next_stage = ["SessionTime", "Lap", "LapDistPct"]
    missing_required = [col for col in required_for_next_stage if col not in df.columns]
    if missing_required:
        print(
            "\nWARNING: Missing baseline channels for lap segmentation/alignment: "
            + ", ".join(missing_required)
        )

    if args.out_dir:
        output_paths = save_session_bundle(
            args.out_dir,
            telemetry_df=df,
            metadata=metadata,
            units_map=units,
            parse_report=parse_report,
        )
        print("\nSaved artifacts:")
        for label, output_path in output_paths.items():
            print(f"  {label}: {output_path}")
