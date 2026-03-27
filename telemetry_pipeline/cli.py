from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .derived_channels import add_derived_units
from .io_mu_csv import load_mu_csv
from .lap_processing import (
    DEFAULT_DISTANCE_STEP_PCT,
    LapValidityConfig,
    align_laps_by_distance,
    build_lap_summary,
    get_valid_lap_ids,
)
from .persistence import (
    build_session_output_dir,
    save_lap_analysis,
    save_session_bundle,
)


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
    args = parser.parse_args()

    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")

    print(f"\nPython interpreter: {sys.executable}")
    print(f"Loading Mu CSV: {csv_path}")
    metadata, units, df, parse_report = load_mu_csv(csv_path)

    print("\nParse summary:")
    for key, value in parse_report.items():
        print(f"  {key}: {value}")

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
        print("\nSelected columns preview (first 5 rows):")
        print(df[preview_cols].head(5))
    else:
        print("\nNo preview columns found in this session.")

    print("\nUnits for selected columns:")
    for col in preview_cols:
        print(f"  {col}: {units.get(col, 'unknown')}")

    required_for_next_stage = ["SessionTime", "Lap", "LapDistPct"]
    missing_required = [col for col in required_for_next_stage if col not in df.columns]

    if missing_required:
        raise ValueError(
            "Missing required channels for deterministic lap alignment: "
            + ", ".join(missing_required)
        )

    validity_config = LapValidityConfig()
    lap_summary = build_lap_summary(df, config=validity_config)
    valid_lap_ids = get_valid_lap_ids(lap_summary)

    print("\nLap validity summary:")
    print(
        f"  total_laps_detected: {len(lap_summary)}\n"
        f"  valid_laps: {len(valid_lap_ids)}\n"
        f"  valid_lap_ids: {valid_lap_ids}"
    )

    invalid_laps = lap_summary.loc[
        ~lap_summary["is_valid"], ["lap_id", "invalid_reasons"]
    ]
    if not invalid_laps.empty:
        print("\nInvalid lap reasons:")
        for _, row in invalid_laps.iterrows():
            print(f"  lap {int(row['lap_id'])}: {row['invalid_reasons']}")

    if valid_lap_ids:
        aligned_laps, alignment_report = align_laps_by_distance(
            df,
            lap_ids=valid_lap_ids,
            distance_step_pct=DEFAULT_DISTANCE_STEP_PCT,
        )
        print("\nDistance alignment summary:")
        print(f"  requested_lap_count: {alignment_report['requested_lap_count']}")
        print(f"  aligned_lap_count: {alignment_report['aligned_lap_count']}")
        print(f"  distance_step_pct: {alignment_report['distance_step_pct']}")
        print(f"  distance_grid_points: {alignment_report['distance_grid_points']}")
        print(
            "  alignment_channels: "
            + ", ".join(alignment_report["alignment_channels"])
        )
    else:
        aligned_laps = None
        alignment_report = {
            "requested_lap_count": 0,
            "aligned_lap_count": 0,
            "distance_step_pct": DEFAULT_DISTANCE_STEP_PCT,
            "distance_grid_points": 0,
            "alignment_channels": [],
        }
        print("\nNo valid laps found, saving empty alignment output.")

    session_out_dir = build_session_output_dir(
        base_dir="outputs",
        source_csv=csv_path,
        metadata=metadata,
    )
    output_paths = save_session_bundle(
        session_out_dir,
        telemetry_df=df,
        metadata=metadata,
        units_map=units,
        parse_report=parse_report,
    )
    lap_paths = save_lap_analysis(
        session_out_dir / "laps",
        lap_summary_df=lap_summary,
        aligned_laps_df=aligned_laps,
        alignment_report=alignment_report,
    )

    print("\nSaved artifacts:")
    print(f"  session_output_dir: {session_out_dir}")
    for label, output_path in output_paths.items():
        print(f"  {label}: {output_path}")
    for label, output_path in lap_paths.items():
        print(f"  {label}: {output_path}")
