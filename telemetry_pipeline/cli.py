from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .corner_metrics import (
    CornerDetectionConfig,
    choose_reference_lap,
    compute_corner_metrics,
    compute_lap_times,
    detect_main_corners,
)
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
    save_corner_analysis,
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
            "BrakeRaw",
            "Throttle",
            "ThrottleRaw",
            "LatAccel",
            "LongAccel",
            "VertAccel",
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

    accel_channels = ["LatAccel", "LongAccel"]
    missing_accel = [col for col in accel_channels if col not in df.columns]
    if missing_accel:
        print(
            "\nWARNING: Missing acceleration channels in this telemetry file: "
            + ", ".join(missing_accel)
        )

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

    if aligned_laps is not None and not aligned_laps.empty:
        lap_times = compute_lap_times(df, valid_lap_ids)
        reference_lap_id = choose_reference_lap(lap_times)
        print(f"\nReference lap selected for corner metrics: {reference_lap_id}")

        corner_config = CornerDetectionConfig()
        corner_definitions = detect_main_corners(
            aligned_laps_df=aligned_laps,
            reference_lap_id=reference_lap_id,
            config=corner_config,
        )
        print(f"Detected main corners: {len(corner_definitions)}")

        corner_lap_metrics, corner_ranking, corner_report = compute_corner_metrics(
            aligned_laps_df=aligned_laps,
            corner_definitions_df=corner_definitions,
            lap_times_df=lap_times,
            reference_lap_id=reference_lap_id,
        )
        print("\nTop coaching-priority corners:")
        top = corner_ranking.head(5)
        if top.empty:
            print("  none")
        else:
            for _, row in top.iterrows():
                print(
                    f"  {row['corner_name']} (rank {int(row['coaching_priority_rank'])}) "
                    f"time_loss={row['mean_time_loss_s']:.4f}s "
                    f"variability={row['inconsistency_time_s']:.4f}s "
                    f"score={row['coaching_relevance_score']:.4f}"
                )
    else:
        corner_definitions = pd.DataFrame(
            columns=[
                "corner_id",
                "corner_name",
                "corner_start_pct",
                "brake_start_pct",
                "brake_end_pct",
                "rotation_start_pct",
                "apex_pct",
                "rotation_end_pct",
                "traction_start_pct",
                "corner_end_pct",
                "reference_apex_speed_kmh",
                "detection_activity_score",
            ]
        )
        corner_lap_metrics = pd.DataFrame()
        corner_ranking = pd.DataFrame()
        corner_report = {
            "reference_lap_id": None,
            "corner_count": 0,
            "laps_in_metrics": 0,
            "trajectory_line_feasible": False,
            "trajectory_line_note": "No aligned laps available.",
            "coaching_score_formula": "",
        }

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
    corner_paths = save_corner_analysis(
        session_out_dir / "corners",
        corner_definitions_df=corner_definitions,
        corner_lap_metrics_df=corner_lap_metrics,
        corner_ranking_df=corner_ranking,
        corner_report=corner_report,
    )

    print("\nSaved artifacts:")
    print(f"  session_output_dir: {session_out_dir}")
    for label, output_path in output_paths.items():
        print(f"  {label}: {output_path}")
    for label, output_path in lap_paths.items():
        print(f"  {label}: {output_path}")
    for label, output_path in corner_paths.items():
        print(f"  {label}: {output_path}")
