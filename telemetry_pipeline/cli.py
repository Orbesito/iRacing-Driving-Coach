from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .corner_metrics import (
    CornerDetectionConfig,
    apply_corner_reference,
    build_best_per_corner_reference,
    build_corner_report,
    choose_reference_lap,
    compute_corner_lap_metrics,
    compute_lap_times,
    detect_main_corners,
)
from .coaching import generate_coaching_outputs
from .coaching_pdf import generate_coaching_pdf
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
    save_coaching_analysis,
    save_corner_analysis,
    save_lap_analysis,
    save_session_bundle,
)


@dataclass
class SessionAnalysis:
    csv_path: Path
    metadata: dict[str, str]
    units: dict[str, str]
    telemetry_df: pd.DataFrame
    parse_report: dict[str, int]
    lap_summary: pd.DataFrame
    valid_lap_ids: list[int]
    aligned_laps: pd.DataFrame
    alignment_report: dict[str, object]
    lap_times: pd.DataFrame
    output_dir: Path


def _progress(message: str) -> None:
    print(f"\n... {message}", flush=True)


def _empty_corner_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    corner_definitions = pd.DataFrame(
        columns=[
            "corner_id",
            "official_turn_number",
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
            "detection_curvature_score",
        ]
    )
    corner_metrics = pd.DataFrame()
    corner_ranking = pd.DataFrame()
    corner_report = {
        "corner_count": 0,
        "laps_in_metrics": 0,
        "trajectory_line_feasible": False,
        "trajectory_line_note": "No aligned laps available.",
        "coaching_score_formula": "",
    }
    return corner_definitions, corner_metrics, corner_ranking, corner_report


def _print_parse_report(parse_report: dict[str, int]) -> None:
    print("\nParse summary:")
    for key, value in parse_report.items():
        print(f"  {key}: {value}")


def _validate_required_channels(df: pd.DataFrame) -> None:
    required_for_next_stage = ["SessionTime", "Lap", "LapDistPct"]
    missing_required = [col for col in required_for_next_stage if col not in df.columns]
    if missing_required:
        raise ValueError(
            "Missing required channels for deterministic lap alignment: "
            + ", ".join(missing_required)
        )


def _analyze_single_session(csv_path: Path, label: str) -> SessionAnalysis:
    print(f"\n[{label}] Loading Mu CSV: {csv_path}")
    _progress(f"[{label}] Parsing Mu CSV metadata and telemetry")
    metadata, units, df, parse_report = load_mu_csv(csv_path)
    _print_parse_report(parse_report)

    _progress(f"[{label}] Applying derived channels")
    df, derived_units = add_derived_units(df)
    units.update(derived_units)
    _validate_required_channels(df)

    _progress(f"[{label}] Classifying valid laps")
    validity_config = LapValidityConfig()
    lap_summary = build_lap_summary(df, config=validity_config)
    valid_lap_ids = get_valid_lap_ids(lap_summary)

    print(f"\n[{label}] Lap validity summary:")
    print(
        f"  total_laps_detected: {len(lap_summary)}\n"
        f"  valid_laps: {len(valid_lap_ids)}\n"
        f"  valid_lap_ids: {valid_lap_ids}"
    )

    if valid_lap_ids:
        _progress(f"[{label}] Aligning valid laps by distance")
        aligned_laps, alignment_report = align_laps_by_distance(
            df,
            lap_ids=valid_lap_ids,
            distance_step_pct=DEFAULT_DISTANCE_STEP_PCT,
        )
        _progress(f"[{label}] Computing lap times for aligned laps")
        lap_times = compute_lap_times(df, valid_lap_ids)
    else:
        aligned_laps = pd.DataFrame(columns=["lap_id", "distance_pct"])
        alignment_report = {
            "requested_lap_count": 0,
            "aligned_lap_count": 0,
            "distance_step_pct": DEFAULT_DISTANCE_STEP_PCT,
            "distance_grid_points": 0,
            "alignment_channels": [],
        }
        lap_times = pd.DataFrame(columns=["lap_id", "start_time_s", "end_time_s", "lap_time_s"])

    print(f"\n[{label}] Distance alignment summary:")
    print(f"  requested_lap_count: {alignment_report['requested_lap_count']}")
    print(f"  aligned_lap_count: {alignment_report['aligned_lap_count']}")
    print(f"  distance_step_pct: {alignment_report['distance_step_pct']}")
    print(f"  distance_grid_points: {alignment_report['distance_grid_points']}")

    out_dir = build_session_output_dir(
        base_dir="outputs",
        source_csv=csv_path,
        metadata=metadata,
    )
    _progress(f"[{label}] Saving session and lap artifacts")
    output_paths = save_session_bundle(
        out_dir,
        telemetry_df=df,
        metadata=metadata,
        units_map=units,
        parse_report=parse_report,
    )
    lap_paths = save_lap_analysis(
        out_dir / "laps",
        lap_summary_df=lap_summary,
        aligned_laps_df=aligned_laps,
        alignment_report=alignment_report,
    )

    print(f"\n[{label}] Saved core artifacts:")
    print(f"  session_output_dir: {out_dir}")
    for key, value in output_paths.items():
        print(f"  {key}: {value}")
    for key, value in lap_paths.items():
        print(f"  {key}: {value}")

    return SessionAnalysis(
        csv_path=csv_path,
        metadata=metadata,
        units=units,
        telemetry_df=df,
        parse_report=parse_report,
        lap_summary=lap_summary,
        valid_lap_ids=valid_lap_ids,
        aligned_laps=aligned_laps,
        alignment_report=alignment_report,
        lap_times=lap_times,
        output_dir=out_dir,
    )


def _normalise_text(value: str) -> str:
    return " ".join(value.lower().split())


def _infer_expected_corner_count(metadata: dict[str, str]) -> int | None:
    """
    Deterministic track metadata lookup for official turn counts where known.
    """
    venue = _normalise_text(metadata.get("Venue", ""))
    if "miami" in venue and "gp" in venue:
        return 19
    return None


def _build_corner_detection_config(metadata: dict[str, str]) -> CornerDetectionConfig:
    expected_count = _infer_expected_corner_count(metadata)
    if expected_count is None:
        return CornerDetectionConfig()
    return CornerDetectionConfig(
        min_corner_count=expected_count,
        max_corner_count=expected_count,
        target_corner_count=expected_count,
    )


def _assert_same_track(driver_metadata: dict[str, str], reference_metadata: dict[str, str]) -> None:
    driver_venue = _normalise_text(driver_metadata.get("Venue", ""))
    reference_venue = _normalise_text(reference_metadata.get("Venue", ""))
    if driver_venue and reference_venue and driver_venue != reference_venue:
        raise ValueError(
            "Driver and reference sessions appear to be from different venues: "
            f"'{driver_metadata.get('Venue', '')}' vs '{reference_metadata.get('Venue', '')}'. "
            "For deterministic cross-session corner comparison, both inputs must be from the same track configuration."
        )


def _format_best_lap_time(lap_times_df: pd.DataFrame) -> str:
    if lap_times_df.empty or "lap_time_s" not in lap_times_df.columns:
        return "n/a"
    lap_series = pd.to_numeric(lap_times_df["lap_time_s"], errors="coerce").dropna()
    if lap_series.empty:
        return "n/a"
    return f"{float(lap_series.min()):.3f} s"


def _build_pdf_session_context(
    coached_session: SessionAnalysis,
    corner_report: dict[str, object],
    reference_session: SessionAnalysis | None = None,
) -> dict[str, object]:
    context: dict[str, object] = {
        "coached_driver": coached_session.metadata.get("Driver", "n/a"),
        "coached_vehicle": coached_session.metadata.get("Vehicle", "n/a"),
        "venue": coached_session.metadata.get("Venue", "n/a"),
        "session_type": coached_session.metadata.get("Session", "n/a"),
        "session_date": coached_session.metadata.get("Session Date", "n/a"),
        "session_time": coached_session.metadata.get("Session Time", "n/a"),
        "sample_rate": coached_session.metadata.get("Sample Rate", "n/a"),
        "session_duration": coached_session.metadata.get("Session Duration", "n/a"),
        "valid_laps": len(coached_session.valid_lap_ids),
        "total_laps": int(len(coached_session.lap_summary)),
        "coached_best_lap_time_s": _format_best_lap_time(coached_session.lap_times),
        "detected_corner_count": corner_report.get(
            "detected_corner_count", corner_report.get("corner_count", "n/a")
        ),
        "expected_corner_count": corner_report.get("expected_corner_count", "n/a"),
    }
    if reference_session is not None:
        context.update(
            {
                "reference_driver": reference_session.metadata.get("Driver", "n/a"),
                "reference_vehicle": reference_session.metadata.get("Vehicle", "n/a"),
                "reference_best_lap_time_s": _format_best_lap_time(reference_session.lap_times),
            }
        )
    return context


def _run_single_session_mode(session: SessionAnalysis) -> None:
    print("\nRunning mode: single-session coaching analysis.")
    if session.aligned_laps.empty or session.lap_times.empty:
        corner_definitions, corner_metrics, corner_ranking, corner_report = _empty_corner_outputs()
        corner_reference = pd.DataFrame()
    else:
        detection_lap_id = choose_reference_lap(session.lap_times)
        print(f"Corner detection lap (fastest valid lap): {detection_lap_id}")
        corner_config = _build_corner_detection_config(session.metadata)
        _progress("Detecting main corners and phase boundaries")
        corner_definitions = detect_main_corners(
            aligned_laps_df=session.aligned_laps,
            reference_lap_id=detection_lap_id,
            config=corner_config,
        )
        detected_count = int(len(corner_definitions))
        expected_count = _infer_expected_corner_count(session.metadata)
        if expected_count is not None:
            print(
                f"Detected corners: {detected_count} (expected official map turns: {expected_count})"
            )
        else:
            print(f"Detected corners: {detected_count}")

        _progress("Computing per-corner metrics for all valid laps")
        raw_corner_metrics = compute_corner_lap_metrics(
            aligned_laps_df=session.aligned_laps,
            corner_definitions_df=corner_definitions,
            lap_times_df=session.lap_times,
        )
        _progress("Building best-per-corner reference profile")
        corner_reference = build_best_per_corner_reference(raw_corner_metrics)
        _progress("Scoring corners against best-per-corner reference")
        corner_metrics, corner_ranking = apply_corner_reference(
            corner_lap_metrics_df=raw_corner_metrics,
            corner_reference_df=corner_reference,
            comparison_label="self_best_per_corner",
            exclude_reference_rows=True,
        )
        corner_report = build_corner_report(
            aligned_laps_df=session.aligned_laps,
            corner_definitions_df=corner_definitions,
            corner_lap_metrics_df=corner_metrics,
            reference_mode="best_per_corner_same_session",
            reference_source=str(session.csv_path),
        )
        corner_report["operation_mode"] = "single_session"
        corner_report["corner_detection_lap_id"] = int(detection_lap_id)
        corner_report["reference_note"] = (
            "Per-corner benchmark uses the best corner performance found across all valid laps, "
            "not only the globally fastest lap."
        )
        corner_report["expected_corner_count"] = expected_count
        corner_report["detected_corner_count"] = detected_count
        corner_report["corner_count_matches_expected"] = (
            expected_count is None or detected_count == expected_count
        )

    _progress("Saving corner coaching artifacts")
    corner_paths = save_corner_analysis(
        session.output_dir / "corners",
        corner_definitions_df=corner_definitions,
        corner_lap_metrics_df=corner_metrics,
        corner_ranking_df=corner_ranking,
        corner_report=corner_report,
        corner_reference_df=corner_reference,
    )

    print("\nSaved corner artifacts:")
    for key, value in corner_paths.items():
        print(f"  {key}: {value}")

    _progress("Building deterministic coaching layer")
    corner_coaching_df, coaching_summary = generate_coaching_outputs(
        corner_lap_metrics_df=corner_metrics,
        corner_ranking_df=corner_ranking,
        corner_report=corner_report,
        mode_name="single_session",
    )
    coaching_paths = save_coaching_analysis(
        session.output_dir / "coaching",
        corner_coaching_df=corner_coaching_df,
        session_summary=coaching_summary,
    )
    _progress("Generating coaching PDF report")
    try:
        pdf_context = _build_pdf_session_context(
            coached_session=session,
            corner_report=corner_report,
            reference_session=None,
        )
        coaching_pdf_path = generate_coaching_pdf(
            out_path=session.output_dir / "coaching" / "coaching_report.pdf",
            mode_name="single_session",
            corner_coaching_df=corner_coaching_df,
            session_summary=coaching_summary,
            corner_definitions_df=corner_definitions,
            aligned_laps_df=session.aligned_laps,
            corner_reference_df=corner_reference,
            reference_aligned_laps_df=None,
            session_context=pdf_context,
        )
    except Exception as exc:
        coaching_pdf_path = None
        print(f"\nPDF report skipped: {exc}")

    print("\nSaved coaching artifacts:")
    for key, value in coaching_paths.items():
        print(f"  {key}: {value}")
    if coaching_pdf_path is not None:
        print(f"  coaching_report_pdf: {coaching_pdf_path}")

    print("\nTop coaching-priority corners:")
    top = corner_ranking.head(5)
    if top.empty:
        print("  none")
        return
    for _, row in top.iterrows():
        print(
            f"  {row['corner_name']} (rank {int(row['coaching_priority_rank'])}) "
            f"time_loss={row['mean_time_loss_s']:.4f}s "
            f"variability={row['inconsistency_time_s']:.4f}s "
            f"score={row['coaching_relevance_score']:.4f}"
        )


def _run_vs_reference_mode(driver_session: SessionAnalysis, reference_session: SessionAnalysis) -> None:
    print("\nRunning mode: driver session vs faster reference session.")
    _assert_same_track(driver_session.metadata, reference_session.metadata)

    if (
        driver_session.aligned_laps.empty
        or driver_session.lap_times.empty
        or reference_session.aligned_laps.empty
        or reference_session.lap_times.empty
    ):
        driver_corner_definitions, driver_corner_metrics, driver_corner_ranking, driver_corner_report = (
            _empty_corner_outputs()
        )
        reference_corner_profile = pd.DataFrame()
    else:
        reference_detection_lap_id = choose_reference_lap(reference_session.lap_times)
        print(f"Reference corner detection lap: {reference_detection_lap_id}")
        corner_config = _build_corner_detection_config(reference_session.metadata)

        # Stable corner IDs: corner definitions are detected once on reference session
        # and then reused for both sessions.
        _progress("Detecting canonical corners from reference session")
        canonical_corner_definitions = detect_main_corners(
            aligned_laps_df=reference_session.aligned_laps,
            reference_lap_id=reference_detection_lap_id,
            config=corner_config,
        )
        detected_count = int(len(canonical_corner_definitions))
        expected_count = _infer_expected_corner_count(reference_session.metadata)
        if expected_count is not None:
            print(
                f"Detected canonical corners: {detected_count} "
                f"(expected official map turns: {expected_count})"
            )
        else:
            print(f"Detected canonical corners: {detected_count}")

        _progress("Computing reference-session per-corner metrics")
        reference_raw_metrics = compute_corner_lap_metrics(
            aligned_laps_df=reference_session.aligned_laps,
            corner_definitions_df=canonical_corner_definitions,
            lap_times_df=reference_session.lap_times,
        )
        _progress("Building reference best-per-corner profile")
        reference_corner_profile = build_best_per_corner_reference(reference_raw_metrics)

        _progress("Scoring reference session against its own profile")
        reference_metrics_scored, reference_ranking = apply_corner_reference(
            corner_lap_metrics_df=reference_raw_metrics,
            corner_reference_df=reference_corner_profile,
            comparison_label="reference_self_best_per_corner",
            exclude_reference_rows=True,
        )
        reference_report = build_corner_report(
            aligned_laps_df=reference_session.aligned_laps,
            corner_definitions_df=canonical_corner_definitions,
            corner_lap_metrics_df=reference_metrics_scored,
            reference_mode="best_per_corner_same_session",
            reference_source=str(reference_session.csv_path),
        )
        reference_report["operation_mode"] = "reference_baseline"
        reference_report["corner_detection_lap_id"] = int(reference_detection_lap_id)
        reference_report["expected_corner_count"] = expected_count
        reference_report["detected_corner_count"] = detected_count
        reference_report["corner_count_matches_expected"] = (
            expected_count is None or detected_count == expected_count
        )

        _progress("Saving reference-session corner artifacts")
        save_corner_analysis(
            reference_session.output_dir / "corners",
            corner_definitions_df=canonical_corner_definitions,
            corner_lap_metrics_df=reference_metrics_scored,
            corner_ranking_df=reference_ranking,
            corner_report=reference_report,
            corner_reference_df=reference_corner_profile,
        )

        _progress("Computing driver-session per-corner metrics on canonical corners")
        driver_raw_metrics = compute_corner_lap_metrics(
            aligned_laps_df=driver_session.aligned_laps,
            corner_definitions_df=canonical_corner_definitions,
            lap_times_df=driver_session.lap_times,
        )
        _progress("Scoring driver session vs external reference profile")
        driver_corner_metrics, driver_corner_ranking = apply_corner_reference(
            corner_lap_metrics_df=driver_raw_metrics,
            corner_reference_df=reference_corner_profile,
            comparison_label="vs_external_reference_driver",
            exclude_reference_rows=False,
        )
        driver_corner_definitions = canonical_corner_definitions
        driver_corner_report = build_corner_report(
            aligned_laps_df=driver_session.aligned_laps,
            corner_definitions_df=driver_corner_definitions,
            corner_lap_metrics_df=driver_corner_metrics,
            reference_mode="external_session_best_per_corner",
            reference_source=str(reference_session.csv_path),
        )
        driver_corner_report["operation_mode"] = "vs_reference_session"
        driver_corner_report["corner_detection_lap_id_reference"] = int(reference_detection_lap_id)
        driver_corner_report["stable_corner_id_note"] = (
            "Corner IDs and boundaries are defined from the reference session and reused "
            "without re-detection on the driver session."
        )
        driver_corner_report["driver_csv"] = str(driver_session.csv_path)
        driver_corner_report["reference_csv"] = str(reference_session.csv_path)
        driver_corner_report["expected_corner_count"] = expected_count
        driver_corner_report["detected_corner_count"] = detected_count
        driver_corner_report["corner_count_matches_expected"] = (
            expected_count is None or detected_count == expected_count
        )

    _progress("Saving driver-vs-reference corner artifacts")
    driver_corner_paths = save_corner_analysis(
        driver_session.output_dir / "corners_vs_reference",
        corner_definitions_df=driver_corner_definitions,
        corner_lap_metrics_df=driver_corner_metrics,
        corner_ranking_df=driver_corner_ranking,
        corner_report=driver_corner_report,
        corner_reference_df=reference_corner_profile,
    )

    print("\nSaved driver-vs-reference corner artifacts:")
    for key, value in driver_corner_paths.items():
        print(f"  {key}: {value}")

    _progress("Building deterministic coaching layer (driver vs reference)")
    corner_coaching_df, coaching_summary = generate_coaching_outputs(
        corner_lap_metrics_df=driver_corner_metrics,
        corner_ranking_df=driver_corner_ranking,
        corner_report=driver_corner_report,
        mode_name="vs_reference_session",
    )
    coaching_paths = save_coaching_analysis(
        driver_session.output_dir / "coaching_vs_reference",
        corner_coaching_df=corner_coaching_df,
        session_summary=coaching_summary,
    )
    _progress("Generating coaching PDF report (driver vs reference)")
    try:
        pdf_context = _build_pdf_session_context(
            coached_session=driver_session,
            corner_report=driver_corner_report,
            reference_session=reference_session,
        )
        coaching_pdf_path = generate_coaching_pdf(
            out_path=driver_session.output_dir / "coaching_vs_reference" / "coaching_report.pdf",
            mode_name="vs_reference_session",
            corner_coaching_df=corner_coaching_df,
            session_summary=coaching_summary,
            corner_definitions_df=driver_corner_definitions,
            aligned_laps_df=driver_session.aligned_laps,
            corner_reference_df=reference_corner_profile,
            reference_aligned_laps_df=reference_session.aligned_laps,
            session_context=pdf_context,
        )
    except Exception as exc:
        coaching_pdf_path = None
        print(f"\nPDF report skipped: {exc}")

    print("\nSaved coaching artifacts:")
    for key, value in coaching_paths.items():
        print(f"  {key}: {value}")
    if coaching_pdf_path is not None:
        print(f"  coaching_report_pdf: {coaching_pdf_path}")

    print("\nTop coaching-priority corners vs reference:")
    top = driver_corner_ranking.head(5)
    if top.empty:
        print("  none")
        return
    for _, row in top.iterrows():
        print(
            f"  {row['corner_name']} (rank {int(row['coaching_priority_rank'])}) "
            f"time_loss={row['mean_time_loss_s']:.4f}s "
            f"variability={row['inconsistency_time_s']:.4f}s "
            f"score={row['coaching_relevance_score']:.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deterministic Mu telemetry analysis for single-session coaching or driver-vs-reference comparison."
    )
    parser.add_argument(
        "csv_file",
        type=str,
        help="Path to the driver's (or self) Mu-exported CSV file.",
    )
    parser.add_argument(
        "--reference-csv",
        type=str,
        default=None,
        help="Optional faster-reference Mu CSV file. If provided, driver-vs-reference mode is used.",
    )
    args = parser.parse_args()

    driver_csv_path = Path(args.csv_file)
    if not driver_csv_path.exists():
        raise FileNotFoundError(f"File not found: {driver_csv_path}")

    reference_csv_path = Path(args.reference_csv) if args.reference_csv else None
    if reference_csv_path is not None and not reference_csv_path.exists():
        raise FileNotFoundError(f"Reference file not found: {reference_csv_path}")

    print(f"\nPython interpreter: {sys.executable}")
    driver_session = _analyze_single_session(driver_csv_path, label="Driver/Self Session")

    if reference_csv_path is None:
        _run_single_session_mode(driver_session)
        return

    reference_session = _analyze_single_session(
        reference_csv_path, label="Reference Session"
    )
    _run_vs_reference_mode(driver_session, reference_session)


if __name__ == "__main__":
    main()
