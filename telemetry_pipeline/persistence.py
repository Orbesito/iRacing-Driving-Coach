from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict

import pandas as pd


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "-", value.strip().lower())
    cleaned = cleaned.strip("-")
    return cleaned or "unknown"


def _format_session_date(raw_date: str) -> str:
    if not raw_date:
        return "unknown-date"
    for fmt in ("%d/%m/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw_date.strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return _slugify(raw_date)


def _format_session_time(raw_time: str) -> str:
    if not raw_time:
        return "unknown-time"
    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            return datetime.strptime(raw_time.strip(), fmt).strftime("%H-%M-%S")
        except ValueError:
            continue
    return _slugify(raw_time)


def build_session_output_dir(
    base_dir: str | Path,
    source_csv: str | Path,
    metadata: dict[str, str],
) -> Path:
    """
    Build a deterministic per-session output directory:
    outputs/<date>__<time>__<vehicle>__<venue>__<session>__<driver>
    Falls back to CSV stem when metadata is incomplete.
    """
    base = Path(base_dir)
    source_stem = _slugify(Path(source_csv).stem)

    session_date = _format_session_date(metadata.get("Session Date", ""))
    session_time = _format_session_time(metadata.get("Session Time", ""))
    vehicle = _slugify(metadata.get("Vehicle", ""))
    venue = _slugify(metadata.get("Venue", ""))
    session_type = _slugify(metadata.get("Session", ""))
    driver = _slugify(metadata.get("Driver", ""))

    folder_name = "__".join(
        [session_date, session_time, vehicle, venue, session_type, driver]
    )
    if not folder_name.replace("-", ""):
        folder_name = source_stem

    # Keep Windows path length reasonable.
    folder_name = folder_name[:140].rstrip("-_")
    if not folder_name:
        folder_name = source_stem

    candidate = base / folder_name
    if not candidate.exists():
        return candidate

    suffix = 2
    while True:
        alt = base / f"{folder_name}__run{suffix:02d}"
        if not alt.exists():
            return alt
        suffix += 1


def save_session_bundle(
    out_dir: str | Path,
    telemetry_df: pd.DataFrame,
    metadata: dict[str, str],
    units_map: dict[str, str],
    parse_report: dict[str, int],
) -> Dict[str, Path]:
    """
    Persist loaded session data for downstream pipeline steps.

    Outputs:
    - telemetry_numeric.csv
    - metadata.json
    - units.json
    - parse_report.json
    """
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    telemetry_path = output_dir / "telemetry_numeric.csv"
    metadata_path = output_dir / "metadata.json"
    units_path = output_dir / "units.json"
    parse_report_path = output_dir / "parse_report.json"

    telemetry_df.to_csv(telemetry_path, index=False)
    _write_json(metadata_path, metadata)
    _write_json(units_path, units_map)
    _write_json(parse_report_path, parse_report)

    return {
        "telemetry_csv": telemetry_path,
        "metadata_json": metadata_path,
        "units_json": units_path,
        "parse_report_json": parse_report_path,
    }


def save_lap_analysis(
    out_dir: str | Path,
    lap_summary_df: pd.DataFrame,
    aligned_laps_df: pd.DataFrame | None,
    alignment_report: dict,
) -> Dict[str, Path]:
    """
    Persist lap validity and distance-aligned traces for comparison workflows.
    """
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lap_summary_path = output_dir / "lap_summary.csv"
    aligned_laps_path = output_dir / "aligned_laps_by_distance.csv"
    alignment_report_path = output_dir / "alignment_report.json"

    lap_summary_df.to_csv(lap_summary_path, index=False)
    if aligned_laps_df is None:
        aligned_laps_df = pd.DataFrame(columns=["lap_id", "distance_pct"])

    aligned_laps_df.to_csv(aligned_laps_path, index=False)
    _write_json(alignment_report_path, alignment_report)

    return {
        "lap_summary_csv": lap_summary_path,
        "aligned_laps_csv": aligned_laps_path,
        "alignment_report_json": alignment_report_path,
    }


def save_corner_analysis(
    out_dir: str | Path,
    corner_definitions_df: pd.DataFrame,
    corner_lap_metrics_df: pd.DataFrame,
    corner_ranking_df: pd.DataFrame,
    corner_report: dict,
) -> Dict[str, Path]:
    """
    Persist per-corner definitions, per-lap corner metrics, and coaching ranking outputs.
    """
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    corner_definitions_path = output_dir / "corner_definitions.csv"
    corner_lap_metrics_path = output_dir / "corner_lap_metrics.csv"
    corner_ranking_path = output_dir / "corner_ranking.csv"
    corner_report_path = output_dir / "corner_report.json"

    corner_definitions_df.to_csv(corner_definitions_path, index=False)
    corner_lap_metrics_df.to_csv(corner_lap_metrics_path, index=False)
    corner_ranking_df.to_csv(corner_ranking_path, index=False)
    _write_json(corner_report_path, corner_report)

    return {
        "corner_definitions_csv": corner_definitions_path,
        "corner_lap_metrics_csv": corner_lap_metrics_path,
        "corner_ranking_csv": corner_ranking_path,
        "corner_report_json": corner_report_path,
    }
