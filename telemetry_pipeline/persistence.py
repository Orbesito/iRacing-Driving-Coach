from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pandas as pd


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


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
