from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import pandas as pd
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: pandas.\n"
        f"Interpreter in use: {sys.executable}\n"
        "Install with:\n"
        f'  "{sys.executable}" -m pip install pandas'
    ) from exc


def _is_blank_row(row: List[str]) -> bool:
    return not any(cell.strip() for cell in row)


def _normalise_metadata_value(values: List[str]) -> str:
    cleaned = [value.strip() for value in values if value.strip()]
    return ", ".join(cleaned)


def _parse_mu_preamble(path: Path) -> Tuple[Dict[str, str], List[str], List[str], int]:
    """
    Read only metadata/header/units and return the first telemetry data row index.
    """
    metadata: Dict[str, str] = {}
    header: List[str] | None = None
    units: List[str] | None = None
    data_start_row: int | None = None
    state = "metadata"

    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.reader(csv_file)
        for row_number, raw_row in enumerate(reader):
            row = [cell.strip() for cell in raw_row]

            if state == "metadata":
                if _is_blank_row(row):
                    state = "seek_header"
                    continue
                if len(row) >= 2 and row[0]:
                    metadata[row[0]] = _normalise_metadata_value(row[1:])
                continue

            if state == "seek_header":
                if _is_blank_row(row):
                    continue
                header = row
                state = "seek_units"
                continue

            if state == "seek_units":
                if _is_blank_row(row):
                    continue
                units = row
                if len(header) != len(units):
                    raise ValueError(
                        "Mu CSV header/units length mismatch: "
                        f"{len(header)} headers vs {len(units)} units."
                    )
                state = "seek_data"
                continue

            if state == "seek_data":
                if _is_blank_row(row):
                    continue
                data_start_row = row_number
                break

    if header is None or units is None:
        raise ValueError("Could not find header and units rows in the Mu CSV.")
    if data_start_row is None:
        raise ValueError("No telemetry samples found after the header/units rows.")

    return metadata, header, units, data_start_row


def load_mu_csv(path: str | Path):
    path = Path(path)
    metadata, header, units, data_start_row = _parse_mu_preamble(path)

    df = pd.read_csv(
        path,
        skiprows=data_start_row,
        names=header,
        header=None,
        encoding="utf-8-sig",
        low_memory=False,
        on_bad_lines="error",
    )

    df = df.apply(pd.to_numeric, errors="coerce")
    raw_rows_loaded = len(df)
    df = df.dropna(how="all").reset_index(drop=True)
    blank_rows_dropped = raw_rows_loaded - len(df)

    units_map = dict(zip(header, units))
    parse_report = {
        "channels": len(header),
        "raw_rows_loaded": int(raw_rows_loaded),
        "blank_rows_dropped": int(blank_rows_dropped),
        "final_rows": int(len(df)),
    }

    return metadata, units_map, df, parse_report


def add_derived_units(df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, str]]:
    df = df.copy()
    derived_units: Dict[str, str] = {}

    conversions = [
        ("Speed", "Speed_kmh", 3.6, "km/h"),
        ("SteeringWheelAngle", "SteeringWheelAngle_deg", 180.0 / math.pi, "deg"),
        ("YawRate", "YawRate_deg_s", 180.0 / math.pi, "deg/s"),
    ]

    for source_col, derived_col, factor, derived_unit in conversions:
        if source_col not in df.columns:
            continue
        if derived_col in df.columns:
            raise ValueError(
                f"Refusing to overwrite existing telemetry column: {derived_col}"
            )
        df[derived_col] = df[source_col] * factor
        derived_units[derived_col] = derived_unit

    return df, derived_units


def main():
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


if __name__ == "__main__":
    main()
