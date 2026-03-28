from __future__ import annotations

import csv
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


MetadataMap = Dict[str, str]
UnitsMap = Dict[str, str]
ParseReport = Dict[str, int]


def _is_blank_row(row: List[str]) -> bool:
    """Return True when a CSV row has no non-whitespace content."""
    return not any(cell.strip() for cell in row)


def _normalise_metadata_value(values: List[str]) -> str:
    """Normalize metadata values into a single printable string."""
    cleaned = [value.strip() for value in values if value.strip()]
    return ", ".join(cleaned)


def _parse_mu_preamble(path: Path) -> Tuple[MetadataMap, List[str], List[str], int]:
    """
    Read only metadata/header/units and return the first telemetry data row index.
    """
    metadata: MetadataMap = {}
    header: List[str] | None = None
    units: List[str] | None = None
    data_start_row: int | None = None
    # Explicit parser state keeps the CSV preamble logic deterministic and easy to audit.
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
                # Allow any number of blank lines before the header row.
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
                # First non-blank row after units is the telemetry data start.
                if _is_blank_row(row):
                    continue
                data_start_row = row_number
                break

    if header is None or units is None:
        raise ValueError("Could not find header and units rows in the Mu CSV.")
    if data_start_row is None:
        raise ValueError("No telemetry samples found after the header/units rows.")

    return metadata, header, units, data_start_row


def load_mu_csv(path: str | Path) -> tuple[MetadataMap, UnitsMap, pd.DataFrame, ParseReport]:
    """
    Load a Mu-exported CSV into:
    - metadata map
    - units map
    - numeric telemetry dataframe
    - parse report summary
    """
    path = Path(path)
    metadata, header, units, data_start_row = _parse_mu_preamble(path)
    print("\n... Reading telemetry samples from CSV", flush=True)
    df = pd.read_csv(
        path,
        skiprows=data_start_row,
        names=header,
        header=None,
        encoding="utf-8-sig",
        low_memory=False,
        on_bad_lines="error",
    )
    print("\n... Converting telemetry values to numeric", flush=True)
    # Coerce telemetry values to numeric while preserving deterministic NaN behavior.
    df = df.apply(pd.to_numeric, errors="coerce")
    raw_rows_loaded = len(df)
    df = df.dropna(how="all").reset_index(drop=True)
    blank_rows_dropped = raw_rows_loaded - len(df)
    print("\n... Building units and parse report", flush=True)

    units_map = dict(zip(header, units))
    parse_report: ParseReport = {
        "channels": len(header),
        "raw_rows_loaded": int(raw_rows_loaded),
        "blank_rows_dropped": int(blank_rows_dropped),
        "final_rows": int(len(df)),
    }
    print("\n... CSV load complete", flush=True)
    return metadata, units_map, df, parse_report
