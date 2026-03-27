from __future__ import annotations

import math
from typing import Dict

import pandas as pd


def add_derived_units(df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, str]]:
    """
    Add derived channels and return both:
    - updated dataframe
    - units for the derived channels
    """
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
        # Never overwrite an existing channel: raw and derived channels stay separate.
        if derived_col in df.columns:
            raise ValueError(
                f"Refusing to overwrite existing telemetry column: {derived_col}"
            )
        df[derived_col] = df[source_col] * factor
        derived_units[derived_col] = derived_unit

    return df, derived_units
