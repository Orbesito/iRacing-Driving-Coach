from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LapValidityConfig:
    """
    Deterministic rules for identifying laps suitable for coaching comparison.
    """

    min_lap_id: int = 1
    min_samples: int = 1000
    min_dist_start_pct: float = 1.0
    min_dist_end_pct: float = 99.0
    min_dist_span_pct: float = 95.0
    max_pit_fraction: float = 0.0
    min_ontrack_fraction: float = 0.99


DEFAULT_ALIGNMENT_CHANNELS = [
    "Speed",
    "Speed_kmh",
    "Throttle",
    "ThrottleRaw",
    "Brake",
    "BrakeRaw",
    "SteeringWheelAngle",
    "SteeringWheelAngle_deg",
    "YawRate",
    "YawRate_deg_s",
    "LatAccel",
    "LongAccel",
    "VertAccel",
    "VelocityX",
    "VelocityY",
    "VelocityZ",
    "LFspeed",
    "RFspeed",
    "LRspeed",
    "RRspeed",
    "Lat",
    "Lon",
    "Gear",
    "RPM",
]

DEFAULT_DISTANCE_STEP_PCT = 0.1


def _lap_id_series(df: pd.DataFrame, lap_col: str) -> pd.Series:
    """
    Build deterministic integer lap IDs from telemetry.
    Rows with non-integer lap values are dropped from lap analysis.
    """
    lap_numeric = pd.to_numeric(df[lap_col], errors="coerce")
    lap_rounded = lap_numeric.round()
    is_integer_like = (lap_numeric - lap_rounded).abs() <= 1e-6
    lap_clean = lap_rounded.where(is_integer_like, np.nan)
    return lap_clean.astype("Int64")


def build_lap_summary(
    df: pd.DataFrame,
    config: LapValidityConfig = LapValidityConfig(),
    lap_col: str = "Lap",
    dist_col: str = "LapDistPct",
) -> pd.DataFrame:
    """
    Aggregate lap-level statistics and classify valid laps.
    """
    if lap_col not in df.columns:
        raise ValueError(f"Required lap column not found: {lap_col}")
    if dist_col not in df.columns:
        raise ValueError(f"Required distance column not found: {dist_col}")

    lap_ids = _lap_id_series(df, lap_col)
    working = pd.DataFrame(
        {
            "_lap_id": lap_ids,
            "_dist_pct": pd.to_numeric(df[dist_col], errors="coerce"),
        }
    )

    if "OnPitRoad" in df.columns:
        working["_on_pit_road"] = pd.to_numeric(df["OnPitRoad"], errors="coerce")
    if "IsOnTrack" in df.columns:
        working["_is_on_track"] = pd.to_numeric(df["IsOnTrack"], errors="coerce")

    working = working.dropna(subset=["_lap_id", "_dist_pct"])
    grouped = working.groupby("_lap_id", dropna=True)

    summary = grouped["_dist_pct"].agg(["size", "min", "max"]).rename(
        columns={"size": "samples", "min": "dist_min_pct", "max": "dist_max_pct"}
    )
    summary["dist_span_pct"] = summary["dist_max_pct"] - summary["dist_min_pct"]

    if "_on_pit_road" in working.columns:
        summary["pit_fraction"] = grouped["_on_pit_road"].apply(
            lambda s: float((s.fillna(0.0) > 0.5).mean())
        )
    else:
        summary["pit_fraction"] = 0.0

    if "_is_on_track" in working.columns:
        summary["ontrack_fraction"] = grouped["_is_on_track"].apply(
            lambda s: float((s.fillna(1.0) > 0.5).mean())
        )
    else:
        summary["ontrack_fraction"] = 1.0

    summary = summary.reset_index().rename(columns={"_lap_id": "lap_id"})
    summary["lap_id"] = summary["lap_id"].astype(int)

    def _evaluate_row(row: pd.Series) -> pd.Series:
        reasons: List[str] = []

        if row["lap_id"] < config.min_lap_id:
            reasons.append(f"lap_id_below_{config.min_lap_id}")
        if row["samples"] < config.min_samples:
            reasons.append(f"samples_below_{config.min_samples}")
        if row["dist_min_pct"] > config.min_dist_start_pct:
            reasons.append(f"dist_start_above_{config.min_dist_start_pct}")
        if row["dist_max_pct"] < config.min_dist_end_pct:
            reasons.append(f"dist_end_below_{config.min_dist_end_pct}")
        if row["dist_span_pct"] < config.min_dist_span_pct:
            reasons.append(f"dist_span_below_{config.min_dist_span_pct}")
        if row["pit_fraction"] > config.max_pit_fraction:
            reasons.append(f"pit_fraction_above_{config.max_pit_fraction}")
        if row["ontrack_fraction"] < config.min_ontrack_fraction:
            reasons.append(f"ontrack_fraction_below_{config.min_ontrack_fraction}")

        return pd.Series(
            {
                "is_valid": len(reasons) == 0,
                "invalid_reasons": "|".join(reasons),
            }
        )

    eval_df = summary.apply(_evaluate_row, axis=1)
    summary = pd.concat([summary, eval_df], axis=1)
    summary = summary.sort_values("lap_id").reset_index(drop=True)

    return summary


def get_valid_lap_ids(lap_summary: pd.DataFrame) -> List[int]:
    """Return deterministic valid lap ID list from a lap summary table."""
    if "lap_id" not in lap_summary.columns or "is_valid" not in lap_summary.columns:
        raise ValueError("lap_summary must contain 'lap_id' and 'is_valid' columns.")
    return (
        lap_summary.loc[lap_summary["is_valid"], "lap_id"]
        .astype(int)
        .sort_values()
        .tolist()
    )


def _prepare_lap_for_alignment(
    lap_df: pd.DataFrame,
    dist_col: str,
    channels: List[str],
) -> pd.DataFrame:
    """
    Prepare a single lap for interpolation by sorting and de-duplicating distance.
    """
    cols = [dist_col] + channels
    lap_data = lap_df[cols].copy()
    lap_data[dist_col] = pd.to_numeric(lap_data[dist_col], errors="coerce")
    lap_data = lap_data.dropna(subset=[dist_col])
    lap_data[dist_col] = lap_data[dist_col].clip(0.0, 100.0)
    lap_data = lap_data.sort_values(dist_col)
    lap_data = lap_data.groupby(dist_col, as_index=False).mean(numeric_only=True)
    return lap_data


def align_laps_by_distance(
    df: pd.DataFrame,
    lap_ids: List[int],
    distance_step_pct: float = 0.1,
    channels: List[str] | None = None,
    lap_col: str = "Lap",
    dist_col: str = "LapDistPct",
) -> tuple[pd.DataFrame, Dict[str, object]]:
    """
    Interpolate telemetry channels onto a shared distance grid for lap-to-lap comparison.
    """
    if distance_step_pct <= 0:
        raise ValueError("distance_step_pct must be > 0.")
    if lap_col not in df.columns:
        raise ValueError(f"Required lap column not found: {lap_col}")
    if dist_col not in df.columns:
        raise ValueError(f"Required distance column not found: {dist_col}")

    requested_channels = channels or DEFAULT_ALIGNMENT_CHANNELS
    alignment_channels = [col for col in requested_channels if col in df.columns]
    if not alignment_channels:
        raise ValueError("No requested alignment channels are present in telemetry.")

    lap_id_col = _lap_id_series(df, lap_col)
    distance_grid = np.arange(0.0, 100.0 + distance_step_pct / 2.0, distance_step_pct)
    distance_grid = np.round(distance_grid, 6)

    aligned_rows: List[pd.DataFrame] = []
    aligned_lap_count = 0

    for lap_id in sorted(set(lap_ids)):
        lap_mask = lap_id_col == lap_id
        lap_df = df.loc[lap_mask, [dist_col] + alignment_channels]
        if lap_df.empty:
            continue

        prepared = _prepare_lap_for_alignment(lap_df, dist_col, alignment_channels)
        x = prepared[dist_col].to_numpy(dtype=float)
        if x.size < 2:
            continue

        aligned_lap = pd.DataFrame(
            {
                "lap_id": lap_id,
                "distance_pct": distance_grid,
            }
        )
        for channel in alignment_channels:
            y = prepared[channel].to_numpy(dtype=float)
            aligned_lap[channel] = np.interp(
                distance_grid,
                x,
                y,
                left=np.nan,
                right=np.nan,
            )

        aligned_rows.append(aligned_lap)
        aligned_lap_count += 1

    if aligned_rows:
        aligned_df = pd.concat(aligned_rows, ignore_index=True)
    else:
        aligned_df = pd.DataFrame(columns=["lap_id", "distance_pct"] + alignment_channels)

    report: Dict[str, object] = {
        "requested_lap_count": int(len(set(lap_ids))),
        "aligned_lap_count": int(aligned_lap_count),
        "distance_step_pct": float(distance_step_pct),
        "distance_grid_points": int(len(distance_grid)),
        "alignment_channels": alignment_channels,
    }

    return aligned_df, report
