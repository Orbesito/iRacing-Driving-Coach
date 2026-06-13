from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CornerDetectionConfig:
    """
    Deterministic corner detection and phase boundary parameters.

    Practical tuning guide:
    - Road/street circuits: keep defaults as baseline.
    - Ovals/high-speed layouts: usually lower min/max/target corner counts and
      increase minimum corner spacing to avoid splitting long arcs.
    - Low-speed technical tracks: sometimes benefit from slightly lower
      spacing and/or lower activity threshold.
    """

    # Signal smoothing / peak pre-processing.
    speed_smoothing_window_points: int = 9
    activity_smoothing_window_points: int = 5
    activity_quantile_threshold: float = 0.50

    # Apex count and spacing controls (main track-type tuning knobs).
    min_corner_count: int = 10
    max_corner_count: int = 25
    min_corner_spacing_pct: float = 0.8
    min_corner_spacing_m: float = 90.0

    # Apex refinement and minimum activity gate.
    speed_apex_search_window_pct: float = 1.5
    min_activity_score: float = 0.12

    # Optional geometric curvature contribution from Lat/Lon.
    curvature_smoothing_window_points: int = 9
    curvature_weight: float = 0.35

    # Phase splitting thresholds.
    brake_threshold_pct: float = 5.0
    throttle_reapply_threshold_pct: float = 10.0
    throttle_reapply_consecutive_points: int = 3

    # Optional hard target corner count for tracks with known official count.
    target_corner_count: int | None = None

    # Optional official turn map. When provided, these apex percentages replace
    # automatic peak selection but still use telemetry-derived phase boundaries.
    manual_apex_pct: List[float] | None = None


BRAKE_CHANNEL_PRIORITY = ["BrakeRaw", "Brake"]
THROTTLE_CHANNEL_PRIORITY = ["ThrottleRaw", "Throttle"]
WHEEL_SPEED_CHANNELS = ["LFspeed", "RFspeed", "LRspeed", "RRspeed"]
REFERENCE_METRIC_MAP = {
    "corner_time_s": "ref_corner_time_s",
    "apex_speed_kmh": "ref_apex_speed_kmh",
    "traction_reapply_delay_pct": "ref_traction_reapply_delay_pct",
    "braking_time_s": "ref_braking_time_s",
    "rotation_time_s": "ref_rotation_time_s",
    "traction_time_s": "ref_traction_time_s",
    "brake_peak_pct": "ref_brake_peak_pct",
    "brake_mean_pct": "ref_brake_mean_pct",
    "brake_raw_peak_pct": "ref_brake_raw_peak_pct",
    "brake_raw_mean_pct": "ref_brake_raw_mean_pct",
    "rotation_mean_abs_yaw_rate_deg_s": "ref_rotation_mean_abs_yaw_rate_deg_s",
    "rotation_mean_abs_lat_accel_g": "ref_rotation_mean_abs_lat_accel_g",
    "rotation_mean_abs_steer_deg": "ref_rotation_mean_abs_steer_deg",
    "rotation_mean_body_slip_ratio": "ref_rotation_mean_body_slip_ratio",
    "traction_exit_throttle_mean_pct": "ref_traction_exit_throttle_mean_pct",
    "traction_exit_throttle_raw_mean_pct": "ref_traction_exit_throttle_raw_mean_pct",
    "traction_exit_long_accel_mean": "ref_traction_exit_long_accel_mean",
    "traction_wheel_speed_std_mps": "ref_traction_wheel_speed_std_mps",
    "apex_lat": "ref_apex_lat",
    "apex_lon": "ref_apex_lon",
}


def _lap_id_series(df: pd.DataFrame, lap_col: str = "Lap") -> pd.Series:
    lap_numeric = pd.to_numeric(df[lap_col], errors="coerce")
    lap_rounded = lap_numeric.round()
    is_integer_like = (lap_numeric - lap_rounded).abs() <= 1e-6
    lap_clean = lap_rounded.where(is_integer_like, np.nan)
    return lap_clean.astype("Int64")


def _first_available_column(df: pd.DataFrame, candidates: List[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def compute_lap_times(
    telemetry_df: pd.DataFrame,
    lap_ids: List[int],
    lap_col: str = "Lap",
    session_time_col: str = "SessionTime",
) -> pd.DataFrame:
    """
    Compute deterministic lap time estimates from SessionTime per valid lap.
    """
    if lap_col not in telemetry_df.columns:
        raise ValueError(f"Required lap column not found: {lap_col}")
    if session_time_col not in telemetry_df.columns:
        raise ValueError(f"Required session time column not found: {session_time_col}")

    lap_id = _lap_id_series(telemetry_df, lap_col)
    session_time = pd.to_numeric(telemetry_df[session_time_col], errors="coerce")

    working = pd.DataFrame({"lap_id": lap_id, "session_time_s": session_time})
    working = working.dropna(subset=["lap_id", "session_time_s"])
    working["lap_id"] = working["lap_id"].astype(int)
    working = working[working["lap_id"].isin(set(lap_ids))]

    grouped = working.groupby("lap_id", dropna=True)["session_time_s"]
    lap_times = grouped.agg(["min", "max"]).rename(
        columns={"min": "start_time_s", "max": "end_time_s"}
    )
    lap_times["lap_time_s"] = lap_times["end_time_s"] - lap_times["start_time_s"]
    lap_times["lap_time_s"] = lap_times["lap_time_s"].where(
        lap_times["lap_time_s"] > 0.0, np.nan
    )

    return lap_times.reset_index().sort_values("lap_id").reset_index(drop=True)


def choose_reference_lap(lap_times_df: pd.DataFrame) -> int:
    """
    Choose reference lap deterministically as the fastest valid lap.
    """
    if lap_times_df.empty:
        raise ValueError("Cannot choose reference lap from empty lap_times_df.")

    valid = lap_times_df.dropna(subset=["lap_time_s"]).copy()
    if valid.empty:
        return int(lap_times_df["lap_id"].min())

    fastest = valid.sort_values(["lap_time_s", "lap_id"]).iloc[0]
    return int(fastest["lap_id"])


def _speed_series_kmh(aligned_lap_df: pd.DataFrame) -> pd.Series:
    if "Speed_kmh" in aligned_lap_df.columns:
        speed_kmh = pd.to_numeric(aligned_lap_df["Speed_kmh"], errors="coerce")
    elif "Speed" in aligned_lap_df.columns:
        speed_kmh = pd.to_numeric(aligned_lap_df["Speed"], errors="coerce") * 3.6
    else:
        raise ValueError("Aligned lap data must include 'Speed_kmh' or 'Speed'.")

    speed_kmh = speed_kmh.replace([np.inf, -np.inf], np.nan)
    speed_kmh = speed_kmh.interpolate(limit_direction="both")
    if speed_kmh.isna().any():
        speed_kmh = speed_kmh.fillna(speed_kmh.median())
    return speed_kmh


def _circular_distance_pct(a: float, b: float) -> float:
    direct = abs(a - b)
    return min(direct, 100.0 - direct)


def _normalise_feature_abs(series: pd.Series, quantile: float = 0.99) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").abs().fillna(0.0)
    scale = float(np.nanquantile(values.to_numpy(dtype=float), quantile))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = float(values.max())
    if not np.isfinite(scale) or scale <= 0.0:
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values / scale).clip(lower=0.0, upper=1.0)


def _estimate_lap_length_m_from_lat_lon(ref_lap_df: pd.DataFrame) -> float:
    """
    Estimate lap length from Lat/Lon samples (deterministic, lap-local).
    """
    if "Lat" not in ref_lap_df.columns or "Lon" not in ref_lap_df.columns:
        return np.nan

    lat_deg = pd.to_numeric(ref_lap_df["Lat"], errors="coerce").to_numpy(dtype=float)
    lon_deg = pd.to_numeric(ref_lap_df["Lon"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(lat_deg) & np.isfinite(lon_deg)
    lat_deg = lat_deg[valid]
    lon_deg = lon_deg[valid]
    if lat_deg.size < 3:
        return np.nan

    lat_rad = np.deg2rad(lat_deg)
    lon_rad = np.deg2rad(lon_deg)
    dlat = np.diff(lat_rad)
    dlon = np.diff(lon_rad)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat_rad[:-1]) * np.cos(lat_rad[1:]) * np.sin(
        dlon / 2.0
    ) ** 2
    c = 2.0 * np.arctan2(np.sqrt(np.clip(a, 0.0, 1.0)), np.sqrt(np.clip(1.0 - a, 0.0, 1.0)))
    return float(np.nansum(6_371_000.0 * c))


def _effective_min_corner_spacing_pct(ref_lap_df: pd.DataFrame, config: CornerDetectionConfig) -> float:
    """
    Compute track-agnostic minimum apex spacing in percent-lap.
    Uses a meter-based spacing target when lap length can be estimated.
    """
    spacing_pct = float(max(config.min_corner_spacing_pct, 0.15))
    spacing_m = float(max(config.min_corner_spacing_m, 10.0))

    lap_length_m = _estimate_lap_length_m_from_lat_lon(ref_lap_df)
    if np.isfinite(lap_length_m) and lap_length_m > 100.0:
        spacing_from_m_pct = (spacing_m / lap_length_m) * 100.0
        spacing_pct = max(spacing_pct, float(spacing_from_m_pct))

    return float(np.clip(spacing_pct, 0.15, 8.0))


def _path_curvature_activity(
    lap_df: pd.DataFrame,
    smoothing_window_points: int,
) -> pd.Series:
    """
    Compute a deterministic curvature proxy from Lat/Lon path geometry.
    Returns zeros if path channels are unavailable.
    """
    if "Lat" not in lap_df.columns or "Lon" not in lap_df.columns:
        return pd.Series(np.zeros(len(lap_df)), index=lap_df.index)

    lat = pd.to_numeric(lap_df["Lat"], errors="coerce").interpolate(limit_direction="both")
    lon = pd.to_numeric(lap_df["Lon"], errors="coerce").interpolate(limit_direction="both")
    if lat.dropna().empty or lon.dropna().empty:
        return pd.Series(np.zeros(len(lap_df)), index=lap_df.index)

    lat = lat.fillna(lat.median())
    lon = lon.fillna(lon.median())
    mean_lat_rad = np.deg2rad(float(lat.mean()))

    # Local tangent-plane approximation is sufficient for one circuit lap.
    x_m = (lon - float(lon.mean())) * (111_320.0 * np.cos(mean_lat_rad))
    y_m = (lat - float(lat.mean())) * 111_320.0

    dx = np.gradient(x_m.to_numpy(dtype=float))
    dy = np.gradient(y_m.to_numpy(dtype=float))
    heading = np.unwrap(np.arctan2(dy, dx))

    distance_pct = pd.to_numeric(lap_df["distance_pct"], errors="coerce").to_numpy(dtype=float)
    distance_pct = np.nan_to_num(distance_pct, nan=0.0)
    if len(distance_pct) >= 2:
        for i in range(1, len(distance_pct)):
            if distance_pct[i] <= distance_pct[i - 1]:
                distance_pct[i] = distance_pct[i - 1] + 1e-6

    curvature = np.abs(np.gradient(heading, distance_pct))
    curvature_series = pd.Series(curvature, index=lap_df.index)
    curvature_series = curvature_series.rolling(
        window=smoothing_window_points, center=True, min_periods=1
    ).mean()
    return _normalise_feature_abs(curvature_series, quantile=0.98)


def _select_spaced_candidates(
    candidates_df: pd.DataFrame,
    score_col: str,
    min_spacing_pct: float,
    max_count: int,
) -> List[float]:
    selected: List[float] = []
    for _, row in candidates_df.sort_values(score_col, ascending=False).iterrows():
        candidate_dist = float(row["distance_pct"])
        if all(
            _circular_distance_pct(candidate_dist, existing) >= min_spacing_pct
            for existing in selected
        ):
            selected.append(candidate_dist)
        if len(selected) >= max_count:
            break
    return sorted(selected)


def _select_target_count_candidates(
    candidates_df: pd.DataFrame,
    score_col: str,
    target_count: int,
    initial_spacing_pct: float,
) -> List[float]:
    spacing = max(0.2, float(initial_spacing_pct))
    selected = _select_spaced_candidates(
        candidates_df,
        score_col=score_col,
        min_spacing_pct=spacing,
        max_count=target_count,
    )

    # If spacing is too strict to reach target_count, progressively relax it.
    while len(selected) < target_count and spacing > 0.2:
        spacing *= 0.85
        selected = _select_spaced_candidates(
            candidates_df,
            score_col=score_col,
            min_spacing_pct=spacing,
            max_count=target_count,
        )

    # Final fallback: add highest-score leftovers with only minimal separation.
    if len(selected) < target_count:
        all_candidates = (
            candidates_df.sort_values(score_col, ascending=False)["distance_pct"]
            .astype(float)
            .tolist()
        )
        for candidate in all_candidates:
            if candidate in selected:
                continue
            if all(_circular_distance_pct(candidate, existing) >= 0.15 for existing in selected):
                selected.append(candidate)
            if len(selected) >= target_count:
                break

    # Keep deterministic size/ordering if fallback overfilled the list.
    if len(selected) > target_count:
        selected_df = candidates_df.loc[
            candidates_df["distance_pct"].isin([round(x, 6) for x in selected])
        ].copy()
        selected = (
            selected_df.sort_values(score_col, ascending=False)
            .head(target_count)["distance_pct"]
            .astype(float)
            .tolist()
        )

    return sorted(selected)


def _find_traction_start_distance(
    post_apex: pd.DataFrame,
    throttle_col: str,
    threshold_pct: float,
    consecutive_points: int,
    fallback_distance: float,
) -> float:
    if post_apex.empty or throttle_col not in post_apex.columns:
        return fallback_distance

    throttle = pd.to_numeric(post_apex[throttle_col], errors="coerce").fillna(0.0)
    mask = (throttle >= threshold_pct).to_numpy(dtype=bool)
    run = 0
    for idx, is_above in enumerate(mask):
        run = run + 1 if is_above else 0
        if run >= consecutive_points:
            first_idx = idx - consecutive_points + 1
            return float(post_apex.iloc[first_idx]["distance_pct"])
    return fallback_distance


def _find_phase_boundaries(
    ref_lap_df: pd.DataFrame,
    corner_start_pct: float,
    apex_pct: float,
    corner_end_pct: float,
    config: CornerDetectionConfig,
) -> Dict[str, float]:
    segment = ref_lap_df.loc[
        (ref_lap_df["distance_pct"] >= corner_start_pct)
        & (ref_lap_df["distance_pct"] <= corner_end_pct)
    ].copy()
    if segment.empty:
        return {
            "brake_start_pct": corner_start_pct,
            "brake_end_pct": apex_pct,
            "rotation_start_pct": apex_pct,
            "rotation_end_pct": apex_pct,
            "traction_start_pct": apex_pct,
        }

    pre_apex = segment.loc[segment["distance_pct"] <= apex_pct]
    post_apex = segment.loc[segment["distance_pct"] >= apex_pct]

    brake_start_pct = corner_start_pct
    brake_end_pct = apex_pct
    braking_detected = False
    brake_col = _first_available_column(pre_apex, BRAKE_CHANNEL_PRIORITY)
    if brake_col is not None and not pre_apex.empty:
        brake = pd.to_numeric(pre_apex[brake_col], errors="coerce").fillna(0.0)
        braking_mask = brake >= config.brake_threshold_pct
        if braking_mask.any():
            braking_detected = True
            braking_rows = pre_apex.loc[braking_mask]
            brake_start_pct = float(braking_rows["distance_pct"].iloc[0])
            brake_end_pct = float(braking_rows["distance_pct"].iloc[-1])
        else:
            # No braking signal before apex: treat this corner as a lift/steer phase,
            # not a braking phase, to avoid misleading downstream coaching.
            brake_start_pct = float(apex_pct)
            brake_end_pct = float(apex_pct)
    else:
        brake_start_pct = float(apex_pct)
        brake_end_pct = float(apex_pct)

    brake_end_pct = min(brake_end_pct, apex_pct)
    rotation_start_pct = brake_end_pct if braking_detected else float(corner_start_pct)
    throttle_col = _first_available_column(post_apex, THROTTLE_CHANNEL_PRIORITY)
    traction_start_pct = _find_traction_start_distance(
        post_apex=post_apex,
        throttle_col=throttle_col if throttle_col is not None else "Throttle",
        threshold_pct=config.throttle_reapply_threshold_pct,
        consecutive_points=config.throttle_reapply_consecutive_points,
        fallback_distance=apex_pct,
    )
    traction_start_pct = max(traction_start_pct, apex_pct)
    rotation_end_pct = min(max(traction_start_pct, rotation_start_pct), corner_end_pct)

    return {
        "brake_start_pct": float(brake_start_pct),
        "brake_end_pct": float(brake_end_pct),
        "rotation_start_pct": float(rotation_start_pct),
        "rotation_end_pct": float(rotation_end_pct),
        "traction_start_pct": float(traction_start_pct),
    }


def detect_main_corners(
    aligned_laps_df: pd.DataFrame,
    reference_lap_id: int,
    config: CornerDetectionConfig = CornerDetectionConfig(),
) -> pd.DataFrame:
    """
    Detect main corner apex points from the reference lap and define deterministic
    corner + phase boundaries.
    """
    if "lap_id" not in aligned_laps_df.columns or "distance_pct" not in aligned_laps_df.columns:
        raise ValueError("aligned_laps_df must contain 'lap_id' and 'distance_pct'.")

    ref = aligned_laps_df.loc[aligned_laps_df["lap_id"] == reference_lap_id].copy()
    if ref.empty:
        raise ValueError(f"Reference lap {reference_lap_id} not found in aligned_laps_df.")

    ref = ref.sort_values("distance_pct").reset_index(drop=True)
    ref["speed_kmh_for_detection"] = _speed_series_kmh(ref)
    ref["speed_kmh_smooth"] = (
        ref["speed_kmh_for_detection"]
        .rolling(window=config.speed_smoothing_window_points, center=True, min_periods=1)
        .mean()
    )

    speed_high = float(np.nanquantile(ref["speed_kmh_smooth"].to_numpy(dtype=float), 0.90))
    speed_high = max(speed_high, 1.0)
    speed_deficit = (speed_high - ref["speed_kmh_smooth"]).clip(lower=0.0) / speed_high

    if "YawRate_deg_s" in ref.columns:
        yaw_abs = pd.to_numeric(ref["YawRate_deg_s"], errors="coerce").abs().fillna(0.0)
    elif "YawRate" in ref.columns:
        yaw_abs = (
            pd.to_numeric(ref["YawRate"], errors="coerce").abs().fillna(0.0) * 180.0 / np.pi
        )
    else:
        yaw_abs = pd.Series(np.zeros(len(ref)))

    if "LatAccel" in ref.columns:
        lat_acc_abs = pd.to_numeric(ref["LatAccel"], errors="coerce").abs().fillna(0.0)
    else:
        lat_acc_abs = pd.Series(np.zeros(len(ref)))

    if "SteeringWheelAngle_deg" in ref.columns:
        steer_abs = pd.to_numeric(ref["SteeringWheelAngle_deg"], errors="coerce").abs().fillna(0.0)
    elif "SteeringWheelAngle" in ref.columns:
        steer_abs = (
            pd.to_numeric(ref["SteeringWheelAngle"], errors="coerce").abs().fillna(0.0)
            * 180.0
            / np.pi
        )
    else:
        steer_abs = pd.Series(np.zeros(len(ref)))

    yaw_norm = _normalise_feature_abs(yaw_abs)
    lat_acc_norm = _normalise_feature_abs(lat_acc_abs)
    steer_norm = _normalise_feature_abs(steer_abs)
    speed_norm = _normalise_feature_abs(speed_deficit)
    curvature_norm = _path_curvature_activity(
        ref, smoothing_window_points=config.curvature_smoothing_window_points
    )

    # Multi-signal activity profile: dynamics-first, curvature as optional aid.
    dynamic_activity = 0.35 * yaw_norm + 0.30 * lat_acc_norm + 0.20 * steer_norm + 0.15 * speed_norm
    curvature_weight = float(np.clip(config.curvature_weight, 0.0, 0.70))
    activity_raw = (1.0 - curvature_weight) * dynamic_activity + curvature_weight * curvature_norm
    ref["corner_activity"] = activity_raw.rolling(
        window=config.activity_smoothing_window_points, center=True, min_periods=1
    ).mean()
    ref["path_curvature_activity"] = curvature_norm

    peak_idx_activity = []
    activity = ref["corner_activity"].to_numpy(dtype=float)
    for i in range(1, len(activity) - 1):
        if activity[i] >= activity[i - 1] and activity[i] > activity[i + 1]:
            peak_idx_activity.append(i)
    if not peak_idx_activity:
        peak_idx_activity = [int(np.nanargmax(activity))]

    curvature = ref["path_curvature_activity"].to_numpy(dtype=float)
    peak_idx_curvature = []
    for i in range(1, len(curvature) - 1):
        if curvature[i] >= curvature[i - 1] and curvature[i] > curvature[i + 1]:
            peak_idx_curvature.append(i)

    # Union of dynamic peaks + geometric peaks improves robustness across layouts.
    peak_idx = sorted(set(peak_idx_activity + peak_idx_curvature))
    if not peak_idx:
        peak_idx = [int(np.nanargmax(activity))]

    peak_candidates = pd.DataFrame(
        {
            "distance_pct": ref.loc[peak_idx, "distance_pct"].to_numpy(dtype=float),
            "activity_score": ref.loc[peak_idx, "corner_activity"].to_numpy(dtype=float),
            "curvature_score": ref.loc[peak_idx, "path_curvature_activity"].to_numpy(dtype=float),
        }
    )
    peak_candidates["composite_score"] = (
        0.75 * peak_candidates["activity_score"] + 0.25 * peak_candidates["curvature_score"]
    )

    threshold_quantile = config.activity_quantile_threshold
    if config.target_corner_count is not None:
        # With fixed target count, keep candidates broad and let spacing selection decide.
        threshold_quantile = min(threshold_quantile, 0.40)
    threshold = max(
        float(np.nanquantile(activity, threshold_quantile)),
        config.min_activity_score,
    )
    peak_candidates = peak_candidates.loc[peak_candidates["activity_score"] >= threshold].copy()
    if peak_candidates.empty:
        peak_candidates = pd.DataFrame(
            {
                "distance_pct": ref.loc[peak_idx, "distance_pct"].to_numpy(dtype=float),
                "activity_score": ref.loc[peak_idx, "corner_activity"].to_numpy(dtype=float),
                "curvature_score": ref.loc[peak_idx, "path_curvature_activity"].to_numpy(dtype=float),
            }
        )
        peak_candidates["composite_score"] = (
            0.75 * peak_candidates["activity_score"] + 0.25 * peak_candidates["curvature_score"]
        )

    refined = []
    for _, peak in peak_candidates.iterrows():
        center = float(peak["distance_pct"])
        # Refine each candidate to a local speed minimum around the activity peak.
        window = ref.loc[
            (ref["distance_pct"] >= center - config.speed_apex_search_window_pct)
            & (ref["distance_pct"] <= center + config.speed_apex_search_window_pct)
        ]
        if window.empty:
            apex_dist = center
            apex_speed = float(ref.loc[(ref["distance_pct"] - center).abs().idxmin(), "speed_kmh_smooth"])
        else:
            min_row = window.loc[window["speed_kmh_smooth"].idxmin()]
            apex_dist = float(min_row["distance_pct"])
            apex_speed = float(min_row["speed_kmh_smooth"])
        refined.append(
            {
                "distance_pct": apex_dist,
                "activity_score": float(peak["activity_score"]),
                "curvature_score": float(peak.get("curvature_score", 0.0)),
                "composite_score": float(peak.get("composite_score", peak["activity_score"])),
                "speed_kmh": apex_speed,
            }
        )

    refined_df = pd.DataFrame(refined).dropna()
    if refined_df.empty:
        raise ValueError("Corner detection failed to produce candidate apex points.")

    refined_df["distance_pct"] = refined_df["distance_pct"].round(6)
    refined_df = (
        refined_df.groupby("distance_pct", as_index=False)
        .agg(
            {
                "activity_score": "max",
                "curvature_score": "max",
                "composite_score": "max",
                "speed_kmh": "min",
            }
        )
        .sort_values("distance_pct")
    )

    manual_apex_pct = config.manual_apex_pct or []
    if manual_apex_pct:
        apex_distances = sorted(
            {
                round(float(value), 6)
                for value in manual_apex_pct
                if np.isfinite(float(value)) and 0.0 <= float(value) <= 100.0
            }
        )
        target_count = config.target_corner_count
        if target_count is not None and len(apex_distances) != int(target_count):
            raise ValueError(
                "manual_apex_pct length must match target_corner_count "
                f"({len(apex_distances)} provided, {target_count} expected)."
            )
    else:
        min_spacing_pct = _effective_min_corner_spacing_pct(ref, config)

        target_count = config.target_corner_count
        if target_count is not None:
            target_count = int(np.clip(target_count, config.min_corner_count, config.max_corner_count))
            apex_distances = _select_target_count_candidates(
                refined_df,
                score_col="composite_score",
                target_count=target_count,
                initial_spacing_pct=min_spacing_pct,
            )
        else:
            apex_distances = _select_spaced_candidates(
                refined_df,
                score_col="composite_score",
                min_spacing_pct=min_spacing_pct,
                max_count=config.max_corner_count,
            )

            if len(apex_distances) < config.min_corner_count:
                # Relax spacing in steps to avoid under-detecting technical complexes.
                spacing = min_spacing_pct
                while len(apex_distances) < config.min_corner_count and spacing > 0.25:
                    spacing *= 0.85
                    apex_distances = _select_spaced_candidates(
                        refined_df,
                        score_col="composite_score",
                        min_spacing_pct=spacing,
                        max_count=config.max_corner_count,
                    )

            if len(apex_distances) < config.min_corner_count:
                # Deterministic fallback: force top-scored candidates up to minimum count.
                apex_distances = (
                    refined_df.sort_values("composite_score", ascending=False)
                    .head(config.min_corner_count)["distance_pct"]
                    .astype(float)
                    .sort_values()
                    .tolist()
                )

            apex_distances = sorted(apex_distances[: config.max_corner_count])
    if not apex_distances:
        raise ValueError("Corner detection failed to produce any apex points.")

    rows = []
    last_idx = len(apex_distances) - 1
    for i, apex_pct in enumerate(apex_distances):
        corner_start_pct = 0.0 if i == 0 else (apex_distances[i - 1] + apex_pct) / 2.0
        corner_end_pct = 100.0 if i == last_idx else (apex_pct + apex_distances[i + 1]) / 2.0
        phase = _find_phase_boundaries(
            ref_lap_df=ref,
            corner_start_pct=corner_start_pct,
            apex_pct=apex_pct,
            corner_end_pct=corner_end_pct,
            config=config,
        )

        apex_row = ref.loc[(ref["distance_pct"] - apex_pct).abs().idxmin()]
        apex_speed = float(apex_row["speed_kmh_for_detection"])
        activity_score = float(apex_row["corner_activity"])

        rows.append(
            {
                "corner_id": i + 1,
                "official_turn_number": i + 1,
                "corner_name": f"T{i + 1}",
                "corner_start_pct": float(corner_start_pct),
                "brake_start_pct": phase["brake_start_pct"],
                "brake_end_pct": phase["brake_end_pct"],
                "rotation_start_pct": phase["rotation_start_pct"],
                "apex_pct": float(apex_pct),
                "rotation_end_pct": phase["rotation_end_pct"],
                "traction_start_pct": phase["traction_start_pct"],
                "corner_end_pct": float(corner_end_pct),
                "reference_apex_speed_kmh": apex_speed,
                "detection_activity_score": activity_score,
                "detection_curvature_score": float(apex_row.get("path_curvature_activity", np.nan)),
            }
        )

    return pd.DataFrame(rows)


def _interval_slice(lap_df: pd.DataFrame, start_pct: float, end_pct: float) -> pd.DataFrame:
    seg = lap_df.loc[
        (lap_df["distance_pct"] >= float(start_pct))
        & (lap_df["distance_pct"] <= float(end_pct))
    ].copy()
    return seg.sort_values("distance_pct")


def _speed_mps_series(lap_df: pd.DataFrame) -> pd.Series:
    if "Speed" in lap_df.columns:
        speed = pd.to_numeric(lap_df["Speed"], errors="coerce")
    elif "Speed_kmh" in lap_df.columns:
        speed = pd.to_numeric(lap_df["Speed_kmh"], errors="coerce") / 3.6
    else:
        raise ValueError("Lap metrics require 'Speed' or 'Speed_kmh' in aligned data.")
    return speed


def _time_integral_unscaled(segment_df: pd.DataFrame) -> float:
    if segment_df.shape[0] < 2:
        return np.nan
    speed_mps = _speed_mps_series(segment_df).to_numpy(dtype=float)
    dist_pct = pd.to_numeric(segment_df["distance_pct"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(speed_mps) & np.isfinite(dist_pct) & (speed_mps > 0.5)
    if valid.sum() < 2:
        return np.nan
    return float(np.trapezoid(1.0 / speed_mps[valid], dist_pct[valid]))


def _segment_time_s(
    lap_df: pd.DataFrame,
    start_pct: float,
    end_pct: float,
    lap_time_s: float,
    lap_unscaled_time: float,
) -> float:
    if not np.isfinite(lap_time_s) or not np.isfinite(lap_unscaled_time) or lap_unscaled_time <= 0:
        return np.nan
    segment = _interval_slice(lap_df, start_pct, end_pct)
    seg_unscaled = _time_integral_unscaled(segment)
    if not np.isfinite(seg_unscaled):
        return np.nan
    # Segment time is obtained by scaling local 1/v integral with measured lap time.
    scale = lap_time_s / lap_unscaled_time
    return float(seg_unscaled * scale)


def _safe_mean(series: pd.Series) -> float:
    series = pd.to_numeric(series, errors="coerce")
    if series.dropna().empty:
        return np.nan
    return float(series.mean())


def _safe_mean_abs(series: pd.Series) -> float:
    series = pd.to_numeric(series, errors="coerce").abs()
    if series.dropna().empty:
        return np.nan
    return float(series.mean())


def _safe_max(series: pd.Series) -> float:
    series = pd.to_numeric(series, errors="coerce")
    if series.dropna().empty:
        return np.nan
    return float(series.max())


def _safe_min(series: pd.Series) -> float:
    series = pd.to_numeric(series, errors="coerce")
    if series.dropna().empty:
        return np.nan
    return float(series.min())


def _safe_mad(series: pd.Series) -> float:
    """
    Median absolute deviation (MAD) as a robust spread metric.
    """
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return np.nan
    median = float(numeric.median())
    return float((numeric - median).abs().median())


def _normalise_positive(series: pd.Series) -> pd.Series:
    clipped = pd.to_numeric(series, errors="coerce").clip(lower=0.0).fillna(0.0)
    max_value = float(clipped.max())
    if max_value <= 0.0:
        return pd.Series(np.zeros(len(clipped)), index=clipped.index)
    return clipped / max_value


def _haversine_distance_m(
    lat1_deg: pd.Series,
    lon1_deg: pd.Series,
    lat2_deg: pd.Series,
    lon2_deg: pd.Series,
) -> pd.Series:
    """
    Deterministic geodesic approximation for small lap-scale position deltas.
    """
    lat1 = np.deg2rad(pd.to_numeric(lat1_deg, errors="coerce").to_numpy(dtype=float))
    lon1 = np.deg2rad(pd.to_numeric(lon1_deg, errors="coerce").to_numpy(dtype=float))
    lat2 = np.deg2rad(pd.to_numeric(lat2_deg, errors="coerce").to_numpy(dtype=float))
    lon2 = np.deg2rad(pd.to_numeric(lon2_deg, errors="coerce").to_numpy(dtype=float))

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(np.clip(a, 0.0, 1.0)), np.sqrt(np.clip(1.0 - a, 0.0, 1.0)))
    distance = 6_371_000.0 * c
    return pd.Series(distance, index=lat1_deg.index)


def _wheel_speed_std_mean(segment_df: pd.DataFrame) -> float:
    wheel_cols = [col for col in WHEEL_SPEED_CHANNELS if col in segment_df.columns]
    if len(wheel_cols) < 2:
        return np.nan
    wheel_df = segment_df[wheel_cols].apply(pd.to_numeric, errors="coerce")
    row_std = wheel_df.std(axis=1, skipna=True)
    if row_std.dropna().empty:
        return np.nan
    return float(row_std.mean())


def _body_slip_ratio_mean(segment_df: pd.DataFrame) -> float:
    if "VelocityX" not in segment_df.columns or "VelocityY" not in segment_df.columns:
        return np.nan
    vx = pd.to_numeric(segment_df["VelocityX"], errors="coerce").abs()
    vy = pd.to_numeric(segment_df["VelocityY"], errors="coerce").abs()
    denom = vx + 1e-3
    slip = vy / denom
    if slip.dropna().empty:
        return np.nan
    return float(slip.mean())


def compute_corner_lap_metrics(
    aligned_laps_df: pd.DataFrame,
    corner_definitions_df: pd.DataFrame,
    lap_times_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute raw per-corner per-lap metrics without applying a reference profile.
    """
    required_aligned = {"lap_id", "distance_pct"}
    if not required_aligned.issubset(aligned_laps_df.columns):
        raise ValueError("aligned_laps_df must contain at least lap_id and distance_pct.")
    if corner_definitions_df.empty:
        raise ValueError("corner_definitions_df is empty.")

    lap_time_map = dict(
        zip(
            lap_times_df["lap_id"].astype(int).tolist(),
            pd.to_numeric(lap_times_df["lap_time_s"], errors="coerce").tolist(),
        )
    )

    metric_rows = []
    available_lap_ids = sorted(aligned_laps_df["lap_id"].dropna().astype(int).unique().tolist())

    for lap_id in available_lap_ids:
        lap_df = aligned_laps_df.loc[aligned_laps_df["lap_id"] == lap_id].copy()
        lap_df = lap_df.sort_values("distance_pct")
        lap_time_s = float(lap_time_map.get(lap_id, np.nan))
        # Lap-local normalization factor used to convert segment integrals to seconds.
        full_unscaled = _time_integral_unscaled(lap_df)

        for _, corner in corner_definitions_df.iterrows():
            c_start = float(corner["corner_start_pct"])
            c_end = float(corner["corner_end_pct"])
            brake_start = float(corner["brake_start_pct"])
            brake_end = float(corner["brake_end_pct"])
            rot_start = float(corner["rotation_start_pct"])
            rot_end = float(corner["rotation_end_pct"])
            apex_pct = float(corner["apex_pct"])
            traction_start = float(corner["traction_start_pct"])

            corner_seg = _interval_slice(lap_df, c_start, c_end)
            brake_seg = _interval_slice(lap_df, brake_start, brake_end)
            rotation_seg = _interval_slice(lap_df, rot_start, rot_end)
            traction_seg = _interval_slice(lap_df, traction_start, c_end)
            apex_row = (
                lap_df.loc[(lap_df["distance_pct"] - apex_pct).abs().idxmin()]
                if not lap_df.empty
                else None
            )

            corner_time_s = _segment_time_s(lap_df, c_start, c_end, lap_time_s, full_unscaled)
            braking_time_s = _segment_time_s(
                lap_df, brake_start, brake_end, lap_time_s, full_unscaled
            )
            rotation_time_s = _segment_time_s(
                lap_df, rot_start, rot_end, lap_time_s, full_unscaled
            )
            traction_time_s = _segment_time_s(
                lap_df, traction_start, c_end, lap_time_s, full_unscaled
            )

            metric_rows.append(
                {
                    "lap_id": int(lap_id),
                    "corner_id": int(corner["corner_id"]),
                    "corner_name": corner["corner_name"],
                    "lap_time_s": lap_time_s,
                    "corner_time_s": corner_time_s,
                    "braking_time_s": braking_time_s,
                    "rotation_time_s": rotation_time_s,
                    "traction_time_s": traction_time_s,
                    "brake_peak_pct": _safe_max(brake_seg["Brake"]) if "Brake" in brake_seg.columns else np.nan,
                    "brake_mean_pct": _safe_mean(brake_seg["Brake"]) if "Brake" in brake_seg.columns else np.nan,
                    "brake_raw_peak_pct": _safe_max(brake_seg["BrakeRaw"]) if "BrakeRaw" in brake_seg.columns else np.nan,
                    "brake_raw_mean_pct": _safe_mean(brake_seg["BrakeRaw"]) if "BrakeRaw" in brake_seg.columns else np.nan,
                    "apex_speed_kmh": (
                        float(apex_row["Speed_kmh"])
                        if apex_row is not None and "Speed_kmh" in apex_row.index
                        else (
                            float(apex_row["Speed"]) * 3.6
                            if apex_row is not None and "Speed" in apex_row.index
                            else np.nan
                        )
                    ),
                    "apex_lat": (
                        float(apex_row["Lat"])
                        if apex_row is not None and "Lat" in apex_row.index
                        else np.nan
                    ),
                    "apex_lon": (
                        float(apex_row["Lon"])
                        if apex_row is not None and "Lon" in apex_row.index
                        else np.nan
                    ),
                    "rotation_min_speed_kmh": (
                        _safe_min(rotation_seg["Speed_kmh"])
                        if "Speed_kmh" in rotation_seg.columns
                        else (
                            _safe_min(rotation_seg["Speed"]) * 3.6
                            if "Speed" in rotation_seg.columns
                            else np.nan
                        )
                    ),
                    "rotation_mean_abs_yaw_rate_deg_s": (
                        _safe_mean(rotation_seg["YawRate_deg_s"].abs())
                        if "YawRate_deg_s" in rotation_seg.columns
                        else (
                            _safe_mean(rotation_seg["YawRate"].abs()) * 180.0 / np.pi
                            if "YawRate" in rotation_seg.columns
                            else np.nan
                        )
                    ),
                    "rotation_mean_abs_lat_accel_g": (
                        _safe_mean_abs(rotation_seg["LatAccel"])
                        if "LatAccel" in rotation_seg.columns
                        else np.nan
                    ),
                    "rotation_mean_abs_steer_deg": (
                        _safe_mean_abs(rotation_seg["SteeringWheelAngle_deg"])
                        if "SteeringWheelAngle_deg" in rotation_seg.columns
                        else (
                            _safe_mean_abs(rotation_seg["SteeringWheelAngle"]) * 180.0 / np.pi
                            if "SteeringWheelAngle" in rotation_seg.columns
                            else np.nan
                        )
                    ),
                    "rotation_mean_body_slip_ratio": _body_slip_ratio_mean(rotation_seg),
                    "braking_mean_long_accel_g": (
                        _safe_mean(brake_seg["LongAccel"])
                        if "LongAccel" in brake_seg.columns
                        else np.nan
                    ),
                    "corner_mean_abs_vert_accel_g": (
                        _safe_mean_abs(corner_seg["VertAccel"])
                        if "VertAccel" in corner_seg.columns
                        else np.nan
                    ),
                    "traction_reapply_delay_pct": traction_start - apex_pct,
                    "traction_exit_throttle_mean_pct": _safe_mean(traction_seg["Throttle"]) if "Throttle" in traction_seg.columns else np.nan,
                    "traction_exit_throttle_raw_mean_pct": _safe_mean(traction_seg["ThrottleRaw"]) if "ThrottleRaw" in traction_seg.columns else np.nan,
                    "traction_exit_long_accel_mean": _safe_mean(traction_seg["LongAccel"]) if "LongAccel" in traction_seg.columns else np.nan,
                    "traction_wheel_speed_std_mps": _wheel_speed_std_mean(traction_seg),
                }
            )

    return pd.DataFrame(metric_rows)


def _build_reference_profile_from_rows(reference_rows: pd.DataFrame) -> pd.DataFrame:
    available_map = {
        source_col: ref_col
        for source_col, ref_col in REFERENCE_METRIC_MAP.items()
        if source_col in reference_rows.columns
    }
    renamed = reference_rows.rename(columns=available_map).copy()
    selected_cols = ["corner_id", "reference_corner_lap_id"] + list(available_map.values())
    profile = renamed[selected_cols].copy()
    return profile.sort_values("corner_id").reset_index(drop=True)


def build_fastest_lap_corner_reference(
    corner_lap_metrics_df: pd.DataFrame,
    reference_lap_id: int,
) -> pd.DataFrame:
    """
    Build a corner reference profile from one selected lap.
    """
    reference_rows = corner_lap_metrics_df.loc[
        corner_lap_metrics_df["lap_id"] == int(reference_lap_id)
    ].copy()
    if reference_rows.empty:
        raise ValueError(
            f"Reference lap {reference_lap_id} is not present in corner_lap_metrics_df."
        )

    reference_rows["reference_corner_lap_id"] = int(reference_lap_id)
    return _build_reference_profile_from_rows(reference_rows)


def build_best_per_corner_reference(corner_lap_metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a per-corner benchmark profile using the best corner_time_s across laps.
    """
    required_cols = {
        "corner_id",
        "lap_id",
        "corner_time_s",
        "apex_speed_kmh",
        "traction_reapply_delay_pct",
    }
    missing = required_cols - set(corner_lap_metrics_df.columns)
    if missing:
        raise ValueError(
            "corner_lap_metrics_df missing required columns for per-corner benchmark: "
            + ", ".join(sorted(missing))
        )

    working = corner_lap_metrics_df.copy()
    working["corner_time_sort"] = pd.to_numeric(working["corner_time_s"], errors="coerce").fillna(
        np.inf
    )
    working["lap_id_sort"] = pd.to_numeric(working["lap_id"], errors="coerce").fillna(np.inf)

    best = (
        working.sort_values(["corner_id", "corner_time_sort", "lap_id_sort"])
        .groupby("corner_id", as_index=False)
        .head(1)
        .copy()
    )
    best["reference_corner_lap_id"] = best["lap_id"].astype(int)
    return _build_reference_profile_from_rows(best)


def apply_corner_reference(
    corner_lap_metrics_df: pd.DataFrame,
    corner_reference_df: pd.DataFrame,
    comparison_label: str,
    exclude_reference_rows: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply a deterministic corner reference profile and build coaching ranking outputs.
    """
    required_ref_cols = [
        "corner_id",
        "reference_corner_lap_id",
        "ref_corner_time_s",
        "ref_apex_speed_kmh",
        "ref_traction_reapply_delay_pct",
    ]
    missing_ref = set(required_ref_cols) - set(corner_reference_df.columns)
    if missing_ref:
        raise ValueError(
            "corner_reference_df missing required columns: " + ", ".join(sorted(missing_ref))
        )

    enriched = corner_lap_metrics_df.merge(
        corner_reference_df, on="corner_id", how="left"
    )
    enriched["comparison_label"] = comparison_label
    enriched["time_loss_vs_ref_s"] = enriched["corner_time_s"] - enriched["ref_corner_time_s"]
    enriched["apex_speed_loss_vs_ref_kmh"] = (
        enriched["ref_apex_speed_kmh"] - enriched["apex_speed_kmh"]
    )
    enriched["traction_delay_vs_ref_pct"] = (
        enriched["traction_reapply_delay_pct"] - enriched["ref_traction_reapply_delay_pct"]
    )
    if {"apex_lat", "apex_lon", "ref_apex_lat", "ref_apex_lon"}.issubset(enriched.columns):
        enriched["apex_position_delta_m"] = _haversine_distance_m(
            enriched["apex_lat"],
            enriched["apex_lon"],
            enriched["ref_apex_lat"],
            enriched["ref_apex_lon"],
        )
    else:
        enriched["apex_position_delta_m"] = np.nan

    if exclude_reference_rows:
        # Exclude benchmark rows so ranking reflects "driver vs reference", not reference vs itself.
        comparison = enriched.loc[
            enriched["lap_id"] != enriched["reference_corner_lap_id"]
        ].copy()
        if comparison.empty:
            comparison = enriched.copy()
    else:
        comparison = enriched.copy()

    agg = comparison.groupby(["corner_id", "corner_name"], dropna=False).agg(
        laps_compared=("lap_id", "nunique"),
        mean_time_loss_s=("time_loss_vs_ref_s", "mean"),
        median_time_loss_s=("time_loss_vs_ref_s", "median"),
        std_time_loss_s=("time_loss_vs_ref_s", "std"),
        mad_time_loss_s=("time_loss_vs_ref_s", _safe_mad),
        mean_abs_time_loss_s=("time_loss_vs_ref_s", lambda s: np.nanmean(np.abs(s))),
        inconsistency_time_s=("corner_time_s", "std"),
        inconsistency_time_mad_s=("corner_time_s", _safe_mad),
        mean_apex_speed_loss_kmh=("apex_speed_loss_vs_ref_kmh", "mean"),
        mean_traction_delay_loss_pct=("traction_delay_vs_ref_pct", "mean"),
        mean_apex_position_delta_m=("apex_position_delta_m", "mean"),
    )
    ranking = agg.reset_index()
    ranking["std_time_loss_s"] = ranking["std_time_loss_s"].fillna(0.0)
    ranking["median_time_loss_s"] = ranking["median_time_loss_s"].fillna(0.0)
    ranking["mad_time_loss_s"] = ranking["mad_time_loss_s"].fillna(0.0)
    ranking["inconsistency_time_s"] = ranking["inconsistency_time_s"].fillna(0.0)
    ranking["inconsistency_time_mad_s"] = ranking["inconsistency_time_mad_s"].fillna(0.0)
    ranking["mean_time_loss_s"] = ranking["mean_time_loss_s"].fillna(0.0)
    ranking["mean_apex_speed_loss_kmh"] = ranking["mean_apex_speed_loss_kmh"].fillna(0.0)
    ranking["mean_traction_delay_loss_pct"] = ranking["mean_traction_delay_loss_pct"].fillna(0.0)
    ranking["mean_apex_position_delta_m"] = ranking["mean_apex_position_delta_m"].fillna(0.0)

    # Robust blends reduce outlier influence from a single anomalous lap.
    ranking["time_loss_blended_s"] = (
        0.60 * ranking["mean_time_loss_s"] + 0.40 * ranking["median_time_loss_s"]
    )
    ranking["inconsistency_blended_s"] = (
        0.60 * ranking["inconsistency_time_s"] + 0.40 * ranking["inconsistency_time_mad_s"]
    )

    norm_time_loss = _normalise_positive(ranking["time_loss_blended_s"])
    norm_variability = _normalise_positive(ranking["inconsistency_blended_s"])
    norm_apex_speed_loss = _normalise_positive(ranking["mean_apex_speed_loss_kmh"])

    # Priority emphasizes time loss first, then repeatability, then speed-at-apex context.
    ranking["coaching_relevance_score"] = (
        0.60 * norm_time_loss + 0.30 * norm_variability + 0.10 * norm_apex_speed_loss
    )
    ranking["coaching_priority_rank"] = (
        ranking["coaching_relevance_score"].rank(method="dense", ascending=False).astype(int)
    )
    ranking = ranking.sort_values(
        ["coaching_relevance_score", "corner_id"], ascending=[False, True]
    ).reset_index(drop=True)

    return enriched, ranking


def build_corner_report(
    aligned_laps_df: pd.DataFrame,
    corner_definitions_df: pd.DataFrame,
    corner_lap_metrics_df: pd.DataFrame,
    reference_mode: str,
    reference_source: str,
) -> Dict[str, object]:
    trajectory_line_feasible = {"Lat", "Lon"}.issubset(set(aligned_laps_df.columns))
    apex_position_proxy_available = "apex_position_delta_m" in set(corner_lap_metrics_df.columns)
    trajectory_note = (
        "Lat/Lon available in aligned data: deterministic line delta metrics can be added."
        if trajectory_line_feasible
        else "Lat/Lon not present in aligned data. Core corner coaching metrics are available; "
        "trajectory/line comparison would require adding Lat/Lon to aligned channels."
    )

    return {
        "reference_mode": reference_mode,
        "reference_source": reference_source,
        "corner_count": int(corner_definitions_df.shape[0]),
        "laps_in_metrics": int(corner_lap_metrics_df["lap_id"].nunique())
        if not corner_lap_metrics_df.empty
        else 0,
        "trajectory_line_feasible": bool(trajectory_line_feasible),
        "apex_position_proxy_available": bool(apex_position_proxy_available),
        "trajectory_line_note": trajectory_note,
        "coaching_score_formula": (
            "0.60*normalized(0.60*mean_time_loss_s + 0.40*median_time_loss_s) + "
            "0.30*normalized(0.60*inconsistency_time_s + 0.40*inconsistency_time_mad_s) + "
            "0.10*normalized(mean_apex_speed_loss_kmh)"
        ),
        "channel_availability": {
            "BrakeRaw": "BrakeRaw" in aligned_laps_df.columns,
            "ThrottleRaw": "ThrottleRaw" in aligned_laps_df.columns,
            "LatAccel": "LatAccel" in aligned_laps_df.columns,
            "LongAccel": "LongAccel" in aligned_laps_df.columns,
            "VertAccel": "VertAccel" in aligned_laps_df.columns,
            "VelocityX": "VelocityX" in aligned_laps_df.columns,
            "VelocityY": "VelocityY" in aligned_laps_df.columns,
            "LFspeed": "LFspeed" in aligned_laps_df.columns,
            "RFspeed": "RFspeed" in aligned_laps_df.columns,
            "LRspeed": "LRspeed" in aligned_laps_df.columns,
            "RRspeed": "RRspeed" in aligned_laps_df.columns,
            "Lat": "Lat" in aligned_laps_df.columns,
            "Lon": "Lon" in aligned_laps_df.columns,
        },
    }


def compute_corner_metrics(
    aligned_laps_df: pd.DataFrame,
    corner_definitions_df: pd.DataFrame,
    lap_times_df: pd.DataFrame,
    reference_lap_id: int,
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    """
    Backward-compatible wrapper: score corners against one reference lap.
    """
    corner_lap_metrics = compute_corner_lap_metrics(
        aligned_laps_df=aligned_laps_df,
        corner_definitions_df=corner_definitions_df,
        lap_times_df=lap_times_df,
    )
    reference_profile = build_fastest_lap_corner_reference(
        corner_lap_metrics_df=corner_lap_metrics,
        reference_lap_id=reference_lap_id,
    )
    scored_metrics, ranking = apply_corner_reference(
        corner_lap_metrics_df=corner_lap_metrics,
        corner_reference_df=reference_profile,
        comparison_label="fastest_lap_reference",
        exclude_reference_rows=True,
    )
    report = build_corner_report(
        aligned_laps_df=aligned_laps_df,
        corner_definitions_df=corner_definitions_df,
        corner_lap_metrics_df=scored_metrics,
        reference_mode="fastest_lap",
        reference_source=f"lap_id={int(reference_lap_id)}",
    )
    report["reference_lap_id"] = int(reference_lap_id)
    return scored_metrics, ranking, report
