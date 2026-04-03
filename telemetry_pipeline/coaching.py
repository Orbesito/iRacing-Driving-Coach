from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CoachingConfig:
    """
    Deterministic thresholds for converting corner metrics into coaching advice.
    """

    max_priority_corners: int = 5
    min_priority_score: float = 0.20
    min_time_loss_s: float = 0.03
    min_inconsistency_s: float = 0.04
    significant_apex_speed_loss_kmh: float = 1.0
    significant_phase_delta_s: float = 0.02
    significant_brake_delta_pct: float = 5.0
    significant_traction_delay_pct: float = 0.20
    significant_yaw_delta_deg_s: float = 2.0
    significant_steer_delta_deg: float = 3.0
    significant_slip_delta: float = 0.02
    significant_exit_long_accel_delta: float = 0.04
    significant_wheel_speed_std_delta: float = 0.12
    significant_apex_position_delta_m: float = 1.5


def _safe_mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce")
    if values.dropna().empty:
        return np.nan
    return float(values.mean())


def _first_finite(values: List[float]) -> float:
    for value in values:
        if np.isfinite(value):
            return float(value)
    return np.nan


def _normalised_positive(value: float, threshold: float) -> float:
    if not np.isfinite(value):
        return 0.0
    if value <= threshold:
        return 0.0
    scale = max(abs(threshold), 1e-6)
    return float((value - threshold) / scale)


def _build_corner_snapshot(corner_rows: pd.DataFrame, ranking_row: pd.Series) -> Dict[str, float]:
    brake_peak_series = None
    if "brake_raw_peak_pct" in corner_rows.columns:
        brake_peak_series = pd.to_numeric(corner_rows["brake_raw_peak_pct"], errors="coerce")
    elif "brake_peak_pct" in corner_rows.columns:
        brake_peak_series = pd.to_numeric(corner_rows["brake_peak_pct"], errors="coerce")

    brake_peak = _first_finite(
        [
            float(brake_peak_series.median()) if brake_peak_series is not None else np.nan,
            _safe_mean(corner_rows["brake_raw_peak_pct"])
            if "brake_raw_peak_pct" in corner_rows.columns
            else np.nan,
            _safe_mean(corner_rows["brake_peak_pct"]) if "brake_peak_pct" in corner_rows.columns else np.nan,
        ]
    )
    ref_brake_peak = _first_finite(
        [
            _safe_mean(corner_rows["ref_brake_raw_peak_pct"])
            if "ref_brake_raw_peak_pct" in corner_rows.columns
            else np.nan,
            _safe_mean(corner_rows["ref_brake_peak_pct"])
            if "ref_brake_peak_pct" in corner_rows.columns
            else np.nan,
        ]
    )

    return {
        "corner_id": float(ranking_row["corner_id"]),
        "mean_time_loss_s": float(ranking_row.get("mean_time_loss_s", 0.0)),
        "inconsistency_time_s": float(ranking_row.get("inconsistency_time_s", 0.0)),
        "priority_score": float(ranking_row.get("coaching_relevance_score", 0.0)),
        "apex_speed_loss_kmh": _safe_mean(corner_rows["apex_speed_loss_vs_ref_kmh"])
        if "apex_speed_loss_vs_ref_kmh" in corner_rows.columns
        else np.nan,
        "brake_peak_delta_pct": brake_peak - ref_brake_peak
        if np.isfinite(brake_peak) and np.isfinite(ref_brake_peak)
        else np.nan,
        "braking_time_delta_s": _safe_mean(corner_rows["braking_time_s"] - corner_rows["ref_braking_time_s"])
        if {"braking_time_s", "ref_braking_time_s"}.issubset(corner_rows.columns)
        else np.nan,
        "rotation_time_delta_s": _safe_mean(corner_rows["rotation_time_s"] - corner_rows["ref_rotation_time_s"])
        if {"rotation_time_s", "ref_rotation_time_s"}.issubset(corner_rows.columns)
        else np.nan,
        "yaw_delta_deg_s": _safe_mean(
            corner_rows["rotation_mean_abs_yaw_rate_deg_s"]
            - corner_rows["ref_rotation_mean_abs_yaw_rate_deg_s"]
        )
        if {"rotation_mean_abs_yaw_rate_deg_s", "ref_rotation_mean_abs_yaw_rate_deg_s"}.issubset(
            corner_rows.columns
        )
        else np.nan,
        "steer_delta_deg": _safe_mean(
            corner_rows["rotation_mean_abs_steer_deg"] - corner_rows["ref_rotation_mean_abs_steer_deg"]
        )
        if {"rotation_mean_abs_steer_deg", "ref_rotation_mean_abs_steer_deg"}.issubset(corner_rows.columns)
        else np.nan,
        "slip_delta": _safe_mean(
            corner_rows["rotation_mean_body_slip_ratio"] - corner_rows["ref_rotation_mean_body_slip_ratio"]
        )
        if {"rotation_mean_body_slip_ratio", "ref_rotation_mean_body_slip_ratio"}.issubset(corner_rows.columns)
        else np.nan,
        "traction_delay_vs_ref_pct": _safe_mean(corner_rows["traction_delay_vs_ref_pct"])
        if "traction_delay_vs_ref_pct" in corner_rows.columns
        else np.nan,
        "traction_time_delta_s": _safe_mean(corner_rows["traction_time_s"] - corner_rows["ref_traction_time_s"])
        if {"traction_time_s", "ref_traction_time_s"}.issubset(corner_rows.columns)
        else np.nan,
        "exit_long_accel_delta": _safe_mean(
            corner_rows["traction_exit_long_accel_mean"] - corner_rows["ref_traction_exit_long_accel_mean"]
        )
        if {"traction_exit_long_accel_mean", "ref_traction_exit_long_accel_mean"}.issubset(corner_rows.columns)
        else np.nan,
        "wheel_speed_std_delta": _safe_mean(
            corner_rows["traction_wheel_speed_std_mps"] - corner_rows["ref_traction_wheel_speed_std_mps"]
        )
        if {"traction_wheel_speed_std_mps", "ref_traction_wheel_speed_std_mps"}.issubset(corner_rows.columns)
        else np.nan,
        "apex_position_delta_m": _safe_mean(corner_rows["apex_position_delta_m"])
        if "apex_position_delta_m" in corner_rows.columns
        else np.nan,
        "brake_event_fraction": float((brake_peak_series.fillna(0.0) >= 5.0).mean())
        if brake_peak_series is not None and not brake_peak_series.empty
        else np.nan,
        "ref_brake_peak_pct": ref_brake_peak,
    }


def _phase_scores(snapshot: Dict[str, float], config: CoachingConfig) -> Dict[str, float]:
    no_brake_corner = (
        np.isfinite(snapshot.get("brake_event_fraction", np.nan))
        and snapshot.get("brake_event_fraction", 1.0) < 0.30
        and (
            not np.isfinite(snapshot.get("ref_brake_peak_pct", np.nan))
            or snapshot.get("ref_brake_peak_pct", 100.0) < config.significant_brake_delta_pct
        )
    )

    entry = (
        _normalised_positive(
            snapshot["braking_time_delta_s"], config.significant_phase_delta_s
        )
        + _normalised_positive(
            snapshot["brake_peak_delta_pct"], config.significant_brake_delta_pct
        )
        + _normalised_positive(
            snapshot["apex_speed_loss_kmh"], config.significant_apex_speed_loss_kmh
        )
    )
    if no_brake_corner:
        entry = 0.30 * _normalised_positive(
            snapshot["apex_speed_loss_kmh"], config.significant_apex_speed_loss_kmh
        )

    mid = (
        _normalised_positive(snapshot["rotation_time_delta_s"], config.significant_phase_delta_s)
        + _normalised_positive(
            snapshot["apex_speed_loss_kmh"], config.significant_apex_speed_loss_kmh
        )
        + _normalised_positive(-snapshot["yaw_delta_deg_s"], config.significant_yaw_delta_deg_s)
        + _normalised_positive(snapshot["steer_delta_deg"], config.significant_steer_delta_deg)
    )
    exit_phase = (
        _normalised_positive(
            snapshot["traction_delay_vs_ref_pct"], config.significant_traction_delay_pct
        )
        + _normalised_positive(snapshot["traction_time_delta_s"], config.significant_phase_delta_s)
        + _normalised_positive(
            -snapshot["exit_long_accel_delta"], config.significant_exit_long_accel_delta
        )
        + _normalised_positive(
            snapshot["wheel_speed_std_delta"], config.significant_wheel_speed_std_delta
        )
    )
    return {"entry": entry, "mid": mid, "exit": exit_phase}


def _entry_advice(snapshot: Dict[str, float], config: CoachingConfig) -> Tuple[str, str, str, str]:
    no_brake_corner = (
        np.isfinite(snapshot.get("brake_event_fraction", np.nan))
        and snapshot.get("brake_event_fraction", 1.0) < 0.30
        and (
            not np.isfinite(snapshot.get("ref_brake_peak_pct", np.nan))
            or snapshot.get("ref_brake_peak_pct", 100.0) < config.significant_brake_delta_pct
        )
    )
    if no_brake_corner:
        return (
            "This corner is effectively a no-brake/lift-and-steer section.",
            "Time is likely lost from line and steering management rather than brake usage.",
            "Prioritize entry positioning and reduce steering corrections so minimum speed is carried cleanly.",
            "Drill: 3 laps with cue 'place car early, one steering arc, no extra correction'.",
        )

    if (
        snapshot["braking_time_delta_s"] > config.significant_phase_delta_s
        and snapshot["brake_peak_delta_pct"] > config.significant_brake_delta_pct
        and snapshot["apex_speed_loss_kmh"] > config.significant_apex_speed_loss_kmh
    ):
        return (
            "Entry speed is being over-suppressed.",
            "Braking is too long and too aggressive versus the benchmark, so speed is dropped before apex.",
            "Keep the initial brake hit but shorten the high-pressure phase and release progressively into turn-in.",
            "Drill: 3 laps focusing on one clean brake release with no second brake squeeze.",
        )
    if (
        snapshot["braking_time_delta_s"] < -config.significant_phase_delta_s
        and snapshot["apex_speed_loss_kmh"] > config.significant_apex_speed_loss_kmh
    ):
        return (
            "Entry is likely rushed into the corner.",
            "A short/late brake phase is forcing a compromised rotation and lower apex speed.",
            "Brake a fraction earlier and prioritize a smoother release to stabilize the front at turn-in.",
            "Drill: 3 laps with a fixed earlier marker and focus on smooth release quality.",
        )
    return (
        "Entry efficiency is below benchmark.",
        "The corner starts with a speed deficit that carries into the rest of the phase.",
        "Focus on cleaner brake-release timing to carry minimum speed without destabilizing the car.",
        "Drill: compare two consecutive laps and target identical release timing each lap.",
    )


def _mid_advice(snapshot: Dict[str, float], config: CoachingConfig) -> Tuple[str, str, str, str]:
    if (
        snapshot["rotation_time_delta_s"] > config.significant_phase_delta_s
        and snapshot["yaw_delta_deg_s"] < -config.significant_yaw_delta_deg_s
    ):
        return (
            "Rotation is delayed in mid-corner.",
            "Yaw build-up is lower than reference while rotation phase time is longer, indicating late rotation.",
            "Start turn-in slightly earlier and keep light trail-brake support until yaw is established.",
            "Drill: 3 laps with focus cue 'rotate the car before apex, then release steering early'.",
        )
    if (
        snapshot["steer_delta_deg"] > config.significant_steer_delta_deg
        and snapshot["apex_speed_loss_kmh"] > config.significant_apex_speed_loss_kmh
    ):
        return (
            "Steering demand is too high through rotation.",
            "Extra steering lock with lower apex speed suggests the car is being asked to turn too late.",
            "Reduce steering rate at turn-in and commit to one smooth arc through apex.",
            "Drill: one-lap reset between attempts, focus on avoiding second steering input.",
        )
    if (
        snapshot["yaw_delta_deg_s"] > config.significant_yaw_delta_deg_s
        and snapshot["slip_delta"] > config.significant_slip_delta
    ):
        return (
            "Mid-corner platform looks unstable.",
            "High yaw with increased slip indicates excessive rotation corrections through the middle phase.",
            "Soften initial turn-in and prioritize steering stability at minimum speed.",
            "Drill: 3 laps at 95% entry aggression, then rebuild speed only if stability is maintained.",
        )
    return (
        "Mid-corner efficiency is below target.",
        "Rotation phase takes longer than benchmark and speed is not recovered early enough.",
        "Prioritize earlier, cleaner rotation and earlier steering release at apex.",
        "Drill: compare yaw trace consistency for this corner across 3 consecutive laps.",
    )


def _exit_advice(snapshot: Dict[str, float], config: CoachingConfig) -> Tuple[str, str, str, str]:
    if (
        snapshot["traction_delay_vs_ref_pct"] > config.significant_traction_delay_pct
        and snapshot["traction_time_delta_s"] > config.significant_phase_delta_s
    ):
        return (
            "Throttle-on is late on exit.",
            "Delayed throttle reapplication and longer traction phase are costing exit speed and time.",
            "Open throttle earlier once steering starts to unwind, then use a progressive ramp.",
            "Drill: 3 laps with cue 'first throttle touch earlier, smoother ramp'.",
        )
    if (
        snapshot["traction_delay_vs_ref_pct"] < -config.significant_traction_delay_pct
        and snapshot["wheel_speed_std_delta"] > config.significant_wheel_speed_std_delta
    ):
        return (
            "Throttle application is too aggressive on initial exit.",
            "Earlier throttle with increased wheel-speed spread suggests traction-limited slip events.",
            "Delay first throttle touch slightly and smooth the initial throttle ramp.",
            "Drill: 3 laps limiting early throttle ramp rate until wheel-speed spread stabilizes.",
        )
    if snapshot["exit_long_accel_delta"] < -config.significant_exit_long_accel_delta:
        return (
            "Exit acceleration is weaker than benchmark.",
            "Longitudinal acceleration is below reference through traction phase.",
            "Prioritize cleaner car placement at apex so throttle can be committed earlier and harder.",
            "Drill: focus on one corner only and maximize clean full-throttle point repeatability.",
        )
    return (
        "Exit conversion is below benchmark.",
        "Time is being retained in the traction phase rather than released onto the following straight.",
        "Prioritize throttle timing and ramp quality while keeping steering unwind continuous.",
        "Drill: 3 laps with attention only on full-throttle point consistency.",
    )


def _build_corner_advice(snapshot: Dict[str, float], config: CoachingConfig) -> Dict[str, object]:
    phase_scores = _phase_scores(snapshot, config)
    primary_phase = max(phase_scores, key=phase_scores.get)
    no_brake_corner = (
        np.isfinite(snapshot.get("brake_event_fraction", np.nan))
        and snapshot.get("brake_event_fraction", 1.0) < 0.30
        and (
            not np.isfinite(snapshot.get("ref_brake_peak_pct", np.nan))
            or snapshot.get("ref_brake_peak_pct", 100.0) < config.significant_brake_delta_pct
        )
    )
    if no_brake_corner and primary_phase == "entry":
        if phase_scores["mid"] >= phase_scores["exit"]:
            primary_phase = "mid"
        else:
            primary_phase = "exit"

    if primary_phase == "entry":
        symptom, cause, action, drill = _entry_advice(snapshot, config)
    elif primary_phase == "mid":
        symptom, cause, action, drill = _mid_advice(snapshot, config)
    else:
        symptom, cause, action, drill = _exit_advice(snapshot, config)

    evidence = []
    metric_order = [
        ("mean_time_loss_s", "mean_time_loss_s", "s"),
        ("inconsistency_time_s", "inconsistency_time_s", "s"),
        ("apex_speed_loss_kmh", "apex_speed_loss_vs_ref", "km/h"),
        ("braking_time_delta_s", "braking_time_delta_vs_ref", "s"),
        ("rotation_time_delta_s", "rotation_time_delta_vs_ref", "s"),
        ("traction_time_delta_s", "traction_time_delta_vs_ref", "s"),
        ("traction_delay_vs_ref_pct", "traction_delay_vs_ref", "% lap"),
        ("brake_peak_delta_pct", "brake_peak_delta_vs_ref", "%"),
        ("yaw_delta_deg_s", "yaw_delta_vs_ref", "deg/s"),
        ("steer_delta_deg", "steer_delta_vs_ref", "deg"),
        ("exit_long_accel_delta", "exit_long_accel_delta_vs_ref", "g"),
        ("wheel_speed_std_delta", "wheel_speed_std_delta_vs_ref", "m/s"),
        ("apex_position_delta_m", "apex_position_delta_vs_ref", "m"),
        ("brake_event_fraction", "brake_event_fraction", "ratio"),
    ]
    for key, label, unit in metric_order:
        value = snapshot.get(key, np.nan)
        if np.isfinite(value):
            evidence.append(f"{label}={value:.4f} {unit}")

    available_signal_count = sum(
        int(np.isfinite(snapshot.get(key, np.nan)))
        for key in [
            "apex_speed_loss_kmh",
            "braking_time_delta_s",
            "rotation_time_delta_s",
            "traction_delay_vs_ref_pct",
            "traction_time_delta_s",
            "yaw_delta_deg_s",
            "steer_delta_deg",
            "exit_long_accel_delta",
            "wheel_speed_std_delta",
        ]
    )
    if primary_phase == "entry":
        primary_signal_count = sum(
            [
                int(snapshot["apex_speed_loss_kmh"] > config.significant_apex_speed_loss_kmh)
                if np.isfinite(snapshot["apex_speed_loss_kmh"])
                else 0,
                int(snapshot["braking_time_delta_s"] > config.significant_phase_delta_s)
                if np.isfinite(snapshot["braking_time_delta_s"])
                else 0,
                int(abs(snapshot["brake_peak_delta_pct"]) > config.significant_brake_delta_pct)
                if np.isfinite(snapshot["brake_peak_delta_pct"])
                else 0,
            ]
        )
    elif primary_phase == "mid":
        primary_signal_count = sum(
            [
                int(snapshot["rotation_time_delta_s"] > config.significant_phase_delta_s)
                if np.isfinite(snapshot["rotation_time_delta_s"])
                else 0,
                int(abs(snapshot["yaw_delta_deg_s"]) > config.significant_yaw_delta_deg_s)
                if np.isfinite(snapshot["yaw_delta_deg_s"])
                else 0,
                int(abs(snapshot["steer_delta_deg"]) > config.significant_steer_delta_deg)
                if np.isfinite(snapshot["steer_delta_deg"])
                else 0,
            ]
        )
    else:
        primary_signal_count = sum(
            [
                int(
                    abs(snapshot["traction_delay_vs_ref_pct"]) > config.significant_traction_delay_pct
                )
                if np.isfinite(snapshot["traction_delay_vs_ref_pct"])
                else 0,
                int(snapshot["traction_time_delta_s"] > config.significant_phase_delta_s)
                if np.isfinite(snapshot["traction_time_delta_s"])
                else 0,
                int(snapshot["exit_long_accel_delta"] < -config.significant_exit_long_accel_delta)
                if np.isfinite(snapshot["exit_long_accel_delta"])
                else 0,
                int(snapshot["wheel_speed_std_delta"] > config.significant_wheel_speed_std_delta)
                if np.isfinite(snapshot["wheel_speed_std_delta"])
                else 0,
            ]
        )

    coverage_ratio = available_signal_count / 9.0
    phase_strength = min(phase_scores[primary_phase], 3.0) / 3.0
    confidence_score = 0.30 + 0.35 * coverage_ratio + 0.20 * phase_strength + 0.10 * min(
        primary_signal_count, 3
    )
    confidence_score = min(1.0, confidence_score)
    if primary_signal_count < 2:
        confidence_score *= 0.75
    if available_signal_count < 3:
        confidence_score *= 0.80

    if confidence_score >= 0.75:
        confidence_level = "high"
    elif confidence_score >= 0.55:
        confidence_level = "medium"
    else:
        confidence_level = "low"

    if np.isfinite(snapshot.get("apex_position_delta_m", np.nan)) and snapshot[
        "apex_position_delta_m"
    ] > config.significant_apex_position_delta_m:
        action = (
            action
            + " Apex placement also differs from benchmark, so prioritize a repeatable entry-to-apex path."
        )
        cause = (
            cause
            + " Positioning proxy indicates measurable apex-path offset versus benchmark."
        )

    return {
        "primary_phase": primary_phase,
        "symptom": symptom,
        "likely_cause": cause,
        "recommended_action": action,
        "drill_focus": drill,
        "confidence_score": float(confidence_score),
        "confidence_level": confidence_level,
        "evidence_json": json.dumps(evidence, ensure_ascii=True),
    }


def _build_track_usage_assessment(corner_report: Dict[str, object]) -> str:
    if corner_report.get("trajectory_line_feasible", False) and corner_report.get(
        "apex_position_proxy_available", False
    ):
        return (
            "Apex position deltas versus reference are available and used as a conservative positioning proxy. "
            "Full entry/apex/exit line-shape analysis is not yet implemented, so positioning conclusions remain limited."
        )
    if corner_report.get("trajectory_line_feasible", False):
        return (
            "Lat/Lon channels are available, but robust geometric line metrics are not yet active in this run. "
            "Coaching remains centered on stronger brake/rotation/traction evidence."
        )
    return (
        "Track-usage and line-quality assessment is limited: required geometric channels are not robustly "
        "available in this run. Coaching focuses on stronger brake/rotation/traction evidence."
    )


def _format_signed(value: float, decimals: int = 3) -> str:
    if not np.isfinite(value):
        return "n/a"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.{decimals}f}"


def _build_coaching_text(
    corner_name: str,
    snapshot: Dict[str, float],
    advice: Dict[str, object],
) -> Tuple[str, str]:
    """
    Build two deterministic coaching text layers:
    1) concise coach note for quick read
    2) detailed evidence-backed note for reports
    """
    phase = str(advice["primary_phase"])
    phase_label = {
        "entry": "Entry",
        "mid": "Mid-corner",
        "exit": "Exit",
    }.get(phase, phase)

    concise = (
        f"{corner_name} [{phase_label}]: {advice['symptom']} "
        f"Action: {advice['recommended_action']}"
    )

    base_evidence = []
    if np.isfinite(snapshot.get("mean_time_loss_s", np.nan)):
        base_evidence.append(f"time loss {_format_signed(snapshot['mean_time_loss_s'], 3)} s")
    if np.isfinite(snapshot.get("inconsistency_time_s", np.nan)):
        base_evidence.append(
            f"variability {_format_signed(snapshot['inconsistency_time_s'], 3)} s"
        )
    if np.isfinite(snapshot.get("apex_speed_loss_kmh", np.nan)):
        base_evidence.append(
            f"apex speed delta {_format_signed(snapshot['apex_speed_loss_kmh'], 2)} km/h"
        )

    phase_evidence = []
    if phase == "entry":
        if np.isfinite(snapshot.get("braking_time_delta_s", np.nan)):
            phase_evidence.append(
                f"braking time delta {_format_signed(snapshot['braking_time_delta_s'], 3)} s"
            )
        if np.isfinite(snapshot.get("brake_peak_delta_pct", np.nan)):
            phase_evidence.append(
                f"brake peak delta {_format_signed(snapshot['brake_peak_delta_pct'], 1)} %"
            )
    elif phase == "mid":
        if np.isfinite(snapshot.get("rotation_time_delta_s", np.nan)):
            phase_evidence.append(
                f"rotation time delta {_format_signed(snapshot['rotation_time_delta_s'], 3)} s"
            )
        if np.isfinite(snapshot.get("yaw_delta_deg_s", np.nan)):
            phase_evidence.append(
                f"yaw delta {_format_signed(snapshot['yaw_delta_deg_s'], 2)} deg/s"
            )
        if np.isfinite(snapshot.get("steer_delta_deg", np.nan)):
            phase_evidence.append(
                f"steering delta {_format_signed(snapshot['steer_delta_deg'], 2)} deg"
            )
    else:
        if np.isfinite(snapshot.get("traction_time_delta_s", np.nan)):
            phase_evidence.append(
                f"traction time delta {_format_signed(snapshot['traction_time_delta_s'], 3)} s"
            )
        if np.isfinite(snapshot.get("traction_delay_vs_ref_pct", np.nan)):
            phase_evidence.append(
                f"throttle reapply delay {_format_signed(snapshot['traction_delay_vs_ref_pct'], 3)} % lap"
            )
        if np.isfinite(snapshot.get("exit_long_accel_delta", np.nan)):
            phase_evidence.append(
                f"exit long accel delta {_format_signed(snapshot['exit_long_accel_delta'], 3)} g"
            )

    if np.isfinite(snapshot.get("apex_position_delta_m", np.nan)):
        phase_evidence.append(
            f"apex position offset {_format_signed(snapshot['apex_position_delta_m'], 2)} m"
        )

    evidence_text = "; ".join(base_evidence + phase_evidence) or "limited telemetry evidence"
    detailed = (
        f"{corner_name} | {phase_label} priority. "
        f"Symptom: {advice['symptom']} "
        f"Likely cause: {advice['likely_cause']} "
        f"Evidence: {evidence_text}. "
        f"Recommended action: {advice['recommended_action']} "
        f"Run focus: {advice['drill_focus']} "
        f"Confidence: {advice['confidence_level']} ({float(advice['confidence_score']):.2f})."
    )
    return concise, detailed


def generate_coaching_outputs(
    corner_lap_metrics_df: pd.DataFrame,
    corner_ranking_df: pd.DataFrame,
    corner_report: Dict[str, object],
    mode_name: str,
    config: CoachingConfig = CoachingConfig(),
) -> tuple[pd.DataFrame, Dict[str, object]]:
    """
    Build deterministic corner coaching outputs from existing per-corner metrics.
    """
    if corner_lap_metrics_df.empty or corner_ranking_df.empty:
        empty = pd.DataFrame(
            columns=[
                "corner_id",
                "corner_name",
                "coaching_priority_rank",
                "priority_score",
                "mean_time_loss_s",
                "inconsistency_time_s",
                "primary_phase",
                "symptom",
                "likely_cause",
                "recommended_action",
                "drill_focus",
                "confidence_level",
                "confidence_score",
                "evidence_json",
                "coaching_summary_concise",
                "coaching_summary_detailed",
            ]
        )
        summary = {
            "mode": mode_name,
            "top_3_priorities": [],
            "main_entry_issue": "No coaching output available (insufficient corner metrics).",
            "main_mid_issue": "No coaching output available (insufficient corner metrics).",
            "main_exit_issue": "No coaching output available (insufficient corner metrics).",
            "most_inconsistent_corners": [],
            "track_usage_assessment": _build_track_usage_assessment(corner_report),
        }
        return empty, summary

    ranking = corner_ranking_df.copy()
    ranking["mean_time_loss_s"] = pd.to_numeric(ranking["mean_time_loss_s"], errors="coerce").fillna(0.0)
    ranking["inconsistency_time_s"] = pd.to_numeric(
        ranking["inconsistency_time_s"], errors="coerce"
    ).fillna(0.0)
    ranking["coaching_relevance_score"] = pd.to_numeric(
        ranking["coaching_relevance_score"], errors="coerce"
    ).fillna(0.0)

    candidate_mask = (
        (ranking["mean_time_loss_s"] >= config.min_time_loss_s)
        | (ranking["inconsistency_time_s"] >= config.min_inconsistency_s)
        | (ranking["coaching_relevance_score"] >= config.min_priority_score)
    )
    selected = ranking.loc[candidate_mask].copy()
    if selected.empty:
        selected = ranking.copy()

    selected = selected.sort_values(
        ["coaching_relevance_score", "mean_time_loss_s", "corner_id"],
        ascending=[False, False, True],
    ).head(config.max_priority_corners)

    coaching_rows: List[Dict[str, object]] = []
    for _, rank_row in selected.iterrows():
        corner_id = int(rank_row["corner_id"])
        corner_name = str(rank_row["corner_name"])
        corner_rows = corner_lap_metrics_df.loc[corner_lap_metrics_df["corner_id"] == corner_id].copy()
        snapshot = _build_corner_snapshot(corner_rows, rank_row)
        advice = _build_corner_advice(snapshot, config)
        concise_note, detailed_note = _build_coaching_text(
            corner_name=corner_name,
            snapshot=snapshot,
            advice=advice,
        )

        coaching_rows.append(
            {
                "corner_id": corner_id,
                "corner_name": corner_name,
                "coaching_priority_rank": int(len(coaching_rows) + 1),
                "priority_score": float(snapshot["priority_score"]),
                "mean_time_loss_s": float(snapshot["mean_time_loss_s"]),
                "inconsistency_time_s": float(snapshot["inconsistency_time_s"]),
                "primary_phase": advice["primary_phase"],
                "symptom": advice["symptom"],
                "likely_cause": advice["likely_cause"],
                "recommended_action": advice["recommended_action"],
                "drill_focus": advice["drill_focus"],
                "confidence_level": advice["confidence_level"],
                "confidence_score": advice["confidence_score"],
                "evidence_json": advice["evidence_json"],
                "coaching_summary_concise": concise_note,
                "coaching_summary_detailed": detailed_note,
            }
        )

    corner_coaching_df = pd.DataFrame(coaching_rows)
    if corner_coaching_df.empty:
        return generate_coaching_outputs(
            corner_lap_metrics_df=pd.DataFrame(),
            corner_ranking_df=pd.DataFrame(),
            corner_report=corner_report,
            mode_name=mode_name,
            config=config,
        )

    top_3 = corner_coaching_df.head(3)
    entry_rows = corner_coaching_df.loc[corner_coaching_df["primary_phase"] == "entry"]
    mid_rows = corner_coaching_df.loc[corner_coaching_df["primary_phase"] == "mid"]
    exit_rows = corner_coaching_df.loc[corner_coaching_df["primary_phase"] == "exit"]

    inconsistency_top = (
        ranking.sort_values(["inconsistency_time_s", "corner_id"], ascending=[False, True])
        .head(3)[["corner_name", "inconsistency_time_s"]]
        .to_dict(orient="records")
    )

    summary = {
        "mode": mode_name,
        "top_3_priorities": top_3[
            [
                "corner_name",
                "primary_phase",
                "recommended_action",
                "drill_focus",
                "confidence_level",
                "coaching_summary_concise",
            ]
        ].to_dict(orient="records"),
        "main_entry_issue": entry_rows.iloc[0]["symptom"]
        if not entry_rows.empty
        else "No dominant entry issue in current priority set.",
        "main_mid_issue": mid_rows.iloc[0]["symptom"]
        if not mid_rows.empty
        else "No dominant mid-corner issue in current priority set.",
        "main_exit_issue": exit_rows.iloc[0]["symptom"]
        if not exit_rows.empty
        else "No dominant exit issue in current priority set.",
        "most_inconsistent_corners": inconsistency_top,
        "track_usage_assessment": _build_track_usage_assessment(corner_report),
    }

    return corner_coaching_df, summary
